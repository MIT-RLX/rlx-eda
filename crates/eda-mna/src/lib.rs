//! `eda-mna` — Modified Nodal Analysis assembler.
//!
//! Wraps a list of `eda_hir::NonlinearDcBehavioral` devices into a
//! [`Circuit`], auto-generates the KCL residual graph in rlx, and
//! exposes parameter / voltage handles for downstream solvers.
//!
//! ## What's in scope (MVP)
//!
//! - `NetId` — handle for an electrical node, with `NetId::GND` reserved
//!   as the reference.
//! - `Circuit` — builder. `alloc_unknown_net()` for nets the solver
//!   iterates; `alloc_boundary_net()` for nets the user provides
//!   (typically driven by a voltage source — modeled here as a fixed
//!   boundary so we don't need a separate VoltageSource device yet).
//! - `build_residual_graph` — emits an rlx graph whose inputs are
//!   per-net voltages (one `Op::Input` named `v_<id>`) and whose
//!   outputs are KCL residuals at the unknown nets only.
//!
//! ## What's not in scope (yet)
//!
//! - Jacobian assembly via `jvp` / forward-mode AD — outer Newton is a
//!   follow-up. The MVP exposes the residual graph; callers can take
//!   gradients themselves via `grad_with_loss` / `jacfwd` for now.
//! - Voltage source devices (with branch-current unknowns) — need a
//!   richer trait shape than `NonlinearDcBehavioral` (algebraic
//!   constraint contributions). Boundary-net trick covers the common
//!   "ideal voltage source" case for now.
//! - Capacitors / inductors — transient assembly is its own crate.

use eda_hir::{MnaDevice, NonlinearDcBehavioral, TransientDelay, TransientStorage};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use std::collections::BTreeSet;

/// Electrical-node handle. Wraps an opaque u32 id; `GND` is reserved
/// (sentinel value `u32::MAX`).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct NetId(pub u32);

impl NetId {
    /// Ground / reference node. Always at 0 V; never appears as an
    /// unknown in the assembled system.
    pub const GND: NetId = NetId(u32::MAX);
    pub fn is_gnd(self) -> bool { self == Self::GND }
}

/// Branch-current handle (1 per branch unknown declared by an
/// `MnaDevice`). Used to thread the unknown into the residual graph
/// alongside net voltages.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BranchId(pub u32);

/// Delay-element handle (1 per [`TransientDelay`] device attached to a
/// circuit). Keys the per-element history buffer and the
/// `v_delayed_<id>` `Op::Input` the BE residual graph reads to stamp
/// the delayed-current contribution.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct DelayId(pub u32);

/// One device attached to a list of nets. `nets[i]` is the net that
/// terminal `i` of the device connects to. `branches` holds the branch
/// IDs the assembler allocated for this device's branch unknowns
/// (length equals `device.n_branches()`).
struct Attachment {
    device: Box<dyn MnaDevice>,
    nets: Vec<NetId>,
    branches: Vec<BranchId>,
}

/// Generic linear capacitor with a caller-chosen Param key. Where
/// `spike-divider-block::Capacitor` keys its capacitance Param via its
/// `Block::name()` (suitable for layout-bound caps), `LinearCap` lets
/// the caller supply an arbitrary string — useful for parasitic /
/// derived caps (MOSFET `Cgs` / `Cgd` / `Cdb`, interconnect
/// extraction caps, …) where the Param key is composed by the parent
/// block, not by the cap's own identity.
///
/// Skips the layout / `Block` machinery — `LinearCap` is purely a
/// transient-side element. Pair it with a behavioral device (resistor,
/// diode, MOSFET) attached to the same nets to form a full primitive.
#[derive(Clone, Debug)]
pub struct LinearCap {
    /// Param key under which the caller will set capacitance value.
    pub name: String,
}

impl LinearCap {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl eda_hir::TransientStorage for LinearCap {
    fn name(&self) -> String { self.name.clone() }
    fn capacitance(&self, g: &mut Graph) -> NodeId {
        g.param(&self.name, Shape::new(&[1], DType::F32))
    }
}

/// Generic ideal one-way transport delay. Models a dispersionless
/// waveguide segment as `i_out(t) = G · v_in(t − τ)`, with no current
/// drawn at the input terminal. Pair with a termination resistor on
/// the output side to set the steady-state gain (`v_out_ss = G · R_term
/// · v_in_ss`).
///
/// The Param key for `G` is `<name>_G` — match the `params` map you
/// pass into `transient_*` against this key.
#[derive(Clone, Debug)]
pub struct IdealDelay {
    pub name: String,
    pub tau: f64,
}

impl IdealDelay {
    pub fn new(name: impl Into<String>, tau_seconds: f64) -> Self {
        Self { name: name.into(), tau: tau_seconds }
    }
}

impl eda_hir::TransientDelay for IdealDelay {
    fn name(&self) -> String { self.name.clone() }
    fn delay_seconds(&self) -> f64 { self.tau }
    fn gain(&self, g: &mut Graph) -> NodeId {
        g.param(&format!("{}_G", self.name),
                Shape::new(&[1], DType::F32))
    }
    fn delay_param(&self, g: &mut Graph) -> NodeId {
        g.param(&format!("{}_tau", self.name),
                Shape::new(&[1], DType::F32))
    }
}

/// Adapter so any `NonlinearDcBehavioral` device drops into a `Circuit`
/// without a wrapper type. Branchless: `n_branches() = 0`,
/// `contributions` returns `(currents, [])`.
struct NonlinearDcAdapter<T: NonlinearDcBehavioral>(T);
impl<T: NonlinearDcBehavioral> MnaDevice for NonlinearDcAdapter<T> {
    fn name(&self) -> String { self.0.name() }
    fn n_terminals(&self) -> usize { self.0.n_terminals() }
    fn n_branches(&self) -> usize { 0 }
    fn contributions(
        &self,
        voltages: &[rlx_ir::NodeId],
        _branches: &[rlx_ir::NodeId],
        graph: &mut rlx_ir::Graph,
    ) -> (Vec<rlx_ir::NodeId>, Vec<rlx_ir::NodeId>) {
        (self.0.currents(voltages, graph), vec![])
    }
}

/// One storage element (capacitor today, inductor tomorrow) attached
/// to a pair of nets. Stamped as a Backward-Euler companion at each
/// transient step:
/// ```text
///   i_C(t+h) = (C/h) · ((v_a − v_b) − (v_a_prev − v_b_prev))
/// ```
/// Contributions to KCL: `−i_C` at terminal `a`, `+i_C` at terminal
/// `b` — same sign convention as `Resistor` (current flows a→b inside
/// the device when `v_a > v_b`).
struct StorageAttachment {
    device: Box<dyn TransientStorage>,
    nets: [NetId; 2],
}

/// One transport-delay element. Index in the Circuit's `delays` Vec is
/// the element's `DelayId`. Nets are `[in, out]` — the device pulls 0
/// current from `in` and pushes `G · v_in(t − τ)` into `out`.
struct DelayAttachment {
    device: Box<dyn TransientDelay>,
    nets: [NetId; 2],
}

#[derive(Default)]
pub struct Circuit {
    n_nets: u32,
    n_branches: u32,
    boundary: BTreeSet<NetId>,
    attachments: Vec<Attachment>,
    storage: Vec<StorageAttachment>,
    delays: Vec<DelayAttachment>,
}

impl Circuit {
    pub fn new() -> Self { Self::default() }

    /// Allocate a new net the solver treats as an unknown — its voltage
    /// is solved for via Newton. Returns the new `NetId`.
    pub fn alloc_unknown_net(&mut self) -> NetId {
        let id = NetId(self.n_nets);
        self.n_nets += 1;
        id
    }

    /// Allocate a new net whose voltage the user provides at runtime.
    /// Acts as an ideal voltage source's positive terminal — KCL at
    /// this node is not enforced; the user is responsible for whatever
    /// current flows in. Useful for `Vin`, `VDD`, etc.
    pub fn alloc_boundary_net(&mut self) -> NetId {
        let id = self.alloc_unknown_net();
        self.boundary.insert(id);
        id
    }

    pub fn n_nets(&self) -> u32 { self.n_nets }
    pub fn n_branches(&self) -> u32 { self.n_branches }
    pub fn n_unknowns(&self) -> usize {
        (self.n_nets as usize) - self.boundary.len() + (self.n_branches as usize)
    }
    pub fn is_boundary(&self, n: NetId) -> bool { self.boundary.contains(&n) }

    /// Attach a device to this circuit. `nets.len()` must equal the
    /// device's `n_terminals()`. Devices that need branch unknowns
    /// (voltage sources, inductors) trigger branch allocation here.
    pub fn add_device<D: NonlinearDcBehavioral + 'static>(
        &mut self,
        device: D,
        nets: &[NetId],
    ) {
        assert_eq!(
            device.n_terminals(),
            nets.len(),
            "device {} expects {} terminals, got {}",
            device.name(), device.n_terminals(), nets.len(),
        );
        self.attachments.push(Attachment {
            device: Box::new(NonlinearDcAdapter(device)),
            nets: nets.to_vec(),
            branches: Vec::new(),
        });
    }

    /// Attach a device that may carry branch-current unknowns
    /// (voltage source, inductor). The assembler allocates one
    /// `BranchId` per `device.n_branches()` and threads it into the
    /// residual graph alongside net voltages.
    pub fn add_mna_device<D: MnaDevice + 'static>(
        &mut self,
        device: D,
        nets: &[NetId],
    ) {
        assert_eq!(
            device.n_terminals(),
            nets.len(),
            "device {} expects {} terminals, got {}",
            device.name(), device.n_terminals(), nets.len(),
        );
        let n_b = device.n_branches();
        let branches: Vec<BranchId> = (0..n_b).map(|_| {
            let id = BranchId(self.n_branches);
            self.n_branches += 1;
            id
        }).collect();
        self.attachments.push(Attachment {
            device: Box::new(device),
            nets: nets.to_vec(),
            branches,
        });
    }

    /// Attach a 2-terminal linear storage device (capacitor today;
    /// inductor when we add `n_branches > 0` storage). Storage devices
    /// don't contribute to the DC residual — `solve_dc` ignores them
    /// (open-circuit at DC for caps, short for inductors). They only
    /// participate in `solve_be_step` / `transient`.
    pub fn add_storage<S: TransientStorage + 'static>(
        &mut self,
        device: S,
        nets: [NetId; 2],
    ) {
        self.storage.push(StorageAttachment {
            device: Box::new(device),
            nets,
        });
    }

    /// Number of storage devices attached. Useful for transient driver
    /// sanity checks ("did I forget to add the cap?").
    pub fn n_storage(&self) -> usize { self.storage.len() }

    /// Set of nets that appear on at least one [`TransientStorage`]
    /// terminal. These are the only nets whose **previous-step
    /// voltage** appears in the BE residual graph (via the cap stamp);
    /// every other net's `prev_voltage_input` is a dangling Op::Input
    /// that nothing consumes, so taking AD gradients WRT them errors
    /// in `rlx_opt::autodiff` ("no gradient flowed").
    ///
    /// Used by `transient_sensitivities` to filter the prev-voltage
    /// gradient set down to actually-relevant nets.
    pub fn storage_coupled_nets(&self) -> std::collections::BTreeSet<NetId> {
        let mut s = std::collections::BTreeSet::new();
        for a in &self.storage {
            for n in &a.nets {
                if !n.is_gnd() { s.insert(*n); }
            }
        }
        s
    }

    /// Attach a 2-terminal transport-delay element. `nets` is `[in,
    /// out]`. Returns the element's [`DelayId`] — needed to wire up the
    /// integrator's per-element history buffer and to read back delayed
    /// values during a transient run.
    pub fn add_delay<D: TransientDelay + 'static>(
        &mut self,
        device: D,
        nets: [NetId; 2],
    ) -> DelayId {
        let id = DelayId(self.delays.len() as u32);
        self.delays.push(DelayAttachment {
            device: Box::new(device),
            nets,
        });
        id
    }

    pub fn n_delays(&self) -> usize { self.delays.len() }

    /// Look up delay parameters by `DelayId`. Returned tuple is
    /// `(in_net, out_net, τ_seconds)`. Used by the integrator to size
    /// history buffers and pick the right Op::Input feed.
    pub fn delay_info(&self, id: DelayId) -> (NetId, NetId, f64) {
        let att = &self.delays[id.0 as usize];
        (att.nets[0], att.nets[1], att.device.delay_seconds())
    }
}

/// Canonical Op::Input names for a delay element's per-step inputs to
/// the unified blend stamp:
///
/// - `v_lo_<id>` / `v_hi_<id>` — surrounding history samples (long
///   delays) or unused (sub-step; integrator passes `0.0`).
/// - `delay_blend_<id>`        — `1.0` for long delays, `0.0` for
///   sub-step. Selects between `(v_lo_hist, v_hi_hist)` and
///   `(v_in_prev, v_in_now)` inside the graph.
/// - `delay_offset_<id>`       — integer `i + 1` where `i = floor(τ/dt)`
///   for long delays; `1.0` for sub-step. Combines with the `<name>_tau`
///   Param to form `α = offset − τ/h` (the in-graph interpolation
///   weight).
pub fn delay_v_lo_name(id: DelayId)   -> String { format!("v_lo_{}",   id.0) }
pub fn delay_v_hi_name(id: DelayId)   -> String { format!("v_hi_{}",   id.0) }
pub fn delay_blend_name(id: DelayId)  -> String { format!("delay_blend_{}",  id.0) }
pub fn delay_offset_name(id: DelayId) -> String { format!("delay_offset_{}", id.0) }

/// Emitted artifacts of [`build_residual_graph`].
pub struct ResidualGraph {
    /// The graph itself. Inputs:
    /// - `v_<id>` for every allocated net (boundary + unknown)
    /// - `i_b<id>` for every allocated branch unknown
    ///
    /// Outputs (concatenated, in order):
    /// - KCL residuals at unknown nets (one per `unknown_nets`)
    /// - Branch residuals (one per `branches`)
    pub graph: Graph,
    pub unknown_nets: Vec<NetId>,
    pub all_nets: Vec<NetId>,
    /// Branch IDs in output-vector order. Branch residuals appear after
    /// KCL residuals in `graph.outputs`.
    pub branches: Vec<BranchId>,
}

/// Assemble the residual graph from `circuit`'s devices.
///
/// Each non-ground net allocates one `Op::Input` named `v_<id>` of
/// shape `[1]` f32. Each device's `currents` contributions are summed
/// into per-net accumulators. Final outputs are residuals at the
/// **unknown** (non-boundary, non-ground) nets — KCL says these must
/// be zero at the DC operating point.
pub fn build_residual_graph(circuit: &Circuit) -> ResidualGraph {
    let mut g = Graph::new("mna_residual");
    let s = Shape::new(&[1], DType::F32);

    let zero_const = g.add_node(
        Op::Constant { data: 0.0_f32.to_le_bytes().to_vec() },
        vec![],
        s.clone(),
    );

    // Per-net Op::Input. GND is implicit at 0 V.
    let mut all_nets: Vec<NetId> = (0..circuit.n_nets).map(NetId).collect();
    all_nets.sort();
    let voltage_inputs: Vec<NodeId> = all_nets.iter().map(|n| {
        g.input(format!("v_{}", n.0), s.clone())
    }).collect();

    // Per-branch Op::Input.
    let branches: Vec<BranchId> = (0..circuit.n_branches).map(BranchId).collect();
    let branch_inputs: Vec<NodeId> = branches.iter().map(|b| {
        g.input(format!("i_b{}", b.0), s.clone())
    }).collect();

    let mut acc: Vec<NodeId> = vec![zero_const; circuit.n_nets as usize];
    let mut branch_residuals_out: Vec<NodeId> = Vec::new();

    for att in &circuit.attachments {
        let voltages: Vec<NodeId> = att.nets.iter().map(|n| {
            if n.is_gnd() { zero_const } else { voltage_inputs[n.0 as usize] }
        }).collect();
        let branches: Vec<NodeId> = att.branches.iter()
            .map(|b| branch_inputs[b.0 as usize])
            .collect();

        let (terminal_currents, branch_residuals) =
            att.device.contributions(&voltages, &branches, &mut g);
        debug_assert_eq!(terminal_currents.len(), att.nets.len());
        debug_assert_eq!(branch_residuals.len(), att.branches.len());

        // Stamp terminal currents into KCL accumulators.
        for (t_idx, &net) in att.nets.iter().enumerate() {
            if net.is_gnd() { continue; }
            acc[net.0 as usize] = g.binary(
                BinaryOp::Add,
                acc[net.0 as usize],
                terminal_currents[t_idx],
                s.clone(),
            );
        }

        // Branch residuals stamp 1:1 into the branch-equation row.
        for (k, &b_id) in att.branches.iter().enumerate() {
            // Branches list ordering: each branch produces one residual.
            // We rely on att.branches being in the same order as the
            // per-device branch index → match by enumerate order.
            let _ = b_id;
            branch_residuals_out.push(branch_residuals[k]);
        }
    }

    // ── DC delay stamp: zero-delay buffer i_out = G · v_in ──
    // At DC all derivatives vanish, so v_in(t − τ) = v_in(t). The
    // delay element collapses to an instantaneous buffer, making
    // `solve_dc` correctly converge with delay loops in topology
    // (waveguide rings, optical resonators).
    for dl in &circuit.delays {
        let [in_net, out_net] = dl.nets;
        let v_in = if in_net.is_gnd() {
            zero_const
        } else {
            voltage_inputs[in_net.0 as usize]
        };
        let gain = dl.device.gain(&mut g);
        let i_inj = g.binary(BinaryOp::Mul, gain, v_in, s.clone());
        if !out_net.is_gnd() {
            acc[out_net.0 as usize] = g.binary(
                BinaryOp::Add, acc[out_net.0 as usize], i_inj, s.clone(),
            );
        }
    }

    // KCL residuals at unknown nets (in NetId order, matching unknown_nets),
    // followed by branch residuals.
    let unknown_nets: Vec<NetId> = all_nets
        .iter()
        .copied()
        .filter(|n| !circuit.is_boundary(*n))
        .collect();
    let mut outputs: Vec<NodeId> = unknown_nets
        .iter()
        .map(|n| acc[n.0 as usize])
        .collect();
    outputs.extend(&branch_residuals_out);

    // Phantom dependency: rlx's `grad_with_loss` panics when a wrt
    // doesn't appear in the forward graph. Some residuals — branch
    // constraints, KCL at nets without a branch attached — don't
    // syntactically depend on every unknown. Adding `0 · sum(...)` to
    // each residual gives a numerically-trivial but graph-visible
    // dependency on every unknown voltage and branch current, so the
    // VJP walk produces a (correct) zero entry instead of panicking.
    let zero_coef = g.add_node(
        Op::Constant { data: 0.0_f32.to_le_bytes().to_vec() },
        vec![], s.clone(),
    );
    let unknown_voltage_nodes: Vec<NodeId> = unknown_nets
        .iter()
        .map(|n| voltage_inputs[n.0 as usize])
        .collect();
    let mut all_unknowns: Vec<NodeId> = unknown_voltage_nodes;
    all_unknowns.extend_from_slice(&branch_inputs);
    if !all_unknowns.is_empty() {
        let mut phantom_sum = all_unknowns[0];
        for &id in &all_unknowns[1..] {
            phantom_sum = g.binary(BinaryOp::Add, phantom_sum, id, s.clone());
        }
        let phantom = g.binary(BinaryOp::Mul, zero_coef, phantom_sum, s.clone());
        for o in outputs.iter_mut() {
            *o = g.binary(BinaryOp::Add, *o, phantom, s.clone());
        }
    }
    g.set_outputs(outputs);

    ResidualGraph { graph: g, unknown_nets, all_nets, branches }
}

/// Vectorize the residual graph along a leading batch axis of size
/// `n_draws`. Every per-draw input — net voltages, branch currents,
/// previous-step voltages, delay history slots — gets a leading
/// `[n_draws, ...]` dim; everything else (Param values like
/// resistances, capacitances, gains, the BE step `h`) stays shared
/// across the batch.
///
/// ## Use case
///
/// Monte Carlo sweeps and corner-batched DC where N independent
/// circuits share topology + device kinds but differ in random Param
/// realisations. After this call, the batched residual evaluates all
/// N draws in one graph dispatch — feeds straight into
/// `Op::BatchedDenseSolve` once K and b are extracted (Newton step or
/// direct linearisation).
///
/// ## Contract
///
/// Input: a scalar `Circuit` with the device set you want to MC.
/// Output: a `ResidualGraph` whose internal graph has `[n_draws, 1]`
/// for every per-draw input and whose outputs are `[n_draws, 1]` per
/// unknown residual. The `unknown_nets`, `all_nets`, `branches` index
/// metadata stays unchanged — callers identify outputs by the same
/// `NetId` / `BranchId` ordering as the scalar graph.
///
/// Built on top of `rlx_opt::vmap`, which has shape rules for every
/// op `build_residual_graph` emits (Input, Param, Constant, Binary,
/// Reduce). Adding new device kinds that emit unfamiliar ops is the
/// only thing that requires extending the vmap rule set.
pub fn build_batched_residual_graph(
    circuit: &Circuit,
    n_draws: usize,
) -> ResidualGraph {
    build_batched_residual_graph_with_mc_params(circuit, n_draws, &[])
}

/// Same as [`build_batched_residual_graph`] but also promotes a list
/// of `Op::Param` names into per-draw `Op::Input`s before vmap'ing.
///
/// Use this when Monte Carlo varies device parameters (Vth mismatch,
/// resistor tolerance, mobility variation) instead of, or in addition
/// to, boundary voltages. Each promoted name appears in the resulting
/// graph as a batched input (`[n_draws, ...]` shape); callers bind
/// per-draw values via the run-time input map rather than `set_param`.
pub fn build_batched_residual_graph_with_mc_params(
    circuit: &Circuit,
    n_draws: usize,
    mc_param_names: &[&str],
) -> ResidualGraph {
    let scalar_orig = build_residual_graph(circuit);

    // Promote any param names the caller wants to MC over into Inputs
    // before vmap. After this, vmap sees them as Inputs and batches
    // them on equal footing with v_<id>.
    let scalar_graph = if mc_param_names.is_empty() {
        scalar_orig.graph
    } else {
        rlx_opt::promote_params_to_inputs(&scalar_orig.graph, mc_param_names)
    };
    let scalar = ResidualGraph {
        graph: scalar_graph,
        unknown_nets: scalar_orig.unknown_nets,
        all_nets: scalar_orig.all_nets,
        branches: scalar_orig.branches,
    };

    // Collect the per-draw input names. Anything in the scalar graph
    // whose value varies per Monte-Carlo draw goes here — the rest
    // (Op::Param like R values, gain G, capacitance C, the BE step h
    // *that the caller didn't ask to MC over*) stays shared. Storage /
    // delay history slots aren't named in `ResidualGraph`'s metadata
    // directly, so probe the actual graph for any input name matching
    // the conventional prefixes.
    use std::collections::HashSet;
    let mut wanted: HashSet<String> = HashSet::new();
    for n in &scalar.all_nets {
        wanted.insert(net_input_name(*n));
        wanted.insert(prev_voltage_input_name(*n));
    }
    for b in &scalar.branches {
        wanted.insert(branch_input_name(*b));
    }
    for name in mc_param_names {
        wanted.insert((*name).to_string());
    }
    // Best-effort coverage of delay history: scan actual inputs for
    // anything matching the prefixes from `delay_*_name()` so callers
    // don't have to remember to thread DelayId lists through.
    for node in scalar.graph.nodes() {
        if let rlx_ir::Op::Input { name } = &node.op {
            if name.starts_with("v_lo_")
                || name.starts_with("v_hi_")
                || name.starts_with("delay_blend_")
                || name.starts_with("delay_offset_")
            {
                wanted.insert(name.clone());
            }
        }
    }

    // Filter to inputs that actually exist in the graph (vmap errors
    // on a name it can't find), preserving the user's intent without
    // requiring the caller to track which inputs the assembler emits.
    let actual: HashSet<String> = scalar.graph.nodes()
        .iter()
        .filter_map(|n| match &n.op {
            rlx_ir::Op::Input { name } => Some(name.clone()),
            _ => None,
        })
        .collect();
    let names_owned: Vec<String> = wanted.into_iter()
        .filter(|n| actual.contains(n))
        .collect();
    let names: Vec<&str> = names_owned.iter().map(|s| s.as_str()).collect();

    let batched = rlx_opt::vmap::vmap(&scalar.graph, &names, n_draws);

    ResidualGraph {
        graph: batched,
        unknown_nets: scalar.unknown_nets,
        all_nets: scalar.all_nets,
        branches: scalar.branches,
    }
}

/// Helper for callers: produce the canonical `Op::Input` name for a net.
pub fn net_input_name(net: NetId) -> String { format!("v_{}", net.0) }

/// Canonical `Op::Input` name for a branch-current unknown.
pub fn branch_input_name(b: BranchId) -> String { format!("i_b{}", b.0) }

/// Canonical Op::Input name for the *previous-step* voltage at a net,
/// used by the BE-step residual graph.
pub fn prev_voltage_input_name(net: NetId) -> String { format!("v_prev_{}", net.0) }

/// Canonical Op::Input name for the timestep `h` (BE-step).
pub const TIMESTEP_INPUT_NAME: &str = "h";

/// BFS over the input-DAG to collect every node reachable via input
/// edges from `root` (inclusive). Used by `transient_sensitivities` to
/// filter its per-row wrt list down to the subset of params/inputs
/// that the row's residual actually depends on — `grad_with_loss`
/// panics on disconnected wrt entries.
fn ancestors_of(g: &rlx_ir::Graph, root: rlx_ir::NodeId) -> std::collections::HashSet<rlx_ir::NodeId> {
    let mut seen: std::collections::HashSet<rlx_ir::NodeId> = std::collections::HashSet::new();
    let mut stack: Vec<rlx_ir::NodeId> = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) { continue; }
        for inp in &g.node(id).inputs {
            if !seen.contains(inp) { stack.push(*inp); }
        }
    }
    seen
}

// ── Transient (Backward-Euler step) ───────────────────────────────────

/// Build the BE-step residual graph: same structure as
/// [`build_residual_graph`] (DC), plus per-storage companion-current
/// contributions to KCL at each terminal.
///
/// New Inputs the caller must provide on every `run`:
/// - `v_prev_<id>` for every net (just gnd → 0; boundary → the same
///   boundary value as this step; unknown → the previous step's
///   solution). Built by [`build_be_step_inputs`].
/// - `h` — the timestep size in seconds.
///
/// New Params (one per storage device): `<storage.name>_C`, contributed
/// by `TransientStorage::capacitance`.
///
/// Each storage device adds these companion-current terms to the per-
/// net KCL accumulators:
/// ```text
///   i_C(t+h) = (C/h) · ((v_a − v_b) − (v_a_prev − v_b_prev))
///   acc[a]  -=  i_C        (device pulls i_C from terminal a)
///   acc[b]  +=  i_C        (device pushes i_C to terminal b)
/// ```
/// At DC steady state (`v_a = v_a_prev`, etc.) this term is zero —
/// transient and DC residuals coincide, which is the whole point of
/// the companion-stamp formulation.
pub fn build_be_step_residual_graph(circuit: &Circuit) -> ResidualGraph {
    let mut g = Graph::new("mna_be_step_residual");
    let s = Shape::new(&[1], DType::F32);

    let zero_const = g.add_node(
        Op::Constant { data: 0.0_f32.to_le_bytes().to_vec() },
        vec![],
        s.clone(),
    );

    // Per-net Op::Input (this-step voltages).
    let mut all_nets: Vec<NetId> = (0..circuit.n_nets).map(NetId).collect();
    all_nets.sort();
    let voltage_inputs: Vec<NodeId> = all_nets.iter().map(|n| {
        g.input(format!("v_{}", n.0), s.clone())
    }).collect();

    // Per-net previous-step voltage Op::Input.
    let prev_voltage_inputs: Vec<NodeId> = all_nets.iter().map(|n| {
        g.input(prev_voltage_input_name(*n), s.clone())
    }).collect();

    // Timestep h.
    let h_input = g.input(TIMESTEP_INPUT_NAME, s.clone());

    // Per-branch Op::Input.
    let branches: Vec<BranchId> = (0..circuit.n_branches).map(BranchId).collect();
    let branch_inputs: Vec<NodeId> = branches.iter().map(|b| {
        g.input(format!("i_b{}", b.0), s.clone())
    }).collect();

    let mut acc: Vec<NodeId> = vec![zero_const; circuit.n_nets as usize];
    let mut branch_residuals_out: Vec<NodeId> = Vec::new();

    // ── DC contributions (identical to build_residual_graph) ──
    for att in &circuit.attachments {
        let voltages: Vec<NodeId> = att.nets.iter().map(|n| {
            if n.is_gnd() { zero_const } else { voltage_inputs[n.0 as usize] }
        }).collect();
        let branches: Vec<NodeId> = att.branches.iter()
            .map(|b| branch_inputs[b.0 as usize])
            .collect();

        let (terminal_currents, branch_residuals) =
            att.device.contributions(&voltages, &branches, &mut g);
        debug_assert_eq!(terminal_currents.len(), att.nets.len());
        debug_assert_eq!(branch_residuals.len(), att.branches.len());

        for (t_idx, &net) in att.nets.iter().enumerate() {
            if net.is_gnd() { continue; }
            acc[net.0 as usize] = g.binary(
                BinaryOp::Add,
                acc[net.0 as usize],
                terminal_currents[t_idx],
                s.clone(),
            );
        }
        for (k, _b) in att.branches.iter().enumerate() {
            branch_residuals_out.push(branch_residuals[k]);
        }
    }

    // ── Storage contributions (BE-step companions) ──
    let neg_one = g.add_node(
        Op::Constant { data: (-1.0_f32).to_le_bytes().to_vec() },
        vec![], s.clone(),
    );
    for st in &circuit.storage {
        let [a, b] = st.nets;
        let v_a = if a.is_gnd() { zero_const } else { voltage_inputs[a.0 as usize] };
        let v_b = if b.is_gnd() { zero_const } else { voltage_inputs[b.0 as usize] };
        let v_a_prev = if a.is_gnd() { zero_const } else { prev_voltage_inputs[a.0 as usize] };
        let v_b_prev = if b.is_gnd() { zero_const } else { prev_voltage_inputs[b.0 as usize] };

        let c_node = st.device.capacitance(&mut g);
        let c_over_h = g.binary(BinaryOp::Div, c_node, h_input, s.clone());
        let v_diff      = g.binary(BinaryOp::Sub, v_a,      v_b,      s.clone());
        let v_diff_prev = g.binary(BinaryOp::Sub, v_a_prev, v_b_prev, s.clone());
        let dv          = g.binary(BinaryOp::Sub, v_diff, v_diff_prev, s.clone());
        // i_C flows a→b inside the cap when v_a > v_b (and rising).
        let i_c = g.binary(BinaryOp::Mul, c_over_h, dv, s.clone());
        let neg_i_c = g.binary(BinaryOp::Mul, neg_one, i_c, s.clone());

        if !a.is_gnd() {
            acc[a.0 as usize] =
                g.binary(BinaryOp::Add, acc[a.0 as usize], neg_i_c, s.clone());
        }
        if !b.is_gnd() {
            acc[b.0 as usize] =
                g.binary(BinaryOp::Add, acc[b.0 as usize], i_c, s.clone());
        }
    }

    // ── Delay contributions (DDE; unified sub-step + long-delay stamp) ──
    //
    //   v_delayed = (1 − α) · v_lo_use  +  α · v_hi_use
    //   α         = offset − τ_param / h
    //
    //   v_lo_use  = (1 − blend) · v_in_prev + blend · v_lo_hist
    //   v_hi_use  = (1 − blend) · v_in_now  + blend · v_hi_hist
    //
    // Sub-step delays (τ < dt) set blend = 0, offset = 1, so v_lo_use →
    // v_in_prev and v_hi_use → v_in_now. AD wrt the in-graph
    // `voltage_inputs[in_net]` correctly couples the live Newton iterate
    // through the Jacobian.
    //
    // Long delays (τ ≥ dt) set blend = 1 and offset = floor(τ/dt) + 1;
    // the (v_lo_hist, v_hi_hist) Op::Inputs are the two surrounding
    // history-buffer samples (linearly interpolated by α).
    //
    // ∂α/∂τ = −1/h flows through the τ_param Param, so AD wrt
    // `<name>_tau` is well-defined inside any integer-step window
    // (offset is held constant for the duration of one solve_be_step).
    let one_const = g.add_node(
        Op::Constant { data: 1.0_f32.to_le_bytes().to_vec() },
        vec![], s.clone(),
    );
    for (idx, dl) in circuit.delays.iter().enumerate() {
        let id = DelayId(idx as u32);
        let [in_net, out_net] = dl.nets;
        let v_in_now  = if in_net.is_gnd() { zero_const }
                        else { voltage_inputs[in_net.0 as usize] };
        let v_in_prev = if in_net.is_gnd() { zero_const }
                        else { prev_voltage_inputs[in_net.0 as usize] };

        let v_lo_hist = g.input(delay_v_lo_name(id),   s.clone());
        let v_hi_hist = g.input(delay_v_hi_name(id),   s.clone());
        let blend     = g.input(delay_blend_name(id),  s.clone());
        let offset    = g.input(delay_offset_name(id), s.clone());
        let tau_param = dl.device.delay_param(&mut g);
        let gain      = dl.device.gain(&mut g);

        let one_m_blend = g.binary(BinaryOp::Sub, one_const, blend, s.clone());
        let lo_a = g.binary(BinaryOp::Mul, one_m_blend, v_in_prev, s.clone());
        let lo_b = g.binary(BinaryOp::Mul, blend,       v_lo_hist, s.clone());
        let v_lo_use = g.binary(BinaryOp::Add, lo_a, lo_b, s.clone());
        let hi_a = g.binary(BinaryOp::Mul, one_m_blend, v_in_now,  s.clone());
        let hi_b = g.binary(BinaryOp::Mul, blend,       v_hi_hist, s.clone());
        let v_hi_use = g.binary(BinaryOp::Add, hi_a, hi_b, s.clone());

        let tau_over_h = g.binary(BinaryOp::Div, tau_param, h_input, s.clone());
        let alpha      = g.binary(BinaryOp::Sub, offset, tau_over_h, s.clone());
        let one_m_a    = g.binary(BinaryOp::Sub, one_const, alpha, s.clone());

        let vd_a = g.binary(BinaryOp::Mul, one_m_a, v_lo_use, s.clone());
        let vd_b = g.binary(BinaryOp::Mul, alpha,   v_hi_use, s.clone());
        let v_delayed = g.binary(BinaryOp::Add, vd_a, vd_b, s.clone());

        let i_inj = g.binary(BinaryOp::Mul, gain, v_delayed, s.clone());
        if !out_net.is_gnd() {
            acc[out_net.0 as usize] = g.binary(
                BinaryOp::Add, acc[out_net.0 as usize], i_inj, s.clone(),
            );
        }
    }

    // Outputs: KCL at unknown nets, then branch residuals.
    let unknown_nets: Vec<NetId> = all_nets
        .iter()
        .copied()
        .filter(|n| !circuit.is_boundary(*n))
        .collect();
    let mut outputs: Vec<NodeId> = unknown_nets
        .iter()
        .map(|n| acc[n.0 as usize])
        .collect();
    outputs.extend(&branch_residuals_out);

    // Phantom-dependency trick — same rationale as DC.
    let zero_coef = g.add_node(
        Op::Constant { data: 0.0_f32.to_le_bytes().to_vec() },
        vec![], s.clone(),
    );
    let unknown_voltage_nodes: Vec<NodeId> = unknown_nets
        .iter()
        .map(|n| voltage_inputs[n.0 as usize])
        .collect();
    let mut all_unknowns: Vec<NodeId> = unknown_voltage_nodes;
    all_unknowns.extend_from_slice(&branch_inputs);
    if !all_unknowns.is_empty() {
        let mut phantom_sum = all_unknowns[0];
        for &id in &all_unknowns[1..] {
            phantom_sum = g.binary(BinaryOp::Add, phantom_sum, id, s.clone());
        }
        let phantom = g.binary(BinaryOp::Mul, zero_coef, phantom_sum, s.clone());
        for o in outputs.iter_mut() {
            *o = g.binary(BinaryOp::Add, *o, phantom, s.clone());
        }
    }
    g.set_outputs(outputs);

    ResidualGraph { graph: g, unknown_nets, all_nets, branches }
}

// ── DC Newton solver ──────────────────────────────────────────────────

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct DcOperatingPoint {
    /// Solved voltage for every allocated net (boundary + unknown).
    pub voltages: HashMap<NetId, f32>,
    /// Solved current through every branch unknown (one per
    /// branch declared by an `MnaDevice` like `VoltageSource`).
    pub branch_currents: HashMap<BranchId, f32>,
    pub iters: usize,
    pub converged: bool,
    /// Final L∞ residual across all unknowns — `< tol` ⇒ `converged`.
    pub final_residual_max: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct NewtonOptions {
    pub max_iters: usize,
    /// Absolute tolerance on the residual (KCL units — amps for current
    /// stamps, dimensionless for branch-equation residuals).
    /// Default 1e-7. **Not sufficient on its own** — see `vntol`.
    pub tol: f32,
    /// Voltage-step tolerance: Newton is converged only when the
    /// most-recent step's max |Δv| over all unknowns is below `vntol`
    /// **AND** the residual is below `tol`. Mirrors SPICE's
    /// `vntol`/`reltol` convention. Default 1e-6 V (1 µV).
    ///
    /// ## Why both checks
    ///
    /// Residual-only convergence silently under-converges on circuits
    /// where the typical residual scale exceeds `tol`. A 1:4 NMOS
    /// mirror at 5 µA reference has KCL residuals in nA at the right
    /// answer — a residual of 35 nA is "small" relative to 5 µA but
    /// would pass an `abstol=1e-7` check, leaving Newton at the
    /// initial guess with v unchanged. Requiring `Δv < vntol` forces
    /// at least one Newton step and catches the case where Newton
    /// would have moved v by more than vntol if it had iterated.
    pub vntol: f32,
    /// Initial guess for every unknown net. The diode test uses
    /// 0.6 to land in the basin where Newton converges from above —
    /// real SPICE uses node-iteration ramping or limiting, but for our
    /// MVP a single shared initial guess covers the simple cases.
    pub init: f32,
    /// Maximum number of backtracking halvings inside a single Newton
    /// iteration. After computing `dv = -J⁻¹·f`, we try the full step
    /// `α = 1`, evaluate the new residual, and if `||f_new||∞ ≥
    /// ||f||∞` (or `f_new` contains NaN/Inf) we halve `α` and retry,
    /// up to `max_backtracks` times. After that, we accept the
    /// smallest tried step regardless. Setting this to 0 disables
    /// damping (pure Newton) — useful for diagnosing convergence
    /// behavior on circuits where bare Newton is known to converge.
    pub max_backtracks: usize,
}
impl Default for NewtonOptions {
    fn default() -> Self {
        Self {
            max_iters: 50,
            tol: 1e-7,
            vntol: 1e-6,
            init: 0.6,
            max_backtracks: 10,
        }
    }
}

/// Solve a `Circuit`'s DC operating point via Newton iteration.
///
/// `params` maps device-`Param` names (`Block::name()` or `<name>_Is` for
/// diodes) to numeric values. `boundary_voltages` maps each boundary
/// `NetId` to its fixed voltage.
///
/// Implementation: per iteration, evaluate the residual graph + N
/// reverse-mode gradient graphs (one per unknown), assemble Jacobian
/// `J ∈ ℝ^(N×N)`, solve `J·dv = −f` via in-house Gauss-Jordan, update.
/// Returns when `‖f‖_∞ < tol` or after `max_iters`.
pub fn solve_dc(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    boundary_voltages: &HashMap<NetId, f32>,
    opt: NewtonOptions,
) -> DcOperatingPoint {
    let rg = build_residual_graph(circuit);
    let unknowns = rg.unknown_nets.clone();
    let branches = rg.branches.clone();
    let n_v = unknowns.len();
    let n_b = branches.len();
    let n = n_v + n_b;

    // Combined unknown wrt list: voltages first, then branch currents.
    let mut unknown_input_ids: Vec<rlx_ir::NodeId> = unknowns
        .iter()
        .map(|net| find_input_node(&rg.graph, &net_input_name(*net))
            .expect("residual graph missing unknown-net Op::Input"))
        .collect();
    for b in &branches {
        unknown_input_ids.push(
            find_input_node(&rg.graph, &branch_input_name(*b))
                .expect("residual graph missing branch Op::Input"),
        );
    }

    use rlx_runtime::{Device, Session};
    let session = Session::new(Device::Cpu);
    let mut compiled_res = session.compile(rg.graph.clone());

    // Per-residual-row reverse-mode graphs. Each output of `rg.graph`
    // (KCL-at-unknown-net, then branch-residual) becomes a separate
    // restricted graph + grad_with_loss.
    let mut compiled_jac_rows: Vec<rlx_runtime::CompiledGraph> = Vec::with_capacity(n);
    for i in 0..n {
        let mut g_i = rg.graph.clone();
        let out_i = g_i.outputs[i];
        g_i.set_outputs(vec![out_i]);
        let bwd = rlx_opt::autodiff::grad_with_loss(&g_i, &unknown_input_ids);
        compiled_jac_rows.push(session.compile(bwd));
    }

    // Set device params on every compiled graph.
    let set_all = |g: &mut rlx_runtime::CompiledGraph| {
        for (k, v) in params {
            g.set_param(k, &[*v]);
        }
    };
    set_all(&mut compiled_res);
    for g in compiled_jac_rows.iter_mut() {
        set_all(g);
    }

    // Newton iteration with Armijo-style backtracking line search.
    // Each iteration: compute Newton direction `dv`; try the full step
    // (α=1); if the resulting residual didn't decrease (or contains
    // NaN/Inf — typical f32 overflow when MOSFET softplus arguments
    // get too large after a Newton overshoot), halve α and retry up
    // to `opt.max_backtracks` times. Falls back to `α = 2^-K` once
    // backtracks are exhausted.
    let eval_residual = |v: &[f32], compiled_res: &mut rlx_runtime::CompiledGraph|
        -> (Vec<f32>, f32)
    {
        let inputs = build_inputs(
            &rg.all_nets, &unknowns, &branches, boundary_voltages,
            &v[..n_v], &v[n_v..],
        );
        let inputs_ref: Vec<(&str, &[f32])> = inputs.iter()
            .map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        let f_outs = compiled_res.run(&inputs_ref);
        let f: Vec<f32> = f_outs.iter().map(|o| o[0]).collect();
        let max_abs = f.iter().fold(0.0_f32, |acc, &x| {
            if x.is_finite() { acc.max(x.abs()) } else { f32::INFINITY }
        });
        (f, max_abs)
    };

    let mut v: Vec<f32> = vec![opt.init; n];
    let (mut f, mut last_max) = eval_residual(&v, &mut compiled_res);
    let mut converged_at: Option<usize> = None;
    // Step size from the previous Newton iter — initialized to +∞ so
    // iter 0 cannot exit early. See `NewtonOptions::vntol` docs for the
    // mosfet-mirror false-converge case this guards against.
    let mut last_step_max = f32::INFINITY;

    for iter in 0..opt.max_iters {
        // Convergence requires both residual < tol AND last Newton
        // step < vntol — see NewtonOptions docs.
        if last_max < opt.tol && last_step_max < opt.vntol {
            converged_at = Some(iter);
            break;
        }

        // Evaluate Jacobian rows at the current v.
        let inputs = build_inputs(
            &rg.all_nets, &unknowns, &branches, boundary_voltages,
            &v[..n_v], &v[n_v..],
        );
        let inputs_ref: Vec<(&str, &[f32])> = inputs.iter()
            .map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        let one_seed = [1.0_f32];
        let mut jac = vec![0.0_f32; n * n];
        for i in 0..n {
            let mut grad_inputs = inputs_ref.clone();
            grad_inputs.push(("d_output", &one_seed[..]));
            let outs = compiled_jac_rows[i].run(&grad_inputs);
            for j in 0..n {
                jac[i * n + j] = outs[1 + j][0];
            }
        }

        // Solve J·dv = −f.
        let neg_f: Vec<f32> = f.iter().map(|x| -x).collect();
        let dv = match linear_solve(&jac, &neg_f, n) {
            Some(dv) => dv,
            None => break,    // singular Jacobian; give up
        };

        // Backtracking line search.
        let mut alpha = 1.0_f32;
        let mut v_trial = vec![0.0_f32; n];
        let (mut f_new, mut max_new) = (Vec::new(), f32::INFINITY);
        for _ in 0..=opt.max_backtracks {
            for i in 0..n { v_trial[i] = v[i] + alpha * dv[i]; }
            let (f_t, max_t) = eval_residual(&v_trial, &mut compiled_res);
            if max_t.is_finite() && max_t < last_max {
                f_new = f_t;
                max_new = max_t;
                break;
            }
            alpha *= 0.5;
            f_new = f_t;
            max_new = max_t;
        }

        // Compute step size before commit (for next iter's convergence check).
        let mut step_max = 0.0_f32;
        for i in 0..n {
            let dv_i = (v_trial[i] - v[i]).abs();
            if dv_i > step_max { step_max = dv_i; }
        }
        last_step_max = step_max;

        // Accept the trial step (whatever α landed on).
        v = v_trial;
        f = f_new;
        last_max = max_new;
    }

    let iters = converged_at.unwrap_or(opt.max_iters);
    let mut voltages = HashMap::new();
    for (idx, net) in unknowns.iter().enumerate() {
        voltages.insert(*net, v[idx]);
    }
    for (net, val) in boundary_voltages {
        voltages.insert(*net, *val);
    }
    let mut branch_currents = HashMap::new();
    for (idx, b) in branches.iter().enumerate() {
        branch_currents.insert(*b, v[n_v + idx]);
    }
    DcOperatingPoint {
        voltages,
        branch_currents,
        iters,
        converged: converged_at.is_some(),
        final_residual_max: last_max,
    }
}

pub mod linear_scan;
pub use linear_scan::{
    build_linear_be_step_body, build_linear_be_step_body_with_mc_params,
    build_nonlinear_be_step_body, build_nonlinear_scan_body,
    LinearBeStepBody,
};
pub mod ac;
pub use ac::{build_ac_response_graph, AcResponseGraph};

// ── Batched DC Newton (vmap'd residual + per-draw convergence) ────────

/// Batched twin of [`DcOperatingPoint`].
///
/// Each per-net entry is a `Vec<f32>` of length `n_draws` — one solved
/// voltage per Monte-Carlo / corner draw. `converged` and
/// `final_residual_max` are per-draw too. `iters` is the global iter
/// count Newton ran for (the loop terminates only once *every* draw
/// has converged, or `max_iters` is hit).
#[derive(Debug, Clone)]
pub struct BatchedDcOperatingPoint {
    pub voltages: HashMap<NetId, Vec<f32>>,
    pub branch_currents: HashMap<BranchId, Vec<f32>>,
    pub iters: usize,
    pub converged: Vec<bool>,
    pub final_residual_max: Vec<f32>,
}

/// Solve a `Circuit`'s DC operating point for `n_draws` independent
/// per-draw boundary conditions in one batched Newton run.
///
/// `params` is shared across the batch (Resistor R, Diode Is, etc. are
/// all global to this MVP — making per-draw Params requires lifting
/// vmap's "params are shared" rule, which is a follow-up). Per-draw
/// variation comes through `boundary_voltages` — each `NetId` key maps
/// to a `Vec<f32>` of length `n_draws`. Net IDs not in the map default
/// to 0.0 V across all draws.
///
/// Convergence is tracked per draw: once a draw's L∞ residual drops
/// below `opt.tol` it gets pinned (`v` no longer updates for that
/// draw, even though the rest of the batch continues iterating). The
/// loop exits once every draw has either converged or `max_iters`
/// iterations have run.
///
/// Implementation:
///   * residual graph: `build_batched_residual_graph` (vmap'd from
///     scalar `build_residual_graph`).
///   * jacobian rows: per output i of the *scalar* residual, run
///     `grad_with_loss` to produce a scalar gradient graph, then vmap
///     that to batched. Same input-name set as the residual, so the
///     same input bindings drive both.
///   * inner solve: macOS path dispatches one
///     `Op::BatchedDenseSolve` through `MlxExecutable`, lowering to the
///     Apple-GPU Metal LU+solve kernel. Non-macOS / n>90 falls back to
///     per-batch Gauss-Jordan.
///   * line search: shared-α Armijo backtracking. Halves α whenever
///     any non-converged draw fails to improve. Per-draw α with
///     masking is a phase-3.1 enhancement.
///
/// `mc_params` carries per-draw values for any device `Op::Param`
/// names you want Monte-Carlo'd (Vth mismatch, R tolerance, …). Each
/// entry's `Vec<f32>` must have length `n_draws`. Internally these
/// params are promoted to `Op::Input`s before vmap, so they bind at
/// run time alongside `v_<id>` instead of going through `set_param`.
/// Pass an empty map for the no-MC-on-params case.
pub fn batched_solve_dc(
    circuit: &Circuit,
    n_draws: usize,
    params: &HashMap<String, f32>,
    mc_params: &HashMap<String, Vec<f32>>,
    boundary_voltages: &HashMap<NetId, Vec<f32>>,
    opt: NewtonOptions,
) -> BatchedDcOperatingPoint {
    use std::collections::HashSet;
    use rlx_runtime::{Device, Session, CompiledGraph};

    // Validate boundary + mc_param lengths.
    for (net, vs) in boundary_voltages {
        assert_eq!(
            vs.len(), n_draws,
            "boundary_voltages[{net:?}] has {} entries, expected n_draws={n_draws}",
            vs.len(),
        );
    }
    for (k, vs) in mc_params {
        assert_eq!(
            vs.len(), n_draws,
            "mc_params[{k}] has {} entries, expected n_draws={n_draws}",
            vs.len(),
        );
    }

    // mc_param names — promote these to Inputs before vmap on both
    // residual and jac-row graphs. Same name set drives both, so the
    // graphs share the same input contract.
    let mc_names_owned: Vec<String> = mc_params.keys().cloned().collect();
    let mc_names: Vec<&str> = mc_names_owned.iter().map(|s| s.as_str()).collect();

    // Scalar source: original eda-mna graph, then promote any MC
    // params to Inputs so grad_with_loss + vmap see them as Inputs.
    let scalar_orig = build_residual_graph(circuit);
    let scalar_promoted_graph = if mc_names.is_empty() {
        scalar_orig.graph.clone()
    } else {
        rlx_opt::promote_params_to_inputs(&scalar_orig.graph, &mc_names)
    };
    let scalar_rg = ResidualGraph {
        graph: scalar_promoted_graph,
        unknown_nets: scalar_orig.unknown_nets.clone(),
        all_nets: scalar_orig.all_nets.clone(),
        branches: scalar_orig.branches.clone(),
    };

    // Batched residual via the same wrapper, threading mc_names through.
    let batched_rg = build_batched_residual_graph_with_mc_params(
        circuit, n_draws, &mc_names,
    );
    let unknowns = batched_rg.unknown_nets.clone();
    let branches = batched_rg.branches.clone();
    let n_v = unknowns.len();
    let n_b = branches.len();
    let n = n_v + n_b;

    let session = Session::new(Device::Cpu);
    let mut compiled_res = session.compile(batched_rg.graph.clone());

    // Per-row jac graphs: scalar grad_with_loss → vmap (same names
    // batched, including MC param names).
    let scalar_unknown_ids: Vec<rlx_ir::NodeId> = unknowns.iter()
        .map(|net| find_input_node(&scalar_rg.graph, &net_input_name(*net))
            .expect("scalar residual missing v_<id> input"))
        .chain(branches.iter().map(|b|
            find_input_node(&scalar_rg.graph, &branch_input_name(*b))
                .expect("scalar residual missing i_b<id> input")))
        .collect();

    let mut wanted: HashSet<String> = HashSet::new();
    for net in &batched_rg.all_nets {
        wanted.insert(net_input_name(*net));
    }
    for b in &branches {
        wanted.insert(branch_input_name(*b));
    }
    for nm in &mc_names {
        wanted.insert((*nm).to_string());
    }

    let mut compiled_jac_rows: Vec<CompiledGraph> = Vec::with_capacity(n);
    for i in 0..n {
        let mut g_i = scalar_rg.graph.clone();
        let out_i = g_i.outputs[i];
        g_i.set_outputs(vec![out_i]);
        let bwd = rlx_opt::autodiff::grad_with_loss(&g_i, &scalar_unknown_ids);
        let actual: HashSet<String> = bwd.nodes().iter()
            .filter_map(|n| match &n.op {
                rlx_ir::Op::Input { name } => Some(name.clone()),
                _ => None,
            })
            .collect();
        let names_owned: Vec<String> = wanted.iter()
            .filter(|nm| actual.contains(*nm))
            .cloned()
            .collect();
        let names: Vec<&str> = names_owned.iter().map(|s| s.as_str()).collect();
        let batched_jac = rlx_opt::vmap::vmap(&bwd, &names, n_draws);
        compiled_jac_rows.push(session.compile(batched_jac));
    }

    // Bind shared params (skipping any names that are now MC Inputs —
    // those bind per-iter via the input map instead).
    let mc_set: HashSet<&str> = mc_names.iter().copied().collect();
    let set_all = |g: &mut CompiledGraph| {
        for (k, v) in params {
            if mc_set.contains(k.as_str()) { continue; }
            g.set_param(k, &[*v]);
        }
    };
    set_all(&mut compiled_res);
    for g in compiled_jac_rows.iter_mut() { set_all(g); }

    // v layout: row-major [n_draws, n] f32. v[d*n + j] = unknown j of draw d.
    let mut v = vec![opt.init; n_draws * n];
    let mut converged = vec![false; n_draws];
    let mut last_max = vec![f32::INFINITY; n_draws];
    let mut iters_run = 0usize;

    let unknown_idx: HashMap<NetId, usize> =
        unknowns.iter().enumerate().map(|(i, n)| (*n, i)).collect();

    // d_output stays scalar [1] — broadcasts through the vmap'd
    // gradient ops without a per-draw allocation.
    let one_seed = [1.0_f32];

    // Read a per-draw value off a vmap'd output. vmap leaves outputs
    // that have no transitive dependency on any batched input at their
    // original shape — for scalar-shape graphs that's `[1]`, length 1.
    // We broadcast those across all draws on read; outputs that did
    // pick up the batch axis come back as length n_draws and we index
    // directly. Same trick covers both residuals and gradients.
    let read = |out: &Vec<f32>, d: usize| -> f32 {
        if out.len() == 1 { out[0] } else { out[d] }
    };

    // Cache for the inner Op::BatchedDenseSolve dispatch — built on
    // first iter, reused thereafter. Saves the MLX compile cost on
    // every Newton iter after the first.
    let mut inner_cache = InnerSolveCache::default();

    // Per-draw max |Δv| from the *previous* Newton iter. Initialized
    // to +∞ so the iter-0 convergence check always fails (forces at
    // least one Newton step before any draw can be declared
    // converged — the SPICE-style "vntol" guard that catches silent
    // under-convergence on circuits with current-scale residuals).
    let mut last_step_max_per_draw = vec![f32::INFINITY; n_draws];

    for iter in 0..opt.max_iters {
        // Build inputs for this iter: per-net Vec<f32> of length n_draws,
        // plus mc_param bindings for any params we promoted to Inputs.
        let mut inputs_owned = batched_inputs_for_iter(
            &batched_rg.all_nets, &unknowns, &branches,
            &unknown_idx, boundary_voltages, &v, n, n_draws,
        );
        for (name, vals) in mc_params {
            inputs_owned.push((name.clone(), vals.clone()));
        }
        let inputs_ref: Vec<(&str, &[f32])> = inputs_owned.iter()
            .map(|(name, vs)| (name.as_str(), vs.as_slice())).collect();

        // Evaluate batched residual.
        let f_outs = compiled_res.run(&inputs_ref);
        for d in 0..n_draws {
            let mut m = 0.0_f32;
            for i in 0..n {
                let val = read(&f_outs[i], d);
                if !val.is_finite() { m = f32::INFINITY; break; }
                m = m.max(val.abs());
            }
            last_max[d] = m;
        }
        // Convergence check requires BOTH residual < tol AND last
        // Newton step < vntol — see the NewtonOptions docs for the
        // motivating mosfet-mirror under-convergence case.
        let mut all_done = true;
        for d in 0..n_draws {
            let res_ok  = last_max[d] < opt.tol;
            let step_ok = last_step_max_per_draw[d] < opt.vntol;
            if res_ok && step_ok { converged[d] = true; }
            else { all_done = false; }
        }
        if all_done {
            iters_run = iter;
            break;
        }
        iters_run = iter + 1;

        // Snapshot v before the line-search update so we can compute
        // the per-draw step size after the commit.
        let v_before = v.clone();

        // Evaluate all jac rows. outs[1 + j][d] = ∂f_i/∂v_j at draw d.
        let mut j_data = vec![0.0_f32; n_draws * n * n];
        for i in 0..n {
            let mut grad_inputs = inputs_ref.clone();
            grad_inputs.push(("d_output", &one_seed[..]));
            let outs = compiled_jac_rows[i].run(&grad_inputs);
            for j in 0..n {
                let col = &outs[1 + j];
                for d in 0..n_draws {
                    j_data[d * n * n + i * n + j] = read(col, d);
                }
            }
        }

        // ── Inner solve: J_d · dv_d = -f_d for every draw d ───────
        //
        // Two paths share one numerical contract — both consume
        // j_data and f_outs, both write into v in place. The MLX
        // path (macOS, n ≤ 90) batches all N solves into one
        // Op::BatchedDenseSolve dispatch, which lowers to the
        // Apple-GPU Metal LU+solve kernel registered in `rlx-mlx`.
        // The fallback path (non-macOS, or n > 90) runs per-batch
        // Gauss-Jordan in Rust. Per-draw "skip if converged" applies
        // on the v-update side either way — the GPU path wastes a
        // little work for converged draws (always solves all N) but
        // never produces wrong values.
        let dv_packed = inner_solve_batch(
            &mut inner_cache, &j_data, &f_outs, n, n_draws, &read,
        );
        let dv = match dv_packed {
            Some(dv) => dv,
            None => continue, // every draw singular or MLX failure;
                              // re-attempt next iter with same v
        };

        // ── Per-draw step acceptance with shared-α backtracking ──
        //
        // Pure Newton overshoots when the residual has steep
        // exponentials (e.g., a forward-biased diode). Scalar
        // `solve_dc` handles that with Armijo backtracking; we mirror
        // the same idea in batched form, but with a *shared* α across
        // all non-converged draws — halve α whenever any non-converged
        // draw fails to improve. Costs one extra residual evaluation
        // per backtrack step. Per-draw α with masking would let easy
        // draws keep α=1 while hard draws shrink — a phase-3.1 enhancement.
        let mut alpha = 1.0_f32;
        let mut v_trial = v.clone();
        let mut accepted_max = last_max.clone();
        for _ in 0..=opt.max_backtracks {
            // Build trial v: only step non-converged draws.
            for d in 0..n_draws {
                if converged[d] {
                    for j in 0..n { v_trial[d * n + j] = v[d * n + j]; }
                } else {
                    for j in 0..n {
                        v_trial[d * n + j] = v[d * n + j] + alpha * dv[d * n + j];
                    }
                }
            }
            // Re-evaluate residual at v_trial — rebuild inputs +
            // mc_params (mc values don't change between iters but
            // share the same input layout as the main eval).
            let mut trial_inputs = batched_inputs_for_iter(
                &batched_rg.all_nets, &unknowns, &branches,
                &unknown_idx, boundary_voltages, &v_trial, n, n_draws,
            );
            for (name, vals) in mc_params {
                trial_inputs.push((name.clone(), vals.clone()));
            }
            let trial_inputs_ref: Vec<(&str, &[f32])> = trial_inputs.iter()
                .map(|(name, vs)| (name.as_str(), vs.as_slice())).collect();
            let f_trial = compiled_res.run(&trial_inputs_ref);
            for d in 0..n_draws {
                let mut m = 0.0_f32;
                for i in 0..n {
                    let val = read(&f_trial[i], d);
                    if !val.is_finite() { m = f32::INFINITY; break; }
                    m = m.max(val.abs());
                }
                accepted_max[d] = m;
            }
            // Accept if every non-converged draw improved.
            let improved = (0..n_draws).all(|d| {
                converged[d]
                    || (accepted_max[d].is_finite() && accepted_max[d] < last_max[d])
            });
            if improved { break; }
            alpha *= 0.5;
        }
        // Commit whatever the line-search settled on (the smallest-α
        // attempt's v_trial if no α improved; that's worse than
        // staying put, so guard the commit by accepted_max actually
        // beating last_max for at least one draw).
        let any_improved = (0..n_draws).any(|d|
            !converged[d] && accepted_max[d] < last_max[d]
        );
        if any_improved {
            v.copy_from_slice(&v_trial);
            last_max.copy_from_slice(&accepted_max);
        }
        // else: leave v unchanged; the next iter will re-derive J at
        // the same v and try again. If this happens repeatedly we'll
        // run out of iters and report not-converged for the stuck
        // draws — same diagnostic as scalar solve_dc.

        // Update per-draw last step size (max |Δv| over unknowns)
        // for the next iter's convergence check.
        for d in 0..n_draws {
            let mut step_max = 0.0_f32;
            for j in 0..n {
                let dv_j = (v[d * n + j] - v_before[d * n + j]).abs();
                if dv_j > step_max { step_max = dv_j; }
            }
            last_step_max_per_draw[d] = step_max;
        }
    }

    // Build output struct: split v back to per-NetId Vec<f32>.
    let mut voltages: HashMap<NetId, Vec<f32>> = HashMap::new();
    for (idx, net) in unknowns.iter().enumerate() {
        let mut col = Vec::with_capacity(n_draws);
        for d in 0..n_draws {
            col.push(v[d * n + idx]);
        }
        voltages.insert(*net, col);
    }
    // Boundary voltages echo through unchanged.
    for (net, vs) in boundary_voltages {
        voltages.insert(*net, vs.clone());
    }
    let mut branch_currents: HashMap<BranchId, Vec<f32>> = HashMap::new();
    for (idx, b) in branches.iter().enumerate() {
        let mut col = Vec::with_capacity(n_draws);
        for d in 0..n_draws {
            col.push(v[d * n + n_v + idx]);
        }
        branch_currents.insert(*b, col);
    }
    BatchedDcOperatingPoint {
        voltages,
        branch_currents,
        iters: iters_run,
        converged,
        final_residual_max: last_max,
    }
}

/// Cached state for the batched inner Newton solve. Holds an
/// `MlxExecutable` for the Metal LU+solve dispatch (compiled once
/// per (n, n_draws) shape; reused across Newton iters and BE steps)
/// plus reusable input/output buffers to skip per-call allocation.
///
/// Kept as `Default + None`-valued so the same struct is allocatable
/// on non-macOS targets without referencing `MlxExecutable`.
///
/// Public so transient drivers can amortize the executable build
/// cost across many `batched_solve_be_step` calls — the field set is
/// intentionally private; construct via `Default::default()`.
#[derive(Default)]
pub struct InnerSolveCache {
    /// Compiled batched-solve graph executable. None until the first
    /// MLX dispatch builds it; reused on every subsequent call.
    #[cfg(target_os = "macos")]
    exe: Option<rlx_mlx::MlxExecutable>,
    /// Captured shape so we rebuild the executable if the caller
    /// changes (n, n_draws) mid-flight (shouldn't happen in normal
    /// use; defensive).
    shape: Option<(usize, usize)>,
    a_buf: Vec<f32>,
    b_buf: Vec<f32>,
}

/// Inner Newton solve for `batched_solve_dc` / `batched_solve_be_step`.
/// Returns `[n_draws, n]` of dv values laid out row-major
/// (`dv[d*n + j]` = j-th unknown's update for draw d), or `None` if
/// every draw failed.
///
/// macOS path: pack A `[N,n,n]` and b `[N,n]` as f32, dispatch the
/// cached `MlxExecutable` for `Op::BatchedDenseSolve` — lowers to the
/// Metal LU+solve kernel registered in `rlx-mlx`. The executable is
/// built on the first call and reused; ~10× speedup on a
/// transient loop where 250 BE steps × ~3 Newton iters each share
/// the same shape.
///
/// Fallback path: per-draw Gauss-Jordan in Rust. Used when the
/// kernel envelope doesn't fit (n > 90 or non-macOS), and as a
/// runtime safety net if MLX init fails at dispatch time.
fn inner_solve_batch(
    cache: &mut InnerSolveCache,
    j_data: &[f32],
    f_outs: &[Vec<f32>],
    n: usize,
    n_draws: usize,
    read: &dyn Fn(&Vec<f32>, usize) -> f32,
) -> Option<Vec<f32>> {
    #[cfg(target_os = "macos")]
    {
        if n > 0 && n <= 90 {
            if let Some(dv) = inner_solve_batch_mlx(cache, j_data, f_outs, n, n_draws, read) {
                return Some(dv);
            }
        }
    }
    let _ = cache;    // unused on non-macOS
    inner_solve_batch_rust(j_data, f_outs, n, n_draws, read)
}

#[cfg(target_os = "macos")]
fn inner_solve_batch_mlx(
    cache: &mut InnerSolveCache,
    j_data: &[f32],
    f_outs: &[Vec<f32>],
    n: usize,
    n_draws: usize,
    read: &dyn Fn(&Vec<f32>, usize) -> f32,
) -> Option<Vec<f32>> {
    use rlx_ir::{DType as IrDType, Graph as IrGraph, Shape as IrShape};
    use rlx_mlx::{MlxExecutable, MlxMode};

    // Drop the cached executable if shape changed (defensive — same
    // (n, n_draws) is reused across all iters of one solve, but this
    // keeps the cache safe if a caller ever recycles it across calls).
    if cache.shape != Some((n, n_draws)) {
        cache.exe = None;
        cache.shape = Some((n, n_draws));
        cache.a_buf.clear();
        cache.b_buf.clear();
    }

    // Pack A [N, n, n] and b [N, n] row-major into the cached buffers.
    let need_a = n_draws * n * n;
    let need_b = n_draws * n;
    cache.a_buf.clear();
    cache.a_buf.reserve(need_a);
    for d in 0..n_draws {
        for i in 0..n {
            for j in 0..n {
                cache.a_buf.push(j_data[d * n * n + i * n + j]);
            }
        }
    }
    cache.b_buf.clear();
    cache.b_buf.reserve(need_b);
    for d in 0..n_draws {
        for i in 0..n {
            cache.b_buf.push(-read(&f_outs[i], d));
        }
    }

    // Build the executable on first use. MLX caches the underlying
    // compiled MTL function by source hash globally, but the
    // MlxExecutable wrapper keeps its own per-instance compile state
    // — building it once per outer solve avoids paying the std::function
    // build cost on every Newton iter (which adds up over a transient
    // loop's hundreds of inner solves).
    if cache.exe.is_none() {
        let mut g = IrGraph::new("inner_dense_solve_batched");
        let a = g.input("A", IrShape::new(&[n_draws, n, n], IrDType::F32));
        let b = g.input("b", IrShape::new(&[n_draws, n],    IrDType::F32));
        let x = g.batched_dense_solve(a, b, IrShape::new(&[n_draws, n], IrDType::F32));
        g.set_outputs(vec![x]);
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            MlxExecutable::compile_with_mode(g, MlxMode::Lazy)
        }));
        cache.exe = match built { Ok(e) => Some(e), Err(_) => return None };
    }
    let exe = cache.exe.as_mut().unwrap();

    // Run. Catch MLX-side panics (e.g., transient GPU error) so we
    // degrade to the Rust fallback for this iter.
    let a_slice = cache.a_buf.as_slice();
    let b_slice = cache.b_buf.as_slice();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let outs = exe.run(&[("A", a_slice), ("b", b_slice)]);
        outs.into_iter().next().unwrap_or_default()
    }));
    match result {
        Ok(v) if v.len() == n_draws * n => Some(v),
        _ => None,
    }
}

fn inner_solve_batch_rust(
    j_data: &[f32],
    f_outs: &[Vec<f32>],
    n: usize,
    n_draws: usize,
    read: &dyn Fn(&Vec<f32>, usize) -> f32,
) -> Option<Vec<f32>> {
    let mut dv = vec![0.0_f32; n_draws * n];
    let mut a_buf = vec![0.0_f32; n * n];
    let mut b_buf = vec![0.0_f32; n];
    let mut any_solved = false;
    for d in 0..n_draws {
        for i in 0..n {
            for j in 0..n {
                a_buf[i * n + j] = j_data[d * n * n + i * n + j];
            }
            b_buf[i] = -read(&f_outs[i], d);
        }
        if let Some(x) = gauss_jordan_solve(&a_buf, &b_buf, n) {
            for j in 0..n {
                dv[d * n + j] = x[j];
            }
            any_solved = true;
        }
        // Singular for this draw → leave dv[d, *] = 0; outer loop
        // re-attempts next iter.
    }
    if any_solved { Some(dv) } else { None }
}

/// Build the input bindings for one batched-Newton iteration. Mirrors
/// `build_inputs` but produces `Vec<f32>` of length `n_draws` per
/// input (the [N, 1] shape after vmap is just N contiguous floats on
/// the wire). For each net, the values come from one of three places:
///   - it's an unknown → take from the current v iterate
///   - it's a boundary → take from `boundary_voltages` (per-draw)
///   - else → 0 (treated as ground)
fn batched_inputs_for_iter(
    all_nets: &[NetId],
    _unknowns: &[NetId],
    branches: &[BranchId],
    unknown_idx: &HashMap<NetId, usize>,
    boundary: &HashMap<NetId, Vec<f32>>,
    v: &[f32],
    n: usize,
    n_draws: usize,
) -> Vec<(String, Vec<f32>)> {
    let mut out: Vec<(String, Vec<f32>)> = all_nets.iter().map(|net| {
        let mut col = Vec::with_capacity(n_draws);
        if let Some(idx) = unknown_idx.get(net) {
            for d in 0..n_draws {
                col.push(v[d * n + idx]);
            }
        } else if let Some(vs) = boundary.get(net) {
            col.extend_from_slice(vs);
        } else {
            col.resize(n_draws, 0.0);
        }
        (net_input_name(*net), col)
    }).collect();
    for (idx, b) in branches.iter().enumerate() {
        let mut col = Vec::with_capacity(n_draws);
        for d in 0..n_draws {
            col.push(v[d * n + (all_nets.len() - boundary.len()) + idx]);
        }
        out.push((branch_input_name(*b), col));
    }
    out
}

// ── Batched BE step (phase-5A) ────────────────────────────────────────

/// Vmap the Backward-Euler residual graph along a leading batch axis
/// of size `n_draws`. Mirror of [`build_batched_residual_graph_with_mc_params`]
/// but on the BE-step graph: every per-draw input — net voltages,
/// branch currents, *previous-step* voltages, delay-step scalars —
/// gets a leading `[n_draws, ...]` dim. The timestep `h` and any
/// device Params not in `mc_param_names` stay shared.
pub fn build_batched_be_step_residual_graph_with_mc_params(
    circuit: &Circuit,
    n_draws: usize,
    mc_param_names: &[&str],
) -> ResidualGraph {
    let scalar_orig = build_be_step_residual_graph(circuit);

    let scalar_graph = if mc_param_names.is_empty() {
        scalar_orig.graph
    } else {
        rlx_opt::promote_params_to_inputs(&scalar_orig.graph, mc_param_names)
    };
    let scalar = ResidualGraph {
        graph: scalar_graph,
        unknown_nets: scalar_orig.unknown_nets,
        all_nets: scalar_orig.all_nets,
        branches: scalar_orig.branches,
    };

    use std::collections::HashSet;
    let mut wanted: HashSet<String> = HashSet::new();
    for net in &scalar.all_nets {
        wanted.insert(net_input_name(*net));
        wanted.insert(prev_voltage_input_name(*net));
    }
    for b in &scalar.branches {
        wanted.insert(branch_input_name(*b));
    }
    // Delay history slots are per-draw (each MC draw can have its own
    // history buffer state). h stays shared and is excluded.
    for node in scalar.graph.nodes() {
        if let rlx_ir::Op::Input { name } = &node.op {
            if name.starts_with("v_lo_")
                || name.starts_with("v_hi_")
                || name.starts_with("delay_blend_")
                || name.starts_with("delay_offset_")
            {
                wanted.insert(name.clone());
            }
        }
    }
    for nm in mc_param_names {
        wanted.insert((*nm).to_string());
    }

    // Filter to inputs the assembler actually emitted.
    let actual: HashSet<String> = scalar.graph.nodes().iter()
        .filter_map(|n| match &n.op {
            rlx_ir::Op::Input { name } => Some(name.clone()),
            _ => None,
        })
        .collect();
    let names_owned: Vec<String> = wanted.into_iter()
        .filter(|n| actual.contains(n))
        .collect();
    let names: Vec<&str> = names_owned.iter().map(|s| s.as_str()).collect();

    let batched = rlx_opt::vmap::vmap(&scalar.graph, &names, n_draws);

    ResidualGraph {
        graph: batched,
        unknown_nets: scalar.unknown_nets,
        all_nets: scalar.all_nets,
        branches: scalar.branches,
    }
}

/// Solved one Backward-Euler step for `n_draws` independent per-draw
/// (boundary V, prev V, MC params) inputs. Output mirrors
/// [`BatchedDcOperatingPoint`] plus the BE-step timestamp `t`.
#[derive(Debug, Clone)]
pub struct BatchedTransientStep {
    pub t: f32,
    pub voltages: HashMap<NetId, Vec<f32>>,
    pub branch_currents: HashMap<BranchId, Vec<f32>>,
    pub iters: usize,
    pub converged: Vec<bool>,
    pub final_residual_max: Vec<f32>,
}

/// Batched Backward-Euler step. Mirror of [`solve_be_step`] for N
/// independent draws. `prev_voltages` is per-draw (each unknown net's
/// `Vec<f32>` length = n_draws). `h` is shared. `mc_params` carries
/// per-draw device-Param values (Vth mismatch, R tolerance, …).
///
/// Returns the per-draw operating point at time `t = t_prev + h`.
/// `t_prev` is supplied by the caller (this fn doesn't track time);
/// pass it through if you're stepping a transient loop yourself.
pub fn batched_solve_be_step(
    circuit: &Circuit,
    n_draws: usize,
    params: &HashMap<String, f32>,
    mc_params: &HashMap<String, Vec<f32>>,
    boundary_voltages: &HashMap<NetId, Vec<f32>>,
    prev_voltages: &HashMap<NetId, Vec<f32>>,
    delay_inputs: &[BatchedDelayStepInputs],
    h: f32,
    t_prev: f32,
    opt: NewtonOptions,
    inner_cache: &mut InnerSolveCache,
) -> BatchedTransientStep {
    // Allocate a fresh context per call. For multi-step transient
    // drivers, build the context ONCE outside the loop and call
    // `batched_solve_be_step_with_ctx` directly — each `new` rebuilds
    // residual + N jac graphs which is the dominant per-step cost.
    let mc_names_owned: Vec<String> = mc_params.keys().cloned().collect();
    let mc_names: Vec<&str> = mc_names_owned.iter().map(|s| s.as_str()).collect();
    let mut ctx = BatchedBeStepContext::new(circuit, n_draws, &mc_names);
    ctx.set_params(circuit, params);
    batched_solve_be_step_with_ctx(
        &mut ctx, n_draws, mc_params, boundary_voltages, prev_voltages,
        delay_inputs, h, t_prev, opt, inner_cache,
    )
}

/// Precompiled state for repeated batched BE steps with the same
/// circuit / `n_draws` / `mc_param_names`. Hoists the residual graph
/// build + N jac-row compiles out of the per-step path — for a
/// 250-step transient that's 250×(1+N) graph compiles avoided.
///
/// Mirror of scalar [`BeStepContext`]. Construct once via
/// [`BatchedBeStepContext::new`], call [`set_params`] once per
/// parameter change, then drive [`batched_solve_be_step_with_ctx`]
/// per step.
pub struct BatchedBeStepContext {
    /// Full per-net + per-branch metadata (NetId ordering preserved
    /// from the scalar residual graph).
    pub all_nets: Vec<NetId>,
    pub unknowns: Vec<NetId>,
    pub branches: Vec<BranchId>,
    n_v: usize,
    n_b: usize,
    /// Compiled batched residual graph (one output per unknown).
    compiled_res: rlx_runtime::CompiledGraph,
    /// One compiled batched gradient graph per residual row.
    compiled_jac_rows: Vec<rlx_runtime::CompiledGraph>,
    /// MC param names (those promoted to Inputs, bound per-iter).
    mc_names_owned: Vec<String>,
}

impl BatchedBeStepContext {
    /// Build the precompiled context. `mc_param_names` lists
    /// `Op::Param` names to be promoted to per-draw `Op::Input`s
    /// before vmap — same convention as `batched_solve_dc`.
    pub fn new(circuit: &Circuit, n_draws: usize, mc_param_names: &[&str]) -> Self {
        use std::collections::HashSet;
        use rlx_runtime::{Device, Session};

        let scalar_orig = build_be_step_residual_graph(circuit);
        let scalar_promoted_graph = if mc_param_names.is_empty() {
            scalar_orig.graph.clone()
        } else {
            rlx_opt::promote_params_to_inputs(&scalar_orig.graph, mc_param_names)
        };
        let scalar_rg = ResidualGraph {
            graph: scalar_promoted_graph,
            unknown_nets: scalar_orig.unknown_nets.clone(),
            all_nets: scalar_orig.all_nets.clone(),
            branches: scalar_orig.branches.clone(),
        };

        let batched_rg = build_batched_be_step_residual_graph_with_mc_params(
            circuit, n_draws, mc_param_names,
        );
        let unknowns = batched_rg.unknown_nets.clone();
        let branches = batched_rg.branches.clone();
        let n_v = unknowns.len();
        let n_b = branches.len();
        let n = n_v + n_b;
        let all_nets = batched_rg.all_nets.clone();

        // Default device is CPU; opt-in to Apple Metal for the WHOLE
        // batched residual + jacobian via `RLX_BATCHED_DEVICE=mlx`.
        // Without this, only the inner LU solve dispatches to MLX
        // (via Op::BatchedDenseSolve) — the residual + jacobian eval
        // stay on CPU, which is what shows up as ~6× speedup instead
        // of the full GPU win.
        let device = match std::env::var("RLX_BATCHED_DEVICE").as_deref() {
            Ok("mlx") => Device::Mlx,
            _ => Device::Cpu,
        };
        let session = Session::new(device);
        let compiled_res = session.compile(batched_rg.graph.clone());

        let scalar_unknown_ids: Vec<rlx_ir::NodeId> = unknowns.iter()
            .map(|net| find_input_node(&scalar_rg.graph, &net_input_name(*net))
                .expect("BE residual missing v_<id>"))
            .chain(branches.iter().map(|b|
                find_input_node(&scalar_rg.graph, &branch_input_name(*b))
                    .expect("BE residual missing i_b<id>")))
            .collect();

        let mut wanted: HashSet<String> = HashSet::new();
        for net in &all_nets {
            wanted.insert(net_input_name(*net));
            wanted.insert(prev_voltage_input_name(*net));
        }
        for b in &branches {
            wanted.insert(branch_input_name(*b));
        }
        for nm in mc_param_names {
            wanted.insert((*nm).to_string());
        }

        let mut compiled_jac_rows = Vec::with_capacity(n);
        for i in 0..n {
            let mut g_i = scalar_rg.graph.clone();
            let out_i = g_i.outputs[i];
            g_i.set_outputs(vec![out_i]);
            let bwd = rlx_opt::autodiff::grad_with_loss(&g_i, &scalar_unknown_ids);
            let actual: HashSet<String> = bwd.nodes().iter()
                .filter_map(|n| match &n.op {
                    rlx_ir::Op::Input { name } => Some(name.clone()),
                    _ => None,
                })
                .collect();
            let names_owned: Vec<String> = wanted.iter()
                .filter(|nm| actual.contains(*nm))
                .cloned()
                .collect();
            let names: Vec<&str> = names_owned.iter().map(|s| s.as_str()).collect();
            let batched_jac = rlx_opt::vmap::vmap(&bwd, &names, n_draws);
            compiled_jac_rows.push(session.compile(batched_jac));
        }

        Self {
            all_nets, unknowns, branches, n_v, n_b,
            compiled_res, compiled_jac_rows,
            mc_names_owned: mc_param_names.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Bind shared `Op::Param` values to every compiled graph.
    /// Skips names listed in `mc_param_names` at construction (those
    /// bind per-iter as inputs). Backfills `<name>_tau` defaults from
    /// each delay device — mirrors the scalar `solve_be_step` setup
    /// so callers who don't care about delay don't have to populate
    /// the param map themselves.
    pub fn set_params(&mut self, circuit: &Circuit, params: &HashMap<String, f32>) {
        use std::collections::HashSet;
        let mut effective_params = params.clone();
        for dl in &circuit.delays {
            let key = format!("{}_tau", dl.device.name());
            effective_params.entry(key).or_insert(
                dl.device.delay_seconds() as f32,
            );
        }
        let mc_set: HashSet<&str> = self.mc_names_owned.iter().map(|s| s.as_str()).collect();
        for (k, v) in &effective_params {
            if mc_set.contains(k.as_str()) { continue; }
            self.compiled_res.set_param(k, &[*v]);
            for g in self.compiled_jac_rows.iter_mut() {
                g.set_param(k, &[*v]);
            }
        }
    }
}

/// Run one batched BE step using a precompiled [`BatchedBeStepContext`].
/// Skips the residual+jac graph build/compile that
/// [`batched_solve_be_step`] pays per call. Use this from multi-step
/// transient drivers to amortize the compile cost across the run.
pub fn batched_solve_be_step_with_ctx(
    ctx: &mut BatchedBeStepContext,
    n_draws: usize,
    mc_params: &HashMap<String, Vec<f32>>,
    boundary_voltages: &HashMap<NetId, Vec<f32>>,
    prev_voltages: &HashMap<NetId, Vec<f32>>,
    delay_inputs: &[BatchedDelayStepInputs],
    h: f32,
    t_prev: f32,
    opt: NewtonOptions,
    inner_cache: &mut InnerSolveCache,
) -> BatchedTransientStep {
    // Validate per-draw input lengths.
    for (net, vs) in boundary_voltages {
        assert_eq!(vs.len(), n_draws,
            "boundary_voltages[{net:?}] has {} entries, expected n_draws={n_draws}", vs.len());
    }
    for (net, vs) in prev_voltages {
        assert_eq!(vs.len(), n_draws,
            "prev_voltages[{net:?}] has {} entries, expected n_draws={n_draws}", vs.len());
    }
    for (k, vs) in mc_params {
        assert_eq!(vs.len(), n_draws,
            "mc_params[{k}] has {} entries, expected n_draws={n_draws}", vs.len());
    }
    for (idx, di) in delay_inputs.iter().enumerate() {
        assert_eq!(di.v_lo.len(), n_draws,
            "delay_inputs[{idx}].v_lo has {} entries, expected n_draws={n_draws}", di.v_lo.len());
        assert_eq!(di.v_hi.len(), n_draws,
            "delay_inputs[{idx}].v_hi has {} entries, expected n_draws={n_draws}", di.v_hi.len());
    }

    let unknowns = &ctx.unknowns;
    let branches = &ctx.branches;
    let n_v = ctx.n_v;
    let n_b = ctx.n_b;
    let n = n_v + n_b;
    let compiled_res = &mut ctx.compiled_res;
    let compiled_jac_rows = &mut ctx.compiled_jac_rows;

    // Initial v: per-draw prev_voltages for unknowns (continuity
    // heuristic). Falls back to opt.init for any unknown not in prev.
    // v layout: row-major [n_draws, n] f32.
    let mut v = vec![0.0_f32; n_draws * n];
    for d in 0..n_draws {
        for (idx, net) in unknowns.iter().enumerate() {
            let val = prev_voltages.get(net)
                .map(|vs| vs[d])
                .unwrap_or(opt.init);
            v[d * n + idx] = val;
        }
        // Branch currents start at 0 across all draws.
    }

    let mut converged = vec![false; n_draws];
    let mut last_max = vec![f32::INFINITY; n_draws];
    let mut iters_run = 0usize;
    // Early-stop guard: if the BATCH-MAX residual hasn't dropped by
    // at least `STALL_RATIO` for `STALL_LIMIT` consecutive iters, the
    // shared-α Newton is plateaued — burning the rest of max_iters
    // is pure waste. Bail and let the next BE step continue from
    // wherever we are; for the SAR ADC this turns 200-iter dead ends
    // (during phase / capture transitions) into ~30-iter exits.
    // Stall-detector patience. Default = 32 iters (give Newton plenty
    // of room before bailing). Override via RLX_BATCHED_STALL_LIMIT.
    // Set to a huge number (e.g. 10000) to effectively disable.
    let stall_limit: usize = std::env::var("RLX_BATCHED_STALL_LIMIT")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    const STALL_RATIO: f32 = 0.99;   // require ≥1% improvement
    let mut prev_batch_max = f32::INFINITY;
    let mut stall_count: usize = 0;

    let unknown_idx: HashMap<NetId, usize> =
        unknowns.iter().enumerate().map(|(i, n)| (*n, i)).collect();
    let one_seed = [1.0_f32];
    let read = |out: &Vec<f32>, d: usize| -> f32 {
        if out.len() == 1 { out[0] } else { out[d] }
    };

    // Per-draw max |Δv| from the previous Newton iter — same vntol
    // guard as `batched_solve_dc`. See `NewtonOptions::vntol` docs.
    let mut last_step_max_per_draw = vec![f32::INFINITY; n_draws];

    // h shared across batch — vmap leaves the [1] input alone since
    // it's not in the batched name set, so we bind a single f32.
    let h_single = [h];

    let build_iter_inputs = |v_state: &[f32]| -> Vec<(String, Vec<f32>)> {
        let mut inputs = batched_inputs_for_iter(
            &ctx.all_nets, unknowns, branches,
            &unknown_idx, boundary_voltages, v_state, n, n_draws,
        );
        // v_prev per net (per-draw): unknown nets from prev_voltages,
        // boundaries echo their this-step value (scalar code does the
        // same).
        for net in &ctx.all_nets {
            let col: Vec<f32> = if let Some(vs) = prev_voltages.get(net) {
                vs.clone()
            } else if let Some(vs) = boundary_voltages.get(net) {
                vs.clone()
            } else {
                vec![0.0; n_draws]
            };
            inputs.push((prev_voltage_input_name(*net), col));
        }
        for (name, vals) in mc_params {
            inputs.push((name.clone(), vals.clone()));
        }
        // Per-element delay scalars: blend / offset stay scalar
        // (shared across draws since τ is shared in v1); v_lo / v_hi
        // bind per-draw vectors. The Op::Inputs they fill come from
        // build_be_step_residual_graph's delay loop (delay_v_lo_<id>,
        // delay_v_hi_<id>, delay_blend_<id>, delay_offset_<id>).
        for (idx, di) in delay_inputs.iter().enumerate() {
            let id = DelayId(idx as u32);
            inputs.push((delay_v_lo_name(id),   di.v_lo.clone()));
            inputs.push((delay_v_hi_name(id),   di.v_hi.clone()));
            inputs.push((delay_blend_name(id),  vec![di.blend]));
            inputs.push((delay_offset_name(id), vec![di.offset]));
        }
        inputs
    };

    for iter in 0..opt.max_iters {
        let inputs_owned = build_iter_inputs(&v);
        let mut inputs_ref: Vec<(&str, &[f32])> = inputs_owned.iter()
            .map(|(name, vs)| (name.as_str(), vs.as_slice())).collect();
        inputs_ref.push((TIMESTEP_INPUT_NAME, &h_single[..]));

        let f_outs = compiled_res.run(&inputs_ref);
        for d in 0..n_draws {
            let mut m = 0.0_f32;
            for i in 0..n {
                let val = read(&f_outs[i], d);
                if !val.is_finite() { m = f32::INFINITY; break; }
                m = m.max(val.abs());
            }
            last_max[d] = m;
        }
        // Convergence requires both res_ok && step_ok per draw — see
        // NewtonOptions::vntol docs.
        let mut all_done = true;
        for d in 0..n_draws {
            let res_ok  = last_max[d] < opt.tol;
            let step_ok = last_step_max_per_draw[d] < opt.vntol;
            if res_ok && step_ok { converged[d] = true; }
            else { all_done = false; }
        }
        if all_done {
            iters_run = iter;
            break;
        }
        iters_run = iter + 1;

        // Stall detector: track the BATCH-MAX residual across iters.
        // If it hasn't dropped by ≥1% for STALL_LIMIT consecutive
        // iters, give up — the shared-α Newton is going nowhere and
        // the remaining iters are pure waste.
        let batch_max = last_max.iter().cloned().fold(0.0_f32, f32::max);
        if batch_max < prev_batch_max * STALL_RATIO {
            stall_count = 0;
        } else {
            stall_count += 1;
        }
        prev_batch_max = batch_max;
        if stall_count >= stall_limit {
            break;
        }

        // Snapshot v before line-search update.
        let v_before = v.clone();

        let mut j_data = vec![0.0_f32; n_draws * n * n];
        for i in 0..n {
            let mut grad_inputs = inputs_ref.clone();
            grad_inputs.push(("d_output", &one_seed[..]));
            let outs = compiled_jac_rows[i].run(&grad_inputs);
            for j in 0..n {
                let col = &outs[1 + j];
                for d in 0..n_draws {
                    j_data[d * n * n + i * n + j] = read(col, d);
                }
            }
        }

        // Reborrow because `inner_cache` is `&mut` from the parameter list
        // (one binding can't be moved/borrowed twice across iter boundaries).
        let dv_packed = inner_solve_batch(&mut *inner_cache, &j_data, &f_outs, n, n_draws, &read);
        let dv = match dv_packed { Some(dv) => dv, None => continue };

        // Backtracking line search — `RLX_BATCHED_PER_CHIP_ALPHA=0`
        // reverts to the original shared-α version (one α for the
        // whole batch; halved when ANY non-converged chip can't
        // improve). Default ON: per-chip α (each chip halves its own
        // α independently). Used by reproducibility runs comparing
        // v0 (shared) vs v1+ (per-chip).
        let use_per_chip_alpha = std::env::var("RLX_BATCHED_PER_CHIP_ALPHA")
            .map(|v| v != "0").unwrap_or(true);

        if use_per_chip_alpha {
            let mut chip_alpha: Vec<f32> = vec![1.0; n_draws];
            let mut chip_accepted: Vec<bool> = vec![false; n_draws];
            let mut accepted_v: Vec<f32> = v.clone();
            let mut accepted_max: Vec<f32> = last_max.clone();
            let mut v_trial = v.clone();
            for _ in 0..=opt.max_backtracks {
                for d in 0..n_draws {
                    if converged[d] {
                        for j in 0..n { v_trial[d * n + j] = v[d * n + j]; }
                    } else if chip_accepted[d] {
                        for j in 0..n { v_trial[d * n + j] = accepted_v[d * n + j]; }
                    } else {
                        let a = chip_alpha[d];
                        for j in 0..n {
                            v_trial[d * n + j] = v[d * n + j] + a * dv[d * n + j];
                        }
                    }
                }
                let trial_inputs_owned = build_iter_inputs(&v_trial);
                let mut trial_inputs_ref: Vec<(&str, &[f32])> = trial_inputs_owned.iter()
                    .map(|(name, vs)| (name.as_str(), vs.as_slice())).collect();
                trial_inputs_ref.push((TIMESTEP_INPUT_NAME, &h_single[..]));
                let f_trial = compiled_res.run(&trial_inputs_ref);
                for d in 0..n_draws {
                    if converged[d] || chip_accepted[d] { continue; }
                    let mut m = 0.0_f32;
                    for i in 0..n {
                        let val = read(&f_trial[i], d);
                        if !val.is_finite() { m = f32::INFINITY; break; }
                        m = m.max(val.abs());
                    }
                    if m.is_finite() && m < last_max[d] {
                        for j in 0..n {
                            accepted_v[d * n + j] = v_trial[d * n + j];
                        }
                        accepted_max[d] = m;
                        chip_accepted[d] = true;
                    } else {
                        chip_alpha[d] *= 0.5;
                    }
                }
                let all_done = (0..n_draws).all(|d| converged[d] || chip_accepted[d]);
                if all_done { break; }
            }
            for d in 0..n_draws {
                if chip_accepted[d] {
                    for j in 0..n { v[d * n + j] = accepted_v[d * n + j]; }
                    last_max[d] = accepted_max[d];
                }
            }
        } else {
            // Original shared-α path (v0 in the solver-version sweep).
            let mut alpha = 1.0_f32;
            let mut v_trial = v.clone();
            let mut accepted_max = last_max.clone();
            for _ in 0..=opt.max_backtracks {
                for d in 0..n_draws {
                    if converged[d] {
                        for j in 0..n { v_trial[d * n + j] = v[d * n + j]; }
                    } else {
                        for j in 0..n {
                            v_trial[d * n + j] = v[d * n + j] + alpha * dv[d * n + j];
                        }
                    }
                }
                let trial_inputs_owned = build_iter_inputs(&v_trial);
                let mut trial_inputs_ref: Vec<(&str, &[f32])> = trial_inputs_owned.iter()
                    .map(|(name, vs)| (name.as_str(), vs.as_slice())).collect();
                trial_inputs_ref.push((TIMESTEP_INPUT_NAME, &h_single[..]));
                let f_trial = compiled_res.run(&trial_inputs_ref);
                for d in 0..n_draws {
                    let mut m = 0.0_f32;
                    for i in 0..n {
                        let val = read(&f_trial[i], d);
                        if !val.is_finite() { m = f32::INFINITY; break; }
                        m = m.max(val.abs());
                    }
                    accepted_max[d] = m;
                }
                let improved = (0..n_draws).all(|d| {
                    converged[d]
                        || (accepted_max[d].is_finite() && accepted_max[d] < last_max[d])
                });
                if improved { break; }
                alpha *= 0.5;
            }
            let any_improved = (0..n_draws).any(|d|
                !converged[d] && accepted_max[d] < last_max[d]
            );
            if any_improved {
                v.copy_from_slice(&v_trial);
                last_max.copy_from_slice(&accepted_max);
            }
        }

        // Update per-draw last step size for next iter's convergence.
        for d in 0..n_draws {
            let mut step_max = 0.0_f32;
            for j in 0..n {
                let dv_j = (v[d * n + j] - v_before[d * n + j]).abs();
                if dv_j > step_max { step_max = dv_j; }
            }
            last_step_max_per_draw[d] = step_max;
        }
    }

    // Pack output.
    let mut voltages: HashMap<NetId, Vec<f32>> = HashMap::new();
    for (idx, net) in unknowns.iter().enumerate() {
        let mut col = Vec::with_capacity(n_draws);
        for d in 0..n_draws { col.push(v[d * n + idx]); }
        voltages.insert(*net, col);
    }
    for (net, vs) in boundary_voltages {
        voltages.insert(*net, vs.clone());
    }
    let mut branch_currents: HashMap<BranchId, Vec<f32>> = HashMap::new();
    for (idx, b) in branches.iter().enumerate() {
        let mut col = Vec::with_capacity(n_draws);
        for d in 0..n_draws { col.push(v[d * n + n_v + idx]); }
        branch_currents.insert(*b, col);
    }

    BatchedTransientStep {
        t: t_prev + h,
        voltages,
        branch_currents,
        iters: iters_run,
        converged,
        final_residual_max: last_max,
    }
}

/// T.11.B — drive `batched_solve_be_step` for `n_steps` steps,
/// threading each step's per-draw solution as the next step's
/// `prev_voltages`. High-level analog of [`transient_pwl`] for
/// Monte Carlo / PVT corner sweeps.
///
/// `boundary_at(t)` returns a `HashMap<NetId, Vec<f32>>` — each entry's
/// `Vec<f32>` length must equal `n_draws`. `mc_params` carries
/// per-draw device-Param overrides (Vth mismatch, R tolerance, …);
/// pass an empty map for shared-param Monte Carlo where the variation
/// lives entirely in the boundary stimulus.
///
/// `initial_voltages_per_draw` seeds the IC for unknown nets per draw;
/// missing entries default to `opt.init` (= 0 V) across all draws.
///
/// **Performance note**: this v1 calls `batched_solve_be_step`
/// per-step, which means the batched residual + jac graphs are
/// recompiled on every BE step (same pre-T.10 issue the scalar
/// `transient_pwl` had before the cached `BeStepContext`). At a few
/// hundred BE steps × ~200 transistor circuit, the per-step compile
/// dominates wall time. T.11.B.2 will lift the cache the same way
/// T.10 did for the scalar path.
pub fn transient_pwl_batched<B>(
    circuit: &Circuit,
    n_draws: usize,
    params: &HashMap<String, f32>,
    mc_params: &HashMap<String, Vec<f32>>,
    boundary_at: B,
    initial_voltages_per_draw: &HashMap<NetId, Vec<f32>>,
    dt: f32,
    n_steps: usize,
    opt: NewtonOptions,
) -> Vec<BatchedTransientStep>
where
    B: Fn(f32) -> HashMap<NetId, Vec<f32>>,
{
    let mut out: Vec<BatchedTransientStep> = Vec::with_capacity(n_steps + 1);

    // t = 0: seed unknown voltages from per-draw IC, boundaries from
    // boundary_at(0), branch currents zero. Mirror of scalar
    // transient_pwl's t=0 emission.
    let bnd0 = boundary_at(0.0);
    let mut v0: HashMap<NetId, Vec<f32>> = initial_voltages_per_draw.clone();
    for (k, v) in &bnd0 {
        v0.insert(*k, v.clone());
    }
    out.push(BatchedTransientStep {
        t: 0.0,
        voltages: v0.clone(),
        branch_currents: HashMap::new(),
        iters: 0,
        converged: vec![true; n_draws],
        final_residual_max: vec![0.0; n_draws],
    });

    let mut histories = init_batched_delay_histories(circuit, n_draws, params, &v0);
    // Build the BE-step context ONCE — this hoists the residual + N
    // jac-graph compile cost out of the per-step loop. Mirrors what
    // scalar transient_from does via BeStepContext.
    let mc_names_owned: Vec<String> = mc_params.keys().cloned().collect();
    let mc_names: Vec<&str> = mc_names_owned.iter().map(|s| s.as_str()).collect();
    let mut ctx = BatchedBeStepContext::new(circuit, n_draws, &mc_names);
    ctx.set_params(circuit, params);
    let mut inner_cache = InnerSolveCache::default();
    // Optional progress emission — gated on RLX_BATCHED_PROGRESS=1 so
    // it stays silent for hot paths but observable when the wall-time
    // is long enough to need it (e.g. full-MLX dispatch on macOS).
    let progress = std::env::var("RLX_BATCHED_PROGRESS").as_deref() == Ok("1");
    // Adaptive sub-stepping: when too few chips converge for the full
    // dt step, automatically retry the same interval as `n_sub` micro-
    // steps. Helps the SAR ADC's bistable SR latch flip during phase
    // transitions: a single 1 ns BE step can't push the cross-couple
    // past its tipping point, but several 0.25 ns sub-steps can.
    // Tunable via env: RLX_BATCHED_ADAPTIVE_DT=1 to enable, default 0.
    let adaptive = std::env::var("RLX_BATCHED_ADAPTIVE_DT").as_deref() == Ok("1");
    let conv_thresh: f32 = std::env::var("RLX_BATCHED_CONV_THRESH")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.90);
    let max_sub_levels: usize = std::env::var("RLX_BATCHED_MAX_SUB_LEVELS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(3);  // up to 8x finer
    let conv_min_count = (n_draws as f32 * conv_thresh) as usize;
    let prog_start = std::time::Instant::now();
    let mut prev_v: HashMap<NetId, Vec<f32>> = v0;
    for k in 1..=n_steps {
        let t = k as f32 * dt;
        let bnd = boundary_at(t);
        let delay_inputs = sample_batched_delay_step(&histories, k, dt as f64, n_draws);
        let mut op = batched_solve_be_step_with_ctx(
            &mut ctx, n_draws, mc_params,
            &bnd, &prev_v, &delay_inputs, dt, t - dt, opt, &mut inner_cache,
        );
        let mut sub_levels_used: usize = 0;
        if adaptive {
            // Retry as 2 → 4 → 8 sub-steps until we hit conv_thresh or
            // give up. Each retry replays the full dt interval from
            // prev_v with smaller dt_sub steps; we accept the final
            // sub-step's voltages as the step output.
            let mut conv_count = op.converged.iter().filter(|&&c| c).count();
            let mut level: usize = 0;
            while conv_count < conv_min_count && level < max_sub_levels {
                level += 1;
                let n_sub: usize = 1 << level;     // 2, 4, 8, …
                let dt_sub: f32 = dt / n_sub as f32;
                let mut sub_v: HashMap<NetId, Vec<f32>> = prev_v.clone();
                let mut last_op = op.clone();
                let mut all_ok = true;
                for s in 0..n_sub {
                    let t_sub_start = t - dt + s as f32 * dt_sub;
                    let t_sub_end   = t_sub_start + dt_sub;
                    let bnd_sub = boundary_at(t_sub_end);
                    let delay_sub = sample_batched_delay_step(
                        &histories, k, dt_sub as f64, n_draws,
                    );
                    let sub_op = batched_solve_be_step_with_ctx(
                        &mut ctx, n_draws, mc_params,
                        &bnd_sub, &sub_v, &delay_sub, dt_sub, t_sub_start,
                        opt, &mut inner_cache,
                    );
                    sub_v = sub_op.voltages.clone();
                    last_op = sub_op;
                    let conv_sub = last_op.converged.iter().filter(|&&c| c).count();
                    if conv_sub < conv_min_count {
                        all_ok = false; break;
                    }
                }
                if all_ok {
                    op = last_op;
                    sub_levels_used = level;
                    break;
                }
            }
            conv_count = op.converged.iter().filter(|&&c| c).count();
            let _ = conv_count;
        }
        if progress {
            let elapsed = prog_start.elapsed().as_secs_f32();
            let pct = 100.0 * (k as f32) / (n_steps as f32);
            let conv = op.converged.iter().filter(|&&c| c).count();
            let eta = if k > 0 { elapsed / (k as f32) * (n_steps - k) as f32 } else { 0.0 };
            let sub_tag = if sub_levels_used > 0 {
                format!(" sub=2^{sub_levels_used}")
            } else { String::new() };
            eprintln!("[batched-step] {k:>4}/{n_steps}  ({pct:5.1}%)  iters={:>2}  converged={conv}/{n_draws}  elapsed={elapsed:>6.1}s  eta={eta:>5.1}s{sub_tag}",
                op.iters);
        }
        for (idx, h) in histories.iter_mut().enumerate() {
            let in_net = circuit.delays[idx].nets[0];
            let v_in: Vec<f32> = op.voltages.get(&in_net).cloned()
                .unwrap_or_else(|| h.initial_v_in.clone());
            for d in 0..n_draws {
                h.samples[d].push(v_in[d]);
            }
        }
        out.push(BatchedTransientStep {
            t,
            voltages: op.voltages.clone(),
            branch_currents: op.branch_currents.clone(),
            iters: op.iters,
            converged: op.converged,
            final_residual_max: op.final_residual_max,
        });
        prev_v = op.voltages;
    }
    out
}

/// Batched per-step delay scalars (phase-5C). Mirror of
/// [`DelayStepInputs`] for `n_draws`. `blend` and `offset` stay
/// scalar (shared τ across the batch in v1 — MC-ing τ would
/// require per-draw `offset` since `offset = floor(τ/dt) + 1`);
/// `v_lo` and `v_hi` are per-draw `Vec<f32>` of length `n_draws`.
#[derive(Clone, Debug, Default)]
pub struct BatchedDelayStepInputs {
    pub blend: f32,
    pub offset: f32,
    pub v_lo: Vec<f32>,
    pub v_hi: Vec<f32>,
}

/// Per-element delay history with one sample buffer per draw.
struct BatchedDelayHistory {
    tau: f64,
    initial_v_in: Vec<f32>,         // length = n_draws
    samples: Vec<Vec<f32>>,         // outer = per-draw; inner = per-step
}

fn init_batched_delay_histories(
    circuit: &Circuit,
    n_draws: usize,
    params: &HashMap<String, f32>,
    v0: &HashMap<NetId, Vec<f32>>,
) -> Vec<BatchedDelayHistory> {
    circuit.delays.iter().map(|att| {
        let in_net = att.nets[0];
        let v_in0: Vec<f32> = v0.get(&in_net).cloned()
            .unwrap_or_else(|| vec![0.0; n_draws]);
        debug_assert_eq!(v_in0.len(), n_draws);
        let tau = effective_tau(&*att.device, params);
        let samples: Vec<Vec<f32>> = (0..n_draws)
            .map(|d| vec![v_in0[d]])
            .collect();
        BatchedDelayHistory { tau, initial_v_in: v_in0, samples }
    }).collect()
}

fn sample_batched_delay_step(
    histories: &[BatchedDelayHistory],
    step: usize,
    dt: f64,
    n_draws: usize,
) -> Vec<BatchedDelayStepInputs> {
    histories.iter().map(|h| {
        if h.tau < dt {
            // Sub-step: graph interpolates between v_in_prev and v_in_now.
            // History scalars unused; v_lo/v_hi must still be n_draws long
            // so the input bind has the right shape.
            BatchedDelayStepInputs {
                blend: 0.0,
                offset: 1.0,
                v_lo: vec![0.0; n_draws],
                v_hi: vec![0.0; n_draws],
            }
        } else {
            let i = (h.tau / dt).floor() as i64;
            let offset = i + 1;
            let lo_step = step as i64 - offset;
            let hi_step = step as i64 - i;
            let v_lo: Vec<f32> = (0..n_draws).map(|d| {
                if lo_step < 0 {
                    h.initial_v_in[d]
                } else {
                    *h.samples[d].get(lo_step as usize)
                        .unwrap_or(&h.initial_v_in[d])
                }
            }).collect();
            let v_hi: Vec<f32> = (0..n_draws).map(|d| {
                if hi_step < 0 {
                    h.initial_v_in[d]
                } else {
                    *h.samples[d].get(hi_step as usize)
                        .unwrap_or(&h.initial_v_in[d])
                }
            }).collect();
            BatchedDelayStepInputs {
                blend: 1.0,
                offset: offset as f32,
                v_lo,
                v_hi,
            }
        }
    }).collect()
}

/// Multi-step batched transient (phase-5B). Mirror of
/// [`transient_from`] for `n_draws` independent per-draw initial
/// conditions. Loops [`batched_solve_be_step`] for `n_steps` BE
/// steps of size `dt`, threading each step's solved voltages into
/// the next step's `prev_voltages`.
///
/// `initial_voltages` is per-draw (each unknown net's `Vec<f32>` of
/// length n_draws). `boundary_voltages` and `mc_params` stay constant
/// across all steps in this v1 — time-varying sources and per-step
/// MC scalars are phase-5C work.
///
/// Returns `n_steps + 1` rows: index 0 is t=0 (the initial conditions
/// echoed through, no Newton solve), indices 1..=n_steps are the
/// successive BE solves at t = k·dt.
///
/// Skips delays in v1 (passes empty delay slice through). The scalar
/// transient_from threads per-element history buffers — that's phase
/// 5C, since each draw needs its own DelayHistory and the integrator
/// has to keep n_draws independent buffers per delay element.
pub fn batched_transient_from(
    circuit: &Circuit,
    n_draws: usize,
    params: &HashMap<String, f32>,
    mc_params: &HashMap<String, Vec<f32>>,
    boundary_voltages: &HashMap<NetId, Vec<f32>>,
    initial_voltages: &HashMap<NetId, Vec<f32>>,
    dt: f32,
    n_steps: usize,
    opt: NewtonOptions,
) -> Vec<BatchedTransientStep> {
    let mut out: Vec<BatchedTransientStep> = Vec::with_capacity(n_steps + 1);

    // t=0 row: just package the initial conditions + boundaries.
    let mut v0: HashMap<NetId, Vec<f32>> = initial_voltages.clone();
    for (k, vs) in boundary_voltages {
        v0.insert(*k, vs.clone());
    }
    out.push(BatchedTransientStep {
        t: 0.0,
        voltages: v0.clone(),
        branch_currents: HashMap::new(),
        iters: 0,
        converged: vec![true; n_draws],
        final_residual_max: vec![0.0; n_draws],
    });

    // Init per-draw delay history buffers (empty Vec if no delays).
    // Backfill <name>_tau defaults so the per-element tau lookup
    // matches what solve_be_step's effective_params later sees.
    let mut effective_params = params.clone();
    for dl in &circuit.delays {
        let key = format!("{}_tau", dl.device.name());
        effective_params.entry(key).or_insert(
            dl.device.delay_seconds() as f32,
        );
    }
    let mut histories = init_batched_delay_histories(
        circuit, n_draws, &effective_params, &v0,
    );

    // BE-step context built once, reused across every step. Hoists
    // the residual + N jac-graph compile cost out of the loop —
    // dominant per-step cost on the transient parity bench.
    let mc_names_owned: Vec<String> = mc_params.keys().cloned().collect();
    let mc_names: Vec<&str> = mc_names_owned.iter().map(|s| s.as_str()).collect();
    let mut ctx = BatchedBeStepContext::new(circuit, n_draws, &mc_names);
    ctx.set_params(circuit, params);
    let mut inner_cache = InnerSolveCache::default();

    let mut prev_v = v0;
    for k in 1..=n_steps {
        let t_prev = (k as f32 - 1.0) * dt;
        let delay_inputs = sample_batched_delay_step(&histories, k, dt as f64, n_draws);
        let step = batched_solve_be_step_with_ctx(
            &mut ctx, n_draws, mc_params, boundary_voltages,
            &prev_v, &delay_inputs, dt, t_prev, opt, &mut inner_cache,
        );
        // Push solved v_in per-draw into each history for the next step.
        for (idx, h) in histories.iter_mut().enumerate() {
            let in_net = circuit.delays[idx].nets[0];
            let v_in_per_draw = step.voltages.get(&in_net)
                .cloned()
                .unwrap_or_else(|| h.initial_v_in.clone());
            for d in 0..n_draws {
                h.samples[d].push(v_in_per_draw[d]);
            }
        }
        prev_v = step.voltages.clone();
        out.push(step);
    }
    out
}

/// Time-varying boundary variant of [`batched_transient_from`].
/// `boundary_at(t)` returns the per-draw boundary-net voltage map at
/// time `t` (each entry's `Vec<f32>` length = `n_draws`).
///
/// Step 0 (t = 0) uses `initial_voltages` for unknowns and
/// `boundary_at(0.0)` for boundaries. Step k uses `boundary_at(k·dt)`.
/// Unlocks pulse / PWL / per-draw-different stimulus shapes — each
/// draw can carry its own waveform amplitude, edge timing, or even
/// completely different signal class.
pub fn batched_transient_pwl<B>(
    circuit: &Circuit,
    n_draws: usize,
    params: &HashMap<String, f32>,
    mc_params: &HashMap<String, Vec<f32>>,
    boundary_at: B,
    initial_voltages: &HashMap<NetId, Vec<f32>>,
    dt: f32,
    n_steps: usize,
    opt: NewtonOptions,
) -> Vec<BatchedTransientStep>
where
    B: Fn(f32) -> HashMap<NetId, Vec<f32>>,
{
    let mut out: Vec<BatchedTransientStep> = Vec::with_capacity(n_steps + 1);

    // t=0 row: union of initial_voltages (unknown nets) and boundary_at(0)
    // (boundary nets). Each per-net Vec must be length n_draws.
    let bnd0 = boundary_at(0.0);
    for (net, vs) in &bnd0 {
        assert_eq!(vs.len(), n_draws,
            "boundary_at(0)[{net:?}] has {} entries, expected n_draws={n_draws}", vs.len());
    }
    let mut v0: HashMap<NetId, Vec<f32>> = initial_voltages.clone();
    for (net, vs) in &bnd0 {
        v0.insert(*net, vs.clone());
    }
    out.push(BatchedTransientStep {
        t: 0.0,
        voltages: v0.clone(),
        branch_currents: HashMap::new(),
        iters: 0,
        converged: vec![true; n_draws],
        final_residual_max: vec![0.0; n_draws],
    });

    // Init per-draw delay history buffers — same backfill of <name>_tau
    // defaults as batched_transient_from so users don't have to populate
    // params themselves for circuits that just use device-default τ.
    let mut effective_params = params.clone();
    for dl in &circuit.delays {
        let key = format!("{}_tau", dl.device.name());
        effective_params.entry(key).or_insert(
            dl.device.delay_seconds() as f32,
        );
    }
    let mut histories = init_batched_delay_histories(
        circuit, n_draws, &effective_params, &v0,
    );

    // BE-step context built once. Same hoist as `batched_transient_from`.
    let mc_names_owned: Vec<String> = mc_params.keys().cloned().collect();
    let mc_names: Vec<&str> = mc_names_owned.iter().map(|s| s.as_str()).collect();
    let mut ctx = BatchedBeStepContext::new(circuit, n_draws, &mc_names);
    ctx.set_params(circuit, params);
    let mut inner_cache = InnerSolveCache::default();

    let mut prev_v = v0;
    for k in 1..=n_steps {
        let t = k as f32 * dt;
        let bnd = boundary_at(t);
        for (net, vs) in &bnd {
            assert_eq!(vs.len(), n_draws,
                "boundary_at({t})[{net:?}] has {} entries, expected n_draws={n_draws}", vs.len());
        }
        let delay_inputs = sample_batched_delay_step(&histories, k, dt as f64, n_draws);
        let step = batched_solve_be_step_with_ctx(
            &mut ctx, n_draws, mc_params, &bnd,
            &prev_v, &delay_inputs, dt, t - dt, opt, &mut inner_cache,
        );
        for (idx, h) in histories.iter_mut().enumerate() {
            let in_net = circuit.delays[idx].nets[0];
            let v_in_per_draw = step.voltages.get(&in_net)
                .cloned()
                .unwrap_or_else(|| h.initial_v_in.clone());
            for d in 0..n_draws {
                h.samples[d].push(v_in_per_draw[d]);
            }
        }
        prev_v = step.voltages.clone();
        out.push(step);
    }
    out
}

// ── Transient (Backward-Euler) solver ─────────────────────────────────

/// One row of the time-domain solution: every net's voltage at `t`,
/// plus every branch current. Same shape as `DcOperatingPoint` plus a
/// timestamp.
#[derive(Debug, Clone)]
pub struct TransientStep {
    pub t: f32,
    pub voltages: HashMap<NetId, f32>,
    pub branch_currents: HashMap<BranchId, f32>,
    pub iters: usize,
    pub converged: bool,
    pub final_residual_max: f32,
}

/// Solve one Backward-Euler step. Identical Newton machinery as
/// `solve_dc`, but on `build_be_step_residual_graph` and threading the
/// previous-step voltages + timestep `h` into the inputs.
///
/// `prev_voltages` must contain entries for every net the circuit
/// allocated (boundary + unknown) — same map shape as
/// `DcOperatingPoint::voltages`. Boundary entries are forwarded as
/// constants (no Newton variable for them); unknown entries seed the
/// `v_prev_*` Op::Inputs.
///
/// Returns the new operating point (voltages at `t+h`).
///
/// `delay_inputs` carries the per-element scalars for the unified
/// blend stamp — see [`DelayStepInputs`]. Pass an empty slice for
/// circuits without delays. `params` is augmented with default
/// `<name>_tau` entries from each delay device's `delay_seconds()`
/// before being forwarded to the compiled graph, so callers who only
/// care about the static-τ case need not populate `params` themselves.
pub fn solve_be_step(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    boundary_voltages: &HashMap<NetId, f32>,
    prev_voltages: &HashMap<NetId, f32>,
    delay_inputs: &[DelayStepInputs],
    h: f32,
    opt: NewtonOptions,
) -> DcOperatingPoint {
    let _ = delay_inputs.len();    // dimension-checked downstream
    // Backfill <name>_tau from device defaults if the caller didn't
    // supply it. Without this, the τ_param Op::Param is uninitialized
    // (rlx defaults to zero) and α blows up.
    let mut effective_params = params.clone();
    for dl in &circuit.delays {
        let key = format!("{}_tau", dl.device.name());
        effective_params.entry(key).or_insert(
            dl.device.delay_seconds() as f32,
        );
    }
    let params = &effective_params;
    let rg = build_be_step_residual_graph(circuit);
    let unknowns = rg.unknown_nets.clone();
    let branches = rg.branches.clone();
    let n_v = unknowns.len();
    let n_b = branches.len();
    let n = n_v + n_b;

    // Combined unknown wrt list: voltages first, then branch currents.
    let mut unknown_input_ids: Vec<rlx_ir::NodeId> = unknowns
        .iter()
        .map(|net| find_input_node(&rg.graph, &net_input_name(*net))
            .expect("BE residual graph missing unknown-net Op::Input"))
        .collect();
    for bx in &branches {
        unknown_input_ids.push(
            find_input_node(&rg.graph, &branch_input_name(*bx))
                .expect("BE residual graph missing branch Op::Input"),
        );
    }

    use rlx_runtime::{Device, Session};
    let session = Session::new(Device::Cpu);
    let mut compiled_res = session.compile(rg.graph.clone());

    let mut compiled_jac_rows: Vec<rlx_runtime::CompiledGraph> = Vec::with_capacity(n);
    for i in 0..n {
        let mut g_i = rg.graph.clone();
        let out_i = g_i.outputs[i];
        g_i.set_outputs(vec![out_i]);
        let bwd = rlx_opt::autodiff::grad_with_loss(&g_i, &unknown_input_ids);
        compiled_jac_rows.push(session.compile(bwd));
    }

    let set_all = |g: &mut rlx_runtime::CompiledGraph| {
        for (k, v) in params {
            g.set_param(k, &[*v]);
        }
    };
    set_all(&mut compiled_res);
    for g in compiled_jac_rows.iter_mut() {
        set_all(g);
    }

    // Initial guess: previous step's solution (continuity heuristic —
    // typical SPICE move). Falls back to opt.init for any unknown not
    // present in prev_voltages.
    let mut v: Vec<f32> = unknowns.iter().map(|net| {
        prev_voltages.get(net).copied().unwrap_or(opt.init)
    }).collect();
    v.extend(std::iter::repeat(0.0_f32).take(n_b));     // start branch currents at 0

    // Newton + Armijo backtracking — same shape as `solve_dc`.
    let eval_residual = |v: &[f32], compiled_res: &mut rlx_runtime::CompiledGraph|
        -> (Vec<f32>, f32)
    {
        let inputs = build_be_step_inputs(
            &rg.all_nets, &unknowns, &branches,
            boundary_voltages, prev_voltages, delay_inputs,
            &v[..n_v], &v[n_v..], h,
        );
        let inputs_ref: Vec<(&str, &[f32])> = inputs.iter()
            .map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        let f_outs = compiled_res.run(&inputs_ref);
        let f: Vec<f32> = f_outs.iter().map(|o| o[0]).collect();
        let max_abs = f.iter().fold(0.0_f32, |acc, &x| {
            if x.is_finite() { acc.max(x.abs()) } else { f32::INFINITY }
        });
        (f, max_abs)
    };

    let (mut f, mut last_max) = eval_residual(&v, &mut compiled_res);
    let mut converged_at: Option<usize> = None;
    let mut last_step_max = f32::INFINITY;    // vntol guard, see NewtonOptions

    for iter in 0..opt.max_iters {
        if last_max < opt.tol && last_step_max < opt.vntol {
            converged_at = Some(iter);
            break;
        }

        let inputs = build_be_step_inputs(
            &rg.all_nets, &unknowns, &branches,
            boundary_voltages, prev_voltages, delay_inputs,
            &v[..n_v], &v[n_v..], h,
        );
        let inputs_ref: Vec<(&str, &[f32])> = inputs.iter()
            .map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        let one_seed = [1.0_f32];
        let mut jac = vec![0.0_f32; n * n];
        for i in 0..n {
            let mut grad_inputs = inputs_ref.clone();
            grad_inputs.push(("d_output", &one_seed[..]));
            let outs = compiled_jac_rows[i].run(&grad_inputs);
            for j in 0..n {
                jac[i * n + j] = outs[1 + j][0];
            }
        }

        let neg_f: Vec<f32> = f.iter().map(|x| -x).collect();
        let dv = match linear_solve(&jac, &neg_f, n) {
            Some(dv) => dv,
            None => break,
        };

        let mut alpha = 1.0_f32;
        let mut v_trial = vec![0.0_f32; n];
        let (mut f_new, mut max_new) = (Vec::new(), f32::INFINITY);
        for _ in 0..=opt.max_backtracks {
            for i in 0..n { v_trial[i] = v[i] + alpha * dv[i]; }
            let (f_t, max_t) = eval_residual(&v_trial, &mut compiled_res);
            if max_t.is_finite() && max_t < last_max {
                f_new = f_t;
                max_new = max_t;
                break;
            }
            alpha *= 0.5;
            f_new = f_t;
            max_new = max_t;
        }
        // Step size for next iter's convergence check.
        let mut step_max = 0.0_f32;
        for i in 0..n {
            let dv_i = (v_trial[i] - v[i]).abs();
            if dv_i > step_max { step_max = dv_i; }
        }
        last_step_max = step_max;
        v = v_trial;
        f = f_new;
        last_max = max_new;
    }
    let _ = f;     // silence unused after the last iter

    let iters = converged_at.unwrap_or(opt.max_iters);
    let mut voltages = HashMap::new();
    for (idx, net) in unknowns.iter().enumerate() {
        voltages.insert(*net, v[idx]);
    }
    for (net, val) in boundary_voltages {
        voltages.insert(*net, *val);
    }
    let mut branch_currents = HashMap::new();
    for (idx, bx) in branches.iter().enumerate() {
        branch_currents.insert(*bx, v[n_v + idx]);
    }
    DcOperatingPoint {
        voltages,
        branch_currents,
        iters,
        converged: converged_at.is_some(),
        final_residual_max: last_max,
    }
}

// ── Cached BE-step context (T.10 fast path) ───────────────────────────

/// Compiled-graph cache for repeated `solve_be_step` calls on the same
/// circuit. The residual graph + per-row gradient graphs depend only
/// on `circuit`, not on `params` or per-step inputs, so we lift the
/// build-and-compile work out of the per-step loop.
pub struct BeStepContext {
    rg: ResidualGraph,
    unknowns: Vec<NetId>,
    branches: Vec<BranchId>,
    n_v: usize,
    n_b: usize,
    #[allow(dead_code)]
    unknown_input_ids: Vec<rlx_ir::NodeId>,
    compiled_res: rlx_runtime::CompiledGraph,
    compiled_jac_rows: Vec<rlx_runtime::CompiledGraph>,
    /// T.11.C — fused Jacobian via `jvp(forward) → vmap(over_tangents)`.
    /// When `Some`, Newton uses ONE compiled-graph run per iter
    /// instead of `n` per-row runs. Opt-in via `RLX_JAC_MODE=fused`.
    compiled_jac_fused: Option<rlx_runtime::CompiledGraph>,
    /// `tangent_<input>` names in unknowns-then-branches order.
    fused_tangent_names: Vec<String>,
}

impl BeStepContext {
    pub fn new(circuit: &Circuit) -> Self {
        use rlx_runtime::{Device, Session};
        let rg = build_be_step_residual_graph(circuit);
        let unknowns = rg.unknown_nets.clone();
        let branches = rg.branches.clone();
        let n_v = unknowns.len();
        let n_b = branches.len();
        let n = n_v + n_b;
        let mut unknown_input_ids: Vec<rlx_ir::NodeId> = unknowns.iter()
            .map(|net| find_input_node(&rg.graph, &net_input_name(*net))
                .expect("BE residual graph missing unknown-net Op::Input"))
            .collect();
        for bx in &branches {
            unknown_input_ids.push(
                find_input_node(&rg.graph, &branch_input_name(*bx))
                    .expect("BE residual graph missing branch Op::Input"),
            );
        }
        let session = Session::new(Device::Cpu);
        let compiled_res = session.compile(rg.graph.clone());
        let mut compiled_jac_rows = Vec::with_capacity(n);
        for i in 0..n {
            let mut g_i = rg.graph.clone();
            let out_i = g_i.outputs[i];
            g_i.set_outputs(vec![out_i]);
            let bwd = rlx_opt::autodiff::grad_with_loss(&g_i, &unknown_input_ids);
            compiled_jac_rows.push(session.compile(bwd));
        }
        // T.11.C — fused path, opt-in. `fused` builds on CPU,
        // `fused-mlx` builds on the Apple Metal/MLX backend so the
        // batched jvp+vmap graph dispatches to GPU at solve time.
        // Falls back to per-row on any build failure (catches op-rule
        // gaps in jvp/vmap coverage).
        let jac_mode = std::env::var("RLX_JAC_MODE").ok();
        let fused_session: rlx_runtime::Session = match jac_mode.as_deref() {
            Some("fused-mlx") => Session::new(Device::Mlx),
            _ => Session::new(Device::Cpu),
        };
        let (compiled_jac_fused, fused_tangent_names) =
            if matches!(jac_mode.as_deref(), Some("fused") | Some("fused-mlx")) {
                build_fused_jacobian(&rg, &unknowns, &branches, &unknown_input_ids, n, &fused_session)
            } else { (None, Vec::new()) };

        Self {
            rg, unknowns, branches, n_v, n_b,
            unknown_input_ids, compiled_res, compiled_jac_rows,
            compiled_jac_fused, fused_tangent_names,
        }
    }

    /// Stamp every (k, v) in `params` onto every compiled graph.
    /// Cheap relative to `new`. Call once per parameter change.
    pub fn set_params(&mut self, params: &HashMap<String, f32>) {
        for (k, v) in params {
            self.compiled_res.set_param(k, &[*v]);
            for g in self.compiled_jac_rows.iter_mut() {
                g.set_param(k, &[*v]);
            }
            if let Some(g) = &mut self.compiled_jac_fused {
                g.set_param(k, &[*v]);
            }
        }
    }
}

/// T.11.C — build one fused Jacobian graph via jvp + vmap. Catches
/// any op-rule gap (jvp/vmap unhandled) and returns `None` so the
/// caller falls back to the per-row path.
fn build_fused_jacobian(
    rg: &ResidualGraph,
    unknowns: &[NetId],
    branches: &[BranchId],
    unknown_input_ids: &[rlx_ir::NodeId],
    n: usize,
    session: &rlx_runtime::Session,
) -> (Option<rlx_runtime::CompiledGraph>, Vec<String>) {
    use std::panic::AssertUnwindSafe;
    let result: std::thread::Result<(rlx_runtime::CompiledGraph, Vec<String>)> =
        std::panic::catch_unwind(AssertUnwindSafe(|| {
            // jvp returns [primal_0..primal_{n-1}, tangent_0..tangent_{n-1}].
            // Tangent input names follow the "tangent_<original>" convention.
            let jvp_g = rlx_opt::autodiff_fwd::jvp(&rg.graph, unknown_input_ids);
            let tangent_names_owned: Vec<String> = unknowns.iter()
                .map(|net| format!("tangent_{}", net_input_name(*net)))
                .chain(branches.iter().map(|b| format!("tangent_{}", branch_input_name(*b))))
                .collect();
            let names: Vec<&str> = tangent_names_owned.iter().map(|s| s.as_str()).collect();
            let batched = rlx_opt::vmap::vmap(&jvp_g, &names, n);
            // Keep only the tangent outputs (last n of the 2n outputs):
            // each becomes a length-n batch = one row of the Jacobian.
            let mut g = batched;
            let outs = g.outputs.clone();
            let tangent_outs: Vec<rlx_ir::NodeId> = outs.iter().skip(n).copied().collect();
            g.set_outputs(tangent_outs);
            let compiled = session.compile(g);
            (compiled, tangent_names_owned)
        }));
    match result {
        Ok((compiled, names)) => (Some(compiled), names),
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() { s.to_string() }
                else if let Some(s) = e.downcast_ref::<String>() { s.clone() }
                else { "unknown panic in jvp+vmap".to_string() };
            eprintln!("T.11.C: fused-Jacobian build failed ({msg}); falling back to per-row.");
            (None, Vec::new())
        }
    }
}

/// Solve one BE step using a pre-compiled [`BeStepContext`]. Caller
/// must have called `ctx.set_params(params)` at least once with the
/// active params. Identical numerics to [`solve_be_step`] — purely a
/// performance optimization.
pub fn solve_be_step_with_ctx(
    ctx: &mut BeStepContext,
    boundary_voltages: &HashMap<NetId, f32>,
    prev_voltages: &HashMap<NetId, f32>,
    delay_inputs: &[DelayStepInputs],
    h: f32,
    opt: NewtonOptions,
) -> DcOperatingPoint {
    let n_v = ctx.n_v;
    let n_b = ctx.n_b;
    let n = n_v + n_b;
    let eval_residual = |v: &[f32], compiled_res: &mut rlx_runtime::CompiledGraph|
        -> (Vec<f32>, f32)
    {
        let inputs = build_be_step_inputs(
            &ctx.rg.all_nets, &ctx.unknowns, &ctx.branches,
            boundary_voltages, prev_voltages, delay_inputs,
            &v[..n_v], &v[n_v..], h,
        );
        let inputs_ref: Vec<(&str, &[f32])> = inputs.iter()
            .map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        let f_outs = compiled_res.run(&inputs_ref);
        let f: Vec<f32> = f_outs.iter().map(|o| o[0]).collect();
        let max_abs = f.iter().fold(0.0_f32, |acc, &x| {
            if x.is_finite() { acc.max(x.abs()) } else { f32::INFINITY }
        });
        (f, max_abs)
    };
    let mut v: Vec<f32> = ctx.unknowns.iter().map(|net| {
        prev_voltages.get(net).copied().unwrap_or(opt.init)
    }).collect();
    v.extend(std::iter::repeat(0.0_f32).take(n_b));
    let (mut f, mut last_max) = eval_residual(&v, &mut ctx.compiled_res);
    let mut converged_at: Option<usize> = None;
    let mut last_step_max = f32::INFINITY;
    let one_seed = [1.0_f32];
    for iter in 0..opt.max_iters {
        if last_max < opt.tol && last_step_max < opt.vntol {
            converged_at = Some(iter); break;
        }
        let inputs = build_be_step_inputs(
            &ctx.rg.all_nets, &ctx.unknowns, &ctx.branches,
            boundary_voltages, prev_voltages, delay_inputs,
            &v[..n_v], &v[n_v..], h,
        );
        let inputs_ref: Vec<(&str, &[f32])> = inputs.iter()
            .map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        let mut jac = vec![0.0_f32; n * n];
        if let Some(fused) = &mut ctx.compiled_jac_fused {
            // T.11.C — one batched run gives the full Jacobian.
            // Bind tangent_<unknown_i> to the i-th row of the n×n
            // identity. Output i (tangent of residual i) comes back
            // as a length-n vector = row i of J.
            let mut tangent_buffers: Vec<Vec<f32>> = (0..n).map(|i| {
                let mut row = vec![0.0_f32; n];
                row[i] = 1.0;
                row
            }).collect();
            let mut grad_inputs = inputs_ref.clone();
            for (i, name) in ctx.fused_tangent_names.iter().enumerate() {
                grad_inputs.push((name.as_str(), tangent_buffers[i].as_slice()));
            }
            let outs = fused.run(&grad_inputs);
            for i in 0..n {
                let row = &outs[i];
                // Output may be shape [n] (batched) or [1] (no tangent
                // dependency); broadcast in the latter case.
                if row.len() == n {
                    for j in 0..n { jac[i * n + j] = row[j]; }
                } else {
                    let v = row[0];
                    for j in 0..n { jac[i * n + j] = v; }
                }
            }
            // Tangent_buffers held by reference until grad_inputs drops; explicit no-op.
            let _ = &mut tangent_buffers;
        } else {
            for i in 0..n {
                let mut grad_inputs = inputs_ref.clone();
                grad_inputs.push(("d_output", &one_seed[..]));
                let outs = ctx.compiled_jac_rows[i].run(&grad_inputs);
                for j in 0..n { jac[i * n + j] = outs[1 + j][0]; }
            }
        }
        let neg_f: Vec<f32> = f.iter().map(|x| -x).collect();
        let dv = match linear_solve(&jac, &neg_f, n) {
            Some(dv) => dv, None => break,
        };
        let mut alpha = 1.0_f32;
        let mut v_trial = vec![0.0_f32; n];
        let (mut f_new, mut max_new) = (Vec::new(), f32::INFINITY);
        for _ in 0..=opt.max_backtracks {
            for i in 0..n { v_trial[i] = v[i] + alpha * dv[i]; }
            let (f_t, max_t) = eval_residual(&v_trial, &mut ctx.compiled_res);
            if max_t.is_finite() && max_t < last_max {
                f_new = f_t; max_new = max_t; break;
            }
            alpha *= 0.5;
            f_new = f_t; max_new = max_t;
        }
        // Step size for next iter's vntol check.
        let mut step_max = 0.0_f32;
        for i in 0..n {
            let dv_i = (v_trial[i] - v[i]).abs();
            if dv_i > step_max { step_max = dv_i; }
        }
        last_step_max = step_max;
        v = v_trial; f = f_new; last_max = max_new;
    }
    let _ = f;
    let iters = converged_at.unwrap_or(opt.max_iters);
    let mut voltages = HashMap::new();
    for (idx, net) in ctx.unknowns.iter().enumerate() {
        voltages.insert(*net, v[idx]);
    }
    for (net, val) in boundary_voltages {
        voltages.insert(*net, *val);
    }
    let mut branch_currents = HashMap::new();
    for (idx, b) in ctx.branches.iter().enumerate() {
        branch_currents.insert(*b, v[n_v + idx]);
    }
    DcOperatingPoint {
        voltages, branch_currents, iters,
        converged: converged_at.is_some(),
        final_residual_max: last_max,
    }
}

/// Drive `solve_be_step` for `n_steps` steps of size `dt`, threading
/// each step's solution as the next step's `prev_voltages`. The
/// initial state is the DC operating point (caps treated as open
/// circuits at DC), computed by an internal `solve_dc` call.
///
/// Returns one `TransientStep` per step (the t=0 DC OP is index 0;
/// step `k` lives at `t = k·dt`). Output length is `n_steps + 1`.
pub fn transient(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    boundary_voltages: &HashMap<NetId, f32>,
    dt: f32,
    n_steps: usize,
    opt: NewtonOptions,
) -> Vec<TransientStep> {
    let dc = solve_dc(circuit, params, boundary_voltages, opt);
    transient_from(circuit, params, boundary_voltages, &dc.voltages,
                   dt, n_steps, opt)
}

/// Like [`transient`], but starts from an explicit initial condition
/// (`initial_voltages`, keyed by `NetId`) rather than the DC OP. Use
/// this for charge/discharge tests where the cap starts at a voltage
/// that doesn't match steady-state — e.g. cap-charges-from-zero,
/// pulse-response from a quiescent state. Boundary nets in
/// `initial_voltages` are ignored (boundaries always come from
/// `boundary_voltages`).
pub fn transient_from(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    boundary_voltages: &HashMap<NetId, f32>,
    initial_voltages: &HashMap<NetId, f32>,
    dt: f32,
    n_steps: usize,
    opt: NewtonOptions,
) -> Vec<TransientStep> {
    let mut out: Vec<TransientStep> = Vec::with_capacity(n_steps + 1);

    // Build the t=0 step from the explicit initial conditions, filling
    // boundary entries from boundary_voltages.
    let mut v0: HashMap<NetId, f32> = initial_voltages.clone();
    for (k, v) in boundary_voltages {
        v0.insert(*k, *v);
    }
    out.push(TransientStep {
        t: 0.0,
        voltages: v0.clone(),
        branch_currents: HashMap::new(),
        iters: 0,
        converged: true,
        final_residual_max: 0.0,
    });

    let mut histories = init_delay_histories(circuit, params, &v0);
    let mut prev_v: HashMap<NetId, f32> = v0;
    for k in 1..=n_steps {
        let delay_inputs = sample_delay_step(&histories, k, dt as f64);
        let op = solve_be_step(circuit, params, boundary_voltages,
                               &prev_v, &delay_inputs, dt, opt);
        for (idx, h) in histories.iter_mut().enumerate() {
            let in_net = circuit.delays[idx].nets[0];
            let v_in = op.voltages.get(&in_net).copied()
                .unwrap_or(h.initial_v_in);
            h.samples.push(v_in);
        }
        out.push(TransientStep {
            t: k as f32 * dt,
            voltages: op.voltages.clone(),
            branch_currents: op.branch_currents.clone(),
            iters: op.iters,
            converged: op.converged,
            final_residual_max: op.final_residual_max,
        });
        prev_v = op.voltages;
    }
    out
}

/// Per-delay history buffer used by the transient drivers. `samples[k]
/// = v_in(k·dt)`, populated incrementally; `initial_v_in` is the
/// constant-history value used for `t < 0` queries.
struct DelayHistory {
    /// `τ` value the integrator uses for buffer indexing this run. Read
    /// once at `init_delay_histories` from `params["<name>_tau"]` (with
    /// `device.delay_seconds()` as fallback) — kept in sync with the
    /// Param the residual graph reads each step. Differentiating wrt τ
    /// across multiple solve calls is the user's responsibility; within
    /// one call it's fixed.
    tau: f64,
    initial_v_in: f32,
    samples: Vec<f32>,
}

fn effective_tau(
    device: &dyn TransientDelay,
    params: &HashMap<String, f32>,
) -> f64 {
    params.get(&format!("{}_tau", device.name()))
        .map(|v| *v as f64)
        .unwrap_or_else(|| device.delay_seconds())
}

fn init_delay_histories(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    v0: &HashMap<NetId, f32>,
) -> Vec<DelayHistory> {
    circuit.delays.iter().map(|att| {
        let in_net = att.nets[0];
        let v_in0 = v0.get(&in_net).copied().unwrap_or(0.0);
        let tau = effective_tau(&*att.device, params);
        DelayHistory {
            tau,
            initial_v_in: v_in0,
            samples: vec![v_in0],
        }
    }).collect()
}

/// Compute the per-step blend-stamp scalars for every delay element at
/// step `k` (time `t_k = k·dt`). Sub-step delays select the in-graph
/// interpolation path (blend = 0); long delays look up the two
/// surrounding history samples (blend = 1). Constant-history convention
/// for `t < 0` queries.
fn sample_delay_step(
    histories: &[DelayHistory],
    step: usize,
    dt: f64,
) -> Vec<DelayStepInputs> {
    histories.iter().map(|h| {
        if h.tau < dt {
            // Sub-step: graph interpolates between v_in_prev and v_in_now.
            // History-side scalars unused; pass zeros so the blended
            // (1 − blend) · history terms vanish cleanly.
            DelayStepInputs {
                blend: 0.0,
                offset: 1.0,
                v_lo: 0.0,
                v_hi: 0.0,
            }
        } else {
            // Long delay: i = floor(τ/dt), offset = i + 1.
            // v_lo = sample at step − offset, v_hi = sample at step − i.
            let i = (h.tau / dt).floor() as i64;
            let offset = i + 1;
            let lo_step = step as i64 - offset;
            let hi_step = step as i64 - i;
            let v_lo = if lo_step < 0 {
                h.initial_v_in
            } else {
                *h.samples.get(lo_step as usize).unwrap_or(&h.initial_v_in)
            };
            let v_hi = if hi_step < 0 {
                h.initial_v_in
            } else {
                *h.samples.get(hi_step as usize).unwrap_or(&h.initial_v_in)
            };
            DelayStepInputs {
                blend: 1.0,
                offset: offset as f32,
                v_lo,
                v_hi,
            }
        }
    }).collect()
}

/// Time-varying boundary variant: `boundary_at(t)` returns the
/// boundary-net voltage map at time `t`. Called once per BE step
/// (at `t = k·dt`) before invoking [`solve_be_step`]. Useful for
/// PULSE / PWL / sinusoidal stimuli without splitting the run into
/// multiple stages.
///
/// Step 0 (t = 0) uses `initial_voltages` for unknowns and
/// `boundary_at(0.0)` for boundaries. Step `k` (k ≥ 1) uses
/// `boundary_at(k·dt)`.
///
/// Helper: see [`pulse_boundary`] for a common rectangular-pulse
/// stimulus on a single boundary net.
pub fn transient_pwl<B>(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    boundary_at: B,
    initial_voltages: &HashMap<NetId, f32>,
    dt: f32,
    n_steps: usize,
    opt: NewtonOptions,
) -> Vec<TransientStep>
where
    B: Fn(f32) -> HashMap<NetId, f32>,
{
    let mut out: Vec<TransientStep> = Vec::with_capacity(n_steps + 1);

    let bnd0 = boundary_at(0.0);
    let mut v0: HashMap<NetId, f32> = initial_voltages.clone();
    for (k, v) in &bnd0 {
        v0.insert(*k, *v);
    }
    out.push(TransientStep {
        t: 0.0,
        voltages: v0.clone(),
        branch_currents: HashMap::new(),
        iters: 0,
        converged: true,
        final_residual_max: 0.0,
    });

    let mut histories = init_delay_histories(circuit, params, &v0);
    let mut prev_v: HashMap<NetId, f32> = v0;

    // T.10 fast path: compile residual + per-row gradient graphs ONCE
    // before the loop, reuse on every BE step. With ~150 unknowns the
    // per-step compile cost (~1-2 s in release) was previously the
    // dominant runtime; this drops it to a single one-time cost +
    // a per-step `set_param` + Newton solve.
    let mut effective_params = params.clone();
    for dl in &circuit.delays {
        let key = format!("{}_tau", dl.device.name());
        effective_params.entry(key).or_insert(
            dl.device.delay_seconds() as f32,
        );
    }
    let mut ctx = BeStepContext::new(circuit);
    ctx.set_params(&effective_params);

    for k in 1..=n_steps {
        let t = k as f32 * dt;
        let bnd = boundary_at(t);
        let delay_inputs = sample_delay_step(&histories, k, dt as f64);
        let op = solve_be_step_with_ctx(
            &mut ctx, &bnd, &prev_v, &delay_inputs, dt, opt,
        );
        for (idx, h) in histories.iter_mut().enumerate() {
            let in_net = circuit.delays[idx].nets[0];
            let v_in = op.voltages.get(&in_net).copied()
                .unwrap_or(h.initial_v_in);
            h.samples.push(v_in);
        }
        out.push(TransientStep {
            t,
            voltages: op.voltages.clone(),
            branch_currents: op.branch_currents.clone(),
            iters: op.iters,
            converged: op.converged,
            final_residual_max: op.final_residual_max,
        });
        prev_v = op.voltages;
    }
    out
}

/// Build a `boundary_at(t)` closure for a single rectangular pulse on
/// `pulse_net`: low (`v_lo`) before `t_rise`, high (`v_hi`) between
/// `t_rise` and `t_fall`, then back to `v_lo`. Other boundary entries
/// in `static_boundary` are returned unchanged at every `t`.
///
/// Use as `transient_pwl(c, params, pulse_boundary(...), ic, dt, n,
/// opt)` to drive an inverter / DFF / SAR cell with a clean clock or
/// data edge through the BE-step solver.
pub fn pulse_boundary(
    static_boundary: HashMap<NetId, f32>,
    pulse_net: NetId,
    v_lo: f32, v_hi: f32,
    t_rise: f32, t_fall: f32,
) -> impl Fn(f32) -> HashMap<NetId, f32> {
    move |t: f32| {
        let mut bnd = static_boundary.clone();
        let v = if t < t_rise || t >= t_fall { v_lo } else { v_hi };
        bnd.insert(pulse_net, v);
        bnd
    }
}

/// At a converged operating point, return `∂v_unknown/∂param` for each
/// requested param. Uses the implicit-function theorem:
///
/// ```text
///   ∂v*/∂p = − J⁻¹ · ∂f/∂p     where J = ∂f/∂v at v*
/// ```
///
/// Compiles N reverse-mode gradient graphs (one per residual) with
/// both unknown voltages **and** the requested params in the `wrt`
/// list. Per residual `i`, reads off row `i` of `J` and the `i`-th
/// entry of every `∂f/∂p_k` column. One Gauss-Jordan solve per param
/// then yields its sensitivity vector.
///
/// Returns `param_name → ∂v_unknown/∂param`, with the inner Vec in
/// `unknown_nets` order (matching `build_residual_graph`).
pub fn sensitivities(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    boundary_voltages: &HashMap<NetId, f32>,
    op: &DcOperatingPoint,
    wrt_params: &[String],
) -> HashMap<String, Vec<f32>> {
    let rg = build_residual_graph(circuit);
    let unknowns = rg.unknown_nets.clone();
    let branches = rg.branches.clone();
    let n_v = unknowns.len();
    let n_b = branches.len();
    let n = n_v + n_b;
    let m = wrt_params.len();
    if n == 0 || m == 0 { return HashMap::new(); }

    let mut unknown_input_ids: Vec<rlx_ir::NodeId> = unknowns
        .iter()
        .map(|net| find_input_node(&rg.graph, &net_input_name(*net))
            .expect("residual graph missing unknown-net Op::Input"))
        .collect();
    for b in &branches {
        unknown_input_ids.push(
            find_input_node(&rg.graph, &branch_input_name(*b))
                .expect("residual graph missing branch Op::Input"),
        );
    }

    let param_ids: Vec<rlx_ir::NodeId> = wrt_params
        .iter()
        .map(|name| find_param_node(&rg.graph, name)
            .unwrap_or_else(|| panic!(
                "param {name:?} not present in the residual graph — \
                 did the device's `currents` impl emit it under a \
                 different name?")))
        .collect();

    // Combined wrt: unknown voltages + branches, then params. Output order follows.
    let mut wrt_ids = unknown_input_ids.clone();
    wrt_ids.extend(&param_ids);

    use rlx_runtime::{Device, Session};
    let session = Session::new(Device::Cpu);

    let mut compiled_rows: Vec<rlx_runtime::CompiledGraph> = Vec::with_capacity(n);
    for i in 0..n {
        let mut g_i = rg.graph.clone();
        let out_i = g_i.outputs[i];
        g_i.set_outputs(vec![out_i]);
        let bwd = rlx_opt::autodiff::grad_with_loss(&g_i, &wrt_ids);
        let mut compiled = session.compile(bwd);
        for (k, v) in params {
            compiled.set_param(k, &[*v]);
        }
        compiled_rows.push(compiled);
    }

    // Operating-point inputs.
    let v_unknowns: Vec<f32> = unknowns.iter()
        .map(|net| op.voltages.get(net).copied().unwrap_or(0.0))
        .collect();
    let i_branches: Vec<f32> = branches.iter()
        .map(|b| op.branch_currents.get(b).copied().unwrap_or(0.0))
        .collect();
    let inputs = build_inputs(
        &rg.all_nets, &unknowns, &branches, boundary_voltages,
        &v_unknowns, &i_branches,
    );
    let inputs_ref: Vec<(&str, &[f32])> = inputs.iter()
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();

    let one_seed = [1.0_f32];
    let mut jac    = vec![0.0_f32; n * n];      // [n, n] row-major
    let mut df_dp  = vec![0.0_f32; n * m];      // [n, m] row-major
    for i in 0..n {
        let mut grad_inputs = inputs_ref.clone();
        grad_inputs.push(("d_output", &one_seed[..]));
        let outs = compiled_rows[i].run(&grad_inputs);
        for j in 0..n {
            jac[i * n + j] = outs[1 + j][0];
        }
        for k in 0..m {
            df_dp[i * m + k] = outs[1 + n + k][0];
        }
    }

    // Solve J · (∂v/∂p_k) = − ∂f/∂p_k per param.
    let mut result: HashMap<String, Vec<f32>> = HashMap::new();
    for k in 0..m {
        let rhs: Vec<f32> = (0..n).map(|i| -df_dp[i * m + k]).collect();
        if let Some(dvdp) = linear_solve(&jac, &rhs, n) {
            result.insert(wrt_params[k].clone(), dvdp);
        }
    }
    result
}

/// Forward sensitivities through a Backward-Euler transient.
///
/// Given a converged transient `trace` (e.g. from `transient_pwl` or
/// `transient_from`), return `∂v_k/∂p` for each unknown net at each
/// step, for each requested parameter. Uses per-step IFT on the BE
/// residual `r(v_k, v_{k-1}, p) = 0`:
///
/// ```text
///   J_k · ds_k/dp = − ∂r/∂p − ∂r/∂v_{k-1} · ds_{k-1}/dp
/// ```
///
/// where `J_k = ∂r/∂v_k` is the same Jacobian Newton converged on at
/// this step. The history coupling `∂r/∂v_{k-1}` comes from cap (and
/// any other storage) stamps; for non-cap nets it's identically zero.
///
/// Initial sensitivity `ds_0/dp` defaults to zero — i.e., we assume
/// the user's initial conditions don't depend on the parameters being
/// optimized. (If they do, seed the recurrence by overwriting
/// `result[param][0]` after this returns.)
///
/// Returns `param → per-step → per-unknown-net`. The inner-most Vec
/// is in `build_residual_graph(circuit).unknown_nets` order; per-step
/// vector has length `trace.len()`.
///
/// Cost: `O(n_steps · n_params · solve)` where `solve` is one
/// Gauss-Jordan pass on the Jacobian (cached per-step). Memory:
/// `O(n_unknowns · n_params)` for the propagating sensitivity vectors.
/// Approach 1 in the T.1 plan; the adjoint variant (Approach 2) lands
/// when this hits a parameter-count wall.
pub fn transient_sensitivities(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    boundary_voltages: &HashMap<NetId, f32>,
    trace: &[TransientStep],
    h: f32,
    wrt_params: &[String],
) -> HashMap<String, Vec<Vec<f32>>> {
    let rg = build_be_step_residual_graph(circuit);
    let unknowns = rg.unknown_nets.clone();
    let branches = rg.branches.clone();
    let all_nets = rg.all_nets.clone();
    let n_v = unknowns.len();
    let n_b = branches.len();
    let n   = n_v + n_b;
    let m   = wrt_params.len();
    if n == 0 || m == 0 || trace.is_empty() {
        return HashMap::new();
    }

    // Map each unknown net to its index in the unknowns vector — used
    // to project ds_{k-1}/dp (defined on unknowns) onto the
    // all-nets-shaped prev-voltage gradient column.
    let unknown_idx: HashMap<NetId, usize> = unknowns.iter()
        .enumerate().map(|(i, n)| (*n, i)).collect();

    // Find input/param node ids in the residual graph (topology fixed
    // across steps, so only resolved once).
    let mut unknown_input_ids: Vec<rlx_ir::NodeId> = unknowns.iter()
        .map(|net| find_input_node(&rg.graph, &net_input_name(*net))
            .expect("BE residual graph missing unknown-net Op::Input"))
        .collect();
    for bx in &branches {
        unknown_input_ids.push(
            find_input_node(&rg.graph, &branch_input_name(*bx))
                .expect("BE residual graph missing branch Op::Input"),
        );
    }
    let param_ids: Vec<rlx_ir::NodeId> = wrt_params.iter()
        .map(|name| find_param_node(&rg.graph, name)
            .unwrap_or_else(|| panic!(
                "param {name:?} not present in the BE residual graph")))
        .collect();
    // Only include prev-voltage inputs for nets that actually appear
    // on a storage terminal — others have no consumer and would trip
    // rlx_opt::autodiff's "no gradient flowed" check.
    let storage_nets: Vec<NetId> = circuit.storage_coupled_nets()
        .into_iter().collect();
    let prev_input_ids: Vec<rlx_ir::NodeId> = storage_nets.iter()
        .map(|net| find_input_node(&rg.graph, &prev_voltage_input_name(*net))
            .expect("BE residual graph missing prev-voltage Op::Input"))
        .collect();

    // Combined wrt: unknowns (n) + params (m) + prev_voltages (n_storage_nets).
    // Residual row `i` (KCL at node i) only depends on a SUBSET of these —
    // chains/multi-stage circuits routinely have rows that don't see
    // every unknown or every param. `grad_with_loss` panics if a wrt
    // id has no path to the output, so we filter per-row by reachability
    // and zero-fill the missing slots downstream.
    let n_storage = storage_nets.len();
    let global_wrt_ids: Vec<rlx_ir::NodeId> = {
        let mut v = unknown_input_ids.clone();
        v.extend(&param_ids);
        v.extend(&prev_input_ids);
        v
    };

    // Compile one gradient graph per residual row (n total). Reused
    // across all timesteps — only the input feeds change. Each row's
    // wrt list is filtered to the subset of global_wrt_ids that are
    // actually ancestors of this row's output. `row_wrt_index[i][k]`
    // maps the k-th wrt slot in compiled_rows[i].run output back to
    // its position in `global_wrt_ids`.
    use rlx_runtime::{Device, Session};
    let session = Session::new(Device::Cpu);
    let mut compiled_rows: Vec<rlx_runtime::CompiledGraph> = Vec::with_capacity(n);
    let mut row_wrt_index: Vec<Vec<usize>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut g_i = rg.graph.clone();
        let out_i = g_i.outputs[i];
        g_i.set_outputs(vec![out_i]);
        let ancestors = ancestors_of(&g_i, out_i);
        let mut row_wrt: Vec<rlx_ir::NodeId> = Vec::new();
        let mut idx_map: Vec<usize> = Vec::new();
        for (gi, id) in global_wrt_ids.iter().enumerate() {
            if ancestors.contains(id) {
                row_wrt.push(*id);
                idx_map.push(gi);
            }
        }
        let bwd = rlx_opt::autodiff::grad_with_loss(&g_i, &row_wrt);
        let mut compiled = session.compile(bwd);
        for (k, v) in params {
            compiled.set_param(k, &[*v]);
        }
        compiled_rows.push(compiled);
        row_wrt_index.push(idx_map);
    }

    // Result allocations: zero-initialized at step 0.
    let mut result: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    for name in wrt_params {
        let mut per_step = Vec::with_capacity(trace.len());
        per_step.push(vec![0.0_f32; n_v]); // ds_0/dp = 0
        result.insert(name.clone(), per_step);
    }

    // Reused buffers across timesteps. df_dvprev is sized to n_storage
    // (only nets touched by caps); all other prev-voltage couplings are
    // structurally zero.
    let mut jac      = vec![0.0_f32; n * n];
    let mut df_dp    = vec![0.0_f32; n * m];
    let mut df_dvprev = vec![0.0_f32; n * n_storage];
    let one_seed = [1.0_f32];

    // Per-step IFT recurrence.
    let no_delays: Vec<DelayStepInputs> = vec![
        DelayStepInputs { blend: 0.0, offset: 0.0, v_lo: 0.0, v_hi: 0.0 };
        circuit.delays.len()
    ];
    for k in 1..trace.len() {
        let v_curr = &trace[k].voltages;
        let v_prev = &trace[k - 1].voltages;
        let i_curr = &trace[k].branch_currents;

        // Voltage / current concrete values for this step's converged
        // operating point (passed into the residual graph).
        let v_unknown_vals: Vec<f32> = unknowns.iter()
            .map(|net| v_curr.get(net).copied().unwrap_or(0.0))
            .collect();
        let i_branch_vals: Vec<f32> = branches.iter()
            .map(|b| i_curr.get(b).copied().unwrap_or(0.0))
            .collect();
        let inputs = build_be_step_inputs(
            &all_nets, &unknowns, &branches,
            boundary_voltages, v_prev, &no_delays,
            &v_unknown_vals, &i_branch_vals, h,
        );
        let inputs_ref: Vec<(&str, &[f32])> = inputs.iter()
            .map(|(n, v)| (n.as_str(), v.as_slice())).collect();

        // Run all n gradient graphs at this timestep, fill J / df_dp /
        // df_dvprev (the latter only over storage-coupled nets). Each
        // row only emits gradients for its own wrt subset; map back
        // through `row_wrt_index[i]` and zero-fill the rest.
        for v in jac.iter_mut() { *v = 0.0; }
        for v in df_dp.iter_mut() { *v = 0.0; }
        for v in df_dvprev.iter_mut() { *v = 0.0; }
        for i in 0..n {
            let mut grad_inputs = inputs_ref.clone();
            grad_inputs.push(("d_output", &one_seed[..]));
            let outs = compiled_rows[i].run(&grad_inputs);
            // outs[0] = residual scalar; outs[1..] = gradients in
            // the order of this row's filtered wrt list.
            for (slot, &gi) in row_wrt_index[i].iter().enumerate() {
                let val = outs[1 + slot][0];
                if gi < n {
                    // Unknown voltage / branch current.
                    jac[i * n + gi] = val;
                } else if gi < n + m {
                    // Param.
                    df_dp[i * m + (gi - n)] = val;
                } else {
                    // Prev-voltage (on a storage-coupled net).
                    df_dvprev[i * n_storage + (gi - n - m)] = val;
                }
            }
        }


        // For each param, solve J · ds_k = -df_dp - df_dvprev · ds_prev.
        // ds_prev is defined only over UNKNOWNS; lift to storage-net
        // index space by mapping each storage net → its unknown idx
        // (boundary/ground storage nets contribute 0 since their
        // voltages don't depend on optimization params).
        for (kp, name) in wrt_params.iter().enumerate() {
            let prev_step_sens = &result[name][k - 1];

            // Build prev-sensitivity values per storage-coupled net.
            let mut s_prev_storage = vec![0.0_f32; n_storage];
            for (jn, sn) in storage_nets.iter().enumerate() {
                if let Some(&u_idx) = unknown_idx.get(sn) {
                    s_prev_storage[jn] = prev_step_sens[u_idx];
                }
            }

            let mut rhs = vec![0.0_f32; n];
            for i in 0..n {
                let mut acc = -df_dp[i * m + kp];
                for j in 0..n_storage {
                    acc -= df_dvprev[i * n_storage + j] * s_prev_storage[j];
                }
                rhs[i] = acc;
            }
            let s_k_full = match linear_solve(&jac, &rhs, n) {
                Some(s) => s,
                None => {
                    // Singular Jacobian — fall back to copying the
                    // previous step's sensitivity (matches the IFT-
                    // undefined behavior in `sensitivities`).
                    let mut s = vec![0.0_f32; n_v];
                    s.copy_from_slice(prev_step_sens);
                    s
                }
            };
            // Extract just the unknown-voltage portion (drop branches).
            let s_k_v: Vec<f32> = s_k_full[..n_v].to_vec();
            result.get_mut(name).expect("param key").push(s_k_v);
        }
    }
    result
}

// ── Inverse design: optimize a parameter to hit a target voltage ──────

#[derive(Debug, Clone, Copy)]
pub struct OptimizeTargetOptions {
    pub max_iters: usize,
    pub tol: f32,
    /// Step damping. `1.0` is a pure outer-Newton step (fast on smooth
    /// monotonic problems, can overshoot on highly nonlinear ones); `0.5`
    /// halves the step for a stable cautious walk.
    pub damping: f32,
    /// Cap each param update to `step_clamp_rel × |param|` so unbounded
    /// Newton steps near vanishing gradients can't blow up. `0.5` lets
    /// the param at most double (or halve) per iter.
    pub step_clamp_rel: f32,
    pub param_min: f32,
    pub solver: NewtonOptions,
}
impl Default for OptimizeTargetOptions {
    fn default() -> Self {
        Self {
            max_iters: 100,
            tol: 1e-3,
            damping: 0.8,
            step_clamp_rel: 0.5,
            param_min: 1.0,
            solver: NewtonOptions::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimizeTargetResult {
    pub final_params: HashMap<String, f32>,
    pub final_op: DcOperatingPoint,
    pub final_v_target: f32,
    pub final_loss: f32,
    pub iters: usize,
    pub converged: bool,
}

/// Drive a single parameter so a chosen unknown net's voltage reaches a
/// target value. Each iteration: `solve_dc` → `sensitivities` → SGD step.
/// Returns when `|v_target − target| < tol` or after `max_iters`.
pub fn optimize_to_target(
    circuit: &Circuit,
    initial_params: &HashMap<String, f32>,
    boundary: &HashMap<NetId, f32>,
    target_net: NetId,
    target_voltage: f32,
    optimize_param: &str,
    opt: OptimizeTargetOptions,
) -> OptimizeTargetResult {
    let mut params = initial_params.clone();
    let target_param_names = vec![optimize_param.to_string()];
    let mut last_op: Option<DcOperatingPoint> = None;
    let mut last_v_target = f32::NAN;
    let mut last_loss = f32::INFINITY;
    let mut converged_at: Option<usize> = None;

    // Pre-compute target_net's index within unknown_nets (stable across
    // iterations since topology doesn't change).
    let rg = build_residual_graph(circuit);
    let target_idx = rg.unknown_nets
        .iter()
        .position(|n| *n == target_net)
        .expect("target_net is not an unknown — boundary nets aren't optimization targets");

    for iter in 0..opt.max_iters {
        let op = solve_dc(circuit, &params, boundary, opt.solver);
        let v_target = op.voltages.get(&target_net).copied().unwrap_or(0.0);
        let err = v_target - target_voltage;
        let loss = err * err;
        last_v_target = v_target;
        last_loss = loss;
        last_op = Some(op.clone());

        if err.abs() < opt.tol {
            converged_at = Some(iter);
            break;
        }

        let sens = sensitivities(circuit, &params, boundary, &op, &target_param_names);
        let dv_dp_vec = match sens.get(optimize_param) {
            Some(v) => v,
            None => break,
        };
        let dv_dp = dv_dp_vec[target_idx];
        if dv_dp.abs() < 1e-30 { break; }   // singular — bail

        // Outer Newton on `v_target(param) − target = 0`:
        //   Δparam = − (v_target − target) / (∂v_target/∂param)
        // Damped + step-size-limited so highly-nonlinear bilevels don't
        // overshoot through their basin of attraction.
        let raw_step = -err / dv_dp;
        let p = params.get_mut(optimize_param).unwrap_or_else(||
            panic!("optimize_param {optimize_param:?} not in initial_params"));
        let max_step = (p.abs() * opt.step_clamp_rel).max(1.0);
        let step = (opt.damping * raw_step).clamp(-max_step, max_step);
        *p += step;
        if *p < opt.param_min { *p = opt.param_min; }
    }

    let iters = converged_at.unwrap_or(opt.max_iters);
    OptimizeTargetResult {
        final_params: params,
        final_op: last_op.expect("at least one iteration"),
        final_v_target: last_v_target,
        final_loss: last_loss,
        iters,
        converged: converged_at.is_some(),
    }
}

fn find_param_node(graph: &rlx_ir::Graph, name: &str) -> Option<rlx_ir::NodeId> {
    for n in graph.nodes() {
        if let rlx_ir::Op::Param { name: nm } = &n.op {
            if nm == name { return Some(n.id); }
        }
    }
    None
}

fn find_input_node(graph: &rlx_ir::Graph, name: &str) -> Option<rlx_ir::NodeId> {
    for n in graph.nodes() {
        if let rlx_ir::Op::Input { name: nm } = &n.op {
            if nm == name { return Some(n.id); }
        }
    }
    None
}

fn build_inputs(
    all_nets: &[NetId],
    unknowns: &[NetId],
    branches: &[BranchId],
    boundary: &HashMap<NetId, f32>,
    v_unknowns: &[f32],
    i_branches: &[f32],
) -> Vec<(String, Vec<f32>)> {
    let unknown_idx: HashMap<NetId, usize> =
        unknowns.iter().enumerate().map(|(i, n)| (*n, i)).collect();
    let mut out: Vec<(String, Vec<f32>)> = all_nets.iter().map(|net| {
        let val = if let Some(idx) = unknown_idx.get(net) {
            v_unknowns[*idx]
        } else if let Some(v) = boundary.get(net) {
            *v
        } else { 0.0 };
        (net_input_name(*net), vec![val])
    }).collect();
    for (idx, b) in branches.iter().enumerate() {
        out.push((branch_input_name(*b), vec![i_branches[idx]]));
    }
    out
}

/// BE-step variant of [`build_inputs`]: appends `v_prev_<id>` for every
/// net (boundary + unknown) and the timestep `h`. Boundary `v_prev`
/// equals the boundary's this-step value (DC-style — boundaries don't
/// move within the simulation unless the caller updates them between
/// steps); unknown `v_prev` comes from `prev_voltages`.
/// Per-delay scalars the integrator sets on each BE step — feed the
/// unified blend stamp in the residual graph. Built per step by
/// [`sample_delay_step`] (private), but the type is public because it
/// appears in [`solve_be_step`]'s signature so external callers can
/// drive a single BE step directly.
#[derive(Copy, Clone, Debug, Default)]
pub struct DelayStepInputs {
    /// 1.0 for long delays (history-buffer path), 0.0 for sub-step
    /// (in-graph interpolation between v_in_prev and v_in_now).
    pub blend: f32,
    /// `floor(τ/dt) + 1`. Combined with the `<name>_tau` Param to form
    /// the in-graph `α = offset − τ/h`.
    pub offset: f32,
    /// Lower / upper history samples — only consulted when `blend = 1`.
    pub v_lo: f32,
    pub v_hi: f32,
}

fn build_be_step_inputs(
    all_nets: &[NetId],
    unknowns: &[NetId],
    branches: &[BranchId],
    boundary: &HashMap<NetId, f32>,
    prev_voltages: &HashMap<NetId, f32>,
    delay_inputs: &[DelayStepInputs],
    v_unknowns: &[f32],
    i_branches: &[f32],
    h: f32,
) -> Vec<(String, Vec<f32>)> {
    let mut out = build_inputs(all_nets, unknowns, branches, boundary,
                               v_unknowns, i_branches);
    for net in all_nets {
        let val = prev_voltages.get(net).copied()
            .or_else(|| boundary.get(net).copied())
            .unwrap_or(0.0);
        out.push((prev_voltage_input_name(*net), vec![val]));
    }
    out.push((TIMESTEP_INPUT_NAME.to_string(), vec![h]));
    for (idx, di) in delay_inputs.iter().enumerate() {
        let id = DelayId(idx as u32);
        out.push((delay_v_lo_name(id),   vec![di.v_lo]));
        out.push((delay_v_hi_name(id),   vec![di.v_hi]));
        out.push((delay_blend_name(id),  vec![di.blend]));
        out.push((delay_offset_name(id), vec![di.offset]));
    }
    out
}

/// Inner-solve dispatch (T.11.A).
///
/// **Lessons learned during T.11.A — kept for the record:**
///
/// 1. At our spike circuit sizes (N ≤ ~250) the inner linear solve is
///    *not* the bottleneck. The per-iter Newton's `N+1` gradient-graph
///    runs through `rlx_runtime` dominate (~100 µs each at N=150), so a
///    faer dense LU swap costs the same wall clock as the hand-rolled
///    Gauss-Jordan. A faer sparse LU pays a dense→sparse conversion
///    overhead that doesn't amortize until N>500.
///
/// 2. The SAR-ADC test surfaced a real correctness hazard: switching
///    pivot strategy (Gauss-Jordan partial pivot ↔ faer's LU partial
///    pivot) yields slightly different float operating points, which
///    can flip a comparator decision near a switching threshold.
///    Both solvers are numerically correct; the SAR is just on a knife
///    edge near the comparator rails.
///
/// **Decision**: keep Gauss-Jordan as the default to preserve the T.10
/// baseline behavior across every existing report. The faer paths
/// remain available as `RLX_LINEAR_SOLVE=faer-dense` /
/// `=faer-sparse` env-var opt-ins for future N>500 work where the
/// sparse asymptotic actually pays off.
fn linear_solve(a_in: &[f32], b_in: &[f32], n: usize) -> Option<Vec<f32>> {
    match std::env::var("RLX_LINEAR_SOLVE").ok().as_deref() {
        Some("faer-dense")  => faer_dense_lu_solve(a_in, b_in, n)
            .or_else(|| gauss_jordan_solve(a_in, b_in, n)),
        Some("faer-sparse") => sparse_lu_solve(a_in, b_in, n)
            .or_else(|| gauss_jordan_solve(a_in, b_in, n)),
        _ => gauss_jordan_solve(a_in, b_in, n),
    }
}

fn faer_dense_lu_solve(a_in: &[f32], b_in: &[f32], n: usize) -> Option<Vec<f32>> {
    use faer::linalg::solvers::Solve;
    use faer::Mat;

    let mat = Mat::<f64>::from_fn(n, n, |i, j| a_in[i * n + j] as f64);
    let lu = mat.partial_piv_lu();
    let mut rhs = Mat::<f64>::zeros(n, 1);
    for i in 0..n { rhs[(i, 0)] = b_in[i] as f64; }
    lu.solve_in_place(&mut rhs);
    let mut out = vec![0.0_f32; n];
    for i in 0..n {
        let v = rhs[(i, 0)];
        if !v.is_finite() { return None; }
        out[i] = v as f32;
    }
    Some(out)
}

fn sparse_lu_solve(a_in: &[f32], b_in: &[f32], n: usize) -> Option<Vec<f32>> {
    use faer::sparse::{SparseColMat, Triplet};
    use faer::linalg::solvers::Solve;
    use faer::Mat;

    // Convert dense → sparse triplets, dropping near-zero entries.
    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::with_capacity(n * 8);
    for r in 0..n {
        for c in 0..n {
            let v = a_in[r * n + c];
            if v.abs() > 1e-30 {
                triplets.push(Triplet::new(r, c, v as f64));
            }
        }
    }
    let mat = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).ok()?;
    let lu = mat.sp_lu().ok()?;
    let mut rhs = Mat::<f64>::zeros(n, 1);
    for i in 0..n { rhs[(i, 0)] = b_in[i] as f64; }
    lu.solve_in_place(&mut rhs);
    let mut out = vec![0.0_f32; n];
    for i in 0..n {
        let v = rhs[(i, 0)];
        if !v.is_finite() { return None; }
        out[i] = v as f32;
    }
    Some(out)
}

/// In-place Gauss-Jordan elimination with partial pivoting. Returns
/// `Some(x)` such that `A·x = b`, or `None` if `A` is singular.
/// O(N³); fine for the small unknown counts our spike circuits hit.
fn gauss_jordan_solve(a_in: &[f32], b_in: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut a: Vec<f32> = a_in.to_vec();
    let mut b: Vec<f32> = b_in.to_vec();
    for k in 0..n {
        // Pivot — swap rows so |a[k][k]| is the largest in column k.
        let mut piv = k;
        for r in (k + 1)..n {
            if a[r * n + k].abs() > a[piv * n + k].abs() {
                piv = r;
            }
        }
        if a[piv * n + k].abs() < 1e-30 {
            return None;
        }
        if piv != k {
            for c in 0..n {
                a.swap(k * n + c, piv * n + c);
            }
            b.swap(k, piv);
        }
        // Eliminate.
        let akk = a[k * n + k];
        for r in 0..n {
            if r == k { continue; }
            let f = a[r * n + k] / akk;
            if f == 0.0 { continue; }
            for c in k..n {
                a[r * n + c] -= f * a[k * n + c];
            }
            b[r] -= f * b[k];
        }
    }
    let mut x = vec![0.0_f32; n];
    for i in 0..n {
        x[i] = b[i] / a[i * n + i];
    }
    Some(x)
}
