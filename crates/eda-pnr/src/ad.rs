//! AD-enabled placement: per-instance `(x, y)` as rlx Params, with
//! a half-perimeter wirelength loss that any rlx optimizer (Adam,
//! SGD, DADO) can drive.
//!
//! ## Why this lives in `eda-pnr`
//!
//! Once instance positions are differentiable, every downstream
//! metric that's a function of geometry — wirelength, bbox area,
//! pin-to-pin Manhattan distance, congestion proxies — turns into
//! a gradient-descent target the same way the LNA's `Lg` and the
//! MZI's `n_eff_A` are. PNR sitting on the rlx graph means
//! placement is just another node in the same ML stack rlx-eda
//! already uses for behavioral inverse design.
//!
//! ## How it works
//!
//! Each instance gets two Params: `<netlist>.<instance>.x` and
//! `.y`, both in **DBU but represented as `f32`** so the rlx graph
//! can multiply / sum / log-sum-exp them cleanly. After
//! optimization, [`DifferentiablePlacement::materialize`] snaps the
//! float values back to `i64` `Trans`es and produces a
//! [`Placement`] the standard [`PnrFlow`](crate::PnrFlow) can stamp.
//!
//! ## HPWL with smooth-max
//!
//! The half-perimeter wirelength of one net is
//!
//! ```text
//!   HPWL(net) = (max_i x_i - min_i x_i) + (max_i y_i - min_i y_i)
//! ```
//!
//! `max` / `min` are non-differentiable at ties, so we approximate
//! both via log-sum-exp with a sharpness `β`:
//!
//! ```text
//!   smooth_max(x_i; β) = (1/β) · log Σ_i exp(β · x_i)
//!   smooth_min(x_i; β) = -(1/β) · log Σ_i exp(-β · x_i)
//! ```
//!
//! `β → ∞` recovers the exact bbox; `β` finite stays smooth and
//! differentiable everywhere. This is the same weighted-average
//! wirelength approximation DREAMPlace uses (Lin et al., 2019),
//! built on rlx-ir Activations rather than a custom kernel.

use std::collections::HashMap;

use klayout_core::{Library, Trans, Vec2};
use rlx_ir::{
    op::{Activation, BinaryOp, ReduceOp},
    DType, Graph, NodeId, Op, Shape as TensorShape,
};

use crate::netlist::{MatchKind, Netlist, SymmetryAxis};
use crate::place::Placement;

// ── Sharpness defaults ────────────────────────────────────────────────
//
// `BETA_HPWL_DEFAULT` is sized for typical micron-scale chips (max
// coord ~ 1e6 DBU → exponent ~ 100); pulls HPWL within ~1 % of the
// true bbox. `BETA_DENSITY_DEFAULT` is sharper because the density
// term operates on small differences (cell-to-cell separations of
// a few thousand DBU), so the smoothing has to bite at that scale.
pub const BETA_HPWL_DEFAULT: f32 = 1e-4;
pub const BETA_DENSITY_DEFAULT: f32 = 1e-3;

/// Per-instance `(x, y)` in DBU, kept as `f32` so the rlx graph
/// stays well-conditioned for reasonable chip sizes (`<= 1e7` DBU
/// = 1 cm at standard 1000-DBU/µm). Initial values come from
/// either [`DifferentiablePlacement::from_placement`] (seed from
/// any concrete placer) or hand-set after construction.
#[derive(Clone, Debug)]
pub struct DifferentiablePlacement {
    pub instance_xy: Vec<(f32, f32)>,
    /// Smooth-max sharpness for HPWL. Larger β ⇒ closer to true
    /// bbox; numerical overflow bites past about β · max_dim ≈ 80.
    /// Default `1e-4` is sized for typical micron-scale chips
    /// (max coord ~ 1e6 DBU → exponent ~ 100). Callers running
    /// nm-scale or mm-scale designs scale β accordingly.
    pub beta: f32,
}

impl DifferentiablePlacement {
    /// Seed the differentiable placement from an existing
    /// [`Placement`] (typically a [`crate::GridPlacer`] pass).
    /// AD then fine-tunes the seed.
    pub fn from_placement(seed: &Placement) -> Self {
        let xy = seed
            .transforms
            .iter()
            .map(|t| {
                let v = t.apply(klayout_core::Point::new(0, 0));
                (v.x as f32, v.y as f32)
            })
            .collect();
        Self { instance_xy: xy, beta: 1e-4 }
    }

    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// Snap the float positions back to integer `Trans`es and
    /// return a [`Placement`] the regular [`crate::PnrFlow`] can
    /// consume.
    pub fn materialize(&self, netlist: &Netlist, lib: &Library) -> Placement {
        let transforms: Vec<Trans> = self
            .instance_xy
            .iter()
            .map(|(x, y)| Trans::translate(Vec2::new(x.round() as i64, y.round() as i64)))
            .collect();
        let bbox = crate::place::union_placed_bbox(netlist, lib, &transforms);
        Placement { transforms, bbox }
    }

    /// Param key for instance `i`'s x-coordinate. Stable across
    /// runs given the same netlist — calls to `Session::set_param`
    /// thread positions back into the graph.
    pub fn x_param_name(&self, netlist: &Netlist, i: usize) -> String {
        format!("{}.{}.x", netlist.name, netlist.instances[i].name)
    }
    pub fn y_param_name(&self, netlist: &Netlist, i: usize) -> String {
        format!("{}.{}.y", netlist.name, netlist.instances[i].name)
    }
}

/// Build a forward graph that returns scalar HPWL summed over
/// every routable net in `netlist`. Differentiable wrt every
/// `<netlist>.<instance>.{x,y}` Param.
///
/// Pin positions are `instance_pos + port_offset`, where the
/// port_offset is read from the cell's port at graph-build time
/// and baked as a constant — placing a cell only translates it,
/// so port offsets are translation-invariant.
///
/// Returns the graph with a single output: scalar `total_hpwl` in
/// DBU. Hand it to [`rlx_opt::autodiff::grad_with_loss`] to get
/// gradients wrt every position Param.
pub fn hpwl_loss_graph(netlist: &Netlist, lib: &Library, beta: f32) -> Graph {
    let mut g = Graph::new(format!("{}_hpwl", netlist.name));
    let s = TensorShape::new(&[1], DType::F32);
    let beta_n = const_f32(&mut g, beta, s.clone());

    // Register every instance's (x, y) Params up front so the
    // resulting graph has a stable param surface.
    let mut x_params: Vec<NodeId> = Vec::with_capacity(netlist.instances.len());
    let mut y_params: Vec<NodeId> = Vec::with_capacity(netlist.instances.len());
    for i in 0..netlist.instances.len() {
        x_params.push(g.param(format!("{}.{}.x", netlist.name, netlist.instances[i].name), s.clone()));
        y_params.push(g.param(format!("{}.{}.y", netlist.name, netlist.instances[i].name), s.clone()));
    }

    // Cache per-instance port offsets so we don't re-look-up the
    // cell for repeated pins on the same net.
    let mut port_offset_cache: HashMap<(usize, String), (f32, f32)> = HashMap::new();
    let mut total: Option<NodeId> = None;

    for net in &netlist.nets {
        if net.pins.len() < 2 { continue; }

        // Collect each pin's (x, y) NodeId in the graph: instance_x + port_offset_x.
        let mut xs: Vec<NodeId> = Vec::with_capacity(net.pins.len());
        let mut ys: Vec<NodeId> = Vec::with_capacity(net.pins.len());
        let mut net_resolvable = true;
        for pin in &net.pins {
            let key = (pin.instance, pin.port.clone());
            let (ox, oy) = match port_offset_cache.get(&key) {
                Some(v) => *v,
                None => {
                    let inst = match netlist.instances.get(pin.instance) {
                        Some(i) => i,
                        None => { net_resolvable = false; break; }
                    };
                    let cell = lib.get(inst.cell);
                    match cell.port(&pin.port) {
                        Some(p) => {
                            let v = (p.center.x as f32, p.center.y as f32);
                            port_offset_cache.insert(key, v);
                            v
                        }
                        None => { net_resolvable = false; break; }
                    }
                }
            };
            let ox_n = const_f32(&mut g, ox, s.clone());
            let oy_n = const_f32(&mut g, oy, s.clone());
            xs.push(g.binary(BinaryOp::Add, x_params[pin.instance], ox_n, s.clone()));
            ys.push(g.binary(BinaryOp::Add, y_params[pin.instance], oy_n, s.clone()));
        }
        if !net_resolvable { continue; }

        let span_x = smooth_span(&mut g, &xs, beta_n, &s);
        let span_y = smooth_span(&mut g, &ys, beta_n, &s);
        let net_hpwl = g.binary(BinaryOp::Add, span_x, span_y, s.clone());
        total = Some(match total {
            Some(acc) => g.binary(BinaryOp::Add, acc, net_hpwl, s.clone()),
            None => net_hpwl,
        });
    }

    let out = total.unwrap_or_else(|| const_f32(&mut g, 0.0, s.clone()));
    g.set_outputs(vec![out]);
    g
}

/// `smooth_max(xs) − smooth_min(xs)` via two log-sum-exp
/// expressions on the rlx graph. The Σ exp(β·x) is built as a
/// chain of Add/Activation::Exp/Activation::Log Ops — no MatMul or
/// reductions needed, which keeps the graph small even for nets
/// with many pins.
fn smooth_span(g: &mut Graph, xs: &[NodeId], beta: NodeId, s: &TensorShape) -> NodeId {
    debug_assert!(!xs.is_empty(), "smooth_span needs at least one input");
    let one = const_f32(g, 1.0, s.clone());
    let neg_one = const_f32(g, -1.0, s.clone());
    let beta_n = beta;
    let neg_beta_n = g.binary(BinaryOp::Mul, beta, neg_one, s.clone());
    let inv_beta = g.binary(BinaryOp::Div, one, beta, s.clone());
    let neg_inv_beta = g.binary(BinaryOp::Mul, inv_beta, neg_one, s.clone());

    // smooth_max = (1/β) · log Σ exp(β x_i)
    let mut acc_max: Option<NodeId> = None;
    for &x in xs {
        let bx = g.binary(BinaryOp::Mul, x, beta_n, s.clone());
        let ex = g.activation(Activation::Exp, bx, s.clone());
        acc_max = Some(match acc_max {
            Some(a) => g.binary(BinaryOp::Add, a, ex, s.clone()),
            None => ex,
        });
    }
    let lse_max = g.activation(Activation::Log, acc_max.unwrap(), s.clone());
    let smax = g.binary(BinaryOp::Mul, lse_max, inv_beta, s.clone());

    // smooth_min = -(1/β) · log Σ exp(-β x_i)
    let mut acc_min: Option<NodeId> = None;
    for &x in xs {
        let bx = g.binary(BinaryOp::Mul, x, neg_beta_n, s.clone());
        let ex = g.activation(Activation::Exp, bx, s.clone());
        acc_min = Some(match acc_min {
            Some(a) => g.binary(BinaryOp::Add, a, ex, s.clone()),
            None => ex,
        });
    }
    let lse_min = g.activation(Activation::Log, acc_min.unwrap(), s.clone());
    let smin = g.binary(BinaryOp::Mul, lse_min, neg_inv_beta, s.clone());

    g.binary(BinaryOp::Sub, smax, smin, s.clone())
}

fn const_f32(g: &mut Graph, val: f32, shape: TensorShape) -> NodeId {
    g.add_node(
        Op::Constant { data: val.to_le_bytes().to_vec() },
        vec![],
        shape,
    )
}

// ── Density / overlap penalty ─────────────────────────────────────────
//
// HPWL alone has a degenerate optimum: every instance wants to
// collapse to a single point (zero wirelength). Real placement
// needs a competing **non-overlap** term that pushes instances
// apart when their bboxes intersect.
//
// The penalty here is a smooth pairwise-overlap area:
//
// ```text
//   overlap_x(i, j) = smooth_relu( (w_i + w_j) / 2 - |x_i - x_j| )
//   overlap_y(i, j) = smooth_relu( (h_i + h_j) / 2 - |y_i - y_j| )
//   penalty(i, j) = overlap_x · overlap_y
//   total = Σ_{i < j} penalty(i, j)
// ```
//
// Where `smooth_relu(z; β) = z · sigmoid(β · z)` (the swish/SiLU
// shape with explicit β) — differentiable everywhere, ≈ z for
// `β·z >> 0`, ≈ 0 for `β·z << 0`. Equivalent to a softplus
// derivative without the always-positive offset that plain
// `(1/β) · log(1 + exp(β·z))` carries.
//
// `|x_i − x_j|` itself is smoothed via `√(Δx² + ε²)` so the
// gradient at coincident positions stays finite.
//
// O(N²) over instance pairs. Fine for the tens-to-low-hundreds of
// blocks rlx-eda spikes care about; a future grid / FFT-based
// density (DREAMPlace's electrostatic-field formulation) becomes
// a sibling function when we put thousands of cells on a chip.

/// Build a forward graph that returns scalar **HPWL + density-weighted
/// overlap penalty**, summed over routable nets and pairwise
/// instance overlaps respectively. Differentiable wrt every
/// `<netlist>.<instance>.{x,y}` Param.
///
/// `density_weight` multiplies the overlap term before it's added
/// to HPWL — values around `1.0 / cell_area_avg` give a balanced
/// loss; larger values push instances apart harder, smaller values
/// let HPWL dominate.
pub fn combined_loss_graph(
    netlist: &Netlist,
    lib: &Library,
    seed: &DifferentiablePlacement,
    density_weight: f32,
) -> Graph {
    combined_loss_graph_with_symmetry(netlist, lib, seed, density_weight, 1.0)
}

/// Same as [`combined_loss_graph`] plus an explicit weight on the
/// matching-constraint penalty (sum over [`crate::netlist::MatchGroup`]).
/// Penalty is in DBU², so a sane starting point is around `1.0` —
/// once a constraint is violated by `~ pitch` it dominates HPWL,
/// pulling the placer back onto the manifold; bump up if the
/// optimizer drifts off symmetry, bump down if matching fights the
/// rest of the loss. Equivalent to `combined_loss_graph` when
/// `netlist.match_groups` is empty (the symmetry term is a graph
/// constant `0`).
pub fn combined_loss_graph_with_symmetry(
    netlist: &Netlist,
    lib: &Library,
    seed: &DifferentiablePlacement,
    density_weight: f32,
    symmetry_weight: f32,
) -> Graph {
    let mut g = Graph::new(format!("{}_combined", netlist.name));
    let s = TensorShape::new(&[1], DType::F32);

    let (x_params, y_params) = register_position_params(&mut g, netlist, seed);
    let port_offsets = collect_port_offsets(netlist, lib);

    // β values arrive at runtime — lets [`crate::ad::HPWL_BETA_INPUT`]
    // / [`DENSITY_BETA_INPUT`] schedules anneal sharpness during
    // optimization without rebuilding the graph.
    let hpwl_beta = g.input(HPWL_BETA_INPUT, s.clone());
    let density_beta = g.input(DENSITY_BETA_INPUT, s.clone());

    let hpwl = hpwl_subgraph(&mut g, netlist, &x_params, &y_params, &port_offsets, hpwl_beta, &s);
    let density = density_subgraph(&mut g, netlist, lib, &x_params, &y_params, density_beta, &s);
    let symmetry = symmetry_subgraph(&mut g, netlist, &x_params, &y_params, &s);

    let dweight = const_f32(&mut g, density_weight, s.clone());
    let weighted_density = g.binary(BinaryOp::Mul, density, dweight, s.clone());
    let sweight = const_f32(&mut g, symmetry_weight, s.clone());
    let weighted_symmetry = g.binary(BinaryOp::Mul, symmetry, sweight, s.clone());

    let hpwl_dens = g.binary(BinaryOp::Add, hpwl, weighted_density, s.clone());
    let total = g.binary(BinaryOp::Add, hpwl_dens, weighted_symmetry, s);

    g.set_outputs(vec![total]);
    g
}

/// Density-only loss. Useful for diagnostic tests / smoke verification.
pub fn density_loss_graph(
    netlist: &Netlist,
    lib: &Library,
    seed: &DifferentiablePlacement,
) -> Graph {
    let mut g = Graph::new(format!("{}_density", netlist.name));
    let s = TensorShape::new(&[1], DType::F32);
    let (x_params, y_params) = register_position_params(&mut g, netlist, seed);
    let beta = g.input(DENSITY_BETA_INPUT, s.clone());
    let total = density_subgraph(&mut g, netlist, lib, &x_params, &y_params, beta, &s);
    g.set_outputs(vec![total]);
    g
}

/// Runtime-input name for the HPWL log-sum-exp sharpness β. Threaded
/// in via `Session::run(&[("hpwl_beta", &[β]), ...])` each step;
/// callers wire it to [`eda_trace::BetaSchedule`] for annealing.
pub const HPWL_BETA_INPUT: &str = "hpwl_beta";
/// Runtime-input name for the density-overlap sharpness β.
pub const DENSITY_BETA_INPUT: &str = "density_beta";
/// Param key for the batched X positions (shape `[N]`) used by
/// [`combined_loss_graph_batched`] / [`position_param_ids_batched`].
pub const POSITIONS_X_PARAM: &str = "positions_x";
/// Param key for the batched Y positions (shape `[N]`).
pub const POSITIONS_Y_PARAM: &str = "positions_y";

// ── Batched / GPU-friendly variant ────────────────────────────────────
//
// `combined_loss_graph_batched` builds the same HPWL + density loss
// as `combined_loss_graph`, but with the per-instance `(x, y)`
// positions packed into two `Param` tensors of shape `[N]` instead
// of `2N` scalar `Param`s of shape `[1]`. The density operator is
// then expressed as outer-difference + reduction over `[N, N]`
// matrices — one ElementwiseRegion + one ReduceSum, instead of
// `O(N²)` scalar ops.
//
// On a GPU backend (Apple Metal via rlx-mlx, CUDA, …) this collapses
// hundreds of `Shape::[1]` kernel launches into a handful of ops on
// non-degenerate tensors, which is what actually amortizes the
// per-launch dispatch cost.
//
// Fixed instances are still supported: their positions get baked
// in as constants in the matrix, and the position Params reflect
// only movable instances. `position_param_ids_batched` returns
// `[positions_x_id, positions_y_id]` (always 2 entries, even with
// fixed pins — fixed-pin position is a graph constant).

/// Build a forward graph that returns scalar **HPWL + density-weighted
/// overlap penalty**, with positions as `[N]` tensors so the density
/// operator vectorizes into one `[N, N]` outer-difference + reduce.
/// Same loss values as [`combined_loss_graph`], same param naming
/// for `set_param` (`POSITIONS_X_PARAM` / `POSITIONS_Y_PARAM`),
/// only the *implementation* batches differently — sized so MLX /
/// GPU backends start winning around `N ≥ 64`.
pub fn combined_loss_graph_batched(
    netlist: &Netlist,
    lib: &Library,
    seed: &DifferentiablePlacement,
    density_weight: f32,
) -> Graph {
    let mut g = Graph::new(format!("{}_combined_batched", netlist.name));
    let s_scalar = TensorShape::new(&[1], DType::F32);

    let n = netlist.instances.len();
    let movable_idx: Vec<usize> = netlist
        .instances
        .iter()
        .enumerate()
        .filter(|(_, inst)| !inst.fixed)
        .map(|(i, _)| i)
        .collect();
    let n_movable = movable_idx.len();

    // Positions live as two `[N]` tensors. Movable positions come
    // from Params; fixed positions are baked as Constants and
    // assembled with the movable ones via Concat (or, in the
    // common all-movable case, the Param tensor IS the position
    // tensor directly).
    let (positions_x, positions_y) = build_positions_tensors(
        &mut g,
        netlist,
        seed,
        &movable_idx,
        n,
        n_movable,
    );

    let port_offsets = collect_port_offsets(netlist, lib);

    // β values arrive at runtime so callers can anneal sharpness.
    let hpwl_beta = g.input(HPWL_BETA_INPUT, s_scalar.clone());
    let density_beta = g.input(DENSITY_BETA_INPUT, s_scalar.clone());

    let hpwl = hpwl_subgraph_batched(
        &mut g, netlist, positions_x, positions_y, &port_offsets, hpwl_beta, n, &s_scalar,
    );
    let density = density_subgraph_batched(
        &mut g, netlist, lib, positions_x, positions_y, density_beta, n, &s_scalar,
    );

    let weight = const_f32(&mut g, density_weight, s_scalar.clone());
    let weighted_density = g.binary(BinaryOp::Mul, density, weight, s_scalar.clone());
    let total = g.binary(BinaryOp::Add, hpwl, weighted_density, s_scalar);

    g.set_outputs(vec![total]);
    g
}

/// Position-Param NodeIds for the batched graph: `[positions_x,
/// positions_y]`. Hand to `grad_with_loss` to get gradient
/// tensors of shape `[N_movable]` per axis. Always 2 entries (or 0
/// if every instance is fixed).
pub fn position_param_ids_batched(g: &Graph) -> Vec<NodeId> {
    let mut out = Vec::with_capacity(2);
    for axis in [POSITIONS_X_PARAM, POSITIONS_Y_PARAM] {
        if let Some(id) = g
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(i, n)| match &n.op {
                Op::Param { name, .. } if name == axis => Some(NodeId(i as u32)),
                _ => None,
            })
        {
            out.push(id);
        }
    }
    out
}

fn build_positions_tensors(
    g: &mut Graph,
    netlist: &Netlist,
    seed: &DifferentiablePlacement,
    movable_idx: &[usize],
    n: usize,
    n_movable: usize,
) -> (NodeId, NodeId) {
    let s_n = TensorShape::new(&[n], DType::F32);
    let s_movable = TensorShape::new(&[n_movable], DType::F32);
    let s_scalar = TensorShape::new(&[1], DType::F32);

    if n_movable == n {
        // Common case: every instance is movable. The Param tensor
        // *is* the position tensor — no fixed-pin assembly needed.
        let px = g.param(POSITIONS_X_PARAM, s_n.clone());
        let py = g.param(POSITIONS_Y_PARAM, s_n);
        return (px, py);
    }

    // Mixed case: assemble [N] from a [N_movable] Param tensor
    // (movable positions, in order) plus per-fixed-instance
    // Constants. Use Concat segment-by-segment so the resulting
    // tensor has the same `instance_index → position_index`
    // mapping the rest of the graph expects.
    let movable_x = if n_movable > 0 {
        Some(g.param(POSITIONS_X_PARAM, s_movable.clone()))
    } else { None };
    let movable_y = if n_movable > 0 {
        Some(g.param(POSITIONS_Y_PARAM, s_movable))
    } else { None };

    let mut x_pieces = Vec::with_capacity(n);
    let mut y_pieces = Vec::with_capacity(n);
    let mut next_movable = 0usize;
    for i in 0..n {
        if netlist.instances[i].fixed {
            let (x0, y0) = seed.instance_xy.get(i).copied().unwrap_or((0.0, 0.0));
            x_pieces.push(const_f32(g, x0, s_scalar.clone()));
            y_pieces.push(const_f32(g, y0, s_scalar.clone()));
        } else {
            let mx = movable_x.expect("movable Param must exist if n_movable > 0");
            let my = movable_y.expect("movable Param must exist if n_movable > 0");
            x_pieces.push(g.add_node(
                Op::Narrow { axis: 0, start: next_movable, len: 1 },
                vec![mx], s_scalar.clone(),
            ));
            y_pieces.push(g.add_node(
                Op::Narrow { axis: 0, start: next_movable, len: 1 },
                vec![my], s_scalar.clone(),
            ));
            next_movable += 1;
        }
    }
    let s_full = TensorShape::new(&[n], DType::F32);
    let positions_x = g.concat(x_pieces, 0, s_full.clone());
    let positions_y = g.concat(y_pieces, 0, s_full);
    (positions_x, positions_y)
}

fn hpwl_subgraph_batched(
    g: &mut Graph,
    netlist: &Netlist,
    positions_x: NodeId,
    positions_y: NodeId,
    port_offsets: &HashMap<(usize, String), (f32, f32)>,
    beta: NodeId,
    _n: usize,
    s: &TensorShape,
) -> NodeId {
    // For each net we narrow out the per-pin slot from the [N]
    // position tensor, add the (constant) port offset, then
    // smooth_max/min over the per-net pin list. Per-net work is
    // still scalar — HPWL is O(NETS · K_pins) which is tiny next
    // to the O(N²) density above. Big GPU wins come from the
    // density subgraph.
    let mut total: Option<NodeId> = None;
    for net in &netlist.nets {
        if net.pins.len() < 2 { continue; }
        let mut xs = Vec::with_capacity(net.pins.len());
        let mut ys = Vec::with_capacity(net.pins.len());
        let mut net_resolvable = true;
        for pin in &net.pins {
            let key = (pin.instance, pin.port.clone());
            let (ox, oy) = match port_offsets.get(&key) {
                Some(v) => *v,
                None => { net_resolvable = false; break; }
            };
            let x_slot = g.add_node(
                Op::Narrow { axis: 0, start: pin.instance, len: 1 },
                vec![positions_x], s.clone(),
            );
            let y_slot = g.add_node(
                Op::Narrow { axis: 0, start: pin.instance, len: 1 },
                vec![positions_y], s.clone(),
            );
            let ox_n = const_f32(g, ox, s.clone());
            let oy_n = const_f32(g, oy, s.clone());
            xs.push(g.binary(BinaryOp::Add, x_slot, ox_n, s.clone()));
            ys.push(g.binary(BinaryOp::Add, y_slot, oy_n, s.clone()));
        }
        if !net_resolvable { continue; }
        let span_x = smooth_span(g, &xs, beta, s);
        let span_y = smooth_span(g, &ys, beta, s);
        let raw = g.binary(BinaryOp::Add, span_x, span_y, s.clone());
        let net_hpwl = if (net.weight - 1.0).abs() < f32::EPSILON {
            raw
        } else {
            let w = const_f32(g, net.weight, s.clone());
            g.binary(BinaryOp::Mul, raw, w, s.clone())
        };
        total = Some(match total {
            Some(acc) => g.binary(BinaryOp::Add, acc, net_hpwl, s.clone()),
            None => net_hpwl,
        });
    }
    total.unwrap_or_else(|| const_f32(g, 0.0, s.clone()))
}

fn density_subgraph_batched(
    g: &mut Graph,
    netlist: &Netlist,
    lib: &Library,
    positions_x: NodeId,
    positions_y: NodeId,
    beta: NodeId,
    n: usize,
    s_scalar: &TensorShape,
) -> NodeId {
    if n < 2 { return const_f32(g, 0.0, s_scalar.clone()); }
    // Per-pair half-bbox sums computed at graph-build time and
    // baked as `[N, N]` Constant tensors so the runtime kernel
    // doesn't have to load per-pair half-dims through a separate
    // input.
    let half_w: Vec<f32> = netlist
        .instances
        .iter()
        .map(|inst| {
            let bbox = lib.get(inst.cell).local_bbox();
            ((bbox.max.x - bbox.min.x) as f32) * 0.5
        })
        .collect();
    let half_h: Vec<f32> = netlist
        .instances
        .iter()
        .map(|inst| {
            let bbox = lib.get(inst.cell).local_bbox();
            ((bbox.max.y - bbox.min.y) as f32) * 0.5
        })
        .collect();
    let mut hw_pairs = vec![0.0_f32; n * n];
    let mut hh_pairs = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            hw_pairs[i * n + j] = half_w[i] + half_w[j];
            hh_pairs[i * n + j] = half_h[i] + half_h[j];
        }
    }

    let s_nn = TensorShape::new(&[n, n], DType::F32);
    let s_n1 = TensorShape::new(&[n, 1], DType::F32);
    let s_1n = TensorShape::new(&[1, n], DType::F32);

    let half_w_const = const_tensor(g, &hw_pairs, s_nn.clone());
    let half_h_const = const_tensor(g, &hh_pairs, s_nn.clone());

    // Outer differences: dx[i, j] = x[i] - x[j] via reshape + expand + sub.
    let x_col = g.reshape(positions_x, vec![n as i64, 1], s_n1.clone());
    let x_row = g.reshape(positions_x, vec![1, n as i64], s_1n.clone());
    let x_col_b = g.add_node(Op::Expand { target_shape: vec![n as i64, n as i64] }, vec![x_col], s_nn.clone());
    let x_row_b = g.add_node(Op::Expand { target_shape: vec![n as i64, n as i64] }, vec![x_row], s_nn.clone());
    let dx = g.binary(BinaryOp::Sub, x_col_b, x_row_b, s_nn.clone());

    let y_col = g.reshape(positions_y, vec![n as i64, 1], s_n1);
    let y_row = g.reshape(positions_y, vec![1, n as i64], s_1n);
    let y_col_b = g.add_node(Op::Expand { target_shape: vec![n as i64, n as i64] }, vec![y_col], s_nn.clone());
    let y_row_b = g.add_node(Op::Expand { target_shape: vec![n as i64, n as i64] }, vec![y_row], s_nn.clone());
    let dy = g.binary(BinaryOp::Sub, y_col_b, y_row_b, s_nn.clone());

    // |Δ| ≈ √(Δ² + ε²)
    let dx2 = g.binary(BinaryOp::Mul, dx, dx, s_nn.clone());
    let dy2 = g.binary(BinaryOp::Mul, dy, dy, s_nn.clone());
    let eps_sq = const_tensor(g, &vec![1.0_f32; n * n], s_nn.clone());
    let eps_sq_y = const_tensor(g, &vec![1.0_f32; n * n], s_nn.clone());
    let dx2_eps = g.binary(BinaryOp::Add, dx2, eps_sq, s_nn.clone());
    let dy2_eps = g.binary(BinaryOp::Add, dy2, eps_sq_y, s_nn.clone());
    let abs_dx = g.activation(Activation::Sqrt, dx2_eps, s_nn.clone());
    let abs_dy = g.activation(Activation::Sqrt, dy2_eps, s_nn.clone());

    // smooth_relu( half_dim_sum - |Δ|; β )
    let z_x = g.binary(BinaryOp::Sub, half_w_const, abs_dx, s_nn.clone());
    let z_y = g.binary(BinaryOp::Sub, half_h_const, abs_dy, s_nn.clone());
    let rx = smooth_relu_batched(g, z_x, beta, &s_nn);
    let ry = smooth_relu_batched(g, z_y, beta, &s_nn);

    // Pair overlap = rx · ry. Sum over the full [N, N] grid would
    // double-count (i,j) + (j,i) AND include the diagonal (where
    // |Δ|=ε so smooth_relu(half_dim − ε) ≈ half_dim, contributing
    // ~4·w² per instance — non-negligible). Zero the diagonal
    // first via a baked mask, then reduce, then halve.
    let pair = g.binary(BinaryOp::Mul, rx, ry, s_nn.clone());
    let mut mask = vec![1.0_f32; n * n];
    for i in 0..n { mask[i * n + i] = 0.0; }
    let mask_const = const_tensor(g, &mask, s_nn.clone());
    let pair_off_diag = g.binary(BinaryOp::Mul, pair, mask_const, s_nn.clone());
    let total = g.reduce(pair_off_diag, ReduceOp::Sum, vec![0, 1], false, s_scalar.clone());
    let half = const_f32(g, 0.5, s_scalar.clone());
    g.binary(BinaryOp::Mul, total, half, s_scalar.clone())
}

fn smooth_relu_batched(g: &mut Graph, z: NodeId, beta: NodeId, s: &TensorShape) -> NodeId {
    // Broadcast scalar β onto the [N, N] z by Expand. rlx-runtime's
    // `mark_elementwise_regions` then folds Mul + Sigmoid + Mul
    // into a single fused kernel.
    let beta_b = g.add_node(
        Op::Expand { target_shape: s.dims().iter().map(|d| d.unwrap_static() as i64).collect() },
        vec![beta],
        s.clone(),
    );
    let bz = g.binary(BinaryOp::Mul, z, beta_b, s.clone());
    let sig = g.activation(Activation::Sigmoid, bz, s.clone());
    g.binary(BinaryOp::Mul, z, sig, s.clone())
}

fn const_tensor(g: &mut Graph, data: &[f32], shape: TensorShape) -> NodeId {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    g.add_node(Op::Constant { data: bytes }, vec![], shape)
}

fn bytes_for_ones(len: usize) -> Vec<u8> {
    let mut b = Vec::with_capacity(len * 4);
    let one = 1.0_f32.to_le_bytes();
    for _ in 0..len { b.extend_from_slice(&one); }
    b
}

// ── Parallel-batch placement ──────────────────────────────────────────
//
// `combined_loss_graph_parallel` runs `B` independent placements
// concurrently as a single rlx graph: positions are `Param[B, N]`,
// the density operator is `[B, N, N]`, and HPWL becomes a `[B]`
// per-batch reduction. Same kernel count and same dispatch count
// as the single-placement batched variant, but each kernel does
// `B×` the compute — so MLX's per-launch overhead amortizes across
// B independent placements at no extra dispatch cost.
//
// Use cases:
//
// * **Best-of-K sampling**: B random seeds, take the lowest final
//   `loss_per_batch[k]`.
// * **Hyperparameter sweep**: B = (n_lr × n_beta) configurations
//   in one Adam invocation, pick the Pareto front.
// * **Monte-Carlo placement**: B process-corner / mismatch draws,
//   produce a placement robust across the distribution.
// * **DADO / RL-style architecture search**: B candidate netlists
//   sharing the same position grid.
//
// Position Params are named `POSITIONS_X_PARAM` / `POSITIONS_Y_PARAM`
// — same names as the unbatched variant but shape `[B, N]` instead
// of `[N]`. Caller does `set_param("positions_x", &flat_BN_data)`
// where the data is row-major `[B, N]`.

/// Build a forward graph that returns `[B]` per-batch loss for `B`
/// independent placements of the same netlist. Density is computed
/// as `[B, N, N]` outer-difference; HPWL as `[B]` per net summed
/// across nets. Loss output shape is `[B]` so the caller can pick
/// the best batch element after Adam converges.
///
/// Numerical equivalence: each batch element produces the exact
/// same loss value as `combined_loss_graph_batched` on its own
/// `[N]` slice. Verified by `tests/ad_parallel.rs`.
pub fn combined_loss_graph_parallel(
    netlist: &Netlist,
    lib: &Library,
    batch_size: usize,
    density_weight: f32,
) -> Graph {
    assert!(batch_size >= 1, "batch_size must be >= 1");
    let mut g = Graph::new(format!("{}_parallel_B{}", netlist.name, batch_size));
    let s_scalar = TensorShape::new(&[1], DType::F32);
    let n = netlist.instances.len();
    let b = batch_size;
    let s_bn = TensorShape::new(&[b, n], DType::F32);
    let s_b = TensorShape::new(&[b], DType::F32);

    // Positions: `[B, N]` Params — one slot per (batch, instance).
    let positions_x = g.param(POSITIONS_X_PARAM, s_bn.clone());
    let positions_y = g.param(POSITIONS_Y_PARAM, s_bn);

    let port_offsets = collect_port_offsets(netlist, lib);

    let hpwl_beta = g.input(HPWL_BETA_INPUT, s_scalar.clone());
    let density_beta = g.input(DENSITY_BETA_INPUT, s_scalar.clone());

    let hpwl_per_batch = hpwl_subgraph_parallel(
        &mut g, netlist, positions_x, positions_y, &port_offsets, hpwl_beta, b, n, &s_b,
    );
    let density_per_batch = density_subgraph_parallel(
        &mut g, netlist, lib, positions_x, positions_y, density_beta, b, n, &s_b,
    );

    // Loss per batch element; reduce to scalar for AD entry point.
    let weight = const_f32(&mut g, density_weight, s_scalar.clone());
    let weight_b = g.add_node(
        Op::Expand { target_shape: vec![b as i64] },
        vec![weight],
        s_b.clone(),
    );
    let weighted_density = g.binary(BinaryOp::Mul, density_per_batch, weight_b, s_b.clone());
    let per_batch = g.binary(BinaryOp::Add, hpwl_per_batch, weighted_density, s_b);
    // Scalar total — what `grad_with_loss` differentiates against.
    let total = g.reduce(per_batch, ReduceOp::Sum, vec![0], false, s_scalar);

    // grad_with_loss requires a single scalar output. To recover
    // per-batch loss after AD, the caller can build a second
    // forward-only graph via `combined_loss_graph_parallel_per_batch`.
    g.set_outputs(vec![total]);
    let _ = per_batch;
    g
}

/// Forward-only variant of [`combined_loss_graph_parallel`] that
/// returns the `[B]` per-batch loss as the only output. Useful
/// for the "best-of-B" readout after `combined_loss_graph_parallel`
/// + Adam has converged: build this graph once at the end, run
/// it on the final positions, pick the lowest-loss batch.
pub fn combined_loss_graph_parallel_per_batch(
    netlist: &Netlist,
    lib: &Library,
    batch_size: usize,
    density_weight: f32,
) -> Graph {
    assert!(batch_size >= 1, "batch_size must be >= 1");
    let mut g = Graph::new(format!("{}_parallel_perbatch_B{}", netlist.name, batch_size));
    let s_scalar = TensorShape::new(&[1], DType::F32);
    let n = netlist.instances.len();
    let b = batch_size;
    let s_bn = TensorShape::new(&[b, n], DType::F32);
    let s_b = TensorShape::new(&[b], DType::F32);

    let positions_x = g.param(POSITIONS_X_PARAM, s_bn.clone());
    let positions_y = g.param(POSITIONS_Y_PARAM, s_bn);
    let port_offsets = collect_port_offsets(netlist, lib);
    let hpwl_beta = g.input(HPWL_BETA_INPUT, s_scalar.clone());
    let density_beta = g.input(DENSITY_BETA_INPUT, s_scalar.clone());

    let hpwl_per_batch = hpwl_subgraph_parallel(
        &mut g, netlist, positions_x, positions_y, &port_offsets, hpwl_beta, b, n, &s_b,
    );
    let density_per_batch = density_subgraph_parallel(
        &mut g, netlist, lib, positions_x, positions_y, density_beta, b, n, &s_b,
    );

    let weight = const_f32(&mut g, density_weight, s_scalar);
    let weight_b = g.add_node(
        Op::Expand { target_shape: vec![b as i64] },
        vec![weight],
        s_b.clone(),
    );
    let weighted_density = g.binary(BinaryOp::Mul, density_per_batch, weight_b, s_b.clone());
    let per_batch = g.binary(BinaryOp::Add, hpwl_per_batch, weighted_density, s_b);
    g.set_outputs(vec![per_batch]);
    g
}

fn hpwl_subgraph_parallel(
    g: &mut Graph,
    netlist: &Netlist,
    positions_x: NodeId,
    positions_y: NodeId,
    port_offsets: &HashMap<(usize, String), (f32, f32)>,
    beta: NodeId,
    b: usize,
    _n: usize,
    s_b: &TensorShape,
) -> NodeId {
    let s_b1 = TensorShape::new(&[b, 1], DType::F32);
    let mut total: Option<NodeId> = None;
    for net in &netlist.nets {
        if net.pins.len() < 2 { continue; }
        // Per-pin slice + offset → list of [B, 1] tensors.
        let mut x_slices = Vec::with_capacity(net.pins.len());
        let mut y_slices = Vec::with_capacity(net.pins.len());
        let mut net_resolvable = true;
        for pin in &net.pins {
            let key = (pin.instance, pin.port.clone());
            let (ox, oy) = match port_offsets.get(&key) {
                Some(v) => *v,
                None => { net_resolvable = false; break; }
            };
            let x_slot = g.add_node(
                Op::Narrow { axis: 1, start: pin.instance, len: 1 },
                vec![positions_x], s_b1.clone(),
            );
            let y_slot = g.add_node(
                Op::Narrow { axis: 1, start: pin.instance, len: 1 },
                vec![positions_y], s_b1.clone(),
            );
            // Port-offset constants broadcast over B via Expand.
            let ox_const = const_f32(g, ox, TensorShape::new(&[1], DType::F32));
            let oy_const = const_f32(g, oy, TensorShape::new(&[1], DType::F32));
            let ox_reshape = g.reshape(ox_const, vec![1, 1], TensorShape::new(&[1, 1], DType::F32));
            let oy_reshape = g.reshape(oy_const, vec![1, 1], TensorShape::new(&[1, 1], DType::F32));
            let ox_b = g.add_node(
                Op::Expand { target_shape: vec![b as i64, 1] },
                vec![ox_reshape], s_b1.clone(),
            );
            let oy_b = g.add_node(
                Op::Expand { target_shape: vec![b as i64, 1] },
                vec![oy_reshape], s_b1.clone(),
            );
            x_slices.push(g.binary(BinaryOp::Add, x_slot, ox_b, s_b1.clone()));
            y_slices.push(g.binary(BinaryOp::Add, y_slot, oy_b, s_b1.clone()));
        }
        if !net_resolvable { continue; }

        // Concat per-pin [B, 1]s into [B, K] and reduce-LSE along axis 1.
        let k = x_slices.len();
        let s_bk = TensorShape::new(&[b, k], DType::F32);
        let xs_bk = g.concat(x_slices, 1, s_bk.clone());
        let ys_bk = g.concat(y_slices, 1, s_bk.clone());

        let span_x = smooth_span_axis1(g, xs_bk, beta, b, k);
        let span_y = smooth_span_axis1(g, ys_bk, beta, b, k);
        let raw = g.binary(BinaryOp::Add, span_x, span_y, s_b.clone());
        let net_hpwl = if (net.weight - 1.0).abs() < f32::EPSILON {
            raw
        } else {
            let w = const_f32(g, net.weight, TensorShape::new(&[1], DType::F32));
            let w_b = g.add_node(
                Op::Expand { target_shape: vec![b as i64] },
                vec![w], s_b.clone(),
            );
            g.binary(BinaryOp::Mul, raw, w_b, s_b.clone())
        };
        total = Some(match total {
            Some(acc) => g.binary(BinaryOp::Add, acc, net_hpwl, s_b.clone()),
            None => net_hpwl,
        });
    }
    total.unwrap_or_else(|| {
        let zero = const_f32(g, 0.0, TensorShape::new(&[1], DType::F32));
        g.add_node(Op::Expand { target_shape: vec![b as i64] }, vec![zero], s_b.clone())
    })
}

/// `smooth_max - smooth_min` reducing along axis 1 of a `[B, K]`
/// tensor → `[B]`. β is a scalar broadcast across the reduction.
fn smooth_span_axis1(
    g: &mut Graph,
    xs: NodeId,
    beta: NodeId,
    b: usize,
    k: usize,
) -> NodeId {
    let s_bk = TensorShape::new(&[b, k], DType::F32);
    let s_b = TensorShape::new(&[b], DType::F32);
    let s_scalar = TensorShape::new(&[1], DType::F32);

    let one = const_f32(g, 1.0, s_scalar.clone());
    let neg_one = const_f32(g, -1.0, s_scalar.clone());
    let inv_beta = g.binary(BinaryOp::Div, one, beta, s_scalar.clone());
    let neg_beta = g.binary(BinaryOp::Mul, beta, neg_one, s_scalar.clone());
    let neg_inv_beta = g.binary(BinaryOp::Mul, inv_beta, neg_one, s_scalar.clone());

    let beta_bk = g.add_node(Op::Expand { target_shape: vec![b as i64, k as i64] }, vec![beta], s_bk.clone());
    let neg_beta_bk = g.add_node(Op::Expand { target_shape: vec![b as i64, k as i64] }, vec![neg_beta], s_bk.clone());

    // smooth_max along axis 1: log_sum_exp(β·xs) / β.
    let bx = g.binary(BinaryOp::Mul, xs, beta_bk, s_bk.clone());
    let ex = g.activation(Activation::Exp, bx, s_bk.clone());
    let sum_ex = g.reduce(ex, ReduceOp::Sum, vec![1], false, s_b.clone());
    let lse = g.activation(Activation::Log, sum_ex, s_b.clone());
    let inv_beta_b = g.add_node(Op::Expand { target_shape: vec![b as i64] }, vec![inv_beta], s_b.clone());
    let smax = g.binary(BinaryOp::Mul, lse, inv_beta_b, s_b.clone());

    // smooth_min along axis 1: -log_sum_exp(-β·xs) / β.
    let nbx = g.binary(BinaryOp::Mul, xs, neg_beta_bk, s_bk.clone());
    let nex = g.activation(Activation::Exp, nbx, s_bk);
    let nsum_ex = g.reduce(nex, ReduceOp::Sum, vec![1], false, s_b.clone());
    let nlse = g.activation(Activation::Log, nsum_ex, s_b.clone());
    let neg_inv_beta_b = g.add_node(Op::Expand { target_shape: vec![b as i64] }, vec![neg_inv_beta], s_b.clone());
    let smin = g.binary(BinaryOp::Mul, nlse, neg_inv_beta_b, s_b.clone());

    g.binary(BinaryOp::Sub, smax, smin, s_b)
}

fn density_subgraph_parallel(
    g: &mut Graph,
    netlist: &Netlist,
    lib: &Library,
    positions_x: NodeId,
    positions_y: NodeId,
    beta: NodeId,
    b: usize,
    n: usize,
    s_b: &TensorShape,
) -> NodeId {
    if n < 2 {
        let zero = const_f32(g, 0.0, TensorShape::new(&[1], DType::F32));
        return g.add_node(Op::Expand { target_shape: vec![b as i64] }, vec![zero], s_b.clone());
    }

    // Per-pair half-dim sums baked as [N, N] and broadcast to [B, N, N].
    let half_w: Vec<f32> = netlist.instances.iter()
        .map(|inst| { let bb = lib.get(inst.cell).local_bbox(); ((bb.max.x - bb.min.x) as f32) * 0.5 })
        .collect();
    let half_h: Vec<f32> = netlist.instances.iter()
        .map(|inst| { let bb = lib.get(inst.cell).local_bbox(); ((bb.max.y - bb.min.y) as f32) * 0.5 })
        .collect();
    let mut hw_pairs = vec![0.0_f32; n * n];
    let mut hh_pairs = vec![0.0_f32; n * n];
    let mut mask    = vec![1.0_f32; n * n];
    for i in 0..n {
        mask[i * n + i] = 0.0;
        for j in 0..n {
            hw_pairs[i * n + j] = half_w[i] + half_w[j];
            hh_pairs[i * n + j] = half_h[i] + half_h[j];
        }
    }
    let s_nn = TensorShape::new(&[n, n], DType::F32);
    let s_1nn = TensorShape::new(&[1, n, n], DType::F32);
    let s_bnn = TensorShape::new(&[b, n, n], DType::F32);
    let half_w_2d = const_tensor(g, &hw_pairs, s_nn.clone());
    let half_h_2d = const_tensor(g, &hh_pairs, s_nn.clone());
    let mask_2d = const_tensor(g, &mask, s_nn.clone());
    let half_w_3d = g.reshape(half_w_2d, vec![1, n as i64, n as i64], s_1nn.clone());
    let half_h_3d = g.reshape(half_h_2d, vec![1, n as i64, n as i64], s_1nn.clone());
    let mask_3d = g.reshape(mask_2d, vec![1, n as i64, n as i64], s_1nn);
    let half_w_b = g.add_node(Op::Expand { target_shape: vec![b as i64, n as i64, n as i64] }, vec![half_w_3d], s_bnn.clone());
    let half_h_b = g.add_node(Op::Expand { target_shape: vec![b as i64, n as i64, n as i64] }, vec![half_h_3d], s_bnn.clone());
    let mask_b   = g.add_node(Op::Expand { target_shape: vec![b as i64, n as i64, n as i64] }, vec![mask_3d], s_bnn.clone());

    // [B, N, 1] and [B, 1, N] reshapes of positions, expanded to [B, N, N].
    let s_bn1 = TensorShape::new(&[b, n, 1], DType::F32);
    let s_b1n = TensorShape::new(&[b, 1, n], DType::F32);
    let x_col = g.reshape(positions_x, vec![b as i64, n as i64, 1], s_bn1.clone());
    let x_row = g.reshape(positions_x, vec![b as i64, 1, n as i64], s_b1n.clone());
    let x_col_e = g.add_node(Op::Expand { target_shape: vec![b as i64, n as i64, n as i64] }, vec![x_col], s_bnn.clone());
    let x_row_e = g.add_node(Op::Expand { target_shape: vec![b as i64, n as i64, n as i64] }, vec![x_row], s_bnn.clone());
    let dx = g.binary(BinaryOp::Sub, x_col_e, x_row_e, s_bnn.clone());

    let y_col = g.reshape(positions_y, vec![b as i64, n as i64, 1], s_bn1);
    let y_row = g.reshape(positions_y, vec![b as i64, 1, n as i64], s_b1n);
    let y_col_e = g.add_node(Op::Expand { target_shape: vec![b as i64, n as i64, n as i64] }, vec![y_col], s_bnn.clone());
    let y_row_e = g.add_node(Op::Expand { target_shape: vec![b as i64, n as i64, n as i64] }, vec![y_row], s_bnn.clone());
    let dy = g.binary(BinaryOp::Sub, y_col_e, y_row_e, s_bnn.clone());

    let dx2 = g.binary(BinaryOp::Mul, dx, dx, s_bnn.clone());
    let dy2 = g.binary(BinaryOp::Mul, dy, dy, s_bnn.clone());
    let eps_x = const_tensor(g, &vec![1.0_f32; b * n * n], s_bnn.clone());
    let eps_y = const_tensor(g, &vec![1.0_f32; b * n * n], s_bnn.clone());
    let dx2_eps = g.binary(BinaryOp::Add, dx2, eps_x, s_bnn.clone());
    let dy2_eps = g.binary(BinaryOp::Add, dy2, eps_y, s_bnn.clone());
    let abs_dx = g.activation(Activation::Sqrt, dx2_eps, s_bnn.clone());
    let abs_dy = g.activation(Activation::Sqrt, dy2_eps, s_bnn.clone());

    let z_x = g.binary(BinaryOp::Sub, half_w_b, abs_dx, s_bnn.clone());
    let z_y = g.binary(BinaryOp::Sub, half_h_b, abs_dy, s_bnn.clone());

    // smooth_relu broadcast: scalar β → [B, N, N]
    let beta_bnn = g.add_node(
        Op::Expand { target_shape: vec![b as i64, n as i64, n as i64] },
        vec![beta], s_bnn.clone(),
    );
    let bz_x = g.binary(BinaryOp::Mul, z_x, beta_bnn, s_bnn.clone());
    let sig_x = g.activation(Activation::Sigmoid, bz_x, s_bnn.clone());
    let rx = g.binary(BinaryOp::Mul, z_x, sig_x, s_bnn.clone());
    let bz_y = g.binary(BinaryOp::Mul, z_y, beta_bnn, s_bnn.clone());
    let sig_y = g.activation(Activation::Sigmoid, bz_y, s_bnn.clone());
    let ry = g.binary(BinaryOp::Mul, z_y, sig_y, s_bnn.clone());

    let pair = g.binary(BinaryOp::Mul, rx, ry, s_bnn.clone());
    let pair_off_diag = g.binary(BinaryOp::Mul, pair, mask_b, s_bnn);
    // Reduce over the two pair axes (1, 2), keeping the batch axis (0).
    let total = g.reduce(pair_off_diag, ReduceOp::Sum, vec![1, 2], false, s_b.clone());
    let half = const_f32(g, 0.5, TensorShape::new(&[1], DType::F32));
    let half_b = g.add_node(Op::Expand { target_shape: vec![b as i64] }, vec![half], s_b.clone());
    g.binary(BinaryOp::Mul, total, half_b, s_b.clone())
}

/// Position-Param NodeIds for the parallel-batch graph: same names
/// as the single-batch variant (`POSITIONS_X_PARAM` / `POSITIONS_Y_PARAM`)
/// but the shapes are `[B, N]`. Hand to `grad_with_loss` to get
/// gradient tensors of the same shape.
pub fn position_param_ids_parallel(g: &Graph) -> Vec<NodeId> {
    position_param_ids_batched(g)
}

/// Per-instance node info for downstream subgraphs.
///
/// Movable instances surface as `(Param, Param)` named
/// `<netlist>.<instance>.{x,y}`. Fixed instances surface as
/// `(Constant, Constant)` materialised from
/// [`DifferentiablePlacement::instance_xy`] at graph-build time —
/// they take no part in [`position_param_ids`] and stay invariant
/// under any optimizer that only updates the position Params.
fn register_position_params(
    g: &mut Graph,
    netlist: &Netlist,
    seed: &DifferentiablePlacement,
) -> (Vec<NodeId>, Vec<NodeId>) {
    let s = TensorShape::new(&[1], DType::F32);
    let mut xs = Vec::with_capacity(netlist.instances.len());
    let mut ys = Vec::with_capacity(netlist.instances.len());
    for (i, inst) in netlist.instances.iter().enumerate() {
        if inst.fixed {
            let (x0, y0) = seed
                .instance_xy
                .get(i)
                .copied()
                .unwrap_or((0.0, 0.0));
            xs.push(const_f32(g, x0, s.clone()));
            ys.push(const_f32(g, y0, s.clone()));
        } else {
            xs.push(g.param(format!("{}.{}.x", netlist.name, inst.name), s.clone()));
            ys.push(g.param(format!("{}.{}.y", netlist.name, inst.name), s.clone()));
        }
    }
    (xs, ys)
}

fn collect_port_offsets(
    netlist: &Netlist,
    lib: &Library,
) -> HashMap<(usize, String), (f32, f32)> {
    let mut cache: HashMap<(usize, String), (f32, f32)> = HashMap::new();
    for net in &netlist.nets {
        for pin in &net.pins {
            let key = (pin.instance, pin.port.clone());
            if cache.contains_key(&key) { continue; }
            let inst = match netlist.instances.get(pin.instance) {
                Some(i) => i,
                None => continue,
            };
            let cell = lib.get(inst.cell);
            if let Some(p) = cell.port(&pin.port) {
                cache.insert(key, (p.center.x as f32, p.center.y as f32));
            }
        }
    }
    cache
}

fn hpwl_subgraph(
    g: &mut Graph,
    netlist: &Netlist,
    x_params: &[NodeId],
    y_params: &[NodeId],
    port_offsets: &HashMap<(usize, String), (f32, f32)>,
    beta: NodeId,
    s: &TensorShape,
) -> NodeId {
    let mut total: Option<NodeId> = None;
    for net in &netlist.nets {
        if net.pins.len() < 2 { continue; }
        let mut xs = Vec::with_capacity(net.pins.len());
        let mut ys = Vec::with_capacity(net.pins.len());
        let mut resolvable = true;
        for pin in &net.pins {
            let key = (pin.instance, pin.port.clone());
            let (ox, oy) = match port_offsets.get(&key) {
                Some(v) => *v,
                None => { resolvable = false; break; }
            };
            let ox_n = const_f32(g, ox, s.clone());
            let oy_n = const_f32(g, oy, s.clone());
            xs.push(g.binary(BinaryOp::Add, x_params[pin.instance], ox_n, s.clone()));
            ys.push(g.binary(BinaryOp::Add, y_params[pin.instance], oy_n, s.clone()));
        }
        if !resolvable { continue; }
        let span_x = smooth_span(g, &xs, beta, s);
        let span_y = smooth_span(g, &ys, beta, s);
        let raw_net_hpwl = g.binary(BinaryOp::Add, span_x, span_y, s.clone());
        let net_hpwl = if (net.weight - 1.0).abs() < f32::EPSILON {
            raw_net_hpwl
        } else {
            let w = const_f32(g, net.weight, s.clone());
            g.binary(BinaryOp::Mul, raw_net_hpwl, w, s.clone())
        };
        total = Some(match total {
            Some(acc) => g.binary(BinaryOp::Add, acc, net_hpwl, s.clone()),
            None => net_hpwl,
        });
    }
    total.unwrap_or_else(|| const_f32(g, 0.0, s.clone()))
}

fn density_subgraph(
    g: &mut Graph,
    netlist: &Netlist,
    lib: &Library,
    x_params: &[NodeId],
    y_params: &[NodeId],
    beta: NodeId,
    s: &TensorShape,
) -> NodeId {
    // Cache each instance's bbox half-width / half-height (in DBU,
    // f32-converted). These are constants — instance moves are
    // pure translation, the bbox dims don't change.
    let half_dims: Vec<(f32, f32)> = netlist
        .instances
        .iter()
        .map(|inst| {
            let bbox = lib.get(inst.cell).local_bbox();
            let hw = ((bbox.max.x - bbox.min.x) as f32) * 0.5;
            let hh = ((bbox.max.y - bbox.min.y) as f32) * 0.5;
            (hw, hh)
        })
        .collect();
    // 1 DBU² added under the sqrt to keep |Δ|'s gradient finite at
    // coincident positions.
    let eps_sq = const_f32(g, 1.0, s.clone());
    let beta_n = beta;

    let mut total: Option<NodeId> = None;
    let n = netlist.instances.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let (hwi, hhi) = half_dims[i];
            let (hwj, hhj) = half_dims[j];
            let half_w_sum = const_f32(g, hwi + hwj, s.clone());
            let half_h_sum = const_f32(g, hhi + hhj, s.clone());

            // |Δx| ≈ √(Δx² + ε²)
            let dx = g.binary(BinaryOp::Sub, x_params[i], x_params[j], s.clone());
            let dy = g.binary(BinaryOp::Sub, y_params[i], y_params[j], s.clone());
            let dx2 = g.binary(BinaryOp::Mul, dx, dx, s.clone());
            let dy2 = g.binary(BinaryOp::Mul, dy, dy, s.clone());
            let dx2_eps = g.binary(BinaryOp::Add, dx2, eps_sq, s.clone());
            let dy2_eps = g.binary(BinaryOp::Add, dy2, eps_sq, s.clone());
            let abs_dx = g.activation(Activation::Sqrt, dx2_eps, s.clone());
            let abs_dy = g.activation(Activation::Sqrt, dy2_eps, s.clone());

            let z_x = g.binary(BinaryOp::Sub, half_w_sum, abs_dx, s.clone());
            let z_y = g.binary(BinaryOp::Sub, half_h_sum, abs_dy, s.clone());
            let rx = smooth_relu(g, z_x, beta_n, s.clone());
            let ry = smooth_relu(g, z_y, beta_n, s.clone());
            let pair_overlap = g.binary(BinaryOp::Mul, rx, ry, s.clone());

            total = Some(match total {
                Some(acc) => g.binary(BinaryOp::Add, acc, pair_overlap, s.clone()),
                None => pair_overlap,
            });
        }
    }
    total.unwrap_or_else(|| const_f32(g, 0.0, s.clone()))
}

// ── Symmetry / matching penalty ───────────────────────────────────────
//
// Analog layout cares about *matched* devices: differential pairs
// share parasitics by sitting at mirrored positions, current
// mirrors share centroid so process gradients average out, ratioed
// arrays interdigitate so a linear gradient cancels.
//
// Each [`crate::netlist::MatchGroup`] turns into a quadratic
// penalty on the position Params. The penalty is zero at the
// constraint manifold and grows as squared deviation in DBU² —
// the natural weight is comparable to HPWL once it dominates by
// the cell-pitch scale.
//
// All terms differentiate cleanly: gradient = 2·deviation, finite
// at the optimum, no smoothing trick needed (unlike `|Δ|` in the
// density term).

/// Build a forward graph that returns scalar matching penalty
/// summed over every [`crate::netlist::MatchGroup`]. Differentiable
/// wrt every position Param. Returns a graph with one scalar
/// output. Useful to inspect / weight independently of HPWL +
/// density; the same term is folded into
/// [`combined_loss_graph_with_symmetry`].
pub fn symmetry_loss_graph(netlist: &Netlist, seed: &DifferentiablePlacement) -> Graph {
    let mut g = Graph::new(format!("{}_symmetry", netlist.name));
    let s = TensorShape::new(&[1], DType::F32);
    let (x_params, y_params) = register_position_params(&mut g, netlist, seed);
    let total = symmetry_subgraph(&mut g, netlist, &x_params, &y_params, &s);
    g.set_outputs(vec![total]);
    g
}

fn symmetry_subgraph(
    g: &mut Graph,
    netlist: &Netlist,
    x_params: &[NodeId],
    y_params: &[NodeId],
    s: &TensorShape,
) -> NodeId {
    let n_inst = netlist.instances.len();
    let mut total: Option<NodeId> = None;
    for grp in &netlist.match_groups {
        let term = match &grp.kind {
            MatchKind::Mirror { a, b, axis, axis_coord } => {
                if *a >= n_inst || *b >= n_inst || a == b { continue; }
                mirror_term(g, x_params, y_params, *a, *b, *axis, *axis_coord, s)
            }
            MatchKind::CommonCentroid { instances, center } => {
                let valid: Vec<usize> = instances.iter().copied()
                    .filter(|i| *i < n_inst).collect();
                if valid.len() < 2 { continue; }
                centroid_term(g, x_params, y_params, &valid, *center, s)
            }
            MatchKind::Interdigitated { instances, axis, origin, pitch } => {
                let valid: Vec<usize> = instances.iter().copied()
                    .filter(|i| *i < n_inst).collect();
                if valid.len() < 2 { continue; }
                interdigitated_term(g, x_params, y_params, &valid, *axis, *origin, *pitch, s)
            }
        };
        let weighted = if (grp.weight - 1.0).abs() < f32::EPSILON {
            term
        } else {
            let w = const_f32(g, grp.weight, s.clone());
            g.binary(BinaryOp::Mul, term, w, s.clone())
        };
        total = Some(match total {
            Some(acc) => g.binary(BinaryOp::Add, acc, weighted, s.clone()),
            None => weighted,
        });
    }
    total.unwrap_or_else(|| const_f32(g, 0.0, s.clone()))
}

/// `(x_a + x_b − 2c)² + (y_a − y_b)²` for vertical axis (and the
/// axes swapped for horizontal). Zero exactly when the two
/// instances are reflections across the axis line `coord`.
fn mirror_term(
    g: &mut Graph,
    x_params: &[NodeId],
    y_params: &[NodeId],
    a: usize,
    b: usize,
    axis: SymmetryAxis,
    axis_coord: f32,
    s: &TensorShape,
) -> NodeId {
    let two_c = const_f32(g, 2.0 * axis_coord, s.clone());
    let (mirror_a, mirror_b, equal_a, equal_b) = match axis {
        SymmetryAxis::Vertical => (x_params[a], x_params[b], y_params[a], y_params[b]),
        SymmetryAxis::Horizontal => (y_params[a], y_params[b], x_params[a], x_params[b]),
    };
    let sum = g.binary(BinaryOp::Add, mirror_a, mirror_b, s.clone());
    let dev = g.binary(BinaryOp::Sub, sum, two_c, s.clone());
    let dev_sq = g.binary(BinaryOp::Mul, dev, dev, s.clone());
    let diff = g.binary(BinaryOp::Sub, equal_a, equal_b, s.clone());
    let diff_sq = g.binary(BinaryOp::Mul, diff, diff, s.clone());
    g.binary(BinaryOp::Add, dev_sq, diff_sq, s.clone())
}

/// `(Σx_i − N·cx)² + (Σy_i − N·cy)²` — equivalent up to N² to the
/// squared-mean-deviation; saves a division on the graph and has
/// the same minimizer.
fn centroid_term(
    g: &mut Graph,
    x_params: &[NodeId],
    y_params: &[NodeId],
    instances: &[usize],
    center: (f32, f32),
    s: &TensorShape,
) -> NodeId {
    let n = instances.len() as f32;
    let mut sum_x: Option<NodeId> = None;
    let mut sum_y: Option<NodeId> = None;
    for &i in instances {
        sum_x = Some(match sum_x {
            Some(acc) => g.binary(BinaryOp::Add, acc, x_params[i], s.clone()),
            None => x_params[i],
        });
        sum_y = Some(match sum_y {
            Some(acc) => g.binary(BinaryOp::Add, acc, y_params[i], s.clone()),
            None => y_params[i],
        });
    }
    let target_x = const_f32(g, n * center.0, s.clone());
    let target_y = const_f32(g, n * center.1, s.clone());
    let dev_x = g.binary(BinaryOp::Sub, sum_x.unwrap(), target_x, s.clone());
    let dev_y = g.binary(BinaryOp::Sub, sum_y.unwrap(), target_y, s.clone());
    let dx_sq = g.binary(BinaryOp::Mul, dev_x, dev_x, s.clone());
    let dy_sq = g.binary(BinaryOp::Mul, dev_y, dev_y, s.clone());
    g.binary(BinaryOp::Add, dx_sq, dy_sq, s.clone())
}

/// Σ_k (pos_axis[k] − (origin + k·pitch))² along the array axis,
/// plus Σ_{k>0} (pos_other[k] − pos_other[0])² to force the row
/// flat on the perpendicular.
fn interdigitated_term(
    g: &mut Graph,
    x_params: &[NodeId],
    y_params: &[NodeId],
    instances: &[usize],
    axis: SymmetryAxis,
    origin: f32,
    pitch: f32,
    s: &TensorShape,
) -> NodeId {
    let (along, perp): (Vec<NodeId>, Vec<NodeId>) = match axis {
        SymmetryAxis::Vertical => (
            instances.iter().map(|&i| x_params[i]).collect(),
            instances.iter().map(|&i| y_params[i]).collect(),
        ),
        SymmetryAxis::Horizontal => (
            instances.iter().map(|&i| y_params[i]).collect(),
            instances.iter().map(|&i| x_params[i]).collect(),
        ),
    };
    let mut total: Option<NodeId> = None;
    for (k, &p) in along.iter().enumerate() {
        let target = const_f32(g, origin + k as f32 * pitch, s.clone());
        let dev = g.binary(BinaryOp::Sub, p, target, s.clone());
        let dev_sq = g.binary(BinaryOp::Mul, dev, dev, s.clone());
        total = Some(match total {
            Some(acc) => g.binary(BinaryOp::Add, acc, dev_sq, s.clone()),
            None => dev_sq,
        });
    }
    let p0 = perp[0];
    for &pk in perp.iter().skip(1) {
        let diff = g.binary(BinaryOp::Sub, pk, p0, s.clone());
        let diff_sq = g.binary(BinaryOp::Mul, diff, diff, s.clone());
        total = Some(match total {
            Some(acc) => g.binary(BinaryOp::Add, acc, diff_sq, s.clone()),
            None => diff_sq,
        });
    }
    total.unwrap_or_else(|| const_f32(g, 0.0, s.clone()))
}

/// `smooth_relu(z; β) = z · sigmoid(β · z)`. The swish/SiLU shape
/// with an explicit sharpness — at large positive `β·z` it
/// approaches `z`; at large negative `β·z` it approaches 0; smooth
/// and differentiable everywhere.
fn smooth_relu(g: &mut Graph, z: NodeId, beta_n: NodeId, s: TensorShape) -> NodeId {
    let bz = g.binary(BinaryOp::Mul, z, beta_n, s.clone());
    let sig = g.activation(Activation::Sigmoid, bz, s.clone());
    g.binary(BinaryOp::Mul, z, sig, s)
}

/// Helper: list every position-Param NodeId so callers can pass it
/// to `grad_with_loss`. Order matches `netlist.instances`, with
/// `x` then `y` per *movable* instance. Fixed instances surface
/// as graph Constants and contribute no Params, so they're skipped.
pub fn position_param_ids(g: &Graph, netlist: &Netlist) -> Vec<NodeId> {
    let mut out = Vec::with_capacity(netlist.instances.len() * 2);
    for inst in &netlist.instances {
        if inst.fixed { continue; }
        for axis in ["x", "y"] {
            let key = format!("{}.{}.{axis}", netlist.name, inst.name);
            let id = g
                .nodes()
                .iter()
                .enumerate()
                .find_map(|(i, n)| match &n.op {
                    Op::Param { name, .. } if *name == key => Some(NodeId(i as u32)),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("position param missing: {key}"));
            out.push(id);
        }
    }
    out
}

/// Map from a position-param-list index (the order
/// [`position_param_ids`] returns) back to `(instance_index, axis)`
/// where axis is 0 for x, 1 for y. Useful for Adam-style updates
/// that read `outs[1 + k]` and need to know which `(x, y)` slot in
/// the placement to mutate.
pub fn position_param_layout(netlist: &Netlist) -> Vec<(usize, u8)> {
    let mut out = Vec::new();
    for (i, inst) in netlist.instances.iter().enumerate() {
        if inst.fixed { continue; }
        out.push((i, 0));
        out.push((i, 1));
    }
    out
}
