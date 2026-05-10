//! `eda-hir` — High-level IR traits for code-defined circuits.
//!
//! The HIR is the user-facing layer of rlx-eda: a small set of traits a
//! Rust type implements to declare itself as a circuit block. From those
//! traits, the framework derives:
//!
//! - **MIR**: rlx graph fragments for differentiable simulation (later).
//! - **LIR**: klayout-rs `Cell`s and `Library` entries for layout / GDS.
//! - **Netlist**: schematic-shaped IR for SPICE export, LVS (later).
//!
//! ## Scope of this crate
//!
//! This pass defines four traits — `Block`, `Layout<P>`, `DcBehavioral`,
//! `NonlinearDcBehavioral` — plus `Schematic<P>` for symbolic schematic
//! emission. `Schematic<P>` is what binds the renderer in `eda-viz` to
//! the same Rust type that produces the layout; without it,
//! schematic-side data drifts away from layout-side reality.
//!
//! The deliberate constraint: each trait carries **only the obligation
//! the user must meet to participate in that flow**. No premature
//! abstractions. New flows (behavioral simulation, formal verification,
//! statistical analysis, …) earn their own trait when they justify it.
//!
//! ## Why so small
//!
//! `Block` is intentionally narrow: any block must have a stable `name`
//! and an equality / hash story (so caching, dedup, and diagnostics work
//! across the framework). `Layout<P>` is intentionally a single method —
//! "given a Library + Pdk, build me a Cell" — so an implementor knows
//! exactly what they're on the hook for. Composition (one block laying
//! out children) happens naturally by calling another block's `layout`.

pub mod hierarchy;
pub mod schematic;
pub use hierarchy::{
    Cell, Core, Device, Die, DiePlacement, IoPad, Lot, Macro, Pin, PinDirection, Reticle, Wafer,
};
pub use schematic::{
    SchemOrient, SchemPort, SchemSymbol, SchemWire, Schematic, SchematicIr, SymbolKind,
};

use klayout_core::{CellId, Library};

/// Identity-and-equality contract every circuit block satisfies.
///
/// A `Block` has a stable `name()` (used as the underlying `CellName`
/// when laid out, and as the diagnostic identifier) and is `Hash + Eq`
/// so that two structurally-identical block values cache to the same
/// downstream artifact (Cell, MIR fragment, …).
///
/// Implementors typically `#[derive(Hash, Eq, PartialEq, Clone)]` on
/// their fields. If a field carries non-equatable state (e.g. an
/// `Rc<dyn Trait>`), wrap it in a structurally-keyed handle.
pub trait Block: std::hash::Hash + Eq + Send + Sync {
    /// Cell-name / diagnostic name. May depend on parameters — e.g.
    /// `"Resistor_L10000"`. Stable across runs given the same params.
    fn name(&self) -> String;
}

/// A block that can produce a layout in a PDK of type `P`.
///
/// `P` is whatever struct the user's PDK macro produces — typically a
/// `klayout_pdk::pdk!` result holding `LayerIndex` fields. The trait
/// imposes no methods on `P`; it's a type tag that lets the same block
/// type be ported to multiple PDKs in the future without changing this
/// trait.
///
/// ## Composition
///
/// `layout` builds *and inserts* its top cell into `lib`, returning the
/// resulting `CellId`. To compose, a parent block calls a child block's
/// `layout` to get a CellId, then adds an `Instance` of that child to
/// its own `CellBuilder`. `Library`'s interior-mutable `insert` makes
/// this nesting natural — no `&mut Library` plumbing through the call
/// graph.
pub trait Layout<P>: Block {
    /// Build this block's layout in `lib` (using `pdk` for layer / port
    /// information) and return the inserted top `CellId`.
    fn layout(&self, lib: &Library, pdk: &P) -> CellId;
}

/// A block that participates in DC simulation by exposing rlx-graph
/// parameters and contributing to a circuit residual.
///
/// ## Scope
///
/// MVP version: this trait covers **2-terminal linear devices** where
/// the block exposes a single resistance-like parameter to a downstream
/// MNA assembler. `add_to_dc(graph)` registers the block's `Param`
/// nodes and returns a handle to the principal one (resistance for a
/// resistor, conductance for an admittance, …) so the parent block can
/// stamp it into a global system.
///
/// 3+-terminal devices (MOSFET, BJT) earn a richer trait when we add
/// them — `add_to_dc` returning a `Vec<NodeId>` of per-terminal
/// admittance contributions, or a structured `DeviceStamp`. This MVP
/// stays narrow on purpose to validate the trait shape end-to-end on
/// the divider before generalizing.
pub trait DcBehavioral: Block {
    /// Add the block's parameters to `graph` and return the principal
    /// parameter `NodeId` (the resistance, for a 2-terminal R). The
    /// param is named via [`Block::name`] so multiple instances stay
    /// distinct.
    fn add_to_dc(&self, graph: &mut rlx_ir::Graph) -> rlx_ir::NodeId;
}

/// Multi-terminal nonlinear DC behavior: takes terminal voltages as
/// graph nodes, returns terminal currents as graph nodes.
///
/// ## Sign convention
///
/// `currents()[i]` is the current flowing **from the device into the
/// external node** at terminal `i`. For a 2-terminal passive device
/// with `v_a > v_b`:
///   - `currents[0]` (at `a`) is **negative** (the device sinks current
///     from `a` into itself)
///   - `currents[1]` (at `b`) is **positive** (the device pushes current
///     out into `b`)
///
/// KCL at any node: sum of `currents[i]` across all device terminals
/// connected to that node equals zero.
///
/// ## Why this trait coexists with `DcBehavioral`
///
/// `DcBehavioral` is the simplest "expose a single resistance Param"
/// shape — useful for closed-form 2-resistor circuits where the loss
/// graph is hand-derived. `NonlinearDcBehavioral` is the general MNA
/// assembly shape: any 2+ terminal device, linear or nonlinear, returns
/// signed currents from voltages, and a downstream MNA assembler stamps
/// them into a residual graph. Both traits can be implemented on the
/// same block — `Resistor` does both.
///
/// ## Limitations
///
/// MVP: assumes pure DC + voltage-controlled current. Capacitors
/// (storage), inductors (dual-storage), and voltage sources (algebraic
/// constraints) need richer trait shapes — those land alongside the
/// transient + AC analyses.
/// **Transient storage** — devices that contribute charge / flux storage
/// terms to the DAE `F(y) + dQ/dt = 0`. Capacitors (Q = C·V), inductors
/// (Φ = L·I), nonlinear MOS junction caps, etc.
///
/// In Backward Euler discretization at timestep `h`, the storage
/// element contributes a "companion" current:
///
/// ```text
///   i_C = C/h · ((v_a − v_b) − (v_a_prev − v_b_prev))
/// ```
///
/// stamped into KCL as `−i_C` at terminal `a` and `+i_C` at terminal
/// `b`. The matrix entries `±C/h` (companion conductances) form the
/// per-step Newton Jacobian.
///
/// MVP scope: **2-terminal linear storage**. The trait returns a single
/// `NodeId` for the capacitance / inductance Param value; the assembler
/// derives all stamp signs from the 2-terminal symmetry. Multi-terminal
/// storage (3-terminal MOS cap, transformers) earns its own trait when
/// we add those devices.
pub trait TransientStorage: Send + Sync {
    fn name(&self) -> String;
    /// Returns the storage Param `NodeId` (capacitance for a cap,
    /// inductance for an inductor) — the single scalar that scales the
    /// BE companion stamp. Param key = `<name>_C` (or `_L`).
    fn capacitance(&self, graph: &mut rlx_ir::Graph) -> rlx_ir::NodeId;
}

/// **Transient transport delay** — a one-way 2-terminal element that
/// injects, at terminal `out`, a current proportional to the voltage at
/// terminal `in` evaluated `τ` seconds in the past:
///
/// ```text
///   i_out(t) = G · v_in(t − τ)        i_in(t) = 0
/// ```
///
/// Models the dispersionless part of an optical / RF waveguide:
/// transport-only, no resonance. Pair with a [`TransientStorage`] cap on
/// the output net and (eventually) a vector-fitted dispersive block to
/// recover the full frequency response (circulax #2 / #3).
///
/// ## Why this is its own trait
///
/// A delay element makes the circuit a **delay differential equation**
/// rather than an ODE: the residual at time `t` depends on the history
/// `{v_in(t − τ)}`, not just the current state. The transient driver
/// has to maintain a per-element history buffer and feed an interpolated
/// past sample as an `Op::Input` to the per-step residual graph. That
/// orchestration is integrator-side, not graph-side — the trait stays
/// minimal and just exposes the delay value (`τ`, plain `f64` since it
/// indexes a buffer) and the gain (a `Param`, so it stays differentiable
/// for inverse design).
///
/// ## DC behavior
///
/// At DC, all derivatives are zero, so the delay collapses to an
/// instantaneous buffer `i_out = G · v_in`. `eda_mna::build_residual_graph`
/// stamps that contribution so `solve_dc` converges with delays in
/// feedback loops (waveguide rings, optical resonators).
///
/// ## Sub-step delays (τ < dt) and differentiability
///
/// The transient assembler unifies long and sub-step delays under one
/// stamp:
///
/// ```text
///   v_delayed = (1 − α) · v_lo + α · v_hi          α = offset − τ/h
/// ```
///
/// where `(v_lo, v_hi)` is either `(v_in_prev, v_in_now)` for
/// sub-step τ or two surrounding history samples for long τ; the
/// switch is a per-step `blend` Op::Input. Because τ flows through
/// the in-graph `α`, AD wrt the `<name>_tau` Param is well-defined
/// inside any integer-step window. (Crossing `τ = k·dt` shifts the
/// `offset` Op::Input for the next call — non-smooth at that
/// boundary, smooth on either side.)
pub trait TransientDelay: Send + Sync {
    fn name(&self) -> String;
    /// Transport delay in seconds. Used integrator-side for history
    /// indexing and as the default value of the `<name>_tau` Param if
    /// the caller's `params` map doesn't override it.
    fn delay_seconds(&self) -> f64;
    /// Gain `G` (siemens — current per delayed-volt). Param key
    /// conventionally `<name>_G`.
    fn gain(&self, graph: &mut rlx_ir::Graph) -> rlx_ir::NodeId;
    /// Transport delay as a graph `Param` (key `<name>_tau`). Lets AD
    /// flow ∂v_out/∂τ through the in-graph `α` formula. The integrator
    /// backfills this Param from `delay_seconds()` whenever the
    /// caller's `params` map omits it.
    fn delay_param(&self, graph: &mut rlx_ir::Graph) -> rlx_ir::NodeId;
}

/// **Generalized MNA device** — a device that contributes to a Modified
/// Nodal Analysis system, allowing both terminal-currents (KCL stamps)
/// and branch-current unknowns (algebraic constraints).
///
/// `NonlinearDcBehavioral` is the special case where `n_branches() = 0`
/// — purely terminal-current devices. Voltage sources, inductors, and
/// any device whose current isn't a function of terminal voltages alone
/// implement `MnaDevice` directly with `n_branches() = 1` or more.
///
/// ## What `contributions` returns
///
/// `(terminal_currents, branch_residuals)`:
///
/// - `terminal_currents[i]` — current flowing from the device into the
///   external node at terminal `i` (same sign convention as
///   `NonlinearDcBehavioral::currents`).
/// - `branch_residuals[k]` — algebraic equation that must be zero at
///   the operating point (e.g. a voltage source's `v_a − v_b − V_src`).
///
/// Branch-current unknowns are passed in via `branches`; the device
/// uses them as if they were variables. The MNA assembler allocates
/// one branch unknown per `n_branches()` declared and threads it
/// through the residual graph.
pub trait MnaDevice: Send + Sync {
    fn name(&self) -> String;
    fn n_terminals(&self) -> usize;
    /// Number of branch-current unknowns this device contributes (in
    /// addition to its terminal-current stamps). Most 2-terminal devices
    /// (resistor, diode, capacitor) need 0; voltage sources and ideal
    /// inductors need 1; transformers need ≥ 2.
    fn n_branches(&self) -> usize { 0 }

    fn contributions(
        &self,
        voltages: &[rlx_ir::NodeId],
        branches: &[rlx_ir::NodeId],
        graph: &mut rlx_ir::Graph,
    ) -> (Vec<rlx_ir::NodeId>, Vec<rlx_ir::NodeId>);
}

pub trait NonlinearDcBehavioral: Send + Sync {
    /// Stable identifier (used as a key for graph `Param` slots).
    /// Implementors that already implement [`Block`] typically do
    /// `fn name(&self) -> String { <Self as Block>::name(self) }`.
    /// We don't make `Block` a supertrait because `Block: Hash + Eq`
    /// isn't dyn-compatible, and MNA assembly wants `Box<dyn ...>`.
    fn name(&self) -> String;

    /// Number of electrical terminals.
    fn n_terminals(&self) -> usize;

    /// Build graph fragments computing the per-terminal currents from
    /// the supplied terminal-voltage `NodeId`s. Length of `voltages`
    /// must equal `n_terminals()`. Returned `Vec` has the same length;
    /// element `i` is the current flowing **from the device into the
    /// external node** at terminal `i`.
    fn currents(
        &self,
        voltages: &[rlx_ir::NodeId],
        graph: &mut rlx_ir::Graph,
    ) -> Vec<rlx_ir::NodeId>;
}

/// Time-domain stimulus shape attached to an ideal voltage (or current)
/// source. Mirrors SPICE's classical source descriptors so that ngspice
/// validation is a thin string-formatting step, while keeping a Rust-side
/// `value_at(t)` that the rlx outer-loop transient driver can sample
/// without going through SPICE.
///
/// ## Why this lives in eda-hir
///
/// A `SourceWaveform` is part of the user-defined circuit (a Block
/// parameter), not an analysis or numerical detail — it sits at the same
/// layer as resistance / capacitance values. The ngspice card emitter
/// (`eda_extern_ngspice::source_card`) consumes this enum without leaking
/// SPICE concepts back upward.
///
/// ## What the variants mean
///
/// - `Dc(v)`: constant `v` for all `t`.
/// - `Pulse`: SPICE PULSE source — `v1` at `t < td`, ramps to `v2` over
///   `tr`, holds `v2` for `pw`, ramps back to `v1` over `tf`, holds `v1`
///   for `per - tr - pw - tf`, repeats every `per`. `per <= 0` → no
///   repetition (single pulse).
/// - `Sine`: SPICE SIN source — `v_off` for `t < td`, then
///   `v_off + v_amp · exp(-(t-td)·theta) · sin(2π·freq·(t-td))`. `theta=0`
///   gives a pure sinusoid; positive `theta` adds exponential damping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SourceWaveform {
    Dc(f64),
    Pulse {
        v1: f64, v2: f64,
        td: f64, tr: f64, tf: f64, pw: f64, per: f64,
    },
    Sine {
        v_off: f64, v_amp: f64, freq: f64, td: f64, theta: f64,
    },
}

impl SourceWaveform {
    /// Sample the waveform at absolute time `t` (seconds). Edge cases:
    /// `tr=0` / `tf=0` give true step transitions; `per=0` (or any
    /// non-positive value) means a single pulse with no repetition.
    pub fn value_at(&self, t: f64) -> f64 {
        match *self {
            SourceWaveform::Dc(v) => v,

            SourceWaveform::Pulse { v1, v2, td, tr, tf, pw, per } => {
                if t < td { return v1; }
                let phase_len = if per > 0.0 { per } else { f64::INFINITY };
                let in_period = (t - td).rem_euclid(phase_len);
                // Region boundaries within one period:
                //   [0, tr)            : rising edge
                //   [tr, tr+pw)        : high
                //   [tr+pw, tr+pw+tf)  : falling edge
                //   [tr+pw+tf, per)    : low
                let rise_end = tr;
                let high_end = tr + pw;
                let fall_end = tr + pw + tf;
                if in_period < rise_end {
                    if tr == 0.0 { v2 } else { v1 + (v2 - v1) * (in_period / tr) }
                } else if in_period < high_end {
                    v2
                } else if in_period < fall_end {
                    if tf == 0.0 { v1 } else {
                        v2 + (v1 - v2) * ((in_period - high_end) / tf)
                    }
                } else {
                    v1
                }
            }

            SourceWaveform::Sine { v_off, v_amp, freq, td, theta } => {
                if t < td { return v_off; }
                let dt = t - td;
                let env = if theta == 0.0 { 1.0 } else { (-dt * theta).exp() };
                v_off + v_amp * env * (2.0 * std::f64::consts::PI * freq * dt).sin()
            }
        }
    }

    /// Constructor convenience — `pulse(v1, v2, td, tr, tf, pw, per)`.
    pub fn pulse(v1: f64, v2: f64, td: f64, tr: f64, tf: f64, pw: f64, per: f64) -> Self {
        SourceWaveform::Pulse { v1, v2, td, tr, tf, pw, per }
    }

    /// Constructor convenience — undamped sine: `sine(v_off, v_amp, freq, td)`.
    pub fn sine(v_off: f64, v_amp: f64, freq: f64, td: f64) -> Self {
        SourceWaveform::Sine { v_off, v_amp, freq, td, theta: 0.0 }
    }
}

#[cfg(test)]
mod waveform_tests {
    use super::*;

    #[test]
    fn dc_is_constant() {
        let w = SourceWaveform::Dc(3.3);
        assert_eq!(w.value_at(-1.0), 3.3);
        assert_eq!(w.value_at(0.0), 3.3);
        assert_eq!(w.value_at(1e9), 3.3);
    }

    #[test]
    fn pulse_step_with_zero_edges() {
        // 0V → 1V step at t=1, no rise/fall, no repetition.
        let w = SourceWaveform::pulse(0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0);
        assert_eq!(w.value_at(0.0), 0.0);
        assert_eq!(w.value_at(0.999), 0.0);
        assert_eq!(w.value_at(1.0), 1.0);
        assert_eq!(w.value_at(1.5), 1.0);
        // After pw, returns to v1 (no period).
        assert_eq!(w.value_at(2.5), 0.0);
    }

    #[test]
    fn pulse_linear_rise_midpoint() {
        // v1=0, v2=2, td=0, tr=1, no pulse-width, no repetition.
        let w = SourceWaveform::pulse(0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let v = w.value_at(0.5);
        assert!((v - 1.0).abs() < 1e-12, "expected 1.0, got {v}");
    }

    #[test]
    fn pulse_periodic_repeats() {
        // Square wave 0/1, td=0, tr=tf=0, pw=1, per=2.
        let w = SourceWaveform::pulse(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0);
        assert_eq!(w.value_at(0.0), 1.0);
        assert_eq!(w.value_at(0.5), 1.0);
        assert_eq!(w.value_at(1.5), 0.0);
        assert_eq!(w.value_at(2.5), 1.0);
        assert_eq!(w.value_at(3.5), 0.0);
    }

    #[test]
    fn sine_zero_at_phase_zero() {
        let w = SourceWaveform::sine(0.0, 1.0, 1.0, 0.0);
        assert!(w.value_at(0.0).abs() < 1e-12);
        let v = w.value_at(0.25);
        assert!((v - 1.0).abs() < 1e-12, "expected 1.0, got {v}");
    }

    #[test]
    fn sine_delay_holds_offset() {
        let w = SourceWaveform::sine(0.5, 1.0, 1e3, 1e-3);
        assert_eq!(w.value_at(0.0), 0.5);
        assert_eq!(w.value_at(0.5e-3), 0.5);
        assert!((w.value_at(1e-3) - 0.5).abs() < 1e-12);
    }
}
