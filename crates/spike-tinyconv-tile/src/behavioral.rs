//! `DcBehavioral` for `Mac8x8Tile` — the differentiable model that
//! the inner Adam loop runs gradient descent through.
//!
//! Mirrors `spike-divider-block`'s 151-iter inverse-design pattern.
//! Topology-dispatched: each `MacTopology` variant stamps a different
//! residual into the rlx graph.
//!
//! For `MacTopology::Digital`, three closed-form residuals share the
//! same `(w_l_n, w_l_p, vdd)` Param nodes:
//!
//! ```text
//!   P_total = α · n_cells · c_per_cell · ((w_l_n + w_l_p) / 2) · vdd² · f_clk
//!           + n_cells · k_leak · (w_l_n + w_l_p) · vdd
//!
//!   delay_ns ≈ k_delay · n_critical_stages · c_per_cell
//!              / ((w_l_n + w_l_p) / 2 · vdd)
//!
//!   area_um2 = n_cells · (a0 + a_per_wl · (w_l_n + w_l_p) / 2)
//! ```
//!
//! `add_to_dc` (trait conformance) returns just `P_total`. The
//! richer `add_loss_to_dc(g, weights)` method shares one set of
//! Params across all three terms and returns the weighted sum
//! `α·P + β·delay + γ·area`. The inner Adam loop calls the latter.
//!
//! Constants are placeholders that map to ngspice-characterized
//! values once the foundry library is checked out.

use eda_hir::{Block, DcBehavioral};
use rlx_ir::infer::GraphExt;
use rlx_ir::{op::BinaryOp, DType, Graph, NodeId, Op, Shape};

use crate::tile::Mac8x8Tile;
use crate::topology::MacTopology;

// ── Closed-form digital MAC model constants ────────────────────────
//
// **v1 placeholders, normalized to optimization-friendly units.**
// All constants are picked so the loss terms at `TileParams::default`
// land in O(1)–O(1000). Real foundry-characterized numbers (in
// physical units like fW, ns, µm²) replace them once ngspice
// characterization runs against sky130_fd_sc_hd corner data; the
// `LossWeights` then absorb the unit-conversion factors.
//
// Why normalize: literal physical units span 12 orders of magnitude
// (fW for power, ns for delay, µm² for area) and f32 loss
// arithmetic loses the smaller terms (especially the accuracy gate)
// to truncation error. Optimization needs *ratios*, not absolutes.

/// Total cell count per tile (PLAN.md "Internal blocks").
const N_CELLS_DIGITAL: f32 = 202.0;
/// Per-cell switched capacitance scale (normalized).
const C_PER_CELL: f32 = 3.0;
/// Activity factor α — fraction of cells switching per clock edge.
const ACTIVITY_FACTOR: f32 = 0.15;
/// Clock-frequency scale (normalized; not literal Hz).
const F_CLK_SCALE: f32 = 1.0;
/// Per-cell leakage scale (normalized).
const K_LEAK: f32 = 0.05;

// Delay model: τ ∝ N_stages · C_load / ((W/L) · V_dd).

/// Number of gate stages on the critical path of an 8×8 MAC.
/// Multiplier carry-save chain (~8) + final adder (~32 ripple).
const N_CRITICAL_STAGES: f32 = 40.0;
/// Delay scaling (normalized).
const K_DELAY: f32 = 0.5;

// Area model: linear in average sizing.

/// Per-cell baseline area (normalized; was µm²).
const A0_PER_CELL: f32 = 0.05;
/// Per-cell area sensitivity to W/L.
const A_PER_WL: f32 = 0.015;

impl DcBehavioral for Mac8x8Tile {
    fn add_to_dc(&self, graph: &mut Graph) -> NodeId {
        match self.topology {
            MacTopology::Digital => {
                let p = build_digital_terms_with_area_baseline(self, graph, None);
                p.p_total
            }
            MacTopology::ChargeRedistribution => {
                unimplemented!("CR residual — deferred")
            }
            MacTopology::CurrentMode => {
                unimplemented!("CM residual — deferred")
            }
        }
    }
}

/// Per-tile loss weights. Mirrors `eda_bench_tinyconv::optimization::LossWeights`
/// without depending on it (kept as a value type so the bench can
/// pass weights into `add_loss_to_dc` without a circular dep).
#[derive(Debug, Clone, Copy)]
pub struct LossWeights {
    pub alpha_energy: f32,
    pub beta_delay: f32,
    pub gamma_area: f32,
    /// Accuracy-gate weight λ. Engages only when the
    /// `noise: Option<&NoiseModel>` argument to `add_loss_to_dc`
    /// is `Some`. Default chosen so a 1 pp accuracy drop above ε
    /// dominates a few percent of the energy term.
    pub lambda_acc: f32,
    /// Accuracy-drop tolerance (pp). Drops within `epsilon_acc_pp`
    /// cost zero; beyond it the gate engages with slope `lambda_acc`.
    pub epsilon_acc_pp: f32,
    /// Optional baseline area in µm². When `Some`, replaces the
    /// placeholder `N_CELLS · A0_PER_CELL` constant in the area
    /// residual — bench harness supplies the real sum from
    /// `ScHdLibrary::sum_area_um2_x1000`. Differentiable scaling
    /// (`a_per_wl · avg_wl · N_cells`) is unaffected, so Adam's
    /// gradient direction stays the same; only the constant offset
    /// shifts toward foundry-truth.
    pub area_baseline_um2: Option<f32>,
    /// Optional cycle count per inference. When `Some`, the
    /// `β·delay` term in the loss is multiplied by this — so the
    /// optimizer minimizes **total silicon time per inference**
    /// (cycles × per-cycle delay), not just per-cycle delay.
    /// Bench harness fills this from RTL sim ground truth or from
    /// the analytic `total_cycles` estimate; either way Adam
    /// gradient-descends against silicon clock time, not host wall
    /// time.
    pub cycles_per_inference: Option<u64>,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            alpha_energy: 1.0,
            beta_delay: 0.5,
            gamma_area: 0.25,
            lambda_acc: 100.0,
            epsilon_acc_pp: 0.5,
            area_baseline_um2: None,
            cycles_per_inference: None,
        }
    }
}

/// Closed-form per-cycle delay at given params, in the same
/// normalized units `behavioral.rs` uses internally. Pure function
/// — no graph, no Adam. Bench harness multiplies by cycles to get
/// silicon-time-per-inference as a derived metric.
pub fn delay_per_cycle_normalized(params: crate::TileParams) -> f64 {
    let avg_wl = (params.w_l_n + params.w_l_p) / 2.0;
    let denom = avg_wl * params.vdd;
    if denom <= 0.0 {
        return f64::INFINITY;
    }
    K_DELAY as f64 * N_CRITICAL_STAGES as f64 * C_PER_CELL as f64 / denom
}

/// Silicon time per inference (ns) at given params, given cycle
/// count. **The headline number Adam optimizes when
/// `cycles_per_inference` is supplied.** Same closed form built
/// into the rlx graph — pure here for reporting.
pub fn silicon_time_ns_per_inference(params: crate::TileParams, cycles: u64) -> f64 {
    let per_cycle = delay_per_cycle_normalized(params);
    per_cycle * cycles as f64
}

impl Mac8x8Tile {
    /// Build the full weighted loss residual into `g`:
    ///
    /// ```text
    ///   loss = α·P + β·delay + γ·area  (always)
    ///        + λ·max(0, k_acc·σ − ε)   (when `noise` is Some)
    /// ```
    ///
    /// All terms share one set of `(w_l_n, w_l_p, vdd)` Param
    /// nodes so autodiff produces one gradient per Adam-targeted
    /// param. The accuracy gate (when supplied) is differentiable
    /// through the noise model's closed form, so Adam can trade
    /// energy/delay/area against staying inside the accuracy
    /// budget.
    ///
    /// Returns the loss `NodeId`. Caller is responsible for
    /// `g.set_outputs(vec![loss])` before autodiff.
    pub fn add_loss_to_dc(
        &self,
        g: &mut Graph,
        weights: LossWeights,
        noise: Option<&crate::noise::NoiseModel>,
    ) -> NodeId {
        match self.topology {
            MacTopology::Digital => {
                let s = Shape::new(&[1], DType::F32);
                let terms = build_digital_terms_with_area_baseline(
                    self,
                    g,
                    weights.area_baseline_um2,
                );
                let alpha = constant(g, weights.alpha_energy, &s);
                let beta = constant(g, weights.beta_delay, &s);
                let gamma = constant(g, weights.gamma_area, &s);
                let weighted_p = g.mul(alpha, terms.p_total);
                // When cycles_per_inference is supplied, multiply
                // per-cycle delay by it to get total silicon time
                // per inference. Adam then optimizes against
                // cycles · period_ns, not just period_ns.
                let delay_term = match weights.cycles_per_inference {
                    Some(cycles) => {
                        let cycles_const = constant(g, cycles as f32, &s);
                        g.mul(terms.delay_ns, cycles_const)
                    }
                    None => terms.delay_ns,
                };
                let weighted_d = g.mul(beta, delay_term);
                let weighted_a = g.mul(gamma, terms.area_um2);
                let pd = g.add(weighted_p, weighted_d);
                let pda = g.add(pd, weighted_a);

                match noise {
                    Some(model) => {
                        let sigma = model.add_to_graph(
                            g,
                            terms.w_l_n,
                            terms.w_l_p,
                            terms.vdd,
                        );
                        let gate = model.add_accuracy_gate(
                            g,
                            sigma,
                            weights.lambda_acc,
                            weights.epsilon_acc_pp,
                        );
                        g.add(pda, gate)
                    }
                    None => pda,
                }
            }
            MacTopology::ChargeRedistribution => {
                unimplemented!("CR loss residual — deferred")
            }
            MacTopology::CurrentMode => {
                unimplemented!("CM loss residual — deferred")
            }
        }
    }
}

/// Closed-form Digital terms, sharing one Param triple. Returned as
/// a struct so callers (`add_to_dc`, `add_loss_to_dc`, future
/// per-term inspectors) can pick which term they need. Param
/// NodeIds are exposed so the noise-model gate can wire its
/// closed-form σ into the same param triple without re-creating
/// the Params (which would be a separate set with different names).
struct DigitalTerms {
    p_total: NodeId,
    delay_ns: NodeId,
    area_um2: NodeId,
    w_l_n: NodeId,
    w_l_p: NodeId,
    vdd: NodeId,
}

fn build_digital_terms_with_area_baseline(
    tile: &Mac8x8Tile,
    g: &mut Graph,
    area_baseline_um2: Option<f32>,
) -> DigitalTerms {
    let s = Shape::new(&[1], DType::F32);
    let name = <Mac8x8Tile as Block>::name(tile);

    // ── Adam-targeted parameters (per tile instance) ──────────────
    let w_l_n = g.param(format!("{name}__w_l_n"), s.clone());
    let w_l_p = g.param(format!("{name}__w_l_p"), s.clone());
    let vdd = g.param(format!("{name}__vdd"), s.clone());

    // ── Constants ─────────────────────────────────────────────────
    let n_cells = constant(g, N_CELLS_DIGITAL, &s);
    let c_per_cell = constant(g, C_PER_CELL, &s);
    let alpha_act = constant(g, ACTIVITY_FACTOR, &s);
    let f_clk = constant(g, F_CLK_SCALE, &s);
    let k_leak = constant(g, K_LEAK, &s);
    let n_stages = constant(g, N_CRITICAL_STAGES, &s);
    let k_delay = constant(g, K_DELAY, &s);
    let a0 = constant(g, A0_PER_CELL, &s);
    let a_per_wl = constant(g, A_PER_WL, &s);
    let two = constant(g, 2.0, &s);

    // ── Shared sub-expressions ────────────────────────────────────
    let sum_wl = g.add(w_l_n, w_l_p); // (w_l_n + w_l_p)
    let avg_wl = g.div(sum_wl, two); // (w_l_n + w_l_p) / 2
    let vdd_sq = g.mul(vdd, vdd);

    // ── Power: Pdyn + Pleak ──────────────────────────────────────
    let p_dyn = chain_mul(g, &[alpha_act, n_cells, c_per_cell, avg_wl, vdd_sq, f_clk]);
    let p_leak = chain_mul(g, &[n_cells, k_leak, sum_wl, vdd]);
    let p_total = g.add(p_dyn, p_leak);

    // ── Delay: k_delay · n_stages · c_per_cell / (avg_wl · vdd) ──
    let delay_num = chain_mul(g, &[k_delay, n_stages, c_per_cell]);
    let avg_wl_vdd = g.mul(avg_wl, vdd);
    let delay_ns = g.div(delay_num, avg_wl_vdd);

    // ── Area: baseline + n_cells · a_per_wl · avg_wl ────────────
    // Baseline is either:
    //   - real Liberty sum from `ScHdLibrary::sum_area_um2_x1000`
    //     (when `area_baseline_um2 = Some(v)`), or
    //   - placeholder `n_cells · a0` when None.
    // The differentiable scaling term stays `a_per_wl · avg_wl ·
    // n_cells` either way, so Adam's gradient direction is
    // unchanged; only the constant offset shifts.
    let baseline = match area_baseline_um2 {
        Some(v) => constant(g, v, &s),
        None => g.mul(n_cells, a0),
    };
    let scaling_per_cell = g.mul(a_per_wl, avg_wl);
    let scaling_total = g.mul(n_cells, scaling_per_cell);
    let area_um2 = g.add(baseline, scaling_total);

    DigitalTerms {
        p_total,
        delay_ns,
        area_um2,
        w_l_n,
        w_l_p,
        vdd,
    }
}

fn constant(g: &mut Graph, value: f32, s: &Shape) -> NodeId {
    g.add_node(
        Op::Constant {
            data: value.to_le_bytes().to_vec(),
        },
        vec![],
        s.clone(),
    )
}

/// Left-fold `Mul` over a slice of NodeIds. Panics on empty input.
fn chain_mul(g: &mut Graph, ids: &[NodeId]) -> NodeId {
    let mut iter = ids.iter().copied();
    let first = iter.next().expect("chain_mul: empty input");
    iter.fold(first, |acc, n| g.mul(acc, n))
}

// `BinaryOp` is in scope via the `use` above so that subsequent analog
// topology bodies (which need `g.binary(BinaryOp::Sub, ...)` for I-V
// residuals) can reuse this module without re-importing.
const _: BinaryOp = BinaryOp::Add;

// Sanity: ensure trait coherence works without pulling Block into scope.
const _: fn() = || {
    fn assert_block<T: Block>() {}
    assert_block::<Mac8x8Tile>();
};
