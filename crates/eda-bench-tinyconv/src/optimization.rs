//! Co-design loop — inner continuous (Adam over `TileParams`) +
//! outer discrete (DADO over `ArrayConfig` axes) + accuracy gate.
//!
//! PLAN.md "Co-design optimization" — the inner loop runs thousands
//! of times per outer step; outer runs ~hundreds total. SPICE only
//! gates the outer loop and final tile validation; the inner loop's
//! accuracy term is sourced from the FPGA backend with the tile's
//! noise model injected, never from SPICE.
//!
//! Two halves intentionally live in one module so the loss formula
//!   loss = α·energy + β·delay + γ·area + λ·max(0, acc_drop_pp − ε)
//! can be inspected and tweaked in one place.

use serde::{Deserialize, Serialize};

/// Loss weights — α, β, γ, λ, ε from the PLAN.md formula.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct LossWeights {
    pub alpha_energy: f64,
    pub beta_delay: f64,
    pub gamma_area: f64,
    pub lambda_acc: f64,
    /// Accuracy-drop tolerance in percentage points. Drops within
    /// `epsilon_acc_pp` cost zero; beyond it the gate engages with
    /// gradient slope `lambda_acc`.
    pub epsilon_acc_pp: f64,
    /// Optional baseline area (µm²) to feed the tile's area
    /// residual. Bench harness usually fills this from
    /// `ScHdLibrary::sum_area_um2_x1000(&inventory)` so the closed
    /// form's constant offset reflects real foundry data instead of
    /// the placeholder `N_CELLS · A0_PER_CELL`. Differentiable
    /// scaling (`a_per_wl · avg_wl · N_cells`) is unaffected.
    pub area_baseline_um2: Option<f64>,
    /// Optional cycle count per inference. When `Some`, Adam
    /// optimizes total silicon time per inference (cycles ×
    /// per-cycle delay) instead of just per-cycle delay. Bench
    /// fills this from RTL sim ground truth (`RtlSimResult.cycles`)
    /// or `total_cycles(model, budget)` analytic estimate.
    pub cycles_per_inference: Option<u64>,
}

impl Default for LossWeights {
    /// Conservative starting point. Heavy `lambda_acc` so the
    /// accuracy gate dominates until calibrated against the noise
    /// model — better to refuse Pareto points than to ship
    /// un-shippable corners (PLAN.md failure-modes section).
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

impl LossWeights {
    /// Builder-style helper: attach an `area_baseline_um2` derived
    /// from a `ScHdLibrary` + cell inventory. Returns `self`
    /// unchanged when the library can't sum the inventory (missing
    /// cell or metadata) — caller decides whether to treat that
    /// as fatal.
    ///
    /// ```ignore
    /// let weights = LossWeights::default()
    ///     .with_inhouse_baseline(&library, &tile.cell_inventory());
    /// ```
    pub fn with_inhouse_baseline(
        mut self,
        library: &eda_stdcells::ScHdLibrary,
        inventory: &[(&str, usize)],
    ) -> Self {
        if let Some(area_x1000) = library.sum_area_um2_x1000(inventory) {
            self.area_baseline_um2 = Some(area_x1000 as f64 / 1000.0);
        }
        self
    }
}

/// Inner Adam loop — continuous optimization of `TileParams` for a
/// fixed `ArrayConfig`. Mirrors `spike-divider-block`'s 151-iter
/// inverse-design pattern.
pub mod inner {
    use eda_hir::Block;
    use rlx_ir::Graph;
    use rlx_opt::autodiff::grad_with_loss;
    use rlx_runtime::{Device, Session};
    use spike_divider_block::{Adam, Optimizer};
    use spike_tinyconv_tile::{
        LossWeights as TileLossWeights, Mac8x8Tile, MacTopology, NoiseModel, TileParams,
    };

    use super::LossWeights;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    #[serde(default)]
    pub struct InnerConfig {
        pub max_steps: usize,
        pub learning_rate: f32,
        pub weights: LossWeights,
        /// Lower bounds on `(w_l_n, w_l_p, vdd)` — clamp after each
        /// Adam step so the optimizer can't wander into
        /// non-physical territory (negative sizing, near-0 supply).
        pub min_params: TileParams,
        /// Upper bounds — same purpose for the high end.
        pub max_params: TileParams,
        /// Accuracy gate. `Some(model)` engages
        /// `λ · max(0, k_acc · σ − ε)` in the loss; `None` disables
        /// the gate (energy + delay + area only). Default = `Some`,
        /// which forces Adam to trade off accuracy against the
        /// other terms — the canonical co-design objective.
        pub noise_model: Option<NoiseModel>,
    }

    impl Default for InnerConfig {
        fn default() -> Self {
            Self {
                max_steps: 200,
                learning_rate: 0.05,
                weights: LossWeights::default(),
                min_params: TileParams {
                    w_l_n: 0.1,
                    w_l_p: 0.1,
                    vdd: 0.6,
                    bias_v: 0.0,
                    weight_bits: 8,
                },
                max_params: TileParams {
                    w_l_n: 5.0,
                    w_l_p: 5.0,
                    vdd: 1.95,
                    bias_v: 0.0,
                    weight_bits: 8,
                },
                noise_model: Some(NoiseModel::default()),
            }
        }
    }

    /// Per-step record from the inner loop. Same shape as
    /// `spike-divider-block::ml_trace`'s `StepRow`, adapted to the
    /// MAC tile's three-param surface.
    #[derive(Debug, Clone, Copy)]
    pub struct InnerStep {
        pub step: usize,
        pub w_l_n: f32,
        pub w_l_p: f32,
        pub vdd: f32,
        /// Total power (Pdyn + Pleak). Same units as the constants
        /// in `spike_tinyconv_tile::behavioral` (placeholder fW for v1
        /// — calibration replaces with real µW once ngspice runs).
        pub p_total: f32,
        pub grad_w_l_n: f32,
        pub grad_w_l_p: f32,
        pub grad_vdd: f32,
    }

    /// Run Adam on `tile`'s power residual to convergence (or until
    /// `max_steps`). Returns the full per-step trace.
    ///
    /// **v1 scope**: minimizes total power (Pdyn + Pleak) directly.
    /// The full PLAN.md loss `α·energy + β·delay + γ·area + λ·gate`
    /// reduces to just `α·energy` here because:
    ///   - `delay` and `area` aren't yet in the residual (would come
    ///     from a richer model card).
    ///   - The accuracy gate term needs the noise-model + FPGA
    ///     inference path, not yet wired (PLAN.md cross-cutting #6).
    ///
    /// So this loop will trivially drive (w_l_n, w_l_p, vdd) to the
    /// lower bounds — that's correct under the partial loss; the
    /// proof of life is "Adam runs end-to-end against a tile's
    /// add_to_dc and follows the gradient." Real co-design lands when
    /// the loss is filled out.
    pub fn run(
        tile: &Mac8x8Tile,
        cfg: &InnerConfig,
    ) -> Result<Vec<InnerStep>, super::OptError> {
        // Only Digital topology has an add_to_dc body in v1.
        if tile.topology != MacTopology::Digital {
            return Err(super::OptError::InnerDiverged { steps: 0 });
        }

        // ── Build forward graph + autodiff ───────────────────────
        // Loss = α·energy + β·delay + γ·area + (optionally)
        //        λ·max(0, k_acc·σ − ε), all sharing one set of
        // (w_l_n, w_l_p, vdd) Params via `add_loss_to_dc`. The
        // accuracy gate engages when `cfg.noise_model` is Some —
        // Adam differentiates through the σ closed form too, so
        // it learns to back off when noise pushes accuracy past
        // the tolerance.
        let mut fwd = Graph::new(format!("{}_inner_loss", <Mac8x8Tile as Block>::name(tile)));
        let tile_weights = TileLossWeights {
            alpha_energy: cfg.weights.alpha_energy as f32,
            beta_delay: cfg.weights.beta_delay as f32,
            gamma_area: cfg.weights.gamma_area as f32,
            lambda_acc: cfg.weights.lambda_acc as f32,
            epsilon_acc_pp: cfg.weights.epsilon_acc_pp as f32,
            area_baseline_um2: cfg.weights.area_baseline_um2.map(|v| v as f32),
            cycles_per_inference: cfg.weights.cycles_per_inference,
        };
        let loss_id = tile.add_loss_to_dc(&mut fwd, tile_weights, cfg.noise_model.as_ref());
        fwd.set_outputs(vec![loss_id]);

        // Find the three param NodeIds by name — `add_to_dc` keys
        // them by `<tile_name>__{w_l_n,w_l_p,vdd}`.
        let prefix = <Mac8x8Tile as Block>::name(tile);
        let name_n = format!("{prefix}__w_l_n");
        let name_p = format!("{prefix}__w_l_p");
        let name_v = format!("{prefix}__vdd");
        let id_n = find_param(&fwd, &name_n);
        let id_p = find_param(&fwd, &name_p);
        let id_v = find_param(&fwd, &name_v);

        let bwd = grad_with_loss(&fwd, &[id_n, id_p, id_v]);
        let mut compiled = Session::new(Device::Cpu).compile(bwd);

        // ── Loop ─────────────────────────────────────────────────
        let mut params = [
            tile.params.w_l_n as f32,
            tile.params.w_l_p as f32,
            tile.params.vdd as f32,
        ];
        let mins = [
            cfg.min_params.w_l_n as f32,
            cfg.min_params.w_l_p as f32,
            cfg.min_params.vdd as f32,
        ];
        let maxes = [
            cfg.max_params.w_l_n as f32,
            cfg.max_params.w_l_p as f32,
            cfg.max_params.vdd as f32,
        ];

        let mut adam = Adam::new(cfg.learning_rate, 3);
        let mut trace = Vec::with_capacity(cfg.max_steps + 1);

        for step in 0..=cfg.max_steps {
            compiled.set_param(&name_n, &[params[0]]);
            compiled.set_param(&name_p, &[params[1]]);
            compiled.set_param(&name_v, &[params[2]]);

            let outs = compiled.run(&[
                ("d_output", &[1.0_f32][..]),
            ]);

            let p_total = outs[0][0];
            let g_n = outs[1][0];
            let g_p = outs[2][0];
            let g_v = outs[3][0];

            trace.push(InnerStep {
                step,
                w_l_n: params[0],
                w_l_p: params[1],
                vdd: params[2],
                p_total,
                grad_w_l_n: g_n,
                grad_w_l_p: g_p,
                grad_vdd: g_v,
            });

            // NaN / Inf — surface as divergence rather than silently
            // corrupting the trace.
            if !p_total.is_finite()
                || ![g_n, g_p, g_v].iter().all(|x| x.is_finite())
            {
                return Err(super::OptError::InnerDiverged { steps: step });
            }

            adam.step(&mut params, &[g_n, g_p, g_v]);
            for i in 0..3 {
                params[i] = params[i].clamp(mins[i], maxes[i]);
            }
        }

        Ok(trace)
    }

    fn find_param(g: &Graph, full_name: &str) -> rlx_ir::NodeId {
        g.nodes()
            .iter()
            .enumerate()
            .find_map(|(i, n)| match &n.op {
                rlx_ir::Op::Param { name } if name == full_name => {
                    Some(rlx_ir::NodeId(i as u32))
                }
                _ => None,
            })
            .unwrap_or_else(|| {
                panic!("inner::run: param {full_name:?} not in forward graph")
            })
    }
}

/// Outer loop — discrete optimization over `ArrayConfig` axes
/// (`weight_bits ∈ {2,4,8}`, parallelism, pipeline_depth, topology).
///
/// **v1 = brute-force grid search** over a caller-supplied
/// `&[ArrayConfig]`. The full DADO algorithm
/// (`spike-dado-r2r`-style tabular categorical on chain JT) drops
/// in later as a pluggable strategy — for the proof-of-life, walking
/// a few dozen candidate configs is enough to exercise the
/// outer→inner→add_loss_to_dc→Adam→clamp pipeline end-to-end.
pub mod outer {
    use spike_tinyconv_array::array::ArrayConfig;
    use spike_tinyconv_tile::{Mac8x8Tile, TileParams};

    use super::inner::{run as run_inner, InnerConfig, InnerStep};
    use super::OptError;

    #[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
    #[serde(default)]
    pub struct OuterConfig {
        pub max_rounds: usize,
        pub orfs_cadence: usize,
    }

    impl Default for OuterConfig {
        fn default() -> Self {
            Self {
                max_rounds: 200,
                orfs_cadence: 10,
            }
        }
    }

    /// Per-candidate result. `final_loss` is `None` when the inner
    /// loop diverged on this candidate (NaN/Inf, non-Digital
    /// topology); the bench reporter renders these as skipped.
    #[derive(Debug, Clone)]
    pub struct CandidateResult {
        pub config: ArrayConfig,
        pub final_loss: Option<f32>,
        pub n_steps: usize,
    }

    /// Final outer-loop verdict. `best_*` reflects the lowest
    /// `final_loss` across non-diverged candidates; `all_results`
    /// preserves input order for reporting.
    #[derive(Debug, Clone)]
    pub struct OuterResult {
        pub best_index: usize,
        pub best_config: ArrayConfig,
        pub best_final_loss: f32,
        pub best_trace: Vec<InnerStep>,
        pub all_results: Vec<CandidateResult>,
    }

    /// Walk every `ArrayConfig` in `candidates`, run the inner Adam
    /// loop on each, return the one with the lowest final loss.
    ///
    /// Diverged candidates (inner returns `Err` or NaN-trips) get
    /// `final_loss = None` in `all_results` and are skipped from
    /// "best" selection. If every candidate diverges, returns
    /// `OptError::OuterBudget`.
    pub fn run(
        candidates: &[ArrayConfig],
        inner_cfg: &InnerConfig,
    ) -> Result<OuterResult, OptError> {
        if candidates.is_empty() {
            return Err(OptError::OuterBudget);
        }

        let mut all = Vec::with_capacity(candidates.len());
        let mut best: Option<(usize, f32, Vec<InnerStep>)> = None;

        for (idx, cfg) in candidates.iter().enumerate() {
            let tile = candidate_to_tile(idx, cfg);
            match run_inner(&tile, inner_cfg) {
                Ok(trace) => {
                    let final_loss = trace.last().map(|s| s.p_total);
                    all.push(CandidateResult {
                        config: cfg.clone(),
                        final_loss,
                        n_steps: trace.len(),
                    });
                    if let Some(f) = final_loss {
                        if f.is_finite()
                            && best.as_ref().map_or(true, |(_, b, _)| f < *b)
                        {
                            best = Some((idx, f, trace));
                        }
                    }
                }
                Err(_) => {
                    all.push(CandidateResult {
                        config: cfg.clone(),
                        final_loss: None,
                        n_steps: 0,
                    });
                }
            }
        }

        let (best_index, best_final_loss, best_trace) = best.ok_or(OptError::OuterBudget)?;
        Ok(OuterResult {
            best_index,
            best_config: candidates[best_index].clone(),
            best_final_loss,
            best_trace,
            all_results: all,
        })
    }

    /// Project an `ArrayConfig` onto a single representative
    /// `Mac8x8Tile` for the inner loop. v1 simplification: optimize
    /// the one-tile residual; per-tile heterogeneity inside a grid
    /// is a v1.5 concern.
    fn candidate_to_tile(idx: usize, cfg: &ArrayConfig) -> Mac8x8Tile {
        Mac8x8Tile::with_topology(
            format!("u_outer_{idx}"),
            TileParams {
                weight_bits: cfg.tile_params.weight_bits,
                ..cfg.tile_params
            },
            cfg.topology,
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OptError {
    #[error("accuracy gate failed: drop={drop_pp:.2}pp exceeds ε={epsilon_pp:.2}pp")]
    AccuracyGate { drop_pp: f64, epsilon_pp: f64 },
    #[error("inner loop diverged after {steps} steps (NaN / Inf in residual)")]
    InnerDiverged { steps: usize },
    #[error("outer loop budget exhausted")]
    OuterBudget,
}
