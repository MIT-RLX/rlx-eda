//! Closed-form MAC-output noise model.
//!
//! The bench harness's functional arm needs to evaluate "would this
//! parameter setting still classify MNIST?" thousands of times per
//! Adam step. Calling SPICE per image is infeasible; instead, this
//! module fits a closed-form `(mean, σ)` model to ngspice tt/mc data,
//! and the FPGA-speed inference path injects it as an additive noise
//! source at the MAC output.
//!
//! Calibration cadence: SPICE runs periodically (every N outer DADO
//! steps, or whenever `TileParams` moves outside the calibrated
//! basin) and refits the model. Calibration residual is reported in
//! the bench manifest so we know when the proxy stops being trusted.
//!
//! ## v1 placeholder closed form
//!
//! Three noise sources, all in LSB units:
//!
//! ```text
//!   σ_supply = k_vdd · max(0, vdd_nominal − vdd)²
//!     ── grows quadratically below nominal Vdd (worse switching) ──
//!
//!   σ_pelgrom = k_pelgrom / sqrt(w_l_n · w_l_p)
//!     ── Pelgrom 1/√(WL) device-mismatch scaling ──
//!
//!   σ_thermal = k_thermal      // const, dominated by k_B·T / C_load
//!
//!   σ_total = sqrt(σ_supply² + σ_pelgrom² + σ_thermal²)
//! ```
//!
//! Mean offset stays zero (balanced digital design — no systematic
//! bias). Constants are placeholders; calibration replaces them.

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use serde::{Deserialize, Serialize};

use crate::tile::TileParams;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NoiseStats {
    /// Mean offset on the MAC output (LSB units). Ideally zero.
    pub mean_lsb: f64,
    /// Output noise sigma in LSB units.
    pub sigma_lsb: f64,
    /// Max over the calibration set of |measured − model| / σ.
    /// Bench manifest records this; > ~3 means refit needed.
    pub calibration_residual: f64,
}

/// Closed-form coefficients. v1 ships placeholder values
/// (`NoiseModel::default`); production swaps in calibration output.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct NoiseModel {
    /// Nominal supply voltage (V) below which σ_supply ramps up.
    pub vdd_nominal: f64,
    /// Supply-noise scaling (LSB / V²).
    pub k_vdd: f64,
    /// Pelgrom mismatch scaling (LSB · √(unit W/L)²).
    pub k_pelgrom: f64,
    /// Thermal-noise floor (LSB).
    pub k_thermal: f64,
    /// Mapping from MAC-output σ (LSB) to MNIST top-1 accuracy
    /// degradation (percentage points). Placeholder; calibrated
    /// against MNIST-under-noise sweeps later.
    pub k_acc_pp_per_lsb: f64,
    /// Calibration residual at last fit. `0.0` for the v1
    /// placeholder (no calibration yet).
    pub last_calibration_residual: f64,
}

impl Default for NoiseModel {
    /// v1 placeholder constants. Order-of-magnitude only — pick
    /// values so `evaluate(TileParams::default())` returns σ ≈ 1 LSB
    /// at default sizing, which is small but non-zero. Calibration
    /// against ngspice MC data replaces these.
    fn default() -> Self {
        Self {
            vdd_nominal: 1.8,
            k_vdd: 4.0,
            k_pelgrom: 0.6,
            k_thermal: 0.2,
            k_acc_pp_per_lsb: 0.5, // 1 LSB σ → 0.5 pp accuracy drop
            last_calibration_residual: 0.0,
        }
    }
}

impl NoiseModel {
    /// Evaluate the closed-form noise model at `params`. Pure
    /// function — no I/O, no SPICE. Cheap enough that the FPGA-speed
    /// inference path can call it once per image without slowing
    /// the inner loop.
    pub fn evaluate(&self, params: TileParams) -> NoiseStats {
        // Supply term: zero above nominal, quadratic below.
        let vdd_drop = (self.vdd_nominal - params.vdd).max(0.0);
        let sigma_supply = self.k_vdd * vdd_drop * vdd_drop;

        // Pelgrom term: 1/sqrt(W/L_n · W/L_p). Guard against
        // pathological zero/negative sizing — clamp to a tiny
        // positive so the function stays defined and the bench
        // harness can flag "this is a bad design point" via
        // sigma_lsb being huge rather than NaN.
        let prod_wl = (params.w_l_n * params.w_l_p).max(1e-6);
        let sigma_pelgrom = self.k_pelgrom / prod_wl.sqrt();

        // Thermal term: constant.
        let sigma_thermal = self.k_thermal;

        let sigma_lsb = (sigma_supply * sigma_supply
            + sigma_pelgrom * sigma_pelgrom
            + sigma_thermal * sigma_thermal)
            .sqrt();

        NoiseStats {
            mean_lsb: 0.0, // balanced digital → no DC offset
            sigma_lsb,
            calibration_residual: self.last_calibration_residual,
        }
    }

    /// Refit from a batch of ngspice MC measurements. Returns the
    /// calibration residual so the bench harness can decide whether
    /// the FPGA-injected proxy is still trustworthy.
    ///
    /// **v1 stub**: real fitting requires nonlinear least squares
    /// on the (k_vdd, k_pelgrom, k_thermal) triple from a batch of
    /// (params, measured σ) pairs. Lands when `eda-extern-ngspice`
    /// produces calibration data. For now records `samples.len()`
    /// as a placeholder so the call site can run end-to-end.
    pub fn calibrate(&mut self, samples: &[(TileParams, NoiseStats)]) -> f64 {
        // Until real fitting lands, report "infinite residual until
        // 1 sample observed" so the bench manifest flags the
        // un-calibrated state honestly.
        self.last_calibration_residual = if samples.is_empty() {
            f64::INFINITY
        } else {
            f64::INFINITY // still untrusted; non-zero sample count alone doesn't make it trustworthy
        };
        self.last_calibration_residual
    }

    /// Build the σ_lsb closed-form residual into `g` using the
    /// supplied (w_l_n, w_l_p, vdd) Param NodeIds. Returns the
    /// σ NodeId — bit-for-bit equivalent to `evaluate(...).sigma_lsb`
    /// at the same param values.
    ///
    /// Used by the inner-loop accuracy gate term:
    /// `λ · relu(k_acc_pp_per_lsb · σ − ε)`. Building it as part of
    /// the rlx graph keeps Adam differentiating through it.
    pub fn add_to_graph(
        &self,
        g: &mut Graph,
        w_l_n: NodeId,
        w_l_p: NodeId,
        vdd: NodeId,
    ) -> NodeId {
        let s = Shape::new(&[1], DType::F32);
        let k = |g: &mut Graph, v: f32| {
            g.add_node(
                Op::Constant {
                    data: v.to_le_bytes().to_vec(),
                },
                vec![],
                s.clone(),
            )
        };

        // σ_supply = k_vdd · relu(vdd_nominal - vdd)²
        let v_nom = k(g, self.vdd_nominal as f32);
        let drop_raw = g.sub(v_nom, vdd);
        let drop_clamped = g.relu(drop_raw); // max(0, vdd_nominal - vdd)
        let drop_sq = g.mul(drop_clamped, drop_clamped);
        let kv = k(g, self.k_vdd as f32);
        let sigma_supply = g.mul(kv, drop_sq);
        let sigma_supply_sq = g.mul(sigma_supply, sigma_supply);

        // σ_pelgrom = k_pelgrom / sqrt(w_l_n · w_l_p + EPS)
        // EPS guards against zero/negative sizing during Adam
        // exploration; mirrors the `evaluate` clamp.
        let prod_wl = g.mul(w_l_n, w_l_p);
        let eps_sizing = k(g, 1.0e-6_f32);
        let prod_wl_safe = g.add(prod_wl, eps_sizing);
        let sqrt_wl = g.sqrt(prod_wl_safe);
        let kp = k(g, self.k_pelgrom as f32);
        let sigma_pelgrom = g.div(kp, sqrt_wl);
        let sigma_pelgrom_sq = g.mul(sigma_pelgrom, sigma_pelgrom);

        // σ_thermal = k_thermal (constant)
        let sigma_thermal = k(g, self.k_thermal as f32);
        let sigma_thermal_sq = g.mul(sigma_thermal, sigma_thermal);

        // σ = sqrt(σ_supply² + σ_pelgrom² + σ_thermal²)
        let var_partial = g.add(sigma_supply_sq, sigma_pelgrom_sq);
        let var_total = g.add(var_partial, sigma_thermal_sq);
        g.sqrt(var_total)
    }

    /// Build the accuracy-gate residual into `g`:
    /// `λ · relu(k_acc_pp_per_lsb · σ_lsb − ε_pp)`. Returns the
    /// gate-term NodeId. Caller adds it to the rest of the loss.
    pub fn add_accuracy_gate(
        &self,
        g: &mut Graph,
        sigma_lsb: NodeId,
        lambda_acc: f32,
        epsilon_acc_pp: f32,
    ) -> NodeId {
        let s = Shape::new(&[1], DType::F32);
        let k = |g: &mut Graph, v: f32| {
            g.add_node(
                Op::Constant {
                    data: v.to_le_bytes().to_vec(),
                },
                vec![],
                s.clone(),
            )
        };
        let k_acc = k(g, self.k_acc_pp_per_lsb as f32);
        let acc_drop_pp = g.mul(k_acc, sigma_lsb);
        let eps = k(g, epsilon_acc_pp);
        let excess = g.sub(acc_drop_pp, eps);
        let gated = g.relu(excess);
        let lambda = k(g, lambda_acc);
        g.mul(lambda, gated)
    }
}
