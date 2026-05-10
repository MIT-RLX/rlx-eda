//! Sample → 5-D normalised input + per-sample residual coefficients.
//!
//! Inputs to the MLP are always in normalised form per §5 of
//! `preregistration.md`. The KCL residual is computed in graph but
//! its physical-units coefficients are precomputed here on Rust side
//! so the graph stays compact (no per-sample physical constants
//! flowing through every Mul).

use crate::config::*;

/// Physical-units sample. The network never sees these directly.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub r:    f32,    // Ω
    pub is_:  f32,    // A
    pub c:    f32,    // F
    pub v_dc: f32,    // V
    pub t:    f32,    // s
}

impl Sample {
    /// Linear time constant `R·C`; the time axis is sampled in
    /// `[T_OVER_TAU_LO, T_OVER_TAU_HI]` multiples of this.
    pub fn tau_ref(&self) -> f32 { self.r * self.c }

    /// Encode to the network's 5-D input per §5: log10-normalised
    /// (R, Is, C), linear-normalised V_dc and t/τ.
    pub fn encode(&self) -> [f32; 5] {
        [
            (self.r    / R_REF ).log10(),
            (self.is_  / IS_REF).log10(),
            (self.c    / C_REF ).log10(),
            self.v_dc  / V_REF,
            self.t / self.tau_ref(),
        ]
    }
}

/// Residual coefficients normalised by the per-sample drive current
/// `V_dc/R` (not `Is_typical`).
///
/// The original framing normalised by `Is_typical = 1e-13`, which is
/// fine in the limit `Is·R/V_dc ≈ 1` but produced `a ≈ 1e9` for the
/// resistor-current term whenever the diode is small relative to the
/// drive — and that dwarfs every other term, killing training even
/// with grad clipping. The right scale is the *actual* drive current
/// at this sample. With `r_phys = (V_dc-Vmid)/R - Is·(exp(Vmid/Vt)-1)
/// - C·dVmid/dt`, dividing by `V_dc/R`:
/// ```text
///   r_n = 1 − α·v − coef_diode·(exp(K1·v) − 1) − α·dv/dt_n
/// ```
/// with `α = V_REF/V_dc` (O(1) across the parameter cube),
/// `coef_diode = Is·R/V_dc` (5 OOM range; bounded by exp magnitude
/// at sample's operating point), `K1 = V_REF/Vt`.
#[derive(Clone, Copy, Debug)]
pub struct ResCoeffs {
    /// `V_REF / V_dc`. O(1) — multiplies both `v` and `dv/dt_n`.
    pub alpha: f32,
    /// `Is · R / V_dc`. 5 OOM range — multiplies `(exp(K1·v) − 1)`.
    pub coef_diode: f32,
}

impl Sample {
    pub fn residual_coeffs(&self) -> ResCoeffs {
        ResCoeffs {
            alpha: V_REF / self.v_dc,
            coef_diode: self.is_ * self.r / self.v_dc,
        }
    }
}

/// `K1 = V_REF / Vt` — the diode-exp argument scale. Constant across
/// all samples (Vt is fixed at 300K). Baked into the residual graph
/// as `Op::Constant`.
pub const K1_DIODE_EXP: f32 = V_REF / VT;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_lands_in_expected_range() {
        let s = Sample { r: 1e4, is_: 1e-13, c: 1e-9, v_dc: 1.0, t: 1e-5 };
        let x = s.encode();
        // log10(1e4 / 1e4) = 0; log10(1e-13/1e-13) = 0; log10(1e-9/1e-9) = 0
        assert!((x[0]).abs() < 1e-6);
        assert!((x[1]).abs() < 1e-6);
        assert!((x[2]).abs() < 1e-6);
        assert!((x[3] - 1.0).abs() < 1e-6);
        // t/τ = 1e-5 / (1e4 · 1e-9) = 1e-5 / 1e-5 = 1.0
        assert!((x[4] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn residual_coeffs_dimensionalise() {
        let s = Sample { r: 1e4, is_: 1e-13, c: 1e-9, v_dc: 1.0, t: 1e-5 };
        let rc = s.residual_coeffs();
        // alpha = V_REF/V_dc = 1.0 / 1.0 = 1.0
        assert!((rc.alpha - 1.0).abs() < 1e-6);
        // coef_diode = Is·R/V_dc = 1e-13 · 1e4 / 1.0 = 1e-9
        assert!((rc.coef_diode - 1.0e-9).abs() / 1e-9 < 1e-5);
    }
}
