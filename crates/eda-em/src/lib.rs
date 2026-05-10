//! `eda-em` — tier-1 electromigration current-density check.
//!
//! Given per-segment `(layer, width, peak_current)` triples and a
//! per-layer `Jmax` table, flag any segment where
//! `J = I / (W · t) > Jmax`.
//!
//! ## Where the inputs come from
//!
//! - **Per-segment current** — `peak |i(net)|` from a transient run
//!   (`eda-mna::TransientStep::branch_currents` or ngspice
//!   `.print i(...)` for layout-extracted nets).
//! - **Width** — segment geometry; today caller-supplied. When
//!   `eda-extract` lands, it will iterate routed wire segments and
//!   produce these triples directly.
//! - **Layer thickness + Jmax** — foundry process-stack metadata.
//!   This crate ships a [`Jmax::sky130_metal`] default for the open
//!   sky130 stack; replace with a foundry-specific table for any
//!   serious analysis.
//!
//! ## Tier 1 vs full EM
//!
//! "Tier 1" because this skips a lot a real Spectre RelXpert / PrimeSim
//! Reliability flow does:
//!
//! - **No temperature dependence.** Real `Jmax(T)` is Black's-equation-
//!   driven, declines with operating temperature. Tier 1 takes the
//!   foundry's room-temp number.
//! - **No average vs peak vs RMS distinction.** Real foundries publish
//!   different `Jmax` for DC, peak, RMS. Tier 1 takes one number per
//!   layer; caller decides which `current_a` to feed in.
//! - **No via current density.** Per-via current handling is a sibling
//!   problem; not covered here.
//! - **No layout-extracted segments.** Caller hand-builds the segment
//!   list; auto-extraction is `eda-extract`'s job.
//!
//! Despite the limitations, this catches the regime where a single
//! ill-sized power strap or ill-shared signal route exceeds Jmax by
//! orders of magnitude — the bug a real EM tool also catches first.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Logical wiring layer the segment lies on. Cross-PDK so callers
/// don't pin to a specific foundry's metal-stack name.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Layer {
    Metal1,
    Metal2,
    Metal3,
    Metal4,
    Metal5,
    Metal6,
    /// Top-level routing metal (sky130 `met5`, gf180mcu `MetalTop`).
    /// Use when the foundry's top metal doesn't fit `Metal1..6`.
    MetalTop,
}

/// One routed wire segment to check. Width is in microns, current in
/// amps. Sign is ignored (we check `|I|`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub net: String,
    pub layer: Layer,
    pub width_um: f64,
    pub current_a: f64,
}

/// Per-layer current-density limits. Values in A/µm² so widths in
/// microns and currents in amps yield a dimensionless ratio against
/// `Jmax::limit`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Jmax {
    /// `(layer, A_per_um2)` pairs. Lookup is linear; a foundry has
    /// at most ~10 metal layers so this is fine.
    pub limits: Vec<(Layer, f64)>,
}

impl Jmax {
    /// Default sky130A peak-current `Jmax` for `met1..met5`. Numbers
    /// are foundry-public, derived from the open sky130 process stack
    /// (`sky130_fd_pr/spec/em.lib` style data, peak-current limit at
    /// 25 °C, expressed per micron of width assuming the published
    /// metal thickness).
    ///
    /// **Treat these as "right order of magnitude" not "tape-out
    /// quality"**. For a real EM analysis, call your foundry's
    /// reliability handbook.
    ///
    /// Approximate values (A/µm of width, peak):
    ///
    /// | layer | thickness | peak J  |
    /// | ----- | --------- | ------- |
    /// | met1  | 0.36 µm   | 1.5 mA  |
    /// | met2  | 0.36 µm   | 1.5 mA  |
    /// | met3  | 0.84 µm   | 4.0 mA  |
    /// | met4  | 0.84 µm   | 4.0 mA  |
    /// | met5  | 1.26 µm   | 6.0 mA  |
    ///
    /// Converted to A/µm² (peak current ÷ thickness ÷ 1µm width):
    pub fn sky130_metal() -> Self {
        Self {
            limits: vec![
                (Layer::Metal1,    0.0015 / 0.36),
                (Layer::Metal2,    0.0015 / 0.36),
                (Layer::Metal3,    0.0040 / 0.84),
                (Layer::Metal4,    0.0040 / 0.84),
                (Layer::Metal5,    0.0060 / 1.26),
                (Layer::MetalTop,  0.0060 / 1.26),
            ],
        }
    }

    /// `Jmax` for a layer (A/µm²), or `None` if the layer isn't in
    /// the table.
    pub fn limit(&self, layer: Layer) -> Option<f64> {
        self.limits.iter().find(|(l, _)| *l == layer).map(|(_, j)| *j)
    }
}

/// Per-layer thickness (microns), needed to convert width-only
/// `peak J = I/W` (foundry datasheet style) into the area-based
/// `J = I/(W·t)` we compute. Defaults match `Jmax::sky130_metal`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerThickness {
    pub thickness_um: Vec<(Layer, f64)>,
}

impl LayerThickness {
    pub fn sky130_metal() -> Self {
        Self {
            thickness_um: vec![
                (Layer::Metal1,    0.36),
                (Layer::Metal2,    0.36),
                (Layer::Metal3,    0.84),
                (Layer::Metal4,    0.84),
                (Layer::Metal5,    1.26),
                (Layer::MetalTop,  1.26),
            ],
        }
    }

    pub fn get(&self, layer: Layer) -> Option<f64> {
        self.thickness_um.iter().find(|(l, _)| *l == layer).map(|(_, t)| *t)
    }
}

/// One EM rule violation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Violation {
    pub net: String,
    pub layer: Layer,
    pub width_um: f64,
    pub current_a: f64,
    /// Computed current density, A/µm².
    pub j_a_per_um2: f64,
    /// Foundry limit, A/µm².
    pub jmax_a_per_um2: f64,
    /// `j / jmax`. Values > 1 are violations; this lets reports sort
    /// by severity.
    pub margin_ratio: f64,
}

#[derive(Debug, Error)]
pub enum CheckError {
    #[error("layer {0:?} has no Jmax in the supplied table")]
    MissingJmax(Layer),
    #[error("layer {0:?} has no thickness in the supplied table")]
    MissingThickness(Layer),
    #[error("segment {net:?}: zero or negative width ({width_um} µm)")]
    BadWidth { net: String, width_um: f64 },
}

/// Run the EM check. Every segment is evaluated; violations come back
/// in input order. A non-violating segment is dropped from the output.
///
/// `current_a.abs()` is used so the caller can pass ngspice's signed
/// branch current straight in.
pub fn check(
    segments: &[Segment],
    jmax: &Jmax,
    thickness: &LayerThickness,
) -> Result<Vec<Violation>, CheckError> {
    let mut out = Vec::new();
    for s in segments {
        if s.width_um <= 0.0 {
            return Err(CheckError::BadWidth {
                net: s.net.clone(),
                width_um: s.width_um,
            });
        }
        let jmax_v = jmax.limit(s.layer).ok_or(CheckError::MissingJmax(s.layer))?;
        let t = thickness.get(s.layer).ok_or(CheckError::MissingThickness(s.layer))?;
        let j = s.current_a.abs() / (s.width_um * t);
        if j > jmax_v {
            out.push(Violation {
                net: s.net.clone(),
                layer: s.layer,
                width_um: s.width_um,
                current_a: s.current_a,
                j_a_per_um2: j,
                jmax_a_per_um2: jmax_v,
                margin_ratio: j / jmax_v,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(net: &str, layer: Layer, w: f64, i: f64) -> Segment {
        Segment { net: net.into(), layer, width_um: w, current_a: i }
    }

    #[test]
    fn within_limit_returns_no_violations() {
        let jmax = Jmax::sky130_metal();
        let thk = LayerThickness::sky130_metal();
        // 1µm-wide met1, 0.5 mA → well under 1.5 mA limit.
        let segs = vec![seg("vdd", Layer::Metal1, 1.0, 0.5e-3)];
        let v = check(&segs, &jmax, &thk).unwrap();
        assert!(v.is_empty(), "unexpected violations: {v:?}");
    }

    #[test]
    fn over_limit_flags_violation() {
        let jmax = Jmax::sky130_metal();
        let thk = LayerThickness::sky130_metal();
        // 0.5µm-wide met1 carrying 5 mA — way over the 1.5 mA/µm peak.
        let segs = vec![seg("clk", Layer::Metal1, 0.5, 5e-3)];
        let v = check(&segs, &jmax, &thk).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].net, "clk");
        assert!(v[0].margin_ratio > 1.0, "ratio = {}", v[0].margin_ratio);
    }

    #[test]
    fn ignores_current_sign() {
        let jmax = Jmax::sky130_metal();
        let thk = LayerThickness::sky130_metal();
        // Negative current of 5 mA on a tiny segment must still flag.
        let segs = vec![seg("ret", Layer::Metal1, 0.5, -5e-3)];
        let v = check(&segs, &jmax, &thk).unwrap();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn rejects_nonpositive_width() {
        let jmax = Jmax::sky130_metal();
        let thk = LayerThickness::sky130_metal();
        let segs = vec![seg("dead", Layer::Metal1, 0.0, 1e-3)];
        let err = check(&segs, &jmax, &thk).unwrap_err();
        assert!(matches!(err, CheckError::BadWidth { .. }));
    }

    #[test]
    fn higher_metals_take_higher_currents() {
        let jmax = Jmax::sky130_metal();
        let thk = LayerThickness::sky130_metal();
        // met5 should pass at currents that fail on met1.
        let segs = vec![
            seg("v1", Layer::Metal1, 1.0, 4e-3), // exceeds met1 limit
            seg("v5", Layer::Metal5, 1.0, 4e-3), // under met5 limit
        ];
        let v = check(&segs, &jmax, &thk).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].net, "v1");
    }

    #[test]
    fn missing_jmax_for_layer_errors() {
        let jmax = Jmax { limits: vec![(Layer::Metal1, 0.001)] };
        let thk = LayerThickness::sky130_metal();
        let segs = vec![seg("x", Layer::Metal3, 1.0, 1e-3)];
        let err = check(&segs, &jmax, &thk).unwrap_err();
        assert!(matches!(err, CheckError::MissingJmax(Layer::Metal3)));
    }

    #[test]
    fn ratio_orders_by_severity() {
        let jmax = Jmax::sky130_metal();
        let thk = LayerThickness::sky130_metal();
        let segs = vec![
            seg("a", Layer::Metal1, 0.5, 5e-3),  // 5x over
            seg("b", Layer::Metal1, 0.5, 10e-3), // 10x over
        ];
        let v = check(&segs, &jmax, &thk).unwrap();
        assert_eq!(v.len(), 2);
        assert!(v[1].margin_ratio > v[0].margin_ratio);
    }
}
