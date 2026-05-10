//! `eda-pex` — tier-1 parallel-plate capacitive parasitic extraction.
//!
//! Walks the merged-polygon set produced by `klayout-connect` and
//! computes:
//!
//! 1. **Cross-layer coupling** between every pair of nets on
//!    different metal layers — `C = ε₀·ε_r·A / d` per overlap.
//! 2. **Substrate coupling** for the bottom-most metal — same
//!    formula against an implicit substrate plane at depth `d_sub`.
//!
//! The result is a list of [`Parasitic`] entries ready to fold into
//! a SPICE deck as `C<index> netA netB <value>` lines via
//! [`emit_spice_caps`].
//!
//! ## Scope vs sign-off PEX
//!
//! This is the parallel-plate term only — what dominates between
//! two large-overlap planes (e.g. M3 routing over a M2 stripe) on a
//! ~µm-scale layout. **Not modelled:**
//!
//! - **Fringe/sidewall capacitance.** A 0.14 µm met1 line over a
//!   substrate has ~50 % of its total cap in the sidewall + fringe
//!   terms; tier-1 misses that. Sign-off PEX uses 2.5D / 3D field
//!   solvers (Calibre xRC, StarRC, magic-with-extract).
//! - **Lateral (same-layer) coupling.** Adjacent met1 lines couple
//!   sideways through the inter-line dielectric. Real flows model
//!   this via a per-spacing capacitance table.
//! - **Series resistance.** PEX is *parasitic R + C*; tier-1 here
//!   is C-only. Resistance fall-out from sheet ρ × line aspect is
//!   straightforward to add — see `Future` in PLAN.md.
//! - **Distributed (segmented) RC.** A real long wire gets carved
//!   into `n` segments, each with a piece of R + C. Tier-1 lumps
//!   to one cap per net pair.
//!
//! ## When tier-1 is enough
//!
//! - First-order coupling estimates ("does M3 over M2 swing affect
//!   the M2 net by more than 10 %?").
//! - Sch-vs-Lay regression: closes the loop with *some* parasitic
//!   model when the alternative is "no parasitics at all" (the
//!   `Rdrain` stub in `spike-lelo-ex` today).
//! - Differentiable inverse design loops where the parasitic value
//!   is one term in the loss; getting it 30 % wrong is fine for
//!   a gradient direction.
//!
//! ## When tier-1 is not enough
//!
//! - Anything claiming silicon correlation. Use real PEX (the
//!   `magic ext2spice` adapter is the next step in PLAN.md).
//! - High-frequency (RF) flows where fringe + lateral dominate.
//! - DRC-clean-but-PEX-tight regimes (sub-100 nm spacing).
//!
//! ## Foundry constants
//!
//! Sky130A defaults via [`Stack::sky130a_default`]. Numbers are the
//! BEOL inter-metal dielectric `ε_r ≈ 4.2` (TEOS-class oxide) and
//! the open `sky130_fd_pr` documented inter-metal spacings:
//!
//! | layer pair | spacing (µm) |
//! | --- | --- |
//! | met1 ↔ met2 | 0.95 |
//! | met2 ↔ met3 | 1.30 |
//! | met3 ↔ met4 | 0.85 |
//! | met4 ↔ met5 | 0.85 |
//! | met1 ↔ substrate | 0.40 (FOX + STI) |
//!
//! These are nominal — process variation shifts each by ±10 %.

use klayout_connect::Net as ConnNet;
use klayout_core::{Bbox, LayerIndex, Point, Polygon};
use klayout_geom::{intersection, Region};
use serde::{Deserialize, Serialize};

/// Logical metal layer (cross-PDK). `Substrate` is the implicit
/// ground plane below the bottom metal.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PexLayer {
    Substrate,
    Metal1,
    Metal2,
    Metal3,
    Metal4,
    Metal5,
}

/// Vacuum permittivity in F/µm (= ε₀ × 1e-6 m/µm). Multiplied by
/// `ε_r` and overlap area in µm² gives capacitance in farads.
const EPS0_F_PER_UM: f64 = 8.854e-18;

/// Process-stack constants for tier-1 PEX. Specific capacitance is
/// `ε₀·ε_r / d` where `d` is the layer-to-layer spacing. Sheet
/// resistance lives on the same struct so a single `Stack` value
/// drives both the C-side and R-side extractors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stack {
    /// Inter-layer dielectric relative permittivity (TEOS oxide
    /// ≈ 4.2 across the BEOL).
    pub eps_r: f64,
    /// Per-layer-pair vertical spacing in µm.
    /// Pairs are unordered: `(Metal1, Metal2)` matches `(Metal2, Metal1)`.
    pub spacings_um: Vec<((PexLayer, PexLayer), f64)>,
    /// Per-layer sheet resistance in Ω/sq (ohms per square). Used
    /// by [`extract_resistance`] to compute lumped per-net series
    /// resistance.
    pub sheet_rho_ohm_per_sq: Vec<(PexLayer, f64)>,
}

impl Stack {
    /// Sky130A defaults — see crate doc comment. Sheet-resistance
    /// numbers from `sky130_fd_pr` device-handbook nominal values:
    ///
    /// | layer | Ω/sq |
    /// | --- | --- |
    /// | met1, met2 | 0.125 |
    /// | met3, met4 | 0.047 |
    /// | met5       | 0.030 |
    pub fn sky130a_default() -> Self {
        Self {
            eps_r: 4.2,
            spacings_um: vec![
                ((PexLayer::Substrate, PexLayer::Metal1), 0.40),
                ((PexLayer::Metal1,    PexLayer::Metal2), 0.95),
                ((PexLayer::Metal2,    PexLayer::Metal3), 1.30),
                ((PexLayer::Metal3,    PexLayer::Metal4), 0.85),
                ((PexLayer::Metal4,    PexLayer::Metal5), 0.85),
            ],
            sheet_rho_ohm_per_sq: vec![
                (PexLayer::Metal1, 0.125),
                (PexLayer::Metal2, 0.125),
                (PexLayer::Metal3, 0.047),
                (PexLayer::Metal4, 0.047),
                (PexLayer::Metal5, 0.030),
            ],
        }
    }

    /// Look up the spacing between two layers in µm. Returns `None`
    /// for non-adjacent or unknown pairs (tier-1 only handles
    /// adjacent layers; non-adjacent coupling is assumed shielded
    /// by the intermediate metals).
    pub fn spacing_um(&self, a: PexLayer, b: PexLayer) -> Option<f64> {
        self.spacings_um.iter().find_map(|((x, y), d)| {
            if (*x == a && *y == b) || (*x == b && *y == a) { Some(*d) } else { None }
        })
    }

    /// Specific capacitance (F/µm²) for two adjacent layers.
    pub fn specific_cap_per_um2(&self, a: PexLayer, b: PexLayer) -> Option<f64> {
        self.spacing_um(a, b).map(|d| EPS0_F_PER_UM * self.eps_r / d)
    }

    /// Sheet resistance (Ω/sq) for a layer.
    pub fn sheet_rho(&self, layer: PexLayer) -> Option<f64> {
        self.sheet_rho_ohm_per_sq.iter().find_map(|(l, r)| if *l == layer { Some(*r) } else { None })
    }
}

/// One extracted parasitic capacitor. Two nets, one value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Parasitic {
    pub net_a: String,
    pub net_b: String,
    pub layer_a: PexLayer,
    pub layer_b: PexLayer,
    /// Overlap area in µm².
    pub overlap_um2: f64,
    /// Lumped capacitance, farads.
    pub cap_f: f64,
}

/// One extracted parasitic resistor. Inserted in series on a net,
/// splitting it into `<net>` and `<net>_pex` at SPICE-emission time.
///
/// Tier-1: lumped per net at the layer's sheet resistance × the
/// net's effective length-over-width ratio. The "effective" L/W is
/// approximated as `total_area_um2 / (min_dim_um)²` — captures the
/// long-skinny regime where IR drop matters, under-counts complex
/// shapes. Real flows segment the wire and place R per segment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParasiticResistor {
    pub net: String,
    pub layer: PexLayer,
    pub squares: f64,
    pub r_ohm: f64,
}

/// Thin wrapper that ties a `klayout_connect::Net` to its
/// extraction-side metadata: which metal layer the polygons sit on,
/// and the user-facing net name (post-relabelling). The crate
/// doesn't try to derive layer-from-polygons because a single net
/// can span multiple layers via vias — caller groups polygons per
/// (net, layer) before calling [`extract`].
#[derive(Clone, Debug)]
pub struct NetOnLayer<'a> {
    pub name: &'a str,
    pub layer: PexLayer,
    pub polygons: &'a [Polygon],
    /// Bbox of the polygons. Used for early-exit when two nets
    /// don't overlap — tier-1 doesn't pay the polygon-intersection
    /// cost when bboxes are disjoint.
    pub bbox: Bbox,
}

/// Extract parallel-plate parasitics across `nets`.
///
/// Returns one [`Parasitic`] per (netA, netB) overlap on adjacent
/// layers, plus one per net on the bottom metal coupling to the
/// implicit substrate plane named `substrate_net_name` (typically
/// `"0"` or `"vss"`).
///
/// `dbu` is the library's DBU per micron (e.g. 1000 ⇒ 1 µm = 1000
/// integer units). Used to convert klayout-internal areas (i64) to
/// physical µm² via `area_dbu / (dbu * dbu)`.
pub fn extract(
    nets: &[NetOnLayer<'_>],
    dbu: i64,
    stack: &Stack,
    substrate_net_name: &str,
) -> Vec<Parasitic> {
    let mut out = Vec::new();
    let dbu_f = dbu as f64;

    // ── Cross-layer coupling ──────────────────────────────────────
    for (i, a) in nets.iter().enumerate() {
        for b in nets.iter().skip(i + 1) {
            // Same net? Skip self-coupling.
            if a.name == b.name {
                continue;
            }
            // Same layer? Tier-1 doesn't model lateral coupling.
            if a.layer == b.layer {
                continue;
            }
            // Non-adjacent layers? Assumed shielded.
            let Some(c_per_um2) = stack.specific_cap_per_um2(a.layer, b.layer) else { continue };
            // Bbox-disjoint? Skip the expensive intersection.
            if !a.bbox.intersects(&b.bbox) {
                continue;
            }
            let area_um2 = overlap_area_um2(a.polygons, b.polygons, dbu_f);
            if area_um2 <= 0.0 {
                continue;
            }
            out.push(Parasitic {
                net_a: a.name.to_string(),
                net_b: b.name.to_string(),
                layer_a: a.layer,
                layer_b: b.layer,
                overlap_um2: area_um2,
                cap_f: area_um2 * c_per_um2,
            });
        }
    }

    // ── Substrate coupling for the bottom metal ───────────────────
    // The bottom metal is whichever layer pairs with `Substrate` in
    // the stack — usually Metal1.
    let bottom_metal: Option<PexLayer> = stack.spacings_um.iter().find_map(|((a, b), _)| {
        match (*a, *b) {
            (PexLayer::Substrate, m) | (m, PexLayer::Substrate) => Some(m),
            _ => None,
        }
    });
    if let Some(m) = bottom_metal {
        let Some(c_per_um2) = stack.specific_cap_per_um2(PexLayer::Substrate, m) else { return out };
        for n in nets.iter().filter(|n| n.layer == m) {
            // Total area on the bottom metal (a net can be multiple
            // disjoint polygons). Substrate is one big sheet so we
            // don't need to intersect — use the polygon area sum.
            let area_dbu = n.polygons.iter().map(polygon_area_dbu_abs).sum::<i128>();
            let area_um2 = (area_dbu as f64) / (dbu_f * dbu_f);
            if area_um2 <= 0.0 { continue; }
            out.push(Parasitic {
                net_a: n.name.to_string(),
                net_b: substrate_net_name.to_string(),
                layer_a: m,
                layer_b: PexLayer::Substrate,
                overlap_um2: area_um2,
                cap_f: area_um2 * c_per_um2,
            });
        }
    }

    out
}

/// Compute the µm² area where two polygon sets overlap. Uses
/// `klayout-geom::intersection` so we get the real merged polygon
/// area, not just bbox.
fn overlap_area_um2(a: &[Polygon], b: &[Polygon], dbu_f: f64) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let ra = Region::from_polygons(a.iter().cloned());
    let rb = Region::from_polygons(b.iter().cloned());
    let inter = intersection(&ra, &rb);
    let area_dbu: i128 = inter.polygons().iter().map(polygon_area_dbu_abs).sum();
    (area_dbu as f64) / (dbu_f * dbu_f)
}

/// Twice the polygon's signed area, absolute value, in DBU². Real
/// area = `result / 2 / dbu²`. Using i128 to avoid overflow at
/// macro-cell scale.
fn polygon_area_dbu_abs(p: &Polygon) -> i128 {
    let mut s: i128 = 0;
    let n = p.hull.len();
    for i in 0..n {
        let a = p.hull[i];
        let b = p.hull[(i + 1) % n];
        s += (a.x as i128) * (b.y as i128) - (b.x as i128) * (a.y as i128);
    }
    s.abs() / 2
}

/// Convenience adapter: turn a `klayout_connect::Net` into a
/// [`NetOnLayer`] given the layer it belongs on. Caller still has
/// to know which layer goes with which net (the merge result
/// flattens layer info into the union-find groups).
pub fn net_on_layer<'a>(n: &'a ConnNet, layer: PexLayer) -> NetOnLayer<'a> {
    NetOnLayer {
        name: n.name.as_str(),
        layer,
        polygons: &n.polygons,
        bbox: n.bbox,
    }
}

/// Render a `Vec<Parasitic>` as SPICE element lines (`C<n> netA
/// netB <value>`). Designators are sequential `Cpex0`, `Cpex1`, …
/// to avoid colliding with caller-side capacitor names.
///
/// Net names get the standard ground rewrite: `gnd`/`GND`/`0` →
/// `0` so the caller doesn't have to pre-process.
pub fn emit_spice_caps(ps: &[Parasitic]) -> Vec<String> {
    ps.iter()
        .enumerate()
        .map(|(i, p)| {
            format!(
                "Cpex{i} {a} {b} {v:.6e}",
                a = spice_net(&p.net_a),
                b = spice_net(&p.net_b),
                v = p.cap_f,
            )
        })
        .collect()
}

/// Extract per-net series resistance from polygon geometry. For
/// each net, sums the squares (`length / width`) over its
/// polygons and multiplies by the layer's sheet resistance.
///
/// Tier-1 approximation: each polygon contributes `area / w²`
/// squares, where `w` is the polygon's smaller bbox dimension. A
/// long-thin rectangle gives `length/width` directly; a square
/// piece contributes ~1 square; complex shapes are under-counted.
/// Real flows segment the wire and place R per segment with
/// per-segment width; tier-1 lumps to one value per net.
pub fn extract_resistance(
    nets: &[NetOnLayer<'_>],
    dbu: i64,
    stack: &Stack,
) -> Vec<ParasiticResistor> {
    let mut out = Vec::new();
    let dbu_f = dbu as f64;
    for n in nets {
        let Some(rho) = stack.sheet_rho(n.layer) else { continue };
        let mut squares = 0.0_f64;
        for p in n.polygons {
            let bb = p.bbox();
            let w_dbu = (bb.max.x - bb.min.x).min(bb.max.y - bb.min.y);
            if w_dbu <= 0 {
                continue;
            }
            let w_um = w_dbu as f64 / dbu_f;
            let area_um2 = (polygon_area_dbu_abs(p) as f64) / (dbu_f * dbu_f);
            // squares = (length × width) / width² = length / width =
            //         = area / width².
            squares += area_um2 / (w_um * w_um);
        }
        if squares <= 0.0 {
            continue;
        }
        out.push(ParasiticResistor {
            net: n.name.to_string(),
            layer: n.layer,
            squares,
            r_ohm: squares * rho,
        });
    }
    out
}

/// Render `ParasiticResistor`s as SPICE in-series elements.
///
/// Each net `<name>` gets split: the original `<name>` becomes the
/// "near" side, and a new node `<name>_pex` becomes the "far" side
/// where the device connects. The caller is responsible for
/// rewriting their device-instance lines so the affected terminal
/// references `<name>_pex` instead of `<name>` — there's no
/// general way to do that from this side without a netlist
/// rewriter. Returned tuples are `(net_name, far_node_name,
/// element_line)` so the caller can do the rewrite.
pub fn emit_spice_resistors(rs: &[ParasiticResistor]) -> Vec<(String, String, String)> {
    rs.iter()
        .enumerate()
        .map(|(i, r)| {
            let near = spice_net(&r.net);
            let far = format!("{}_pex", r.net);
            let line = format!("Rpex{i} {near} {far} {v:.6e}", v = r.r_ohm);
            (r.net.clone(), far, line)
        })
        .collect()
}

fn spice_net(n: &str) -> String {
    match n {
        "gnd" | "GND" | "0" => "0".to_string(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use klayout_core::{Bbox, Point};

    fn rect_poly(x0: i64, y0: i64, x1: i64, y1: i64) -> Polygon {
        Polygon::rect(Bbox::new(Point::new(x0, y0), Point::new(x1, y1)))
    }

    fn dummy_bbox(p: &Polygon) -> Bbox { p.bbox() }

    #[test]
    fn sky130_specific_cap_in_expected_range() {
        let s = Stack::sky130a_default();
        // met1↔met2 at 0.95 µm: c ≈ 8.854e-18 × 4.2 / 0.95 ≈ 3.91e-17 F/µm².
        let c = s.specific_cap_per_um2(PexLayer::Metal1, PexLayer::Metal2).unwrap();
        assert!((c - 3.91e-17).abs() < 1e-18, "got {c}");
    }

    #[test]
    fn no_overlap_means_no_parasitic() {
        // Two nets, met1 and met2, but their bboxes don't intersect.
        let p_a = rect_poly(0, 0, 1_000, 1_000);
        let p_b = rect_poly(5_000, 5_000, 6_000, 6_000);
        let nets = vec![
            NetOnLayer { name: "a", layer: PexLayer::Metal1, polygons: std::slice::from_ref(&p_a), bbox: dummy_bbox(&p_a) },
            NetOnLayer { name: "b", layer: PexLayer::Metal2, polygons: std::slice::from_ref(&p_b), bbox: dummy_bbox(&p_b) },
        ];
        let p = extract(&nets, 1000, &Stack::sky130a_default(), "0");
        // No cross-coupling. Substrate coupling for `a` (met1) only.
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].net_a, "a");
        assert_eq!(p[0].net_b, "0");
        assert_eq!(p[0].layer_b, PexLayer::Substrate);
    }

    #[test]
    fn stacked_metals_couple_through_overlap() {
        // 1 µm × 1 µm met1 stripe directly under a 1 µm × 1 µm met2
        // stripe — full overlap, 1 µm² = 1e-12 m² = 1 µm² area.
        let p_a = rect_poly(0, 0, 1_000, 1_000);
        let p_b = rect_poly(0, 0, 1_000, 1_000);
        let nets = vec![
            NetOnLayer { name: "a", layer: PexLayer::Metal1, polygons: std::slice::from_ref(&p_a), bbox: dummy_bbox(&p_a) },
            NetOnLayer { name: "b", layer: PexLayer::Metal2, polygons: std::slice::from_ref(&p_b), bbox: dummy_bbox(&p_b) },
        ];
        let p = extract(&nets, 1000, &Stack::sky130a_default(), "0");
        // Expect: met1↔met2 cap + a-substrate cap. Met2 doesn't
        // get a substrate cap because Metal2 isn't adjacent to
        // Substrate.
        let cross: &Parasitic = p.iter().find(|x| x.layer_a == PexLayer::Metal1 && x.layer_b == PexLayer::Metal2).unwrap();
        assert!((cross.overlap_um2 - 1.0).abs() < 1e-9);
        // c = 1 µm² × 3.91e-17 F/µm² ≈ 3.91e-17 F = 39.1 aF.
        assert!((cross.cap_f - 3.91e-17).abs() < 1e-18, "got {}", cross.cap_f);
    }

    #[test]
    fn partial_overlap_uses_intersection_area() {
        // 2 µm × 2 µm met1 over a 2 µm × 2 µm met2 stripe offset
        // by 1 µm in x → 1 µm × 2 µm = 2 µm² overlap.
        let p_a = rect_poly(0, 0, 2_000, 2_000);
        let p_b = rect_poly(1_000, 0, 3_000, 2_000);
        let nets = vec![
            NetOnLayer { name: "a", layer: PexLayer::Metal1, polygons: std::slice::from_ref(&p_a), bbox: dummy_bbox(&p_a) },
            NetOnLayer { name: "b", layer: PexLayer::Metal2, polygons: std::slice::from_ref(&p_b), bbox: dummy_bbox(&p_b) },
        ];
        let p = extract(&nets, 1000, &Stack::sky130a_default(), "0");
        let cross = p.iter().find(|x| x.layer_a != PexLayer::Substrate && x.layer_b != PexLayer::Substrate).unwrap();
        assert!((cross.overlap_um2 - 2.0).abs() < 1e-9, "got {}", cross.overlap_um2);
    }

    #[test]
    fn substrate_cap_proportional_to_metal1_area() {
        // 5 µm × 2 µm met1 stripe.
        let p_a = rect_poly(0, 0, 5_000, 2_000);
        let nets = vec![
            NetOnLayer { name: "rail", layer: PexLayer::Metal1, polygons: std::slice::from_ref(&p_a), bbox: dummy_bbox(&p_a) },
        ];
        let p = extract(&nets, 1000, &Stack::sky130a_default(), "vss");
        assert_eq!(p.len(), 1);
        let sub = &p[0];
        assert_eq!(sub.net_b, "vss");
        assert_eq!(sub.layer_b, PexLayer::Substrate);
        assert!((sub.overlap_um2 - 10.0).abs() < 1e-9);
    }

    #[test]
    fn non_adjacent_layers_assumed_shielded() {
        // met1 and met3 are non-adjacent in the sky130 stack.
        let p_a = rect_poly(0, 0, 1_000, 1_000);
        let p_b = rect_poly(0, 0, 1_000, 1_000);
        let nets = vec![
            NetOnLayer { name: "a", layer: PexLayer::Metal1, polygons: std::slice::from_ref(&p_a), bbox: dummy_bbox(&p_a) },
            NetOnLayer { name: "b", layer: PexLayer::Metal3, polygons: std::slice::from_ref(&p_b), bbox: dummy_bbox(&p_b) },
        ];
        let p = extract(&nets, 1000, &Stack::sky130a_default(), "0");
        // No m1↔m3 coupling. Substrate cap for `a` (met1) only.
        assert!(p.iter().all(|x| !(x.layer_a == PexLayer::Metal1 && x.layer_b == PexLayer::Metal3)));
    }

    #[test]
    fn same_layer_no_coupling() {
        // Two nets on met1 — tier-1 doesn't model lateral coupling.
        let p_a = rect_poly(0, 0, 1_000, 1_000);
        let p_b = rect_poly(2_000, 0, 3_000, 1_000);
        let nets = vec![
            NetOnLayer { name: "a", layer: PexLayer::Metal1, polygons: std::slice::from_ref(&p_a), bbox: dummy_bbox(&p_a) },
            NetOnLayer { name: "b", layer: PexLayer::Metal1, polygons: std::slice::from_ref(&p_b), bbox: dummy_bbox(&p_b) },
        ];
        let p = extract(&nets, 1000, &Stack::sky130a_default(), "0");
        assert!(p.iter().all(|x| !(x.layer_a == PexLayer::Metal1 && x.layer_b == PexLayer::Metal1)));
    }

    #[test]
    fn sheet_rho_present_for_metals() {
        let s = Stack::sky130a_default();
        assert!((s.sheet_rho(PexLayer::Metal1).unwrap() - 0.125).abs() < 1e-9);
        assert!((s.sheet_rho(PexLayer::Metal3).unwrap() - 0.047).abs() < 1e-9);
        assert!(s.sheet_rho(PexLayer::Substrate).is_none());
    }

    #[test]
    fn long_skinny_resistor_matches_aspect_ratio() {
        // 10 µm × 1 µm met1 stripe — 10 squares × 0.125 Ω/sq = 1.25 Ω.
        let p = rect_poly(0, 0, 10_000, 1_000);
        let nets = vec![NetOnLayer {
            name: "rail", layer: PexLayer::Metal1,
            polygons: std::slice::from_ref(&p),
            bbox: dummy_bbox(&p),
        }];
        let rs = extract_resistance(&nets, 1000, &Stack::sky130a_default());
        assert_eq!(rs.len(), 1);
        assert!((rs[0].squares - 10.0).abs() < 1e-6, "got {}", rs[0].squares);
        assert!((rs[0].r_ohm - 1.25).abs() < 1e-6, "got {}", rs[0].r_ohm);
    }

    #[test]
    fn square_resistor_contributes_one_square() {
        // 1 µm × 1 µm met3 — 1 square × 0.047 = 0.047 Ω.
        let p = rect_poly(0, 0, 1_000, 1_000);
        let nets = vec![NetOnLayer {
            name: "tap", layer: PexLayer::Metal3,
            polygons: std::slice::from_ref(&p),
            bbox: dummy_bbox(&p),
        }];
        let rs = extract_resistance(&nets, 1000, &Stack::sky130a_default());
        assert!((rs[0].r_ohm - 0.047).abs() < 1e-6);
    }

    #[test]
    fn emit_spice_resistors_creates_far_node() {
        let r = ParasiticResistor {
            net: "vbias".into(), layer: PexLayer::Metal1,
            squares: 5.0, r_ohm: 0.625,
        };
        let lines = emit_spice_resistors(&[r]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].0, "vbias");
        assert_eq!(lines[0].1, "vbias_pex");
        assert!(lines[0].2.starts_with("Rpex0 vbias vbias_pex "));
    }

    #[test]
    fn emit_spice_caps_renders_with_ground_alias() {
        let ps = vec![Parasitic {
            net_a: "vbias".into(),
            net_b: "gnd".into(),
            layer_a: PexLayer::Metal1,
            layer_b: PexLayer::Substrate,
            overlap_um2: 10.0,
            cap_f: 1.5e-15,
        }];
        let lines = emit_spice_caps(&ps);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("Cpex0 vbias 0 "));
        assert!(lines[0].contains("1.500000e-15"));
    }
}
