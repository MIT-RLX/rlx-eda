//! Mach-Zehnder interferometer composed from two `Waveguide` arms and
//! two ideal 50/50 directional couplers.
//!
//! The couplers are modeled algebraically rather than as separate
//! `Block`s — they're lossless, balanced, and frequency-flat by
//! construction:
//!
//! ```text
//!   S_coupler = (1/√2) · [[1, i], [i, 1]]
//! ```
//!
//! All wavelength-dependent behavior lives in the arm `Waveguide`s.
//! Their existing `loss_dB_cm` / `neff` params remain the autodiff
//! handles — the MZI just wires them through a complex 2×2×2 product.
//!
//! Closed-form (lossless). Both 50/50 90° couplers act as a "bar/cross
//! switch": balanced arms (Δφ = 0) put all light on the **through** port
//! by our labeling convention; antiphase (Δφ = π) puts it all on cross.
//!
//! ```text
//!   |T_through|² = cos²(Δφ/2)
//!   |T_cross  |² = sin²(Δφ/2)
//!   Δφ = (2π/λ) · (n_A · L_A − n_B · L_B)
//! ```
//!
//! With per-arm loss `T_A`, `T_B` (real, positive):
//!
//! ```text
//!   |T_through|² = (T_A² + T_B² + 2·T_A·T_B · cos Δφ) / 4
//!   |T_cross  |² = (T_A² + T_B² − 2·T_A·T_B · cos Δφ) / 4
//! ```
//!
//! Energy is conserved when `T_A = T_B = 1`; otherwise the deficit
//! `1 − (T_A² + T_B²)/2` shows up as arm loss.

use eda_hir::{Block, Layout};
use klayout_core::{
    Angle90, Bbox, CellBuilder, CellId, Library, Point, Port, Rect, Shape, Trans, Vec2,
};
use rlx_ir::{op::BinaryOp, DType, Graph, NodeId, Shape as TensorShape};

use crate::{const_f32, OpticalPdk, OpticalScattering, Waveguide};

/// Two-arm Mach-Zehnder interferometer. The couplers are implicit
/// (ideal 50/50, 90°). Both arms inherit their behavioral params from
/// the underlying [`Waveguide`]s, so a session built from
/// [`Mzi::build_intensity_graph`] takes the union of both arms' params:
/// `<arm>.loss_dB_cm` and `<arm>.neff`.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Mzi {
    pub arm_a: Waveguide,
    pub arm_b: Waveguide,
}

impl Mzi {
    /// Construct an MZI with two straight arms of the given widths and
    /// lengths (DBU). `id` is woven into both arms' `Waveguide.id` so
    /// the resulting param names stay unique across multiple MZIs in
    /// the same graph.
    pub fn new(width: i64, length_a: i64, length_b: i64, id: impl Into<String>) -> Self {
        let id: String = id.into();
        Self {
            arm_a: Waveguide { width, length: length_a, id: format!("{id}_armA") },
            arm_b: Waveguide { width, length: length_b, id: format!("{id}_armB") },
        }
    }

    /// Returns the through and cross complex amplitudes
    /// `(T_re, T_im, C_re, C_im)` at the given runtime wavelength node.
    pub fn s_outputs(
        &self,
        wavelength_nm: NodeId,
        g: &mut Graph,
    ) -> (NodeId, NodeId, NodeId, NodeId) {
        let s = TensorShape::new(&[1], DType::F32);
        let (a_re, a_im) = self.arm_a.s21(wavelength_nm, g);
        let (b_re, b_im) = self.arm_b.s21(wavelength_nm, g);

        // Two cascaded 50/50 90° couplers contribute (1/√2)·(1/√2) = 1/2
        // to every term. Working out the matrix product with input on
        // port 1 of the first coupler:
        //
        //   E_(port-3, opposite side) = (i/2) · (a + b)   ← "through"
        //   E_(port-4, same side)     = (1/2) · (a − b)   ← "cross"
        //
        // Convention here: the "through" port is the one that carries
        // *all* the light when arms are balanced (Δφ = 0). For two
        // 90° 3-dB couplers in series that's the opposite-side port.
        //
        // Expanding i·(a+b) = -Im(a+b) + i·Re(a+b):
        let half = const_f32(g, 0.5, s.clone());
        let neg_half = const_f32(g, -0.5, s.clone());

        // Through = (i/2)·(a + b)
        let sr = g.binary(BinaryOp::Add, a_re, b_re, s.clone());
        let si = g.binary(BinaryOp::Add, a_im, b_im, s.clone());
        let t_re = g.binary(BinaryOp::Mul, si, neg_half, s.clone());
        let t_im = g.binary(BinaryOp::Mul, sr, half, s.clone());

        // Cross = (1/2)·(a − b)
        let dr = g.binary(BinaryOp::Sub, a_re, b_re, s.clone());
        let di = g.binary(BinaryOp::Sub, a_im, b_im, s.clone());
        let c_re = g.binary(BinaryOp::Mul, dr, half, s.clone());
        let c_im = g.binary(BinaryOp::Mul, di, half, s);
        (t_re, t_im, c_re, c_im)
    }

    /// Build a forward graph that returns `[|T_through|², |T_cross|²]`
    /// at runtime `wavelength_nm`. The two outputs sum to `(T_A² +
    /// T_B²)/2` — exactly 1 in the lossless case, energy-bounded
    /// otherwise.
    pub fn build_intensity_graph(&self) -> Graph {
        let mut g = Graph::new(format!(
            "Mzi_{}_{}_intensity",
            self.arm_a.length, self.arm_b.length
        ));
        let s = TensorShape::new(&[1], DType::F32);
        let lambda = g.input("wavelength_nm", s.clone());
        let (t_re, t_im, c_re, c_im) = self.s_outputs(lambda, &mut g);
        let through = sq_mag(&mut g, t_re, t_im, &s);
        let cross = sq_mag(&mut g, c_re, c_im, &s);
        g.set_outputs(vec![through, cross]);
        g
    }

    /// Build an inverse-design **notch loss** graph: scalar
    /// `|T_through(λ)|²`, which the caller drives toward 0 to place a
    /// transmission zero at the operating wavelength.
    ///
    /// Inputs: `wavelength_nm`. Outputs: scalar loss.
    /// Differentiable through every `loss_dB_cm` / `neff` param the
    /// two arms registered. Typical use freezes one arm and runs Adam
    /// on the other arm's `neff` (a stand-in for a thermo-optic phase
    /// shifter).
    pub fn build_notch_loss_graph(&self) -> Graph {
        let mut g = Graph::new(format!(
            "Mzi_{}_{}_notch_loss",
            self.arm_a.length, self.arm_b.length
        ));
        let s = TensorShape::new(&[1], DType::F32);
        let lambda = g.input("wavelength_nm", s.clone());
        let (t_re, t_im, _, _) = self.s_outputs(lambda, &mut g);
        let intensity = sq_mag(&mut g, t_re, t_im, &s);
        g.set_outputs(vec![intensity]);
        g
    }
}

fn sq_mag(g: &mut Graph, re: NodeId, im: NodeId, s: &TensorShape) -> NodeId {
    let r2 = g.binary(BinaryOp::Mul, re, re, s.clone());
    let i2 = g.binary(BinaryOp::Mul, im, im, s.clone());
    g.binary(BinaryOp::Add, r2, i2, s.clone())
}

// ── Floorplan layout ───────────────────────────────────────────────────
//
// `Mzi` lays out a heated 2×2 Mach-Zehnder floorplan: two parallel
// `Waveguide` arms of equal physical length, joined by two rectangular
// directional-coupler regions on the WG layer, with a heater (HEATER
// layer) and two M1 contact pads on arm A representing the
// thermo-optic phase shifter.
//
// Real asymmetric MZIs route a meander on the shorter arm to add
// optical path-length while keeping coupler endpoints aligned. We
// render the equal-length topology and let the behavioral `n_eff_A`
// tune carry the asymmetry — it's what an active MZI actually does in
// silicon photonics, where ΔL is fixed by mask geometry and Δφ comes
// from the heater.

/// Floorplan parameters. Picked to comfortably pass typical 220 nm-SOI
/// photonic DRC: WG width ≥ 400 nm, WG–WG spacing ≥ 1 µm, HEATER width
/// ≥ 1 µm, M1 width ≥ 1 µm.
const ARM_SEPARATION: i64 = 5_000; // 5 µm centre-to-centre
const COUPLER_LEN: i64 = 4_000;    // 4 µm
const STUB_LEN: i64 = 5_000;       // 5 µm input/output bus
const HEATER_OVERHANG: i64 = 5_000; // distance from arm end to heater end
const HEATER_WIDTH: i64 = 2_000;   // 2 µm
const M1_PAD: i64 = 2_000;         // 2 µm square contacts
const M1_PAD_OFFSET: i64 = 4_000;  // pad centre offset above arm A
const M1_LEAD_W: i64 = 1_000;      // 1 µm wide vertical M1 lead pad → heater

impl Block for Mzi {
    fn name(&self) -> String {
        format!(
            "Mzi_{}_{}_{}_floorplan",
            self.arm_a.id, self.arm_a.length, self.arm_b.length
        )
    }
}

impl<P: OpticalPdk> Layout<P> for Mzi {
    fn layout(&self, lib: &Library, pdk: &P) -> CellId {
        // Symmetric floorplan length — see module note above.
        let arm_len = self.arm_a.length.max(self.arm_b.length);
        let half_w = self.arm_a.width / 2;

        // Y bounds of the coupler bridges (covers both arm extents).
        let coup_y_min = -ARM_SEPARATION / 2 - half_w;
        let coup_y_max = ARM_SEPARATION / 2 + half_w;

        // X regions, left → right.
        let x_in_stub_lo = -STUB_LEN - COUPLER_LEN;
        let x_in_stub_hi = -COUPLER_LEN;
        let x_in_coup_lo = -COUPLER_LEN;
        let x_in_coup_hi = 0;
        let x_out_coup_lo = arm_len;
        let x_out_coup_hi = arm_len + COUPLER_LEN;
        let x_out_stub_lo = arm_len + COUPLER_LEN;
        let x_out_stub_hi = arm_len + COUPLER_LEN + STUB_LEN;

        // Instantiate symmetric arm cells of length `arm_len` (distinct
        // from the behavioral arm_a/arm_b lengths) so the floorplan
        // remains physically consistent.
        let arm_top = Waveguide {
            width: self.arm_a.width,
            length: arm_len,
            id: format!("{}_floorplan_top", self.arm_a.id),
        };
        let arm_bot = Waveguide {
            width: self.arm_b.width,
            length: arm_len,
            id: format!("{}_floorplan_bot", self.arm_b.id),
        };
        let arm_top_id = arm_top.layout(lib, pdk);
        let arm_bot_id = arm_bot.layout(lib, pdk);

        let mut top = CellBuilder::new(<Self as Block>::name(self));
        top.instantiate(arm_top_id, Trans::translate(Vec2::new(0, ARM_SEPARATION / 2)));
        top.instantiate(arm_bot_id, Trans::translate(Vec2::new(0, -ARM_SEPARATION / 2)));

        // Coupler bridges (WG layer rectangles spanning both arms).
        let wg = pdk.wg();
        top.add_shape(
            wg,
            Shape::Box(Rect::new(Bbox::new(
                Point::new(x_in_coup_lo, coup_y_min),
                Point::new(x_in_coup_hi, coup_y_max),
            ))),
        );
        top.add_shape(
            wg,
            Shape::Box(Rect::new(Bbox::new(
                Point::new(x_out_coup_lo, coup_y_min),
                Point::new(x_out_coup_hi, coup_y_max),
            ))),
        );

        // Two input + two output stub waveguides (true 2×2 MZI ports).
        for &y_centre in &[ARM_SEPARATION / 2, -ARM_SEPARATION / 2] {
            top.add_shape(
                wg,
                Shape::Box(Rect::new(Bbox::new(
                    Point::new(x_in_stub_lo, y_centre - half_w),
                    Point::new(x_in_stub_hi, y_centre + half_w),
                ))),
            );
            top.add_shape(
                wg,
                Shape::Box(Rect::new(Bbox::new(
                    Point::new(x_out_stub_lo, y_centre - half_w),
                    Point::new(x_out_stub_hi, y_centre + half_w),
                ))),
            );
        }

        // Heater on arm A (top arm), with two M1 contact pads pulled
        // away vertically so they don't sit directly on the WG.
        let heater_lo_x = HEATER_OVERHANG;
        let heater_hi_x = arm_len - HEATER_OVERHANG;
        let heater_y_mid = ARM_SEPARATION / 2;
        top.add_shape(
            pdk.heater(),
            Shape::Box(Rect::new(Bbox::new(
                Point::new(heater_lo_x, heater_y_mid - HEATER_WIDTH / 2),
                Point::new(heater_hi_x, heater_y_mid + HEATER_WIDTH / 2),
            ))),
        );

        let pad_y = ARM_SEPARATION / 2 + HEATER_WIDTH / 2 + M1_PAD_OFFSET;
        let elec = pdk.electrical_kind();
        // M1 lead bottom = heater top edge (the "via" landing); M1 lead
        // top = pad bottom edge. The lead overlaps the heater by
        // HEATER_WIDTH/2 vertically so the contact area is real.
        let lead_top_y = pad_y - M1_PAD / 2;
        let lead_bot_y = ARM_SEPARATION / 2 - HEATER_WIDTH / 2;
        for (px, name) in [(heater_lo_x, "heater_neg"), (heater_hi_x, "heater_pos")] {
            // Square M1 contact pad.
            top.add_shape(
                pdk.m1(),
                Shape::Box(Rect::new(Bbox::new(
                    Point::new(px - M1_PAD / 2, pad_y - M1_PAD / 2),
                    Point::new(px + M1_PAD / 2, pad_y + M1_PAD / 2),
                ))),
            );
            // Vertical M1 lead from pad down to (and overlapping) the
            // heater end — represents the routing + via stack.
            top.add_shape(
                pdk.m1(),
                Shape::Box(Rect::new(Bbox::new(
                    Point::new(px - M1_LEAD_W / 2, lead_bot_y),
                    Point::new(px + M1_LEAD_W / 2, lead_top_y),
                ))),
            );
            top.add_port(
                Port::new(name, pdk.m1(), Point::new(px, pad_y), Angle90::N, M1_PAD)
                    .with_kind(elec),
            );
        }

        // Optical ports: 2 input (left), 2 output (right).
        let opt = pdk.optical_kind();
        top.add_port(
            Port::new("in1", wg, Point::new(x_in_stub_lo, ARM_SEPARATION / 2), Angle90::W, self.arm_a.width)
                .with_kind(opt),
        );
        top.add_port(
            Port::new("in2", wg, Point::new(x_in_stub_lo, -ARM_SEPARATION / 2), Angle90::W, self.arm_a.width)
                .with_kind(opt),
        );
        top.add_port(
            Port::new("through", wg, Point::new(x_out_stub_hi, ARM_SEPARATION / 2), Angle90::E, self.arm_a.width)
                .with_kind(opt),
        );
        top.add_port(
            Port::new("cross", wg, Point::new(x_out_stub_hi, -ARM_SEPARATION / 2), Angle90::E, self.arm_a.width)
                .with_kind(opt),
        );

        lib.insert(top)
    }
}
