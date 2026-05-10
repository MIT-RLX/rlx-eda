//! `spike-waveguide-block` — the photonic analog of `spike-divider-block`.
//!
//! Two pieces:
//!
//! 1. The [`OpticalPdk`] trait — a small layer-and-port-kind contract
//!    every photonic PDK is expected to satisfy. Mirrors `RcLikePdk` in
//!    `spike-divider-block`: parametric Block + Layout code stays
//!    foundry-agnostic; the trait pins down what the PDK must expose.
//!
//! 2. A parametric [`Waveguide`] block. `Block` + `Layout<P: OpticalPdk>`
//!    impls. Lays out a single rectangular strip on the PDK's `wg()`
//!    layer, with two `optical_kind`-typed ports at the input/output
//!    ends. Width and length are runtime parameters in DBU.
//!
//! ## What this is
//!
//! A spike: minimum viable code that proves rlx-eda's "code-defined
//! photonics" pitch composes — same `Waveguide` Rust type, three
//! foundry PDKs, three different GDS layer pairs, all driven by a
//! generic `Layout<P: OpticalPdk>` impl.
//!
//! ## What this is not
//!
//! Not a real waveguide model: no propagation loss, no bend handling,
//! no taper. Those land when there's a real photonic-circuit consumer
//! to drive the design — see the README for the broader optimization
//! story.

use eda_hir::{Block, Layout};
use klayout_core::{
    Angle90, Bbox, CellBuilder, CellId, LayerIndex, Library, Point, Port, PortKindId, Rect, Shape,
};
use rlx_ir::{
    op::{Activation, BinaryOp},
    DType, Graph, NodeId, Op, Shape as TensorShape,
};

mod mzi;
pub use mzi::Mzi;

// ── PDK abstraction ────────────────────────────────────────────────────

/// Layers + port-kinds a photonic strip-waveguide block needs.
///
/// MVP scope: the strip waveguide layer (`wg`), a heater layer for
/// thermal phase shifters (`heater`), and a first metal layer for
/// pad/routing (`m1`). Plus the two port kinds — optical and
/// electrical — that distinguish a fiber-coupled port from a wirebond
/// pad.
///
/// Slab / partial-etch layers aren't in this trait yet because not
/// every photonic PDK exposes one cleanly under a unique short name
/// (SiEPIC EBeam being the obvious example). When a consumer actually
/// needs slab geometry, a sibling `SlabLayer` trait can be bolted on
/// and only the PDKs that have it implement it.
pub trait OpticalPdk {
    /// Strip-waveguide layer.
    fn wg(&self) -> LayerIndex;
    /// Thermal-heater metal (titanium nitride / NiCr / poly heater).
    fn heater(&self) -> LayerIndex;
    /// First routing metal — pad / heater contact.
    fn m1(&self) -> LayerIndex;
    /// Port-kind id for optical ports (fiber edge couplers, grating couplers).
    fn optical_kind(&self) -> PortKindId;
    /// Port-kind id for electrical ports (heater pads, bonding pads).
    fn electrical_kind(&self) -> PortKindId;
}

// ── OpticalPdk impls for each photonic PDK from eda-pdks ──────────────
//
// One trivial pass-through per method — mirrors how `spike-divider-block`
// hosts the `RcLikePdk` impls for the auto-generated `Sky130` /
// `Gf180mcu` from `eda-pdks`. Each impl is feature-gated so consumers
// only pull in the foundries they enabled in eda-pdks.

#[cfg(feature = "gdsfactory-generic")]
impl OpticalPdk for eda_pdks::GdsfactoryGeneric {
    fn wg(&self)              -> LayerIndex { self.WG }
    fn heater(&self)          -> LayerIndex { self.HEATER }
    fn m1(&self)              -> LayerIndex { self.M1 }
    fn optical_kind(&self)    -> PortKindId { Self::Optical }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

#[cfg(feature = "cornerstone-si220")]
impl OpticalPdk for eda_pdks::CornerstoneSi220 {
    fn wg(&self)              -> LayerIndex { self.WG }
    fn heater(&self)          -> LayerIndex { self.HEATER }
    fn m1(&self)              -> LayerIndex { self.M1 }
    fn optical_kind(&self)    -> PortKindId { Self::Optical }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

#[cfg(feature = "siepic-ebeam")]
impl OpticalPdk for eda_pdks::SiepicEbeam {
    fn wg(&self)              -> LayerIndex { self.WG }
    fn heater(&self)          -> LayerIndex { self.HEATER }
    fn m1(&self)              -> LayerIndex { self.M1 }
    fn optical_kind(&self)    -> PortKindId { Self::Optical }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

// ── Waveguide block ────────────────────────────────────────────────────

/// A straight rectangular strip waveguide.
///
/// `width` and `length` are in DBU (1000 DBU = 1 µm at the PDKs we
/// support). The block lays a single `Box` shape on the PDK's `wg()`
/// layer and exposes two optical ports — `in` at `(0, 0)` and `out` at
/// `(length, 0)` — both tagged with the PDK's `optical_kind()`.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Waveguide {
    pub width: i64,
    pub length: i64,
    pub id: String,
}

impl Block for Waveguide {
    fn name(&self) -> String {
        format!("Waveguide_{}_W{}_L{}", self.id, self.width, self.length)
    }
}

impl<P: OpticalPdk> Layout<P> for Waveguide {
    fn layout(&self, lib: &Library, pdk: &P) -> CellId {
        let mut cb = CellBuilder::new(<Self as Block>::name(self));
        // Centered on the x-axis: y ∈ [-w/2, w/2].
        let half_w = self.width / 2;
        let rect = Rect::new(Bbox::new(
            Point::new(0, -half_w),
            Point::new(self.length, half_w),
        ));
        cb.add_shape(pdk.wg(), Shape::Box(rect));
        // Two optical ports at the strip endpoints. Port `width` reflects
        // the waveguide width — downstream connectivity tooling can use
        // it to verify mating waveguides have compatible cross-sections.
        let kind = pdk.optical_kind();
        cb.add_port(
            Port::new("in", pdk.wg(), Point::new(0, 0), Angle90::W, self.width)
                .with_kind(kind),
        );
        cb.add_port(
            Port::new("out", pdk.wg(), Point::new(self.length, 0), Angle90::E, self.width)
                .with_kind(kind),
        );
        lib.insert(cb)
    }
}

// ── Optical scattering ─────────────────────────────────────────────────
//
// Photonic counterpart to `DcBehavioral`: register optical params and
// add ops computing the block's S-parameter response to the rlx graph.
//
// Two methods, two scopes:
//
// * `t_mag` — transmission magnitude `|T|`, real, scalar. Differentiable
//   through `Exp`/`Mul` only. Suffices for loss-side optimization
//   (`loss_dB_cm` → target `|T|`).
//
// * `s21` — full complex `S₂₁ = T_mag · exp(-i·phi)` as a `(Re, Im)`
//   pair of `NodeId`. Wavelength is a runtime `Input` so one graph
//   covers sweeps. Differentiable through `Sin`/`Cos`/`Exp`/`Mul` —
//   enables phase-side optimization (`neff`, dispersion).
//
// Multi-port blocks (splitters, couplers, rings) extend the surface
// with an `s_matrix` method when they land.

/// Optical scattering: register params + ops computing this block's
/// S-parameter response as an rlx graph. Two-port symmetric case for
/// now (waveguides, straight sections). Multi-port blocks (splitters,
/// couplers, rings) will get an `s_matrix(_)` method when they land.
pub trait OpticalScattering: Block {
    /// Register this block's loss-side parameter(s) into `g` and add
    /// the ops to compute scalar `|T|` (transmission magnitude, real,
    /// dimensionless, in `[0, 1]`). Returns the `|T|` `NodeId`.
    ///
    /// `length` and any other deterministic geometry get baked in as
    /// constants — only the materials/process side (loss per unit
    /// length) is exposed as an autodiff-able param. Used when the
    /// caller cares about loss-side optimization only.
    fn t_mag(&self, g: &mut Graph) -> NodeId;

    /// Register this block's full optical response and return
    /// `(S₂₁_re, S₂₁_im)` as `NodeId`s. `wavelength_nm` is an existing
    /// graph node — typically `g.input("wavelength_nm", ...)` — so a
    /// single graph compiles into a wavelength-sweepable session.
    ///
    /// The 2-port symmetric model treats `S₁₁ = S₂₂ = 0` (lossless on
    /// reflection, first-order); the cross terms equal `T = T_mag ·
    /// exp(-i·phi)`. Magnitude carries the propagation loss, phase
    /// carries `2π · n_eff · L / λ` (with first-order dispersion if
    /// `n_eff(λ)` is wired up downstream).
    fn s21(&self, wavelength_nm: NodeId, g: &mut Graph) -> (NodeId, NodeId);
}

/// `-ln(10) / 20` — converts dB to natural-log magnitude:
/// `T = 10^(-loss_dB / 20) = exp(-loss_dB · ln(10) / 20)`.
const NEPER_PER_DB: f32 = -2.302_585_093 / 20.0;

/// Convert DBU to centimetres. PDKs we ship use `dbu = 1000` → 1 µm =
/// 1000 DBU, so 1 DBU = 1 nm = 1e-7 cm.
const DBU_TO_CM: f32 = 1.0e-7;

pub(crate) fn const_f32(g: &mut Graph, val: f32, shape: TensorShape) -> NodeId {
    g.add_node(
        Op::Constant { data: val.to_le_bytes().to_vec() },
        vec![],
        shape,
    )
}

impl OpticalScattering for Waveguide {
    fn t_mag(&self, g: &mut Graph) -> NodeId {
        let s = TensorShape::new(&[1], DType::F32);
        // Process-side autodiff param: propagation loss in dB/cm.
        // Keyed by the block's name so each `Waveguide` instance gets
        // its own slot. Same identity that drives `Layout::layout`.
        let alpha_db_per_cm = g.param(
            format!("{}.loss_dB_cm", <Self as Block>::name(self)),
            s.clone(),
        );
        // Geometry is fixed per-block (`length` is part of the Block
        // name): bake it as a constant so it doesn't pollute the param
        // surface. Total dB loss = α [dB/cm] · length [cm].
        let length_cm = const_f32(g, (self.length as f32) * DBU_TO_CM, s.clone());
        let total_db = g.binary(BinaryOp::Mul, alpha_db_per_cm, length_cm, s.clone());
        // |T| = exp(-total_dB · ln(10) / 20)
        let scale = const_f32(g, NEPER_PER_DB, s.clone());
        let exponent = g.binary(BinaryOp::Mul, total_db, scale, s.clone());
        g.activation(Activation::Exp, exponent, s)
    }

    fn s21(&self, wavelength_nm: NodeId, g: &mut Graph) -> (NodeId, NodeId) {
        let s = TensorShape::new(&[1], DType::F32);

        // T_mag — same expression as `t_mag`. Inlined rather than
        // calling self.t_mag(g) so each `s21` build is self-contained
        // and downstream tests can graph-introspect a single S₂₁.
        let alpha = g.param(
            format!("{}.loss_dB_cm", <Self as Block>::name(self)),
            s.clone(),
        );
        let length_cm = const_f32(g, (self.length as f32) * DBU_TO_CM, s.clone());
        let total_db  = g.binary(BinaryOp::Mul, alpha, length_cm, s.clone());
        let neper_per_db = const_f32(g, NEPER_PER_DB, s.clone());
        let exponent  = g.binary(BinaryOp::Mul, total_db, neper_per_db, s.clone());
        let t_mag     = g.activation(Activation::Exp, exponent, s.clone());

        // Phase. `length_nm = length` because PDKs use DBU = 1000
        // (1 µm = 1000 DBU = 1000 nm). `phi = 2π · n_eff · L_nm / λ_nm`.
        let neff = g.param(
            format!("{}.neff", <Self as Block>::name(self)),
            s.clone(),
        );
        let length_nm = const_f32(g, self.length as f32, s.clone());
        let two_pi    = const_f32(g, std::f32::consts::TAU, s.clone());
        // Numerator: 2π · n_eff · L_nm
        let two_pi_neff = g.binary(BinaryOp::Mul, two_pi, neff, s.clone());
        let num         = g.binary(BinaryOp::Mul, two_pi_neff, length_nm, s.clone());
        let phi         = g.binary(BinaryOp::Div, num, wavelength_nm, s.clone());
        // S₂₁ = T_mag · exp(-i·phi) = T_mag·(cos(phi) − i·sin(phi)).
        let cos_phi = g.activation(Activation::Cos, phi, s.clone());
        let sin_phi = g.activation(Activation::Sin, phi, s.clone());
        let re      = g.binary(BinaryOp::Mul, t_mag, cos_phi, s.clone());
        let neg_t_mag_sin = g.binary(BinaryOp::Mul, t_mag, sin_phi, s.clone());
        let neg_one = const_f32(g, -1.0, s.clone());
        let im      = g.binary(BinaryOp::Mul, neg_t_mag_sin, neg_one, s);
        (re, im)
    }
}

impl Waveguide {
    /// Param name for `loss_dB_cm`, suitable for `Session::set_param`.
    pub fn loss_param_name(&self) -> String {
        format!("{}.loss_dB_cm", <Self as Block>::name(self))
    }

    /// Param name for the phase-side parameter `neff` (the effective
    /// refractive index). Set this together with `loss_param_name`
    /// when running an `s21`-built graph.
    pub fn neff_param_name(&self) -> String {
        format!("{}.neff", <Self as Block>::name(self))
    }

    /// Build a phase-side **loss graph**: `(∠S₂₁ − target_phase)²` over
    /// the `neff` param. Wavelength is a runtime input so a single
    /// compiled session sweeps λ.
    ///
    /// Inputs: `wavelength_nm` (operating wavelength, real), `target_phase`
    /// (desired `∠S₂₁` in radians), `d_output` (autodiff seed, set to `1`).
    /// Params: `loss_dB_cm`, `neff`.
    /// Output 0: scalar loss. Output 1: `∂L/∂neff`.
    ///
    /// `∠S₂₁` is computed via `atan2`-equivalent through Sub/Mul: since
    /// the phase model is `phi = 2π·neff·L/λ` and `S₂₁ = T_mag·exp(-iφ)`,
    /// we use `phi` directly (rather than recovering it from `re`/`im`)
    /// — algebraically identical for the model and avoids needing
    /// `atan2` ops in rlx-ir.
    pub fn build_phase_loss_graph(&self) -> (Graph, NodeId) {
        let mut g = Graph::new(format!("{}_phase_loss", <Self as Block>::name(self)));
        let s = TensorShape::new(&[1], DType::F32);

        // We intentionally rebuild the phase expression here (rather
        // than calling `s21` and recovering φ from re/im) because the
        // loss is naturally expressed in terms of φ itself: `(φ − φ*)²`.
        // S₂₁ is materialized below for downstream consumers that want
        // it; the loss output is the first graph output.
        let neff = g.param(self.neff_param_name(), s.clone());
        let length_nm = const_f32(&mut g, self.length as f32, s.clone());
        let two_pi    = const_f32(&mut g, std::f32::consts::TAU, s.clone());
        let wavelength_nm = g.input("wavelength_nm", s.clone());
        let two_pi_neff = g.binary(BinaryOp::Mul, two_pi, neff, s.clone());
        let num         = g.binary(BinaryOp::Mul, two_pi_neff, length_nm, s.clone());
        let phi_pos     = g.binary(BinaryOp::Div, num, wavelength_nm, s.clone());
        // Phase of S₂₁ = -phi (model uses `exp(-iφ)`).
        let neg_one = const_f32(&mut g, -1.0, s.clone());
        let phi     = g.binary(BinaryOp::Mul, phi_pos, neg_one, s.clone());

        let target_phase = g.input("target_phase", s.clone());
        let diff = g.binary(BinaryOp::Sub, phi, target_phase, s.clone());
        let loss = g.binary(BinaryOp::Mul, diff, diff, s);

        let neff_id = g
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(i, n)| match &n.op {
                Op::Param { name, .. } if *name == self.neff_param_name() => {
                    Some(NodeId(i as u32))
                }
                _ => None,
            })
            .expect("neff param missing");
        g.set_outputs(vec![loss]);
        (g, neff_id)
    }

    /// Build a scalar **loss graph** for inverse design: `(|T| − target)²`
    /// over the waveguide's loss param.
    ///
    /// Inputs: `target` (desired `|T|`, real ∈ `[0, 1]`).
    /// Params: `loss_dB_cm` (per the [`OpticalScattering`] impl).
    /// Output: scalar loss. `grad_with_loss(g, &[loss_id])` yields
    /// `[loss, ∂L/∂α]`.
    ///
    /// Returns `(graph, loss_param_id)` so the caller can wire the
    /// param into `grad_with_loss`.
    pub fn build_loss_graph(&self) -> (Graph, NodeId) {
        let mut g = Graph::new(format!("{}_loss", <Self as Block>::name(self)));
        let s = TensorShape::new(&[1], DType::F32);
        let t = self.t_mag(&mut g);
        let target = g.input("target", s.clone());
        let diff = g.binary(BinaryOp::Sub, t, target, s.clone());
        let loss = g.binary(BinaryOp::Mul, diff, diff, s);
        // Find the loss-param NodeId by looking it up — `t_mag` registered
        // it but didn't expose it. Walk the graph for the matching Param.
        let param_name = self.loss_param_name();
        let loss_id = g
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(i, n)| match &n.op {
                Op::Param { name, .. } if *name == param_name => {
                    Some(NodeId(i as u32))
                }
                _ => None,
            })
            .expect("OpticalScattering::t_mag must register the loss param");
        g.set_outputs(vec![loss]);
        (g, loss_id)
    }
}
