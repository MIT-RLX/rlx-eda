//! MLP-based MOSFET I/V surrogate with the same calling convention as
//! `spike_mosfet_dc::id_subgraph`.
//!
//! Goal: a function `id_subgraph(g, vgs, vds, vth, kp, lam) -> NodeId`
//! that returns drain current as a NodeId in `g`, but routed through a
//! learned MLP rather than the analytic LEVEL-1 form. Because the call
//! shape is identical, every consumer of the analytic version
//! (`spike_divider_block::Mosfet`, `spike-cmos-gates`, etc.) can swap
//! to the surrogate behind a feature flag with no further code changes.
//!
//! Status: scaffold. The wiring (Mlp construction + parameter handles)
//! is real; the per-call `id_subgraph` body is a placeholder that needs
//! the rlx scalar-wiring decision (see TODO below).

use eda_nn::Mlp;
use rlx_ir::op::Activation;
use rlx_ir::{Graph, NodeId};

/// Hidden MLP that maps `[vgs, vds, vth, kp, lam] -> Id`.
///
/// The MLP is *batched* — when used inside an MNA residual graph each
/// transistor's KCL contribution is one element of the batch. The
/// trainer also batches over a sweep grid. Default hidden topology:
/// `[5, 32, 32, 1]`. Two hidden layers is enough to capture the smooth
/// triode/saturation transition; deeper nets just overfit.
pub struct MosfetSurrogate {
    pub mlp: Mlp,
    pub batch: usize,
}

impl MosfetSurrogate {
    /// Allocate the surrogate's parameters in `g`. `prefix` namespaces
    /// the Linear weight names so multiple surrogates can coexist
    /// (e.g. one per device flavour: NMOS_TT, NMOS_FF, ...).
    pub fn new(g: &mut Graph, prefix: &str, batch: usize) -> Self {
        let mlp = Mlp::new(g, prefix, &[5, 32, 32, 1], Activation::Tanh);
        Self { mlp, batch }
    }

    /// Drop-in replacement for `spike_mosfet_dc::id_subgraph`.
    ///
    /// TODO: assemble the `[B, 5]` input tensor by stacking the five
    /// scalar `NodeId`s along the feature axis. rlx's IR uses concat /
    /// stack ops for this; pick the canonical one once the analogous
    /// pattern in `spike-surrogate` is generalised. Until then this
    /// function is unimplemented; the validation plan in `PLAN.md`
    /// covers what to test once it lands.
    pub fn id_subgraph(
        &self,
        _g: &mut Graph,
        _vgs: NodeId,
        _vds: NodeId,
        _vth: NodeId,
        _kp: NodeId,
        _lam: NodeId,
    ) -> NodeId {
        unimplemented!(
            "MosfetSurrogate::id_subgraph: scalar-stacking convention TBD; \
             see PLAN.md → Differentiable surrogates and PINNs"
        )
    }
}

/// Training-time configuration for the surrogate trainer.
///
/// The trainer itself (sweep generation → ngspice batch → graph compile
/// → Adam loop) is a follow-on; this struct fixes the knobs so the
/// downstream binary doesn't need to invent them.
pub struct TrainConfig {
    /// Number of (vgs, vds) grid points per parameter-corner sample.
    pub grid: usize,
    /// Adam learning rate. 1e-3 is the default that worked on
    /// spike-surrogate's divider problem.
    pub lr: f32,
    /// Total training steps. spike-surrogate converged in ~1000.
    pub n_steps: usize,
    /// Weight on the analytic shape-anchor regulariser. 0 disables it.
    pub anchor_lambda: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self { grid: 32, lr: 1e-3, n_steps: 2000, anchor_lambda: 0.1 }
    }
}
