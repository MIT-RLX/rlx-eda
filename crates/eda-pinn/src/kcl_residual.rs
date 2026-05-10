//! KCL-residual losses for PINN training.
//!
//! Workflow: an NN predicts node voltages `v_pred = f_θ(params, t)` for
//! a circuit, and the loss is the squared sum of KCL residuals at every
//! internal node, evaluated at `v_pred`. No data labels needed — the
//! supervisory signal is "physics must hold".
//!
//! This module reuses `eda_mna::build_residual_graph` so KCL is not
//! duplicated. The current `build_residual_graph` API takes an
//! `&Circuit` and emits a graph whose inputs are the unknown-net
//! voltages. For PINN training we want to feed *NN-predicted* voltages
//! into those input slots — that is exactly the shape rlx already
//! supports via `Graph` composition. The TODO is to expose a small
//! wrapper that:
//!
//! 1. Builds the residual graph for `circuit` (existing call).
//! 2. Substitutes each unknown-net input with the corresponding column
//!    of `v_pred` from the NN forward pass (concat / index op).
//! 3. Sums squared residuals into a scalar loss NodeId.
//!
//! Once that wrapper exists, `grad_with_loss(loss, mlp.param_ids())`
//! gives gradients of the physics loss w.r.t. NN weights, and the
//! `eda-nn::Adam` driver from `mosfet_surrogate` reuses verbatim.

use rlx_ir::{Graph, NodeId};

/// Build a scalar PINN loss = Σ KCL_i(v_pred)² for an MNA circuit.
///
/// Status: scaffold. See module docs for the staged plan.
pub fn kcl_loss(
    _g: &mut Graph,
    // _circuit: &eda_mna::Circuit,
    _v_pred_per_node: &[NodeId],
    _params: &[NodeId],
) -> NodeId {
    unimplemented!(
        "kcl_loss: needs eda-mna::build_residual_graph_at(v_pred) hook; \
         see PLAN.md → Differentiable surrogates and PINNs"
    )
}
