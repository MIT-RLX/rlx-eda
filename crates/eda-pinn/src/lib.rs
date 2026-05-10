//! Physics-informed training on top of the rlx + eda-mna stack.
//!
//! Two distinct workflows live here:
//!
//! 1. **Differentiable surrogates** (`mosfet_surrogate`) — train an MLP
//!    to mimic an existing physics subgraph (e.g. `spike-mosfet-dc`'s
//!    smooth LEVEL=1 `id_subgraph`) and expose a drop-in replacement
//!    with the same signature. The surrogate retains AD, so it slots
//!    into `eda-mna::Circuit` and `transient_sensitivities` keeps
//!    working unchanged. Loss = data MSE against ngspice/analytic
//!    samples + a shape-anchor regulariser against the analytic form.
//!
//! 2. **PINN-style residual training** (`kcl_residual`) — let an NN
//!    predict node voltages directly, with the loss derived from KCL
//!    residuals at NN-predicted state. Reuses `eda-mna`'s residual
//!    graph builder so physics is not duplicated.
//!
//! Both are scaffolds today; see `PLAN.md → Differentiable surrogates
//! and PINNs` for the staged plan and validation pyramid.

pub mod mosfet_surrogate;
pub mod kcl_residual;
