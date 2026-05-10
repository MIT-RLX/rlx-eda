//! Optimization helpers — Adam state + LR / β schedules — used by
//! every trace bin's inner loop.
//!
//! Until this module landed, `lna_match_trace`, `mzi_match_trace`,
//! and `hpwl_optim_trace` each hand-rolled the same Adam mechanics
//! (`m`, `v`, bias correction, manual `set_param` / `run` plumbing,
//! constant LR with no decay). Three places to fix when the loss
//! tail oscillated, three places to add LR decay or β-anneal. This
//! lifts that into one tested implementation:
//!
//! * [`AdamState`] — `(m, v)` per-parameter buffers, `step()` for
//!   one bias-corrected update.
//! * [`LrSchedule`] — constant / cosine / step-decay / linear-decay.
//! * [`BetaSchedule`] — for log-sum-exp smoothing in placement
//!   loss graphs (start smooth, sharpen near convergence — the
//!   standard DREAMPlace-style β-anneal trick).
//!
//! Nothing here knows about rlx — both Adam and the schedules are
//! plain f32 math, callable from any optimizer loop. The trace bins
//! still own the rlx Session, the loss graph, and whatever
//! per-domain bookkeeping (which Param keys to update, what to
//! log per step). The harness owns timing-correct bias correction
//! and LR / β decay.

use std::f32::consts::PI;

use rlx_runtime::{is_available, Device};

// ── Device selection ──────────────────────────────────────────────────
//
// On macOS hosts where the MLX backend is registered, point new
// Sessions at `Device::Mlx` so the rlx graph's matmul / elementwise
// chains run on the Apple-GPU. Anywhere else, fall back to
// `Device::Cpu`. Callers wire it as:
//
// ```ignore
// let sess = Session::new(eda_trace::optim::default_device()).compile(graph);
// ```
//
// One line, no target-cfg in the bin.

/// Returns the best available `Device` for new Sessions:
/// `Device::Mlx` if the MLX backend is registered (typically on
/// macOS via `rlx-mlx`), otherwise `Device::Cpu`. Designed so
/// trace bins read the same regardless of host — the GPU path
/// just lights up when MLX is in scope.
pub fn default_device() -> Device {
    if is_available(Device::Mlx) {
        Device::Mlx
    } else {
        Device::Cpu
    }
}

/// Same as [`default_device`] but with an explicit override for
/// CI runs where the host has MLX but you want the deterministic
/// CPU path (set `RLX_FORCE_CPU=1`).
pub fn default_device_or_env() -> Device {
    if std::env::var("RLX_FORCE_CPU").map(|v| v != "0").unwrap_or(false) {
        return Device::Cpu;
    }
    default_device()
}

// ── Adam state ────────────────────────────────────────────────────────

/// Adam first/second moments + hyperparameters. One [`AdamState`]
/// covers the whole parameter vector — call [`AdamState::step`]
/// once per Adam iteration with the current grads + LR.
#[derive(Clone, Debug)]
pub struct AdamState {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
}

impl AdamState {
    /// New zero-initialised state for `n` parameters with the
    /// canonical Kingma & Ba 2014 hyperparameters.
    pub fn new(n: usize) -> Self {
        Self::with_betas(n, 0.9, 0.999, 1e-8)
    }

    pub fn with_betas(n: usize, b1: f32, b2: f32, eps: f32) -> Self {
        Self { m: vec![0.0; n], v: vec![0.0; n], b1, b2, eps }
    }

    /// One Adam update: `params[k] -= lr · m̂[k] / (√v̂[k] + eps)`,
    /// where `m̂`, `v̂` are bias-corrected first/second moments.
    /// `step` is the 1-based iteration counter (use `t` from
    /// `for t in 1..=N`).
    pub fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32, step: u32) {
        debug_assert_eq!(params.len(), self.m.len(), "Adam: params length mismatch");
        debug_assert_eq!(grads.len(), self.m.len(), "Adam: grads length mismatch");
        let t = step.max(1) as i32;
        let b1_t = self.b1.powi(t);
        let b2_t = self.b2.powi(t);
        for k in 0..self.m.len() {
            let g = grads[k];
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g;
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * g * g;
            let m_hat = self.m[k] / (1.0 - b1_t);
            let v_hat = self.v[k] / (1.0 - b2_t);
            params[k] -= lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ── Learning-rate schedule ────────────────────────────────────────────

/// Per-step LR multiplier on a base learning rate. The trace
/// harness asks the schedule for `lr_at(base, step, total)` once
/// per iteration; the schedule decides what fraction of `base` to
/// return. Independent of optimizer choice — works equally well
/// with [`AdamState`] and bare SGD.
#[derive(Clone, Debug)]
pub enum LrSchedule {
    /// Always return `base`.
    Constant,
    /// Half-cosine decay from `base` down to `base · min_factor`
    /// over `total` steps. Smooth, no hyperparameters beyond the
    /// floor — same shape every PyTorch / JAX consumer reaches for.
    Cosine { min_factor: f32 },
    /// Linear decay from `base` to `base · min_factor`.
    LinearDecay { min_factor: f32 },
    /// Multiply by `factor` at each `drop_at` step. Simple, easy
    /// to read in trace logs.
    StepDecay { drop_at: Vec<u32>, factor: f32 },
}

impl LrSchedule {
    pub fn lr_at(&self, base: f32, step: u32, total: u32) -> f32 {
        if total == 0 { return base; }
        let t = (step.min(total) as f32) / (total as f32);
        match self {
            LrSchedule::Constant => base,
            LrSchedule::Cosine { min_factor } => {
                let factor = min_factor + (1.0 - min_factor) * 0.5 * (1.0 + (PI * t).cos());
                base * factor
            }
            LrSchedule::LinearDecay { min_factor } => {
                base * (1.0 - (1.0 - min_factor) * t)
            }
            LrSchedule::StepDecay { drop_at, factor } => {
                let drops = drop_at.iter().filter(|&&d| step >= d).count() as i32;
                base * factor.powi(drops)
            }
        }
    }
}

// ── β / sharpness schedule ────────────────────────────────────────────

/// Smoothing-sharpness schedule, used by placement losses where
/// the LSE β trades approximation accuracy for gradient-signal
/// strength. Standard DREAMPlace-style trick: start with a small
/// β so the loss surface is broad and gradients carry far, then
/// sharpen β so the optimum approaches the true (non-smoothed)
/// HPWL minimum.
#[derive(Clone, Debug)]
pub enum BetaSchedule {
    /// Always return `beta`.
    Constant(f32),
    /// Linear interpolation between `start` and `end` over
    /// `total` steps. Fastest to write, decent enough in practice.
    LinearAnneal { start: f32, end: f32 },
    /// Geometric (log-space) interpolation. Better fit when
    /// `end / start` spans multiple orders of magnitude.
    GeometricAnneal { start: f32, end: f32 },
    /// Half-cosine in log-space — combines geometric anneal with a
    /// smooth ramp, matches the LR cosine schedule's shape.
    CosineAnneal { start: f32, end: f32 },
}

impl BetaSchedule {
    pub fn beta_at(&self, step: u32, total: u32) -> f32 {
        if total == 0 {
            return match self {
                BetaSchedule::Constant(b) => *b,
                BetaSchedule::LinearAnneal { start, .. }
                | BetaSchedule::GeometricAnneal { start, .. }
                | BetaSchedule::CosineAnneal { start, .. } => *start,
            };
        }
        let t = (step.min(total) as f32) / (total as f32);
        match self {
            BetaSchedule::Constant(b) => *b,
            BetaSchedule::LinearAnneal { start, end } => start + (end - start) * t,
            BetaSchedule::GeometricAnneal { start, end } => {
                let ls = start.ln();
                let le = end.ln();
                (ls + (le - ls) * t).exp()
            }
            BetaSchedule::CosineAnneal { start, end } => {
                // 0..1 cosine ramp in log-space: t' = 0.5 * (1 - cos(π · t)).
                let tp = 0.5 * (1.0 - (PI * t).cos());
                let ls = start.ln();
                let le = end.ln();
                (ls + (le - ls) * tp).exp()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_decreases_loss_on_quadratic() {
        // Minimize f(x) = (x - 3)² with Adam; x should approach 3.
        let mut x = vec![0.0_f32];
        let mut adam = AdamState::new(1);
        let lr = 0.1_f32;
        for t in 1..=200 {
            let grad = 2.0 * (x[0] - 3.0);
            adam.step(&mut x, &[grad], lr, t);
        }
        assert!((x[0] - 3.0).abs() < 0.05, "x = {} after Adam", x[0]);
    }

    #[test]
    fn cosine_lr_decays_to_floor() {
        let s = LrSchedule::Cosine { min_factor: 0.1 };
        assert!((s.lr_at(1.0, 0, 100) - 1.0).abs() < 1e-6);
        assert!((s.lr_at(1.0, 100, 100) - 0.1).abs() < 1e-6);
        // Halfway: between 1.0 and 0.1, smoothly.
        let mid = s.lr_at(1.0, 50, 100);
        assert!(mid > 0.1 && mid < 1.0, "mid lr = {mid}");
    }

    #[test]
    fn step_decay_drops_at_milestones() {
        let s = LrSchedule::StepDecay { drop_at: vec![100, 200], factor: 0.1 };
        assert!((s.lr_at(1.0, 50, 300) - 1.0).abs() < 1e-6);
        assert!((s.lr_at(1.0, 150, 300) - 0.1).abs() < 1e-6);
        assert!((s.lr_at(1.0, 250, 300) - 0.01).abs() < 1e-7);
    }

    #[test]
    fn geometric_anneal_log_space_interpolation() {
        let s = BetaSchedule::GeometricAnneal { start: 1e-5, end: 1e-3 };
        assert!((s.beta_at(0, 100) - 1e-5).abs() < 1e-9);
        let mid = s.beta_at(50, 100);
        // Log-space midpoint of (1e-5, 1e-3) is sqrt(1e-5 * 1e-3) = 1e-4.
        assert!((mid - 1e-4).abs() / 1e-4 < 1e-3, "mid β = {mid}");
        assert!((s.beta_at(100, 100) - 1e-3).abs() / 1e-3 < 1e-3);
    }
}
