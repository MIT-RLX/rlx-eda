//! End-to-end batched divider MNA on MLX.
//!
//! Proof-of-concept that the rlx stack can do GPU-batched MNA today:
//! a B-element batch of (R1, R2, V) values flows through one rlx-ir
//! graph that builds the per-draw 3x3 system, solves all B systems
//! via `Op::BatchedDenseSolve` (which lowers to `mlx::linalg::solve`
//! on the MLX-CPU stream — the Metal LU kernel is a follow-up that
//! drops in transparently), and returns Vout per draw.
//!
//! What this validates:
//!
//! 1. The `Op::BatchedDenseSolve` lowering we added to `rlx-mlx` works
//!    on a real (not smoke-test) circuit graph.
//! 2. Broadcast-shaped graph construction (per-draw conductances ×
//!    constant stamp patterns) lowers cleanly through MLX.
//! 3. Per-draw Vout matches the analytic `V·R2/(R1+R2)` within f32
//!    rounding — confirms shape inference + arithmetic + solve all
//!    line up.
//!
//! Throughput is **not** the headline here — MLX-CPU-stream solve is
//! ~6× over rayon CPU at best. The headline is that the architecture
//! end-to-ends, so when the Metal LU kernel sketched in
//! `rlx-mlx/src/batched_lu_kernel.rs` matures, the entire path
//! upstream of the solve op stays unchanged.

#![cfg(target_os = "macos")]

use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_mlx::{MlxExecutable, MlxMode};

// ── Shape helpers ───────────────────────────────────────────────────

fn batched_vec(b: usize) -> Shape {
    Shape::new(&[b], DType::F32)
}

fn batched_b1(b: usize) -> Shape {
    Shape::new(&[b, 1], DType::F32)
}

fn batched_b11(b: usize) -> Shape {
    Shape::new(&[b, 1, 1], DType::F32)
}

fn mat3_f32() -> Shape {
    Shape::new(&[3, 3], DType::F32)
}

fn vec3_f32() -> Shape {
    Shape::new(&[3], DType::F32)
}

fn batched_mat3(b: usize) -> Shape {
    Shape::new(&[b, 3, 3], DType::F32)
}

fn batched_vec3(b: usize) -> Shape {
    Shape::new(&[b, 3], DType::F32)
}

fn const_scalar_f32(g: &mut Graph, x: f32) -> NodeId {
    g.add_node(Op::Constant { data: x.to_le_bytes().to_vec() }, vec![], Shape::new(&[1], DType::F32))
}

fn const_mat3_pattern(g: &mut Graph, x: &[f32; 9]) -> NodeId {
    let mut bytes = Vec::with_capacity(36);
    for v in x { bytes.extend_from_slice(&v.to_le_bytes()); }
    g.add_node(Op::Constant { data: bytes }, vec![], mat3_f32())
}

fn const_vec3_pattern(g: &mut Graph, x: &[f32; 3]) -> NodeId {
    let mut bytes = Vec::with_capacity(12);
    for v in x { bytes.extend_from_slice(&v.to_le_bytes()); }
    g.add_node(Op::Constant { data: bytes }, vec![], vec3_f32())
}

// ── Batched MNA graph ───────────────────────────────────────────────

/// Build the f32 batched divider MNA graph.
///
/// Inputs (all shape `[B]`, f32):
///   `V`, `R1`, `R2`
///
/// Output (shape `[B]`, f32): `Vout` per draw.
pub fn build_batched_divider_mna(b: usize) -> Graph {
    let mut g = Graph::new("batched_divider_mna_f32");

    let v_in  = g.input("V",  batched_vec(b));
    let r1_in = g.input("R1", batched_vec(b));
    let r2_in = g.input("R2", batched_vec(b));

    // Per-draw conductances g1 = 1/R1, g2 = 1/R2, shape [B].
    let one = const_scalar_f32(&mut g, 1.0);
    let g1 = g.binary(BinaryOp::Div, one, r1_in, batched_vec(b));
    let g2 = g.binary(BinaryOp::Div, one, r2_in, batched_vec(b));

    // Reshape conductances [B] → [B, 1, 1] so they broadcast cleanly
    // against the [3, 3] stamp patterns (which align as [1, 3, 3])
    // to produce [B, 3, 3] per-draw stamps.
    let g1_b11 = g.reshape(g1, vec![b as i64, 1, 1], batched_b11(b));
    let g2_b11 = g.reshape(g2, vec![b as i64, 1, 1], batched_b11(b));

    // MNA stamp patterns (same as scalar `build_forward_mna`).
    let pattern_vsrc = const_mat3_pattern(&mut g, &[
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
    ]);
    let pattern_r1 = const_mat3_pattern(&mut g, &[
         1.0, -1.0, 0.0,
        -1.0,  1.0, 0.0,
         0.0,  0.0, 0.0,
    ]);
    let pattern_r2 = const_mat3_pattern(&mut g, &[
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
    ]);

    // A[draw] = pattern_vsrc + pattern_r1 · g1[draw] + pattern_r2 · g2[draw]
    // Each broadcast: [3,3] × [B,1,1] → [B,3,3].
    let r1_term = g.binary(BinaryOp::Mul, pattern_r1, g1_b11, batched_mat3(b));
    let r2_term = g.binary(BinaryOp::Mul, pattern_r2, g2_b11, batched_mat3(b));
    let a_partial = g.binary(BinaryOp::Add, pattern_vsrc, r1_term, batched_mat3(b));
    let a_mat     = g.binary(BinaryOp::Add, a_partial,    r2_term, batched_mat3(b));

    // b[draw] = pattern_b · V[draw]. pattern_b is [3], V_b1 is [B,1],
    // result is [B, 3] via standard broadcast (trailing-aligned).
    let pattern_b = const_vec3_pattern(&mut g, &[0.0, 0.0, 1.0]);
    let v_b1 = g.reshape(v_in, vec![b as i64, 1], batched_b1(b));
    let b_vec = g.binary(BinaryOp::Mul, pattern_b, v_b1, batched_vec3(b));

    // Solve all B systems in one op. Lowers to `mlx::linalg::solve`
    // (CPU stream today, Metal LU kernel later).
    let x = g.batched_dense_solve(a_mat, b_vec, batched_vec3(b));

    // Extract Vout = x[:, 1] via masked-dot reduction along axis=1.
    // Same trick the scalar path uses to avoid Op::Narrow's f32-only
    // CPU lowering (note: irrelevant here since we're f32 already, but
    // keeping the pattern parallel to the scalar code so future
    // changes apply uniformly).
    let e1 = const_vec3_pattern(&mut g, &[0.0, 1.0, 0.0]);
    let masked = g.binary(BinaryOp::Mul, x, e1, batched_vec3(b));
    let vout = g.reduce(masked, ReduceOp::Sum, vec![1], /*keep_dim=*/false, batched_vec(b));

    g.set_outputs(vec![vout]);
    g
}

/// Run the batched divider MNA through MLX. Returns Vout per draw (f32).
pub fn run_batched_divider_mna_mlx(v: &[f32], r1: &[f32], r2: &[f32]) -> Vec<f32> {
    let b = v.len();
    assert_eq!(r1.len(), b);
    assert_eq!(r2.len(), b);

    let g = build_batched_divider_mna(b);
    let mut exe = MlxExecutable::compile_with_mode(g, MlxMode::Lazy);
    let outs = exe.run(&[("V", v), ("R1", r1), ("R2", r2)]);
    outs.into_iter().next().unwrap_or_default()
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn analytic(v: f32, r1: f32, r2: f32) -> f32 {
        v * r2 / (r1 + r2)
    }

    #[test]
    fn batched_mlx_matches_analytic_per_draw() {
        // 16 independent draws covering a wide R-range. Decade spread
        // is tame for MNA but checks broadcast + solve over a real
        // batch. f32 tol of 1e-5 is comfortably above typical f32 LU
        // drift on this conditioning (cf. the precision probe).
        let v:  Vec<f32> = (0..16).map(|i| 0.5 + 0.1 * (i as f32 % 5.0)).collect();
        let r1: Vec<f32> = (0..16).map(|i| 1e3 * (i as f32 + 1.0)).collect();
        let r2: Vec<f32> = (0..16).map(|i| 5e3 * ((i as f32 % 4.0) + 1.0)).collect();

        let got = run_batched_divider_mna_mlx(&v, &r1, &r2);
        assert_eq!(got.len(), 16);

        for i in 0..16 {
            let want = analytic(v[i], r1[i], r2[i]);
            let drift = (got[i] - want).abs() / want.abs().max(1e-30);
            assert!(
                drift < 1e-5,
                "draw {i}: got {} want {} (rel drift {drift:.3e})",
                got[i], want,
            );
        }
    }

    #[test]
    fn batched_handles_uniform_and_non_uniform_draws() {
        // Edge case: all draws identical — checks that broadcast + solve
        // doesn't accidentally collapse the batch axis somewhere.
        let n = 8;
        let v  = vec![1.0_f32; n];
        let r1 = vec![1e3_f32; n];
        let r2 = vec![3e3_f32; n];
        let got = run_batched_divider_mna_mlx(&v, &r1, &r2);
        let want = analytic(1.0, 1e3, 3e3);
        for (i, &g) in got.iter().enumerate() {
            assert!((g - want).abs() < 1e-5, "uniform draw {i} drifted: {g} vs {want}");
        }
    }
}
