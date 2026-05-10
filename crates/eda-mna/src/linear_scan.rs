//! Auto-derive an `Op::Scan` body for linear circuits.
//!
//! For any linear MNA, the BE-step residual has the form
//!
//!   `r(v, v_prev) = K · v − b(v_prev)`
//!
//! where `K` is constant in v (linearity) and `b` is linear in v_prev.
//! Decomposing `b(v_prev) = M_b · v_prev + c`, the BE step solution
//! is `v_new = K⁻¹ · b(v_prev) = (K⁻¹ M_b) · v_prev + K⁻¹ c`. Both
//! `step_matrix = K⁻¹ M_b` and `step_const = K⁻¹ c` are constants we
//! can extract once via the existing residual-graph + AD machinery
//! and embed as `Op::Constant` nodes in the scan body.
//!
//! The body that comes out is a tiny graph: one matmul + one add per
//! step. Wrap it in [`rlx_ir::Graph::scan_trajectory`] to get a
//! whole-transient-on-GPU dispatch in a single MLX call.
//!
//! ## Scope (this module)
//!
//! * **Linear-only**: `K` extracted once at v=0; if the circuit's
//!   residual is non-linear in v, `K` won't be constant and the body
//!   is wrong. Caller is responsible for supplying a linear circuit.
//! * **Params baked in at body-build time**: changing R / C means
//!   rebuilding the body. Per-draw MC over device params would
//!   require a different body shape (params as bcasts) and is a
//!   follow-up.
//! * **No branches**: voltage sources / inductors that introduce
//!   branch unknowns aren't supported yet — would need stamping into
//!   K's branch rows / columns. Same `build_be_step_residual_graph`
//!   produces them; the extraction below would just need to handle
//!   `n_v + n_b` unknowns instead of just `n_v`.

use std::collections::HashMap;

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::{
    build_be_step_residual_graph, branch_input_name, find_input_node,
    net_input_name, prev_voltage_input_name, BranchId, Circuit, NetId,
    TIMESTEP_INPUT_NAME,
};

/// Result of [`build_linear_be_step_body`].
#[derive(Debug)]
pub struct LinearBeStepBody {
    /// Body graph: input `"carry"` shape `[n]`, output shape `[n]`.
    /// Plug into [`rlx_ir::Graph::scan_trajectory`] (carry first, body
    /// second, length third) to get a whole-transient graph.
    pub body: Graph,
    /// Number of unknowns (voltage + branch). Carry shape is `[n]`.
    pub n: usize,
    /// Net + branch ordering. `unknowns[i]` is the i-th component
    /// of the carry vector for `i < n_v`; `branches[i - n_v]` for
    /// `i >= n_v`.
    pub unknowns: Vec<NetId>,
    pub branches: Vec<BranchId>,
}

/// Auto-derive a linear BE-step body from `circuit + params + dt`.
///
/// Internally:
///   1. Builds the existing scalar BE-step residual graph
///      ([`build_be_step_residual_graph`]).
///   2. Extracts `K` (Jacobian wrt unknown voltages, evaluated at
///      v=0) via `grad_with_loss` + rlx-cpu evaluation.
///   3. Extracts `M_b` and `c` similarly: `r(v=0, v_prev=0) = -c`,
///      `∂r/∂v_prev_j` at (v=0, v_prev=0) = -M_b column j.
///   4. Solves `K · step_matrix = M_b` and `K · step_const = c` on
///      the host (Gauss-Jordan).
///   5. Builds a body graph that does `v_new = step_matrix · carry +
///      step_const` via one matmul + one add.
/// Same as [`build_linear_be_step_body`] but with `mc_param_names`
/// promoted to per-draw inputs in the resulting body.
///
/// The body graph gains one `Op::Input` per name in `mc_param_names`
/// (each shape `[1]`). After wrapping in `scan_trajectory` and
/// `vmap`'ing with those names in the batched-input list, each
/// becomes a `[B, 1]` per-draw value (a vmap-Scan **bcast**: held
/// constant across time, varying per draw). Inside the body, K and
/// M_b are symbolic graph expressions of those inputs — different
/// draws solve different per-step systems.
///
/// Non-mc params (the rest of `params`) stay as `Op::Param` slots
/// in the body and bind once via `set_param` on the compiled
/// executable, same as the no-mc-params variant.
pub fn build_linear_be_step_body_with_mc_params(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    mc_param_names: &[&str],
    dt: f32,
) -> Result<LinearBeStepBody, String> {
    if mc_param_names.is_empty() {
        return build_linear_be_step_body(circuit, params, dt);
    }
    use std::collections::HashSet;
    use rlx_ir::op::ReduceOp;

    let scalar_orig = build_be_step_residual_graph(circuit);
    let scalar_promoted = rlx_opt::promote_params_to_inputs(
        &scalar_orig.graph, mc_param_names,
    );
    let unknowns = scalar_orig.unknown_nets.clone();
    let branches = scalar_orig.branches.clone();
    let n_v = unknowns.len();
    let n_b = branches.len();
    let n = n_v + n_b;
    if n == 0 {
        return Err("circuit has no unknowns".into());
    }
    if n_b > 0 {
        return Err(format!(
            "build_linear_be_step_body_with_mc_params: branches not yet \
             supported (circuit has {n_b} branch unknown(s))."
        ));
    }

    // unknowns input ids in promoted graph (need to check the
    // promoted graph since IDs may have shifted).
    let unknown_input_ids: Vec<NodeId> = unknowns.iter()
        .map(|net| find_input_node(&scalar_promoted, &net_input_name(*net))
            .ok_or_else(|| format!("residual missing v_{}", net.0)))
        .collect::<Result<_, _>>()?;
    let prev_input_ids: Vec<NodeId> = unknowns.iter()
        .map(|net| find_input_node(&scalar_promoted, &prev_voltage_input_name(*net))
            .ok_or_else(|| format!("residual missing v_prev_{}", net.0)))
        .collect::<Result<_, _>>()?;
    let mut grad_targets = unknown_input_ids.clone();
    grad_targets.extend(prev_input_ids.iter().copied());

    // Build per-row grad graphs (one per residual output).
    let mut row_grad_graphs: Vec<Graph> = Vec::with_capacity(n_v);
    for i in 0..n_v {
        let mut g_i = scalar_promoted.clone();
        let out_i = g_i.outputs[i];
        g_i.set_outputs(vec![out_i]);
        let bwd = rlx_opt::autodiff::grad_with_loss(&g_i, &grad_targets);
        row_grad_graphs.push(bwd);
    }

    // Backfill <name>_tau defaults — same as the eager builder.
    let mut effective_params = params.clone();
    for dl in &circuit.delays {
        let key = format!("{}_tau", dl.device.name());
        effective_params.entry(key).or_insert(
            dl.device.delay_seconds() as f32,
        );
    }
    let mc_set: HashSet<&str> = mc_param_names.iter().copied().collect();

    // ── Build the body graph. ──
    let s_scalar = Shape::new(&[1], DType::F32);
    let s_vec    = Shape::new(&[n], DType::F32);
    let s_col    = Shape::new(&[n, 1], DType::F32);
    let s_mat    = Shape::new(&[n, n], DType::F32);
    let mut body = Graph::new("linear_be_body_with_mc");

    // Body inputs: carry [n] first, then each mc_param as [1]
    // (declaration order matters — vmap-Scan reads inputs as
    // [carry, bcast_0, ..., bcast_B-1, x_t_0, ...]).
    let carry = body.input("carry", s_vec.clone());
    let mut mc_input_ids: HashMap<String, NodeId> = HashMap::new();
    for nm in mc_param_names {
        mc_input_ids.insert(
            (*nm).to_string(),
            body.input((*nm).to_string(), s_scalar.clone()),
        );
    }

    // Constants the inlined grad graphs share: zero (for v / v_prev
    // bindings inside grad), one (for d_output), and dt (for h).
    let zero_scalar = body.add_node(
        Op::Constant { data: 0.0_f32.to_le_bytes().to_vec() },
        vec![], s_scalar.clone(),
    );
    let one_scalar = body.add_node(
        Op::Constant { data: 1.0_f32.to_le_bytes().to_vec() },
        vec![], s_scalar.clone(),
    );
    let dt_scalar = body.add_node(
        Op::Constant { data: dt.to_le_bytes().to_vec() },
        vec![], s_scalar.clone(),
    );

    // Per-row inlining. We inline each grad graph with v_<id>=0 and
    // v_prev_<id>=0 (so K and M_b extract symbolically as graph
    // expressions of the mc_param inputs); we don't need v_prev to
    // be `carry` here because for linear circuits b(v_prev) is
    // exactly M_b · v_prev + c, and we can build that affine form
    // inside the body using the M_b columns extracted at v_prev=0.
    let mut k_rows:   Vec<Vec<NodeId>> = Vec::with_capacity(n_v);
    let mut mb_rows:  Vec<Vec<NodeId>> = Vec::with_capacity(n_v);
    let mut c_rows:   Vec<NodeId>      = Vec::with_capacity(n_v);
    for i in 0..n_v {
        let bwd = &row_grad_graphs[i];
        // Build the input bindings map for this inline.
        let mut bindings: HashMap<String, NodeId> = HashMap::new();
        // All net voltage inputs (boundary + unknown) → 0.
        for net in &scalar_orig.all_nets {
            bindings.insert(net_input_name(*net),         zero_scalar);
            bindings.insert(prev_voltage_input_name(*net), zero_scalar);
        }
        // Branches → 0.
        for b in &branches {
            bindings.insert(branch_input_name(*b), zero_scalar);
        }
        // Timestep + d_output.
        bindings.insert(TIMESTEP_INPUT_NAME.to_string(), dt_scalar);
        bindings.insert("d_output".to_string(),          one_scalar);
        // Delay scalars at zero defaults.
        for (idx, _) in circuit.delays.iter().enumerate() {
            let id = crate::DelayId(idx as u32);
            bindings.insert(crate::delay_v_lo_name(id),   zero_scalar);
            bindings.insert(crate::delay_v_hi_name(id),   zero_scalar);
            bindings.insert(crate::delay_blend_name(id),  zero_scalar);
            bindings.insert(crate::delay_offset_name(id), zero_scalar);
        }
        // mc_params: bind to body inputs (per-draw bcasts in vmap).
        for nm in mc_param_names {
            bindings.insert((*nm).to_string(), mc_input_ids[*nm]);
        }
        // Non-mc params: source still has them as Op::Param; let
        // inline_into re-emit them as Op::Param in body. Caller will
        // bind via set_param at compile time.

        let outs = rlx_opt::inline_into(&mut body, bwd, &bindings, None)
            .map_err(|e| format!("inline_into row {i}: {e}"))?;
        // outs[0] = r_i at zero point = -c[i] → c_i = -outs[0]
        let neg_one = body.add_node(
            Op::Constant { data: (-1.0_f32).to_le_bytes().to_vec() },
            vec![], s_scalar.clone(),
        );
        let c_i = body.binary(BinaryOp::Mul, neg_one, outs[0], s_scalar.clone());
        c_rows.push(c_i);
        // outs[1..1+n_v] = K row i
        let mut k_row = Vec::with_capacity(n_v);
        for j in 0..n_v {
            k_row.push(outs[1 + j]);
        }
        k_rows.push(k_row);
        // outs[1+n_v..1+2*n_v] = -M_b row i  →  M_b row = -outs
        let mut mb_row = Vec::with_capacity(n_v);
        for j in 0..n_v {
            let neg = body.binary(BinaryOp::Mul, neg_one, outs[1 + n_v + j], s_scalar.clone());
            mb_row.push(neg);
        }
        mb_rows.push(mb_row);
    }

    // Compose K matrix [n, n] and M_b matrix [n, n] from rows.
    // Each k_rows[i][j] is shape [1]; concat along axis 0 gives [n]
    // for one row; reshape to [1, n]; concat all rows along axis 0
    // to get [n, n].
    let k_mat = stack_scalars_into_matrix(&mut body, &k_rows, n);
    let mb_mat = stack_scalars_into_matrix(&mut body, &mb_rows, n);
    let c_vec = stack_scalars_into_vector(&mut body, &c_rows, n);

    // v_new = K^-1 · (M_b · carry + c). Two paths:
    //   n == 1:  scalar — use binary div directly (works on both
    //            rlx-cpu (F32) and rlx-mlx (F32) backends).
    //   n  > 1:  build b as M_b · carry + c, then `dense_solve(K, b)`.
    //            Works on MLX (F32 supported); rlx-cpu lacks F32
    //            DenseSolve today, so unit tests at n > 1 require MLX.
    let v_new = if n == 1 {
        // K, M_b, c are all [1] scalars in this case.
        let k_scalar = k_rows[0][0];
        let mb_scalar = mb_rows[0][0];
        let c_scalar = c_rows[0];
        let mb_carry = body.binary(BinaryOp::Mul, mb_scalar, carry, s_vec.clone());
        let b_scalar = body.binary(BinaryOp::Add, mb_carry, c_scalar, s_vec.clone());
        body.binary(BinaryOp::Div, b_scalar, k_scalar, s_vec.clone())
    } else {
        let carry_col = body.reshape(carry, vec![n as i64, 1], s_col.clone());
        let mb_carry_col = body.matmul(mb_mat, carry_col, s_col.clone());
        let mb_carry = body.reshape(mb_carry_col, vec![n as i64], s_vec.clone());
        let b_vec = body.binary(BinaryOp::Add, mb_carry, c_vec, s_vec.clone());
        body.dense_solve(k_mat, b_vec, s_vec.clone())
    };
    body.set_outputs(vec![v_new]);

    // Bind non-mc params on subsequent compile via set_param —
    // caller's job. Param re-emit happens inside inline_into when
    // param_bindings is None, so the body has the right Op::Param
    // slots.
    let _ = effective_params;    // recorded for documentation; bind happens externally
    let _ = mc_set;
    let _ = ReduceOp::Sum;       // suppress unused-import lint

    Ok(LinearBeStepBody {
        body, n, unknowns, branches,
    })
}

fn stack_scalars_into_vector(g: &mut Graph, scalars: &[NodeId], n: usize) -> NodeId {
    debug_assert_eq!(scalars.len(), n);
    if n == 1 {
        return scalars[0];
    }
    g.add_node(
        Op::Concat { axis: 0 },
        scalars.to_vec(),
        Shape::new(&[n], DType::F32),
    )
}

fn stack_scalars_into_matrix(g: &mut Graph, rows: &[Vec<NodeId>], n: usize) -> NodeId {
    debug_assert_eq!(rows.len(), n);
    let s_row    = Shape::new(&[n],     DType::F32);
    let s_row_2d = Shape::new(&[1, n], DType::F32);
    // Concat each row's scalars into a [n] vector, reshape to [1, n].
    let mut row_2ds: Vec<NodeId> = Vec::with_capacity(n);
    for i in 0..n {
        debug_assert_eq!(rows[i].len(), n);
        let row_1d = if n == 1 {
            rows[i][0]
        } else {
            g.add_node(
                Op::Concat { axis: 0 },
                rows[i].clone(),
                s_row.clone(),
            )
        };
        let row_2d = g.reshape(row_1d, vec![1, n as i64], s_row_2d.clone());
        row_2ds.push(row_2d);
    }
    if n == 1 {
        return row_2ds[0];
    }
    g.add_node(
        Op::Concat { axis: 0 },
        row_2ds,
        Shape::new(&[n, n], DType::F32),
    )
}

/// Unified nonlinear scan body builder. Handles every dimension we
/// support: any number of unknowns, optional per-draw `mc_params`,
/// optional boundary voltages exposed as body `Op::Input`s instead
/// of bound to 0.
///
/// Body interface (Op::Input declaration order; matters for vmap'd
/// Scan bcasts):
///   `[carry, *mc_param_names, *boundary_input_names]`
///
/// All extras are scalar `[1]` and become per-draw bcasts after
/// vmap. The body output (next carry) is shape `[n]`.
///
/// Per Newton iter inside the body:
///   1. For each output row i, inline the grad graph (residual
///      restricted to row i, differentiated wrt the unknown v_<id>
///      inputs). v_<unknown_j> binds to a slice of `v_iter`,
///      v_prev_<unknown_j> binds to a slice of `carry`, mc_params
///      bind to body inputs, boundary v_<id> binds to body inputs
///      (if listed) or to the const-0 slot.
///   2. Stack r [n] from per-row outputs, K [n,n] from per-row K
///      slices.
///   3. dv = K⁻¹·(-r) via Op::DenseSolve (n>1) or Binary::Div (n=1).
///   4. v_iter += dv.
///
/// After `fixed_iters`, `v_iter` becomes the next carry. Wasteful
/// for circuits that converge in 2 iters but always pays
/// `fixed_iters`; conservative for hard cases. Variable iter count
/// would need `Op::While` lowering on rlx-mlx (multi-week).
///
/// Note: n>1 uses Op::DenseSolve which is F32-MLX-only today
/// (rlx-cpu has F32 path missing). Single-unknown bodies work on
/// both backends via the binary-div fast path.
pub fn build_nonlinear_scan_body(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    mc_param_names: &[&str],
    boundary_input_names: &[&str],
    dt: f32,
    fixed_iters: usize,
) -> Result<LinearBeStepBody, String> {
    use std::collections::HashSet;
    assert!(fixed_iters >= 1, "fixed_iters must be ≥ 1");

    let scalar_orig = build_be_step_residual_graph(circuit);
    let unknowns = scalar_orig.unknown_nets.clone();
    let branches = scalar_orig.branches.clone();
    let n_v = unknowns.len();
    let n_b = branches.len();
    let n = n_v + n_b;
    if n_v == 0 {
        return Err("circuit has no unknowns".into());
    }
    if n_b > 0 {
        return Err(format!(
            "build_nonlinear_scan_body: branches not yet supported \
             (circuit has {n_b} branch unknown(s))."
        ));
    }

    // Promote mc_params to Inputs in the residual graph (so the
    // grad/inline path treats them as bindable inputs, not Params
    // baked at compile).
    let scalar_promoted = if mc_param_names.is_empty() {
        scalar_orig.graph.clone()
    } else {
        rlx_opt::promote_params_to_inputs(&scalar_orig.graph, mc_param_names)
    };

    let unknown_input_ids: Vec<NodeId> = unknowns.iter()
        .map(|net| find_input_node(&scalar_promoted, &net_input_name(*net))
            .ok_or_else(|| format!("residual missing v_{}", net.0)))
        .collect::<Result<_, _>>()?;

    // Build per-row grad graphs: residual restricted to output i,
    // differentiated wrt all unknown v_<id> inputs (NOT v_prev — we
    // don't need those since we substitute carry directly).
    let mut row_grad_graphs: Vec<Graph> = Vec::with_capacity(n_v);
    for i in 0..n_v {
        let mut g_i = scalar_promoted.clone();
        let out_i = g_i.outputs[i];
        g_i.set_outputs(vec![out_i]);
        let bwd = rlx_opt::autodiff::grad_with_loss(&g_i, &unknown_input_ids);
        // outputs: [r_i, dr_i/dv_0, ..., dr_i/dv_{n_v-1}]
        row_grad_graphs.push(bwd);
    }

    // ── Build the body. ──
    let s_scalar = Shape::new(&[1], DType::F32);
    let s_vec    = Shape::new(&[n_v], DType::F32);
    let s_col    = Shape::new(&[n_v, 1], DType::F32);
    let mut body = Graph::new("nonlinear_scan_body");

    // Inputs in declaration order: carry first, then mc_params,
    // then boundary inputs (matches scan Op::Input ordering for
    // [carry, *bcasts]).
    let carry = body.input("carry", s_vec.clone());
    let mut mc_input_ids: HashMap<String, NodeId> = HashMap::new();
    for nm in mc_param_names {
        mc_input_ids.insert(
            (*nm).to_string(),
            body.input((*nm).to_string(), s_scalar.clone()),
        );
    }
    let bnd_set: HashSet<&str> = boundary_input_names.iter().copied().collect();
    let mut boundary_input_ids: HashMap<String, NodeId> = HashMap::new();
    for nm in boundary_input_names {
        boundary_input_ids.insert(
            (*nm).to_string(),
            body.input((*nm).to_string(), s_scalar.clone()),
        );
    }

    // Pre-emit each non-mc Op::Param ONCE so duplicate inlines share
    // the slot (set_param hits one node).
    let mc_set: HashSet<&str> = mc_param_names.iter().copied().collect();
    let mut param_bindings: HashMap<String, NodeId> = HashMap::new();
    for n in scalar_promoted.nodes() {
        if let Op::Param { name } = &n.op {
            if mc_set.contains(name.as_str()) { continue; }
            param_bindings.entry(name.clone()).or_insert_with(|| {
                body.param(name.clone(), n.shape.clone())
            });
        }
    }

    // Common constants.
    let zero_scalar = body.add_node(
        Op::Constant { data: 0.0_f32.to_le_bytes().to_vec() },
        vec![], s_scalar.clone(),
    );
    let one_scalar = body.add_node(
        Op::Constant { data: 1.0_f32.to_le_bytes().to_vec() },
        vec![], s_scalar.clone(),
    );
    let neg_one_scalar = body.add_node(
        Op::Constant { data: (-1.0_f32).to_le_bytes().to_vec() },
        vec![], s_scalar.clone(),
    );
    let dt_scalar = body.add_node(
        Op::Constant { data: dt.to_le_bytes().to_vec() },
        vec![], s_scalar.clone(),
    );

    // Helper: slice a [n_v] vector at index j into a [1] scalar.
    // Op::Narrow returns a [1]-shape view; reshape collapses it.
    let slice_scalar = |body: &mut Graph, vec_node: NodeId, j: usize| -> NodeId {
        let narrow = body.add_node(
            Op::Narrow { axis: 0, start: j, len: 1 },
            vec![vec_node],
            Shape::new(&[1], DType::F32),
        );
        // Already [1] so reshape is a no-op for n_v=1; keep for clarity.
        narrow
    };

    let mut v_iter = carry;     // v_iter is [n_v]; for n_v=1 it's [1].

    for _k in 0..fixed_iters {
        // Slice v_iter and carry at each unknown index.
        let v_iter_slices: Vec<NodeId> = (0..n_v)
            .map(|j| slice_scalar(&mut body, v_iter, j))
            .collect();
        let carry_slices: Vec<NodeId> = (0..n_v)
            .map(|j| slice_scalar(&mut body, carry, j))
            .collect();

        // Inline each row's grad graph; collect r and K row.
        let mut r_scalars: Vec<NodeId> = Vec::with_capacity(n_v);
        let mut k_rows: Vec<Vec<NodeId>> = Vec::with_capacity(n_v);
        for i in 0..n_v {
            let mut bindings: HashMap<String, NodeId> = HashMap::new();
            // Per-net v_<id> + v_prev_<id> bindings. Unknowns get
            // v_iter / carry slices; boundaries get either body
            // inputs (if listed) or 0.
            for net in &scalar_orig.all_nets {
                let v_name = net_input_name(*net);
                let vp_name = prev_voltage_input_name(*net);
                // Find this net's index in unknowns (if any).
                let unknown_idx = unknowns.iter().position(|u| u == net);
                if let Some(j) = unknown_idx {
                    bindings.insert(v_name,  v_iter_slices[j]);
                    bindings.insert(vp_name, carry_slices[j]);
                } else {
                    // Boundary or other-net — bind to body input
                    // (if listed) or to constant 0.
                    let bnd_node = if bnd_set.contains(v_name.as_str()) {
                        boundary_input_ids[&v_name]
                    } else {
                        zero_scalar
                    };
                    bindings.insert(v_name.clone(),  bnd_node);
                    // v_prev for boundary = same as v (boundary doesn't move).
                    bindings.insert(vp_name, bnd_node);
                }
            }
            for b in &branches {
                bindings.insert(branch_input_name(*b), zero_scalar);
            }
            bindings.insert(TIMESTEP_INPUT_NAME.to_string(), dt_scalar);
            bindings.insert("d_output".to_string(),          one_scalar);
            for (idx, _) in circuit.delays.iter().enumerate() {
                let id = crate::DelayId(idx as u32);
                bindings.insert(crate::delay_v_lo_name(id),   zero_scalar);
                bindings.insert(crate::delay_v_hi_name(id),   zero_scalar);
                bindings.insert(crate::delay_blend_name(id),  zero_scalar);
                bindings.insert(crate::delay_offset_name(id), zero_scalar);
            }
            for nm in mc_param_names {
                bindings.insert((*nm).to_string(), mc_input_ids[*nm]);
            }
            let outs = rlx_opt::inline_into(
                &mut body, &row_grad_graphs[i], &bindings, Some(&param_bindings),
            ).map_err(|e| format!("inline row {i}: {e}"))?;
            r_scalars.push(outs[0]);
            let mut k_row = Vec::with_capacity(n_v);
            for j in 0..n_v {
                k_row.push(outs[1 + j]);
            }
            k_rows.push(k_row);
        }

        // Compose r [n_v] and K [n_v, n_v]; solve K·dv = -r; update.
        let dv = if n_v == 1 {
            // Scalar fast path: dv = -r / K
            let neg_r = body.binary(BinaryOp::Mul, neg_one_scalar, r_scalars[0], s_scalar.clone());
            body.binary(BinaryOp::Div, neg_r, k_rows[0][0], s_scalar.clone())
        } else {
            let r_vec = stack_scalars_into_vector(&mut body, &r_scalars, n_v);
            let k_mat = stack_scalars_into_matrix(&mut body, &k_rows, n_v);
            let neg_r = body.binary(BinaryOp::Mul, neg_one_scalar, r_vec, s_vec.clone());
            // Op::DenseSolve(K, -r) → dv [n_v]
            body.dense_solve(k_mat, neg_r, s_vec.clone())
        };

        v_iter = body.binary(BinaryOp::Add, v_iter, dv, s_vec.clone());
    }

    body.set_outputs(vec![v_iter]);
    let _ = s_col; let _ = mc_set; let _ = bnd_set;

    Ok(LinearBeStepBody {
        body, n: n_v, unknowns, branches,
    })
}

/// Build a scan body for a **nonlinear** circuit by unrolling
/// `fixed_iters` Newton iterations inline.
///
/// Each per-step Newton iter:
///   `dv_k = -K(v_k, v_prev)⁻¹ · r(v_k, v_prev)`
///   `v_{k+1} = v_k + dv_k`
///
/// starting at `v_0 = v_prev` (continuity guess). After `fixed_iters`
/// iters, the last `v_k` becomes the next carry.
///
/// Wasteful for circuits that converge in 2 iters but always run
/// `fixed_iters`; conservative for circuits where convergence is
/// hard. The natural cure (variable iter count via `Op::While`) is
/// a multi-week project on rlx-mlx — fixed-iter is the MVP that
/// proves the architecture.
///
/// ## Scope (this MVP)
///
/// * **Single unknown net** only (`n_v == 1`). Multi-unknown is
///   the same shape but needs `Op::DenseSolve` per Newton iter and
///   F32 DenseSolve on whichever backend you run on (MLX ✓; rlx-cpu
///   F32 ⬜ upstream).
/// * **No branches**. Same restriction as the linear builder.
/// * **Params bound at build time**: pass via `params`. Per-draw MC
///   over device params is the same recipe as
///   [`build_linear_be_step_body_with_mc_params`]: promote them to
///   Inputs and bind body `Op::Input`s at vmap time. Not wired in
///   this MVP.
pub fn build_nonlinear_be_step_body(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    dt: f32,
    fixed_iters: usize,
) -> Result<LinearBeStepBody, String> {
    assert!(fixed_iters >= 1, "fixed_iters must be ≥ 1");

    let scalar_rg = build_be_step_residual_graph(circuit);
    let unknowns = scalar_rg.unknown_nets.clone();
    let branches = scalar_rg.branches.clone();
    let n_v = unknowns.len();
    let n_b = branches.len();
    if n_v != 1 || n_b != 0 {
        return Err(format!(
            "build_nonlinear_be_step_body: MVP supports n_v=1 + no \
             branches; got n_v={n_v}, n_b={n_b}"
        ));
    }
    let unknown_input_id = find_input_node(
        &scalar_rg.graph, &net_input_name(unknowns[0]),
    ).ok_or("residual missing v_<unknown>")?;
    let prev_input_id = find_input_node(
        &scalar_rg.graph, &prev_voltage_input_name(unknowns[0]),
    ).ok_or("residual missing v_prev_<unknown>")?;

    // Build grad graph: residual (output 0) + dr/dv_unknown (output 1).
    let mut g_for_grad = scalar_rg.graph.clone();
    let out0 = g_for_grad.outputs[0];
    g_for_grad.set_outputs(vec![out0]);
    let grad_graph = rlx_opt::autodiff::grad_with_loss(
        &g_for_grad, &[unknown_input_id],
    );
    // grad_graph outputs: [r(v, v_prev), dr/dv_unknown]

    // ── Build body. ──
    let s_scalar = Shape::new(&[1], DType::F32);
    let mut body = Graph::new("nonlinear_be_body_scan");
    let carry = body.input("carry", s_scalar.clone());

    // Backfill <name>_tau defaults.
    let mut effective_params = params.clone();
    for dl in &circuit.delays {
        let key = format!("{}_tau", dl.device.name());
        effective_params.entry(key).or_insert(
            dl.device.delay_seconds() as f32,
        );
    }

    // Pre-emit each Op::Param ONCE in the body so multiple inlines
    // share the same NodeId — otherwise we'd get duplicate-named
    // Op::Param nodes and `set_param` would only bind one.
    let mut param_bindings: HashMap<String, NodeId> = HashMap::new();
    for n in scalar_rg.graph.nodes() {
        if let Op::Param { name } = &n.op {
            param_bindings.entry(name.clone()).or_insert_with(|| {
                body.param(name.clone(), n.shape.clone())
            });
        }
    }

    // Common constants used across Newton iters.
    let zero_scalar = body.add_node(
        Op::Constant { data: 0.0_f32.to_le_bytes().to_vec() },
        vec![], s_scalar.clone(),
    );
    let one_scalar = body.add_node(
        Op::Constant { data: 1.0_f32.to_le_bytes().to_vec() },
        vec![], s_scalar.clone(),
    );
    let neg_one_scalar = body.add_node(
        Op::Constant { data: (-1.0_f32).to_le_bytes().to_vec() },
        vec![], s_scalar.clone(),
    );
    let dt_scalar = body.add_node(
        Op::Constant { data: dt.to_le_bytes().to_vec() },
        vec![], s_scalar.clone(),
    );

    // Newton continuation: start at carry (v_prev as initial guess).
    let mut v_iter = carry;

    for _k in 0..fixed_iters {
        // Build input bindings for inlining grad_graph at this iter.
        let mut bindings: HashMap<String, NodeId> = HashMap::new();
        // All other v_<id> inputs (boundary + grounded) → 0.
        for net in &scalar_rg.all_nets {
            if find_input_node(&scalar_rg.graph, &net_input_name(*net))
                .map(|id| id == unknown_input_id).unwrap_or(false)
            {
                bindings.insert(net_input_name(*net), v_iter);
            } else {
                bindings.insert(net_input_name(*net), zero_scalar);
            }
            // v_prev for our unknown → carry; everywhere else → 0.
            if find_input_node(&scalar_rg.graph, &prev_voltage_input_name(*net))
                .map(|id| id == prev_input_id).unwrap_or(false)
            {
                bindings.insert(prev_voltage_input_name(*net), carry);
            } else {
                bindings.insert(prev_voltage_input_name(*net), zero_scalar);
            }
        }
        for b in &branches {
            bindings.insert(branch_input_name(*b), zero_scalar);
        }
        bindings.insert(TIMESTEP_INPUT_NAME.to_string(), dt_scalar);
        bindings.insert("d_output".to_string(),          one_scalar);
        for (idx, _) in circuit.delays.iter().enumerate() {
            let id = crate::DelayId(idx as u32);
            bindings.insert(crate::delay_v_lo_name(id),   zero_scalar);
            bindings.insert(crate::delay_v_hi_name(id),   zero_scalar);
            bindings.insert(crate::delay_blend_name(id),  zero_scalar);
            bindings.insert(crate::delay_offset_name(id), zero_scalar);
        }

        let outs = rlx_opt::inline_into(
            &mut body, &grad_graph, &bindings, Some(&param_bindings),
        ).map_err(|e| format!("inline_into Newton iter: {e}"))?;
        let r_at_iter = outs[0];
        let k_at_iter = outs[1];

        // dv = -r / K
        let neg_r = body.binary(BinaryOp::Mul, neg_one_scalar, r_at_iter, s_scalar.clone());
        let dv = body.binary(BinaryOp::Div, neg_r, k_at_iter, s_scalar.clone());
        // v_{k+1} = v_k + dv
        v_iter = body.binary(BinaryOp::Add, v_iter, dv, s_scalar.clone());
    }

    body.set_outputs(vec![v_iter]);

    // Drop unused locals to silence potential warnings on the
    // backfill backplane that's read only for its side effect of
    // populating `effective_params` (we bind via set_param later;
    // for the body we only need the param NodeIds).
    let _ = effective_params;

    Ok(LinearBeStepBody {
        body, n: 1, unknowns, branches,
    })
}

pub fn build_linear_be_step_body(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    dt: f32,
) -> Result<LinearBeStepBody, String> {
    use rlx_runtime::{Device, Session};

    let rg = build_be_step_residual_graph(circuit);
    let unknowns = rg.unknown_nets.clone();
    let branches = rg.branches.clone();
    let n_v = unknowns.len();
    let n_b = branches.len();
    let n = n_v + n_b;
    if n == 0 {
        return Err("circuit has no unknowns".into());
    }

    // Collect the Op::Input NodeIds we need to bind / differentiate
    // against. v_<id> for unknowns, v_prev_<id> for ALL nets (boundary
    // + unknown), branch i_b<id>, plus h.
    let unknown_input_ids: Vec<NodeId> = unknowns.iter()
        .map(|net| find_input_node(&rg.graph, &net_input_name(*net))
            .ok_or_else(|| format!("residual missing v_{}", net.0)))
        .collect::<Result<_, _>>()?;
    let prev_input_ids: Vec<NodeId> = unknowns.iter()
        .map(|net| find_input_node(&rg.graph, &prev_voltage_input_name(*net))
            .ok_or_else(|| format!("residual missing v_prev_{}", net.0)))
        .collect::<Result<_, _>>()?;
    let branch_input_ids: Vec<NodeId> = branches.iter()
        .map(|b| find_input_node(&rg.graph, &branch_input_name(*b))
            .ok_or_else(|| format!("residual missing i_b{}", b.0)))
        .collect::<Result<_, _>>()?;
    if !branches.is_empty() {
        return Err(format!(
            "build_linear_be_step_body: branches not yet supported \
             (circuit has {} branch unknown(s)). Open follow-up to extend \
             K/M_b extraction over the [n_v + n_b] system.", branches.len()));
    }

    // ── Helpers to evaluate / differentiate the residual at chosen
    //    (v, v_prev) bindings, with all params backfilled. ──

    // Backfill <name>_tau defaults for delays — same as solve_be_step.
    let mut effective_params = params.clone();
    for dl in &circuit.delays {
        let key = format!("{}_tau", dl.device.name());
        effective_params.entry(key).or_insert(
            dl.device.delay_seconds() as f32,
        );
    }

    // Build N graphs: residual restricted to output i, with grads
    // wrt all v_<unknown> AND all v_prev_<unknown> inputs. One
    // compile each — we reuse them across K and M_b extraction.
    let session = Session::new(Device::Cpu);
    let mut grad_targets: Vec<NodeId> = unknown_input_ids.clone();
    grad_targets.extend(prev_input_ids.iter().copied());
    grad_targets.extend(branch_input_ids.iter().copied());
    let mut compiled_jac_rows: Vec<rlx_runtime::CompiledGraph> = Vec::with_capacity(n);
    for i in 0..n {
        let mut g_i = rg.graph.clone();
        let out_i = g_i.outputs[i];
        g_i.set_outputs(vec![out_i]);
        let bwd = rlx_opt::autodiff::grad_with_loss(&g_i, &grad_targets);
        let mut compiled = session.compile(bwd);
        for (k, v) in &effective_params {
            compiled.set_param(k, &[*v]);
        }
        compiled_jac_rows.push(compiled);
    }

    // Run each row at v=0, v_prev=0 (everything zero). The graph has:
    //   outputs: [r_i, dr_i/dv_0, ..., dr_i/dv_{n_v-1},
    //             dr_i/dv_prev_0, ..., dr_i/dv_prev_{n_v-1}]
    let zero = [0.0_f32];
    let dt_arr = [dt];
    let one = [1.0_f32];
    let mut k_mat   = vec![0.0_f32; n * n];      // K[i, j]
    let mut m_b_mat = vec![0.0_f32; n * n];      // M_b[i, j]
    let mut c_vec   = vec![0.0_f32; n];          // c[i]
    for i in 0..n {
        // Build the input list: every v_<id> = 0, every v_prev_<id> = 0,
        // h = dt, d_output = 1, plus delay scalars at zero defaults.
        let mut inputs: Vec<(String, Vec<f32>)> = Vec::new();
        for net in &rg.all_nets {
            inputs.push((net_input_name(*net), zero.to_vec()));
            inputs.push((prev_voltage_input_name(*net), zero.to_vec()));
        }
        for b in &branches {
            inputs.push((branch_input_name(*b), zero.to_vec()));
        }
        inputs.push((TIMESTEP_INPUT_NAME.to_string(), dt_arr.to_vec()));
        inputs.push(("d_output".to_string(), one.to_vec()));
        // Delay scalars (zero-history defaults).
        for (idx, _dl) in circuit.delays.iter().enumerate() {
            let id = crate::DelayId(idx as u32);
            inputs.push((crate::delay_v_lo_name(id),  zero.to_vec()));
            inputs.push((crate::delay_v_hi_name(id),  zero.to_vec()));
            inputs.push((crate::delay_blend_name(id), zero.to_vec()));
            inputs.push((crate::delay_offset_name(id), zero.to_vec()));
        }

        let inputs_ref: Vec<(&str, &[f32])> = inputs.iter()
            .map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        let outs = compiled_jac_rows[i].run(&inputs_ref);

        // outs[0] = r_i at zero point = -c[i]
        c_vec[i] = -outs[0][0];

        // outs[1..1+n_v] = dr_i/dv_unknown_j → row i of K
        for j in 0..n_v {
            k_mat[i * n + j] = outs[1 + j][0];
        }
        // outs[1+n_v..1+2*n_v] = dr_i/dv_prev_j → row i of M_b but
        // sign-inverted: r = K v - b(v_prev), so dr/dv_prev = -dM_b/dv_prev.
        // For linear b = M_b · v_prev + c, dr/dv_prev_j = -M_b[i,j].
        for j in 0..n_v {
            m_b_mat[i * n + j] = -outs[1 + n_v + j][0];
        }
        // Branch unknowns intentionally unhandled (asserted out above).
    }

    // ── Solve K · step_matrix = M_b and K · step_const = c ──
    let step_matrix = solve_columns(&k_mat, &m_b_mat, n, n)?;
    let step_const  = solve_columns(&k_mat, &c_vec, n, 1)?;

    // ── Build the body graph ──
    let s_vec = Shape::new(&[n], DType::F32);
    let s_col = Shape::new(&[n, 1], DType::F32);
    let mut body = Graph::new("linear_be_body");
    let carry = body.input("carry", s_vec.clone());

    let step_mat_node = const_mat(&mut body, &step_matrix, n, n);
    let step_const_node = const_vec(&mut body, &step_const, n);

    // matmul([n,n], [n,1]) → [n,1] then reshape → [n].
    let carry_col = body.reshape(
        carry, vec![n as i64, 1], s_col.clone(),
    );
    let prod_col = body.matmul(step_mat_node, carry_col, s_col.clone());
    let prod_vec = body.reshape(prod_col, vec![n as i64], s_vec.clone());
    let v_new = body.binary(BinaryOp::Add, prod_vec, step_const_node, s_vec);
    body.set_outputs(vec![v_new]);

    Ok(LinearBeStepBody {
        body, n, unknowns, branches,
    })
}

// ── Helpers ───────────────────────────────────────────────────────

fn const_mat(g: &mut Graph, data: &[f32], rows: usize, cols: usize) -> NodeId {
    debug_assert_eq!(data.len(), rows * cols);
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for x in data { bytes.extend_from_slice(&x.to_le_bytes()); }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[rows, cols], DType::F32),
    )
}

fn const_vec(g: &mut Graph, data: &[f32], n: usize) -> NodeId {
    debug_assert_eq!(data.len(), n);
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for x in data { bytes.extend_from_slice(&x.to_le_bytes()); }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[n], DType::F32),
    )
}

/// Column-by-column solve K · X = B for X. X has `n_cols` columns;
/// for `c` (single RHS), call with n_cols=1. Uses the crate-private
/// `gauss_jordan_solve` from the parent module — same numerical
/// kernel scalar `solve_dc` falls back to.
fn solve_columns(
    k: &[f32], b: &[f32], n: usize, n_cols: usize,
) -> Result<Vec<f32>, String> {
    let mut out = vec![0.0_f32; n * n_cols];
    let mut col = vec![0.0_f32; n];
    for c in 0..n_cols {
        for i in 0..n {
            col[i] = b[i * n_cols + c];
        }
        let x = super::gauss_jordan_solve(k, &col, n)
            .ok_or_else(|| format!(
                "K is singular at column {c} — circuit's BE-step matrix \
                 won't invert at v=0. Either non-physical or non-linear."))?;
        for i in 0..n {
            out[i * n_cols + c] = x[i];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eda_hir::Block;
    use spike_divider_block::{Capacitor, Resistor};

    #[test]
    fn unified_nonlinear_body_with_mc_param_and_boundary_input_matches_scalar() {
        use spike_divider_block::Diode;
        // Same RC + diode topology but with V_in as boundary input
        // AND R as mc_param. For each (V_in, R) pair, run one BE
        // step from a known v_prev and check parity vs scalar
        // solve_be_step.
        let c_farads = 1e-9_f32;
        let dt       = 1e-9_f32;
        let v_prev_val = 0.7_f32;

        let mut c = Circuit::new();
        let v_in = c.alloc_boundary_net();
        let vmid = c.alloc_unknown_net();
        let r   = Resistor { length: 10_000, id: "Rmid".into() };
        let cap = Capacitor { plate_size: 2_000, id: "Cmid".into() };
        let d   = Diode { size: 2_000, is_value: 1e-15, id: "Dmid".into() };
        c.add_device(r.clone(),    &[v_in, vmid]);
        c.add_storage(cap.clone(), [vmid, NetId::GND]);
        c.add_device(d.clone(),    &[vmid, NetId::GND]);

        let r_key = Block::name(&r);
        let c_key = format!("{}_C", Block::name(&cap));
        let d_is_key = format!("{}_Is", Block::name(&d));
        let v_in_input_name = net_input_name(v_in);

        let mut params = std::collections::HashMap::new();
        params.insert(c_key.clone(), c_farads);
        params.insert(d_is_key.clone(), 1e-15_f32);
        // R is mc'd, NOT in `params`.

        let body = build_nonlinear_scan_body(
            &c, &params,
            /*mc_param_names=*/        &[r_key.as_str()],
            /*boundary_input_names=*/  &[v_in_input_name.as_str()],
            dt, /*fixed_iters=*/ 12,
        ).expect("unified nonlinear body");
        assert_eq!(body.n, 1);

        use rlx_runtime::{Device, Session};
        let mut compiled = Session::new(Device::Cpu).compile(body.body);
        for (k, v) in &params {
            compiled.set_param(k, &[*v]);
        }

        // Test grid: 3 (V_in, R) combinations.
        for &(v_in_val, r_value) in &[
            (0.0_f32, 1_000.0_f32),
            (1.0,     500.0),
            (1.5,     2_000.0),
        ] {
            let outs = compiled.run(&[
                ("carry",                       &[v_prev_val]),
                (r_key.as_str(),                &[r_value]),
                (v_in_input_name.as_str(),      &[v_in_val]),
            ]);
            let v_new_scan = outs[0][0];

            // Scalar reference: per-config solve_be_step.
            use crate::{NewtonOptions, solve_be_step};
            let mut p_scalar = params.clone();
            p_scalar.insert(r_key.clone(), r_value);
            let mut prev: HashMap<NetId, f32> = HashMap::new();
            prev.insert(vmid, v_prev_val);
            let mut bnd: HashMap<NetId, f32> = HashMap::new();
            bnd.insert(v_in, v_in_val);
            let opt = NewtonOptions { init: v_prev_val, ..NewtonOptions::default() };
            let scalar_step = solve_be_step(&c, &p_scalar, &bnd, &prev, &[], dt, opt);
            assert!(scalar_step.converged,
                "scalar (V_in={v_in_val} R={r_value}) didn't converge");
            let v_new_scalar = scalar_step.voltages[&vmid];
            let drift = (v_new_scan - v_new_scalar).abs();
            assert!(
                drift < 5e-4,
                "(V_in={v_in_val}, R={r_value}): scan v_new={v_new_scan} \
                 scalar v_new={v_new_scalar} (Δ {drift:.3e})",
            );
        }
    }

    #[test]
    fn nonlinear_body_rc_plus_diode_one_step_matches_scalar_solve() {
        use spike_divider_block::Diode;
        // RC with shunt diode at the unknown node.
        //   gnd -- R -- vmid -- C -- gnd
        //                   `-- D --  gnd
        // Per-step BE residual at vmid is nonlinear in v due to the
        // diode. One scan-body step with fixed_iters Newton iters
        // should match scalar solve_be_step for one step from the
        // same v_prev.
        let r_ohms   = 1_000.0_f32;
        let c_farads = 1e-9_f32;
        let dt       = 1e-9_f32;        // 1 ns
        let v_prev_val = 0.7_f32;       // diode forward-biased

        let mut c = Circuit::new();
        let v_in = c.alloc_boundary_net();
        let vmid = c.alloc_unknown_net();
        let r   = Resistor { length: 10_000, id: "R".into() };
        let cap = Capacitor { plate_size: 2_000, id: "C".into() };
        let d   = Diode { size: 2_000, is_value: 1e-15, id: "D".into() };
        c.add_device(r.clone(),    &[v_in, vmid]);
        c.add_storage(cap.clone(), [vmid, NetId::GND]);
        c.add_device(d.clone(),    &[vmid, NetId::GND]);

        let mut params = std::collections::HashMap::new();
        params.insert(Block::name(&r), r_ohms);
        params.insert(format!("{}_C", Block::name(&cap)), c_farads);
        params.insert(format!("{}_Is", Block::name(&d)), 1e-15_f32);

        let body = build_nonlinear_be_step_body(&c, &params, dt, /*fixed_iters=*/ 12)
            .expect("nonlinear body");
        assert_eq!(body.n, 1);

        // Compile body. Bind ALL params (including those re-emitted
        // inside the inlined grad graph). The Resistor + Diode +
        // Capacitor each re-emit Op::Param via the inlined residual
        // graph; param_bindings dedupes so set_param hits one slot.
        use rlx_runtime::{Device, Session};
        let mut compiled = Session::new(Device::Cpu).compile(body.body);
        for (k, v) in &params {
            compiled.set_param(k, &[*v]);
        }
        // The boundary v_in shows up as v_<id> in the inlined graph;
        // we hard-bound it to 0 inside the body. Need to set the
        // boundary's actual voltage somehow — for this test we
        // bound v_in implicitly via the Op::Constant 0 binding, so
        // V_in = 0. With V_in = 0 the diode pulls vmid down toward
        // 0; at v_prev = 0.7 V the cap discharges.
        // (Generalising binding to per-iter boundary is followup.)

        let outs = compiled.run(&[("carry", &[v_prev_val])]);
        let v_new_scan = outs[0][0];

        // Scalar reference via solve_be_step.
        use crate::{NewtonOptions, solve_be_step};
        let mut prev: HashMap<NetId, f32> = HashMap::new();
        prev.insert(vmid, v_prev_val);
        let mut bnd: HashMap<NetId, f32> = HashMap::new();
        bnd.insert(v_in, 0.0_f32);
        let opt = NewtonOptions { init: v_prev_val, ..NewtonOptions::default() };
        let scalar_step = solve_be_step(&c, &params, &bnd, &prev, &[], dt, opt);
        assert!(scalar_step.converged, "scalar solve_be_step didn't converge");
        let v_new_scalar = scalar_step.voltages[&vmid];

        let drift = (v_new_scan - v_new_scalar).abs();
        assert!(
            drift < 5e-4,
            "nonlinear body v_new {v_new_scan} vs scalar {v_new_scalar} \
             (Δ {drift:.3e})",
        );
    }

    #[test]
    fn auto_body_with_mc_params_supports_per_draw_R() {
        // RC discharge with R as a per-draw mc_param. Body should
        // expose "R1" as an Op::Input (bcast in vmap'd Scan), and
        // re-running it at different R values should give different
        // step coefficients matching the closed form.
        let r_nominal = 1_000.0_f32;
        let c_farads  = 1e-9_f32;
        let dt        = r_nominal * c_farads / 50.0;

        let mut c = Circuit::new();
        let vmid = c.alloc_unknown_net();
        let r   = Resistor { length: 10_000, id: "R1".into() };
        let cap = Capacitor { plate_size: 2_000, id: "C1".into() };
        c.add_device(r.clone(),    &[vmid, NetId::GND]);
        c.add_storage(cap.clone(), [vmid, NetId::GND]);

        let mut params = std::collections::HashMap::new();
        // R is MC'd, but include it in `params` for the non-mc-path
        // assertion below; the with-mc path skips it via mc_set.
        let r_key = Block::name(&r);
        let c_key = format!("{}_C", Block::name(&cap));
        params.insert(c_key.clone(), c_farads);

        let body = build_linear_be_step_body_with_mc_params(
            &c, &params, &[r_key.as_str()], dt,
        ).expect("auto-build with mc_params");
        assert_eq!(body.n, 1);

        // Run the body at several R values; verify each matches its
        // closed-form step coefficient.
        use rlx_runtime::{Device, Session};
        let mut compiled = Session::new(Device::Cpu).compile(body.body);
        // Bind the non-mc cap value (Op::Param re-emitted by inline).
        compiled.set_param(&c_key, &[c_farads]);

        for r_value in [500.0_f32, 1_000.0, 2_000.0, 5_000.0] {
            let h_over_rc = dt / (r_value * c_farads);
            let expected_step = 1.0 / (1.0 + h_over_rc);
            // Apply v_prev=1 → v_new = step
            let outs = compiled.run(&[
                ("carry", &[1.0_f32]),
                (r_key.as_str(), &[r_value]),
            ]);
            let v_new = outs[0][0];
            let drift = (v_new - expected_step).abs();
            assert!(
                drift < 5e-6,
                "R={r_value}: auto v_new={v_new} expected {expected_step} \
                 (Δ {drift:.3e})",
            );
        }
    }

    #[test]
    fn auto_body_for_rc_matches_closed_form() {
        let r_ohms   = 1_000.0_f32;
        let c_farads = 1e-9_f32;
        let dt       = r_ohms * c_farads / 50.0;
        let h_over_rc = dt / (r_ohms * c_farads);
        let expected_step_coeff = 1.0 / (1.0 + h_over_rc);

        let mut c = Circuit::new();
        let vmid = c.alloc_unknown_net();
        let r   = Resistor { length: 10_000, id: "R1".into() };
        let cap = Capacitor { plate_size: 2_000, id: "C1".into() };
        c.add_device(r.clone(),    &[vmid, NetId::GND]);
        c.add_storage(cap.clone(), [vmid, NetId::GND]);

        let mut params = std::collections::HashMap::new();
        params.insert(Block::name(&r), r_ohms);
        params.insert(format!("{}_C", Block::name(&cap)), c_farads);

        let body = build_linear_be_step_body(&c, &params, dt)
            .expect("auto-build");
        assert_eq!(body.n, 1);
        assert_eq!(body.unknowns, vec![vmid]);

        // Sanity: run the body manually to verify it produces the
        // expected step coefficient. Apply v_prev = 1 → v_new = step.
        use rlx_runtime::{Device, Session};
        let mut compiled = Session::new(Device::Cpu).compile(body.body);
        let outs = compiled.run(&[("carry", &[1.0_f32])]);
        let v_new = outs[0][0];
        let drift = (v_new - expected_step_coeff).abs();
        assert!(
            drift < 1e-6,
            "auto-derived step coeff {v_new} differs from closed-form \
             {expected_step_coeff} (Δ {drift:.3e})",
        );
    }
}

