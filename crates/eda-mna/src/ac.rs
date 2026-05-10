//! Batched small-signal AC analysis on the GPU.
//!
//! For a linear MNA at a DC operating point, the AC response is the
//! solution of the complex linear system
//!
//!   `(G + jω·C) · V(jω) = b_ac`
//!
//! where `G` is the DC conductance Jacobian, `C` is the capacitance
//! matrix (built from `TransientStorage` companions divided by step
//! size, since BE companion gives `C/h` and we re-derive `C = M_b·h`),
//! and `b_ac` carries AC stimulus into KCL.
//!
//! ## Why batched
//!
//! The natural AC sweep has hundreds of frequency points. Each is a
//! tiny solve, but the per-point Rust + dispatch overhead dominates
//! when run sequentially. Vmap'ing over the frequency axis makes the
//! whole sweep one MLX dispatch.
//!
//! ## Scope (this MVP)
//!
//! * **Single-unknown linear circuit** — the body computes the
//!   complex 1×1 inverse symbolically (`V = b_ac/(G + jωC)` →
//!   `V_re, V_im`) without needing `Op::DenseSolve`. Multi-unknown
//!   would build the 2N×2N real-block real system and solve via
//!   `Op::DenseSolve` (F32 MLX path).
//! * **One AC stimulus boundary, one output net** — fixed at 1V on
//!   the boundary, output read at the unknown net. Multi-stim /
//!   multi-output is a thin generalization.
//! * **Linearization is at v=0** — fine for circuits whose DC
//!   operating point is the trivial solution (passive RC/LC). For
//!   nonlinear circuits, linearize at the actual DC point first
//!   (call `solve_dc`, then evaluate Jacobian there).

use std::collections::HashMap;

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

use crate::{
    build_be_step_residual_graph, build_residual_graph, branch_input_name,
    delay_blend_name, delay_offset_name, delay_v_hi_name, delay_v_lo_name,
    find_input_node, net_input_name, prev_voltage_input_name, BranchId,
    Circuit, DelayId, NetId, TIMESTEP_INPUT_NAME,
};

/// Output of [`build_ac_response_graph`].
pub struct AcResponseGraph {
    /// Graph: input `"omega"` shape `[1]`, outputs `[V_re, V_im]`
    /// each shape `[1]`. After vmap over `["omega"]` the graph
    /// computes the per-frequency response in one MLX dispatch.
    pub graph: Graph,
    /// The unknown net the response is read at.
    pub output_net: NetId,
    /// Numerical extracted stamps — exposed for diagnostics /
    /// witnessing. (Calling `.run()` on the graph computes the same
    /// values via the embedded constants.)
    pub g: f32,
    pub c: f32,
    pub b_ac: f32,
}

/// Multi-unknown AC response graph. Builds the 2N×2N real-block
/// complex system and solves it via `Op::DenseSolve` per frequency.
///
/// System layout: stack `V_re` then `V_im` into one length-2N vector
/// `V_block`. The complex equation `(G + jωC) · V = b_ac` becomes
/// the real block:
///
/// ```text
///   [G   −ωC] [V_re]   [b_re]
///   [ωC   G ] [V_im] = [b_im]
/// ```
///
/// where `b_im = 0` for our convention (AC stimulus is real-valued
/// at the boundary).
///
/// Returns `(graph, output_net_indices)`. The graph takes `omega [1]`
/// and outputs `V_block [2N]`; row i is V_re[i], row N+i is V_im[i].
/// Caller indexes by the unknown ordering on `output_net_indices`.
pub fn build_ac_response_graph_multi(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    boundary_net_with_ac: NetId,
) -> Result<(Graph, Vec<NetId>), String> {
    let scalar_dc = build_residual_graph(circuit);
    let unknowns = scalar_dc.unknown_nets.clone();
    let n = unknowns.len();
    if n == 0 || !scalar_dc.branches.is_empty() {
        return Err(format!(
            "build_ac_response_graph_multi: need n_v ≥ 1 + no branches; \
             got n_v={n}, n_b={}", scalar_dc.branches.len()
        ));
    }

    // Extract G [n,n], C [n,n], b_ac [n] numerically.
    // G via grad wrt unknowns (every row depends on every unknown
    // via the residual builder's phantom-dependency trick). b_ac
    // via direct forward eval at v_unknown=0, v_prev=0, v_boundary=1
    // — for linear circuits r is linear in (v, v_prev, v_boundary) so
    // r_i(v_b=1, rest=0) = -b_ac[i] (no AD needed; sidesteps the
    // "no gradient flowed" assertion when a row's residual doesn't
    // transitively reference the boundary input).
    let session = Session::new(Device::Cpu);
    let unknown_input_ids: Vec<NodeId> = unknowns.iter()
        .map(|net| find_input_node(&scalar_dc.graph, &net_input_name(*net))
            .ok_or_else(|| format!("DC residual missing v_{}", net.0)))
        .collect::<Result<_, _>>()?;

    let zero = [0.0_f32];
    let one = [1.0_f32];

    let mut g_mat = vec![0.0_f32; n * n];
    let mut b_ac_vec = vec![0.0_f32; n];
    // First pass: G from grad-wrt-unknowns + r_i(zero) for sanity.
    for i in 0..n {
        let mut g_i = scalar_dc.graph.clone();
        g_i.set_outputs(vec![g_i.outputs[i]]);
        let bwd = rlx_opt::autodiff::grad_with_loss(&g_i, &unknown_input_ids);
        let mut compiled = session.compile(bwd);
        for (k, v) in params { compiled.set_param(k, &[*v]); }
        let mut inputs: Vec<(String, Vec<f32>)> = Vec::new();
        for net in &scalar_dc.all_nets {
            inputs.push((net_input_name(*net), zero.to_vec()));
        }
        for b in &scalar_dc.branches {
            inputs.push((branch_input_name(*b), zero.to_vec()));
        }
        inputs.push(("d_output".to_string(), one.to_vec()));
        let refs: Vec<(&str, &[f32])> = inputs.iter()
            .map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        let outs = compiled.run(&refs);
        for j in 0..n {
            g_mat[i * n + j] = -outs[1 + j][0];
        }
    }

    // Second pass: forward-eval the residual at v_in=1, all else 0,
    // to read out b_ac directly. r_i evaluates to -b_ac[i] under the
    // sign convention r = K·v - b(v_prev, v_in).
    {
        let mut compiled_fwd = session.compile(scalar_dc.graph.clone());
        for (k, v) in params { compiled_fwd.set_param(k, &[*v]); }
        let mut inputs: Vec<(String, Vec<f32>)> = Vec::new();
        for net in &scalar_dc.all_nets {
            let val = if *net == boundary_net_with_ac { one } else { zero };
            inputs.push((net_input_name(*net), val.to_vec()));
        }
        for b in &scalar_dc.branches {
            inputs.push((branch_input_name(*b), zero.to_vec()));
        }
        let refs: Vec<(&str, &[f32])> = inputs.iter()
            .map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        let outs = compiled_fwd.run(&refs);
        for i in 0..n {
            b_ac_vec[i] = -outs[i][0];
        }
    }

    // C[i, j] extraction: forward-eval the BE residual at
    // v_unknown=0, v_prev_j=1 (all other v_prev=0), h=1. Linear
    // residual gives r_i = +C[i,j]/h · 1 = C[i,j] (with h=1). Same
    // sidestep as b_ac to avoid grad-AD over rows that don't reach
    // every v_prev input.
    let scalar_be = build_be_step_residual_graph(circuit);
    let h_val = 1.0_f32;
    let h_arr = [h_val];
    let mut c_mat = vec![0.0_f32; n * n];
    let mut effective_params = params.clone();
    for dl in &circuit.delays {
        let key = format!("{}_tau", dl.device.name());
        effective_params.entry(key).or_insert(
            dl.device.delay_seconds() as f32,
        );
    }
    let mut compiled_be_fwd = session.compile(scalar_be.graph.clone());
    for (k, v) in &effective_params { compiled_be_fwd.set_param(k, &[*v]); }
    for j in 0..n {
        // Build inputs: all v=0, v_prev_<unknowns[j]>=1, rest 0.
        let target_prev_net = unknowns[j];
        let mut inputs: Vec<(String, Vec<f32>)> = Vec::new();
        for net in &scalar_be.all_nets {
            inputs.push((net_input_name(*net), zero.to_vec()));
            let val = if *net == target_prev_net { one } else { zero };
            inputs.push((prev_voltage_input_name(*net), val.to_vec()));
        }
        for b in &scalar_be.branches {
            inputs.push((branch_input_name(*b), zero.to_vec()));
        }
        inputs.push((TIMESTEP_INPUT_NAME.to_string(), h_arr.to_vec()));
        for (idx, _) in circuit.delays.iter().enumerate() {
            let id = DelayId(idx as u32);
            inputs.push((delay_v_lo_name(id),   zero.to_vec()));
            inputs.push((delay_v_hi_name(id),   zero.to_vec()));
            inputs.push((delay_blend_name(id),  zero.to_vec()));
            inputs.push((delay_offset_name(id), zero.to_vec()));
        }
        let refs: Vec<(&str, &[f32])> = inputs.iter()
            .map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        let outs = compiled_be_fwd.run(&refs);
        // outs[i] = r_i at this point = C[i, j] / h · 1 = C[i, j] (h=1).
        for i in 0..n {
            c_mat[i * n + j] = outs[i][0] * h_val;
        }
    }

    // ── Build the AC graph with Op::DenseSolve on a 2N system. ──
    let two_n = 2 * n;
    let s_scalar  = Shape::new(&[1], DType::F32);
    let s_block   = Shape::new(&[two_n], DType::F32);
    let s_block_m = Shape::new(&[two_n, two_n], DType::F32);
    let mut g = Graph::new("ac_response_multi");
    let omega = g.input("omega", s_scalar.clone());

    // Embed G (constant), C (constant), b_re (constant) as scalars
    // / vectors / matrices. Build [G −ωC; ωC G] as a 2N×2N graph
    // expression: top-left = G_const, top-right = -ω·C_const,
    // bottom-left = ω·C_const, bottom-right = G_const. Compose via
    // explicit Constant tiles + Concat.

    // For each entry [i, j] of the 2N×2N block matrix, build a
    // scalar graph node, then concat all into the matrix. This is
    // O(n²) Op nodes per build, which is fine for n ≤ a few dozen
    // (typical analog circuits).

    let neg_one = const_scalar_node(&mut g, -1.0);

    // ωC matrix: entries are ω · C[i,j].
    let omega_c: Vec<NodeId> = (0..(n * n)).map(|k| {
        let c_const = const_scalar_node(&mut g, c_mat[k]);
        g.binary(BinaryOp::Mul, omega, c_const, s_scalar.clone())
    }).collect();
    // -ωC entries.
    let neg_omega_c: Vec<NodeId> = omega_c.iter().map(|&n_id| {
        g.binary(BinaryOp::Mul, neg_one, n_id, s_scalar.clone())
    }).collect();
    // G entries (constants).
    let g_consts: Vec<NodeId> = g_mat.iter().map(|v| const_scalar_node(&mut g, *v))
        .collect();

    // Compose [2N, 2N] block matrix from the four n×n blocks.
    // Matrix layout: row r ∈ 0..n is the "real" half; row n + r is
    // the "imag" half. Same for columns.
    let mut block_rows: Vec<NodeId> = Vec::with_capacity(two_n);
    for r in 0..n {
        // Real row r: [G[r, *], -ωC[r, *]].
        let mut row_scalars: Vec<NodeId> = Vec::with_capacity(two_n);
        for j in 0..n { row_scalars.push(g_consts[r * n + j]); }
        for j in 0..n { row_scalars.push(neg_omega_c[r * n + j]); }
        block_rows.push(concat_scalars(&mut g, &row_scalars, two_n));
    }
    for r in 0..n {
        // Imag row r: [ωC[r, *], G[r, *]].
        let mut row_scalars: Vec<NodeId> = Vec::with_capacity(two_n);
        for j in 0..n { row_scalars.push(omega_c[r * n + j]); }
        for j in 0..n { row_scalars.push(g_consts[r * n + j]); }
        block_rows.push(concat_scalars(&mut g, &row_scalars, two_n));
    }
    // Stack rows into [2N, 2N] matrix via reshape + concat.
    let block_mat = stack_rows_to_matrix(&mut g, &block_rows, two_n);

    // RHS: [b_re; 0]. b_im is 0 (real-valued AC stimulus).
    let b_re_consts: Vec<NodeId> = b_ac_vec.iter().map(|v| const_scalar_node(&mut g, *v))
        .collect();
    let zero_const = const_scalar_node(&mut g, 0.0);
    let mut rhs_scalars: Vec<NodeId> = Vec::with_capacity(two_n);
    for j in 0..n { rhs_scalars.push(b_re_consts[j]); }
    for _ in 0..n { rhs_scalars.push(zero_const); }
    let rhs_block = concat_scalars(&mut g, &rhs_scalars, two_n);

    // Solve.
    let v_block = g.dense_solve(block_mat, rhs_block, s_block);
    g.set_outputs(vec![v_block]);
    let _ = s_block_m;
    Ok((g, unknowns))
}

fn const_scalar_node(g: &mut Graph, x: f32) -> NodeId {
    g.add_node(
        Op::Constant { data: x.to_le_bytes().to_vec() },
        vec![],
        Shape::new(&[1], DType::F32),
    )
}

fn concat_scalars(g: &mut Graph, scalars: &[NodeId], n: usize) -> NodeId {
    debug_assert_eq!(scalars.len(), n);
    if n == 1 { return scalars[0]; }
    g.add_node(
        Op::Concat { axis: 0 },
        scalars.to_vec(),
        Shape::new(&[n], DType::F32),
    )
}

fn stack_rows_to_matrix(g: &mut Graph, rows: &[NodeId], n: usize) -> NodeId {
    debug_assert_eq!(rows.len(), n);
    let mut row_2ds: Vec<NodeId> = Vec::with_capacity(n);
    for &r in rows {
        let r2d = g.reshape(r, vec![1, n as i64], Shape::new(&[1, n], DType::F32));
        row_2ds.push(r2d);
    }
    g.add_node(
        Op::Concat { axis: 0 },
        row_2ds,
        Shape::new(&[n, n], DType::F32),
    )
}

/// Build a per-frequency AC response graph for a linear circuit
/// with one unknown net. The graph takes `omega` (= 2π·f) as its
/// only input and outputs `[V_re, V_im]` at `boundary_net + AC stimulus = 1V`.
///
/// `boundary_net_with_ac` carries the AC test signal at unit
/// amplitude — caller scales the result to the desired stimulus
/// amplitude on the host.
pub fn build_ac_response_graph(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    boundary_net_with_ac: NetId,
) -> Result<AcResponseGraph, String> {
    // Validate single-unknown.
    let scalar_dc = build_residual_graph(circuit);
    let unknowns = scalar_dc.unknown_nets.clone();
    if unknowns.len() != 1 || !scalar_dc.branches.is_empty() {
        return Err(format!(
            "build_ac_response_graph: MVP supports n_v=1 + no branches; \
             got n_v={}, n_b={}", unknowns.len(), scalar_dc.branches.len(),
        ));
    }
    let output_net = unknowns[0];
    let unknown_input = find_input_node(&scalar_dc.graph, &net_input_name(output_net))
        .ok_or("DC residual missing v_<unknown>")?;
    let boundary_input = find_input_node(
        &scalar_dc.graph, &net_input_name(boundary_net_with_ac),
    ).ok_or_else(|| format!(
        "DC residual missing v_<boundary={}> — is the boundary net allocated?",
        boundary_net_with_ac.0,
    ))?;

    // Extract G via grad of DC residual wrt v_<unknown> at v=0,
    // also -b_ac via grad wrt v_<boundary>.
    let session = Session::new(Device::Cpu);
    let mut g_dc = scalar_dc.graph.clone();
    g_dc.set_outputs(vec![g_dc.outputs[0]]);
    let bwd_dc = rlx_opt::autodiff::grad_with_loss(
        &g_dc, &[unknown_input, boundary_input],
    );
    let mut compiled_dc = session.compile(bwd_dc);
    for (k, v) in params {
        compiled_dc.set_param(k, &[*v]);
    }
    // Bind every Op::Input in DC graph to 0; d_output to 1.
    let zero = [0.0_f32];
    let one = [1.0_f32];
    let mut dc_inputs: Vec<(String, Vec<f32>)> = Vec::new();
    for net in &scalar_dc.all_nets {
        dc_inputs.push((net_input_name(*net), zero.to_vec()));
    }
    for b in &scalar_dc.branches {
        dc_inputs.push((branch_input_name(*b), zero.to_vec()));
    }
    dc_inputs.push(("d_output".to_string(), one.to_vec()));
    let dc_refs: Vec<(&str, &[f32])> = dc_inputs.iter()
        .map(|(n, v)| (n.as_str(), v.as_slice())).collect();
    let dc_outs = compiled_dc.run(&dc_refs);
    // The MNA residual rlx uses sums device terminal_currents
    // straight into KCL, so the Jacobian sign is OPPOSITE to the
    // physical admittance convention: dr/dv_unknown = -G,
    // dr/dv_boundary = +G_b (the boundary's KCL contribution).
    // Negate the unknown-side gradient to recover physical G; keep
    // the boundary-side gradient as the b_ac coefficient as-is.
    let g_val    = -dc_outs[1][0];
    let b_ac_val = dc_outs[2][0];

    // Extract C via BE residual M_b at h=1: M_b = C/h, so C = M_b·1 = M_b.
    let scalar_be = build_be_step_residual_graph(circuit);
    let unknown_input_be = find_input_node(&scalar_be.graph, &net_input_name(output_net))
        .ok_or("BE residual missing v_<unknown>")?;
    let prev_input_be = find_input_node(
        &scalar_be.graph, &prev_voltage_input_name(output_net),
    ).ok_or("BE residual missing v_prev_<unknown>")?;
    let mut g_be = scalar_be.graph.clone();
    g_be.set_outputs(vec![g_be.outputs[0]]);
    let bwd_be = rlx_opt::autodiff::grad_with_loss(
        &g_be, &[unknown_input_be, prev_input_be],
    );
    let mut compiled_be = session.compile(bwd_be);
    // Backfill <name>_tau defaults (scalar solve_be_step does this too).
    let mut effective_params = params.clone();
    for dl in &circuit.delays {
        let key = format!("{}_tau", dl.device.name());
        effective_params.entry(key).or_insert(
            dl.device.delay_seconds() as f32,
        );
    }
    for (k, v) in &effective_params {
        compiled_be.set_param(k, &[*v]);
    }
    let h_val = 1.0_f32;
    let h_arr = [h_val];
    let mut be_inputs: Vec<(String, Vec<f32>)> = Vec::new();
    for net in &scalar_be.all_nets {
        be_inputs.push((net_input_name(*net), zero.to_vec()));
        be_inputs.push((prev_voltage_input_name(*net), zero.to_vec()));
    }
    for b in &scalar_be.branches {
        be_inputs.push((branch_input_name(*b), zero.to_vec()));
    }
    be_inputs.push((TIMESTEP_INPUT_NAME.to_string(), h_arr.to_vec()));
    be_inputs.push(("d_output".to_string(),          one.to_vec()));
    for (idx, _) in circuit.delays.iter().enumerate() {
        let id = DelayId(idx as u32);
        be_inputs.push((delay_v_lo_name(id),   zero.to_vec()));
        be_inputs.push((delay_v_hi_name(id),   zero.to_vec()));
        be_inputs.push((delay_blend_name(id),  zero.to_vec()));
        be_inputs.push((delay_offset_name(id), zero.to_vec()));
    }
    let be_refs: Vec<(&str, &[f32])> = be_inputs.iter()
        .map(|(n, v)| (n.as_str(), v.as_slice())).collect();
    let be_outs = compiled_be.run(&be_refs);
    // BE residual cap contribution at unknown: +(C/h)·v_prev term.
    // So dr/dv_prev = +C/h → C = +outs[2]·h. (Sign convention same
    // as the boundary-side derivation above: dr/dv_INPUT vars is
    // positive of the admittance contribution they make.)
    let c_val = be_outs[2][0] * h_val;

    // ── Build the AC graph. ──
    let s = Shape::new(&[1], DType::F32);
    let mut g = Graph::new("ac_response");
    let omega = g.input("omega", s.clone());
    let g_node = const_scalar(&mut g, g_val);
    let c_node = const_scalar(&mut g, c_val);
    let b_node = const_scalar(&mut g, b_ac_val);
    let neg_one_node = const_scalar(&mut g, -1.0);

    // omega·C
    let omega_c = g.binary(BinaryOp::Mul, omega, c_node, s.clone());
    // |G + jωC|² = G² + (ωC)²
    let g_sq = g.binary(BinaryOp::Mul, g_node, g_node, s.clone());
    let omega_c_sq = g.binary(BinaryOp::Mul, omega_c, omega_c, s.clone());
    let denom = g.binary(BinaryOp::Add, g_sq, omega_c_sq, s.clone());
    // Numerator: V_re = b·G / |…|²,  V_im = -b·ωC / |…|²
    let v_re_num = g.binary(BinaryOp::Mul, b_node, g_node, s.clone());
    let v_re = g.binary(BinaryOp::Div, v_re_num, denom, s.clone());
    let neg_omega_c = g.binary(BinaryOp::Mul, neg_one_node, omega_c, s.clone());
    let v_im_num = g.binary(BinaryOp::Mul, b_node, neg_omega_c, s.clone());
    let v_im = g.binary(BinaryOp::Div, v_im_num, denom, s.clone());
    g.set_outputs(vec![v_re, v_im]);

    Ok(AcResponseGraph {
        graph: g, output_net,
        g: g_val, c: c_val, b_ac: b_ac_val,
    })
}

fn const_scalar(g: &mut Graph, x: f32) -> NodeId {
    g.add_node(
        Op::Constant { data: x.to_le_bytes().to_vec() },
        vec![],
        Shape::new(&[1], DType::F32),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use eda_hir::Block;
    use spike_divider_block::{Capacitor, Resistor};

    #[test]
    fn multi_unknown_ac_two_stage_rc_ladder_matches_analytic() {
        // V_in -- R1 -- v1 -- R2 -- v2 -- C2 -- gnd
        //                  `-- C1 -- gnd                `-- C2 covers v2
        // Two unknowns (v1, v2). Build multi-unknown AC graph,
        // sweep, compare |V2(jω)| to two-stage analytic transfer:
        //
        //   Y at v1: G1 + G2 + jωC1
        //   Y at v2: G2     + jωC2
        //   Coupling: -G2 each off-diagonal
        //   |G + jωC| matrix → solve.
        //
        // For the test we just spot-check a few frequencies and
        // verify the magnitude is bounded and decreases past the
        // higher pole — the architectural claim is "matches scalar
        // matrix solve". Strict closed form for cascaded RC is
        // nontrivial; we verify shape by comparing to a per-freq
        // analytic solver (host-side complex 2x2 solve).
        let r1_o   = 1_000.0_f32;
        let r2_o   = 2_000.0_f32;
        let c1_f   = 1e-9_f32;
        let c2_f   = 0.5e-9_f32;
        let mut c = Circuit::new();
        let v_in = c.alloc_boundary_net();
        let v1   = c.alloc_unknown_net();
        let v2   = c.alloc_unknown_net();
        let r1 = Resistor { length: 10_000, id: "R1".into() };
        let r2 = Resistor { length: 30_000, id: "R2".into() };
        let cap1 = Capacitor { plate_size: 2_000, id: "C1".into() };
        let cap2 = Capacitor { plate_size: 2_000, id: "C2".into() };
        c.add_device(r1.clone(),   &[v_in, v1]);
        c.add_device(r2.clone(),   &[v1,   v2]);
        c.add_storage(cap1.clone(), [v1,   NetId::GND]);
        c.add_storage(cap2.clone(), [v2,   NetId::GND]);

        let mut params = HashMap::new();
        params.insert(Block::name(&r1), r1_o);
        params.insert(Block::name(&r2), r2_o);
        params.insert(format!("{}_C", Block::name(&cap1)), c1_f);
        params.insert(format!("{}_C", Block::name(&cap2)), c2_f);

        let (graph, output_nets) = build_ac_response_graph_multi(&c, &params, v_in)
            .expect("multi-unknown AC graph");
        assert_eq!(output_nets.len(), 2);

        // Compile + sweep.
        let mut compiled = Session::new(Device::Cpu).compile(graph);

        // Per-freq host-side analytic complex 2x2 solve as reference.
        for &f in &[1.0e3_f32, 1.0e4, 1.0e5, 1.0e6, 1.0e7] {
            let omega = 2.0 * std::f32::consts::PI * f;
            let outs = compiled.run(&[("omega", &[omega])]);
            let v_block = &outs[0];
            // V_re indexes 0..n, V_im indexes n..2n.
            let v1_re = v_block[0]; let v1_im = v_block[2];
            let v2_re = v_block[1]; let v2_im = v_block[3];
            let v2_mag = (v2_re*v2_re + v2_im*v2_im).sqrt();

            // Reference: solve [G + jωC] · V = b in host f64 complex.
            let g1 = 1.0_f64 / r1_o as f64;
            let g2 = 1.0_f64 / r2_o as f64;
            let omg = omega as f64;
            // Block 2x2 system at v1, v2:
            //  [(g1+g2)+jωc1   −g2          ] [V1]   [g1·V_in]
            //  [   −g2         g2+jωc2     ] [V2] = [   0   ]
            let a11 = (g1 + g2, omg * c1_f as f64);
            let a12 = (-g2, 0.0);
            let a21 = (-g2, 0.0);
            let a22 = (g2,   omg * c2_f as f64);
            let b1  = (g1 * 1.0, 0.0);
            let b2  = (0.0, 0.0);
            // 2x2 complex Cramer's rule.
            let det = csub(cmul(a11, a22), cmul(a12, a21));
            let v1_ref = cdiv(csub(cmul(b1, a22), cmul(a12, b2)), det);
            let v2_ref = cdiv(csub(cmul(a11, b2), cmul(b1, a21)), det);
            let v2_ref_mag = (v2_ref.0*v2_ref.0 + v2_ref.1*v2_ref.1).sqrt() as f32;

            let drift = (v2_mag - v2_ref_mag).abs() / v2_ref_mag.max(1e-12);
            assert!(
                drift < 5e-4,
                "f={f}: scan |V2|={v2_mag} ref |V2|={v2_ref_mag} (rel {drift:.3e}) \
                 v1=({v1_re}+{v1_im}j)",
            );
        }
    }

    fn cmul(a: (f64,f64), b: (f64,f64)) -> (f64,f64) {
        (a.0*b.0 - a.1*b.1, a.0*b.1 + a.1*b.0)
    }
    fn csub(a: (f64,f64), b: (f64,f64)) -> (f64,f64) {
        (a.0 - b.0, a.1 - b.1)
    }
    fn cdiv(a: (f64,f64), b: (f64,f64)) -> (f64,f64) {
        let denom = b.0*b.0 + b.1*b.1;
        ((a.0*b.0 + a.1*b.1) / denom, (a.1*b.0 - a.0*b.1) / denom)
    }

    #[test]
    fn rc_low_pass_ac_response_matches_analytic() {
        // Topology: V_in -- R -- vmid -- C -- gnd
        // H(jω) = 1 / (1 + jωRC); |H| = 1/sqrt(1+(ωRC)²)
        let r_ohms   = 1_000.0_f32;
        let c_farads = 1e-9_f32;
        let mut c = Circuit::new();
        let v_in = c.alloc_boundary_net();
        let vmid = c.alloc_unknown_net();
        let r = Resistor { length: 10_000, id: "R".into() };
        let cap = Capacitor { plate_size: 2_000, id: "C".into() };
        c.add_device(r.clone(), &[v_in, vmid]);
        c.add_storage(cap.clone(), [vmid, NetId::GND]);

        let mut params = HashMap::new();
        params.insert(Block::name(&r), r_ohms);
        params.insert(format!("{}_C", Block::name(&cap)), c_farads);

        let ac = build_ac_response_graph(&c, &params, v_in)
            .expect("build AC graph");

        // Sanity: extracted G = 1/R, C = C_farads.
        assert!((ac.g - 1.0 / r_ohms).abs() < 1e-9,
            "G extraction: {} vs 1/R={}", ac.g, 1.0 / r_ohms);
        assert!((ac.c - c_farads).abs() < 1e-15,
            "C extraction: {} vs C={}", ac.c, c_farads);
        assert!((ac.b_ac - 1.0 / r_ohms).abs() < 1e-9,
            "b_ac extraction: {} vs 1/R", ac.b_ac);

        // Compile + sweep over a decade of frequencies; compare |V|
        // to analytic 1/sqrt(1 + (ωRC)²).
        let mut compiled = Session::new(Device::Cpu).compile(ac.graph);
        let f0 = 1.0 / (2.0 * std::f32::consts::PI * r_ohms * c_farads); // -3 dB freq
        for &mult in &[0.01_f32, 0.1, 1.0, 10.0, 100.0] {
            let f = f0 * mult;
            let omega = 2.0 * std::f32::consts::PI * f;
            let outs = compiled.run(&[("omega", &[omega])]);
            let v_re = outs[0][0];
            let v_im = outs[1][0];
            let mag = (v_re * v_re + v_im * v_im).sqrt();
            let analytic = 1.0_f32 / (1.0 + (omega * r_ohms * c_farads).powi(2)).sqrt();
            let drift = (mag - analytic).abs();
            let rel  = drift / analytic.max(1e-12);
            assert!(
                rel < 5e-5,
                "f/f0={mult}: |V|={mag} analytic={analytic} (rel {rel:.3e})",
            );
        }
    }
}
