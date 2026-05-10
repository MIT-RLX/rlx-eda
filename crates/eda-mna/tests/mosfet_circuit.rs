//! End-to-end: a 4-terminal `Mosfet` (square-law) attached to the
//! `Circuit` framework and solved through `solve_dc` — proves that the
//! same trait machinery that drives `Resistor`/`Diode`/`VoltageSource`
//! handles a 4-port nonlinear device with no MNA-side changes.
//!
//! Diode-connected NMOS load:
//!
//! ```text
//!     V_DD ──[R]── vmid ──[NMOS, D=G=vmid; S=B=gnd]── gnd
//! ```
//!
//! KCL at `vmid`: current sourced through R = drain current of NMOS.
//! ```text
//!     (V_DD − vmid) / R   =   ½·K_p·(W/L)·(vmid − V_th)²
//! ```
//! (NMOS is in saturation since V_DS = V_GS = vmid ≥ V_GS − V_th.)
//!
//! With V_DD=2V, R=10 kΩ, K_p=100 µA/V², W/L=1, V_th=0.5: the quadratic
//! collapses to `vmid − 0.5 = 1`, so `vmid = 1.5 V` and i_drain = 50 µA.

use std::collections::HashMap;
use eda_mna::{solve_dc, Circuit, NetId, NewtonOptions};
use spike_divider_block::{MosModel, MosPolarity, Mosfet, Resistor};

#[test]
fn diode_connected_nmos_load_resistor_settles_to_closed_form() {
    let mut c = Circuit::new();
    let v_dd = c.alloc_boundary_net();
    let vmid = c.alloc_unknown_net();

    let r    = Resistor { length: 10_000, id: "R1".into() };
    let nmos = Mosfet {
        polarity: MosPolarity::Nmos,
        model: MosModel::SquareLaw,
        w: 1_000,
        l: 1_000,
        id: "M1".into(),
    };

    c.add_device(r.clone(),    &[v_dd, vmid]);
    // Terminal order [D, G, S, B] — diode-connected: D=G=vmid; S=B=gnd.
    c.add_device(nmos.clone(), &[vmid, vmid, NetId::GND, NetId::GND]);

    // 1 unknown net (vmid), 0 branches.
    assert_eq!(c.n_unknowns(), 1);

    // Seed all five MOSFET params via the Mosfet helper; Kp/Vth match
    // the closed-form, λ=γ=0 → reduces to bare square-law.
    let mut params = nmos.default_params();
    params.insert(eda_hir::Block::name(&r), 10_000.0_f32);

    let mut boundary = HashMap::new();
    boundary.insert(v_dd, 2.0_f32);

    let op = solve_dc(&c, &params, &boundary, NewtonOptions::default());
    assert!(op.converged, "Newton failed: residual_max = {:.3e}", op.final_residual_max);

    let vmid_v = op.voltages[&vmid];
    // 1e-3 absolute (≈0.07% rel) — f32 MNA at these microamp scales tops
    // out around 4-5 sig figs.
    assert!((vmid_v - 1.5).abs() < 1e-3,
        "vmid = {}, expected 1.5", vmid_v);
}
