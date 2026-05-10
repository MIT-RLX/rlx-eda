//! eda-mna composition for the CMOS sample-and-hold.
//!
//! Mirrors the `SpiceEmit` impl: a transmission-gate (NMOS in
//! parallel with PMOS) controlled by `clk_sh` with a hold cap on the
//! `vhold` net. Adds primitives directly to an `eda_mna::Circuit`
//! from `spike_divider_block::Mosfet` so the result runs under
//! `transient_pwl` + `transient_sensitivities`.

use std::collections::HashMap;

use eda_mna::{Circuit, LinearCap, NetId};
use spike_cmos_gates::mna::add_inverter;
use spike_divider_block::Mosfet;

const W_NMOS: i64 = 2_000;
const W_PMOS: i64 = 4_000;
const L:      i64 = 1_000;

/// Sample-and-hold transmission-gate + hold cap.
/// Nets: `[vin, vhold, clk_sh, vdd, gnd]`.
/// `c_hold` in farads (e.g. `100e-15` for the textbook 100 fF SAR S/H).
pub fn add_sample_hold(
    c: &mut Circuit, nets: [NetId; 5], id: &str,
    c_hold_f: f32, params: &mut HashMap<String, f32>,
) {
    let (vin, vhold, clk_sh, vdd, gnd) = (nets[0], nets[1], nets[2], nets[3], nets[4]);
    let clk_sh_n = c.alloc_unknown_net();
    add_inverter(c, [clk_sh, clk_sh_n, vdd, gnd], &format!("{id}_iv"), params);

    let nmos = Mosfet::nmos(W_NMOS, L, format!("{id}_n"));
    let pmos = Mosfet::pmos(W_PMOS, L, format!("{id}_p"));
    // NMOS pass: drain=vin, gate=clk_sh, source=vhold, bulk=gnd.
    c.add_device(nmos.clone(), &[vin, clk_sh, vhold, gnd]);
    // PMOS pass: drain=vin, gate=clk_sh_n, source=vhold, bulk=vdd.
    c.add_device(pmos.clone(), &[vin, clk_sh_n, vhold, vdd]);
    params.extend(nmos.default_params());
    params.extend(pmos.default_params());

    let cap_key = format!("{id}_chold");
    c.add_storage(LinearCap::new(cap_key.clone()), [vhold, NetId::GND]);
    params.insert(cap_key, c_hold_f);
}
