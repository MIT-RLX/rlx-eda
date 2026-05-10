//! eda-mna composition for the N-bit SAR register.
//!
//! Mirrors the `SpiceEmit` impl exactly: per bit `i`,
//!   - 1 inverter producing `set_b[i] = !phase[i]`
//!   - 1 DffSR with `d = cmp`, `clk = capture[i]`, the inverter's
//!     `set_b`, and shared `reset_b`
//!   - 2 cascaded inverters buffering `q_int → bit[i]` for low DAC
//!     drive impedance
//!
//! IC seeds are threaded through to `add_dff_sr` so the SR cross-
//! couples settle to a stable `Q=0, Qb=Vdd` state at t = 0.

use std::collections::HashMap;

use eda_mna::{Circuit, NetId};
use spike_cmos_gates::mna::{add_dff_sr, add_inverter};

/// Add an N-bit SAR register.
/// Net order matches `SpiceEmit`:
///   `[phase_0..phase_{N-1}, capture_0..capture_{N-1}, cmp, reset_b,
///     bit_0..bit_{N-1}, vdd, gnd]`
pub fn add_sar_register(
    c: &mut Circuit,
    phases:   &[NetId],   // length N
    captures: &[NetId],   // length N
    cmp:      NetId,
    reset_b:  NetId,
    bits:     &[NetId],   // length N (outputs to drive DAC)
    vdd:      NetId,
    gnd:      NetId,
    id:       &str,
    params:   &mut HashMap<String, f32>,
    mut ic:   Option<&mut HashMap<NetId, f32>>,
) {
    let n = phases.len();
    assert_eq!(captures.len(), n);
    assert_eq!(bits.len(),     n);
    for i in 0..n {
        let set_b   = c.alloc_unknown_net();
        let q_int   = c.alloc_unknown_net();
        let qb      = c.alloc_unknown_net();
        let buf_mid = c.alloc_unknown_net();

        // set_b[i] = !phase[i]
        add_inverter(c, [phases[i], set_b, vdd, gnd], &format!("{id}_inv_{i}"), params);
        // DffSR: [d=cmp, clk=capture[i], set_b, reset_b, q=q_int, qb, vdd, gnd]
        add_dff_sr(
            c, [cmp, captures[i], set_b, reset_b, q_int, qb, vdd, gnd],
            &format!("{id}_dff_{i}"), params, ic.as_deref_mut(),
        );
        // Buffer chain: q_int → buf_mid → bit[i]
        add_inverter(c, [q_int,   buf_mid, vdd, gnd], &format!("{id}_buf_a_{i}"), params);
        add_inverter(c, [buf_mid, bits[i], vdd, gnd], &format!("{id}_buf_b_{i}"), params);

        if let Some(ic) = ic.as_deref_mut() {
            // Pre-edge: phase[i] = 0 → set_b = Vdd. q_int seeded to 0
            // by add_dff_sr; buf_mid = !q_int = Vdd; bit[i] = !buf_mid = 0.
            ic.insert(set_b,   1.8);
            ic.insert(buf_mid, 1.8);
            ic.insert(bits[i], 0.0);
        }
    }
}
