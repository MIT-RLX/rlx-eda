//! eda-mna composition for the R-2R DAC. Pure passive — just
//! resistors. Mirrors the `SpiceEmit` impl, layered identically:
//!
//! - 2R termination at LSB end (vlow → n_0)
//! - per-bit 2R input feeders (in_i → n_i)
//! - R spine (n_i → n_{i+1})
//!
//! All resistors use `spike_divider_block::Resistor` so values come
//! through eda-mna's standard param map. `Resistor::name()` returns
//! `"Resistor_<id>_L<length>"` where `length` is just a tag — the
//! actual resistance is set under the same key in `params`.

use std::collections::HashMap;

use eda_mna::{Circuit, NetId};
use eda_hir::Block;
use spike_divider_block::Resistor;

/// Add an N-bit R-2R ladder.
/// Nets order: `[in_0, in_1, ..., in_{N-1}, vlow, vout]` — `N + 2` total.
/// `r_ohms` is the value of the spine resistor (R); each input feeder
/// and the LSB termination are 2R.
pub fn add_r2r_dac(
    c: &mut Circuit,
    inputs: &[NetId],     // length N
    vlow: NetId,
    vout: NetId,
    r_ohms: f32,
    id: &str,
    params: &mut HashMap<String, f32>,
) {
    let n = inputs.len();
    if n == 0 { return; }

    // Internal nodes n_0 .. n_{N-2}; n_{N-1} == vout.
    let mut nodes: Vec<NetId> = (0..n - 1).map(|_| c.alloc_unknown_net()).collect();
    nodes.push(vout);   // n_{N-1}

    let add_r = |c: &mut Circuit, params: &mut HashMap<String, f32>,
                 a: NetId, b: NetId, value: f32, sub_id: String| {
        // length is just a tag for unique naming.
        let r = Resistor { length: value as i64, id: sub_id };
        let key = Block::name(&r);
        c.add_device(r.clone(), &[a, b]);
        params.insert(key, value);
    };

    // Termination 2R: vlow → n_0.
    add_r(c, params, vlow, nodes[0], 2.0 * r_ohms, format!("{id}_term"));

    // Per-bit 2R feeders.
    for i in 0..n {
        add_r(c, params, inputs[i], nodes[i], 2.0 * r_ohms, format!("{id}_in{i}"));
    }

    // Spine R between consecutive nodes.
    for i in 0..n - 1 {
        add_r(c, params, nodes[i], nodes[i + 1], r_ohms, format!("{id}_sp{i}"));
    }
}
