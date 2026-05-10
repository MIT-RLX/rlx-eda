//! eda-mna composition functions for the gate-level digital library.
//!
//! Mirrors the existing `SpiceEmit` impls (Inverter / Nand2 / Nand3 /
//! And2 / DLatch / Dff / DLatchSR / DffSR) but emits into an
//! `eda_mna::Circuit` from `spike_divider_block::Mosfet` primitives,
//! so the resulting circuit runs under the differentiable BE solver +
//! `transient_sensitivities`.
//!
//! Same net-order conventions as the SpiceEmit path. Each `add_*`
//! function takes the circuit, the terminal nets in spec order, an
//! id prefix used for transistor name namespacing, and a mutable
//! `params` map into which it writes default Mosfet param values.
//!
//! ## Conventions
//!
//! - All transistors L = 1 µm.
//! - Default NMOS W = 2 µm; PMOS W = 4 µm (β-matched ≈ 2:1 for our
//!   square-law model with Kp_p ≈ Kp_n / 2).
//! - Series stacks (Nand2 NMOS pair, Nand3 NMOS triad) get their
//!   width scaled by the stack depth so the on-conductance per gate
//!   stays comparable to a standalone inverter — matches the
//!   `SpiceEmit::Default` sizing in this crate.
//! - Each digital node also gets a small to-ground load cap (~5 fF)
//!   to give the BE-step solver finite history coupling. Without
//!   these the digital nodes have no storage and the transient
//!   collapses to instantaneous DC.

use std::collections::HashMap;

use eda_mna::{Circuit, LinearCap, NetId};
use spike_divider_block::Mosfet;

const W_NMOS: i64 = 2_000;
const W_PMOS: i64 = 4_000;
const L:      i64 = 1_000;
const C_NODE_F: f32 = 5e-15;     // 5 fF per digital net

fn add_node_cap(c: &mut Circuit, id: &str, net: NetId, params: &mut HashMap<String, f32>) {
    let key = format!("Cnode_{id}");
    c.add_storage(LinearCap::new(key.clone()), [net, NetId::GND]);
    params.insert(key, C_NODE_F);
}

fn stamp_default_params(m: &Mosfet, params: &mut HashMap<String, f32>) {
    params.extend(m.default_params());
}

/// Inverter: PMOS pull-up + NMOS pull-down, gates tied at `in`.
/// Nets: `[in, out, vdd, gnd]`.
pub fn add_inverter(c: &mut Circuit, nets: [NetId; 4], id: &str,
                    params: &mut HashMap<String, f32>)
{
    let (in_, out, vdd, gnd) = (nets[0], nets[1], nets[2], nets[3]);
    let n = Mosfet::nmos(W_NMOS, L, format!("{id}_n"));
    let p = Mosfet::pmos(W_PMOS, L, format!("{id}_p"));
    c.add_device(n.clone(), &[out, in_, gnd, gnd]);
    c.add_device(p.clone(), &[out, in_, vdd, vdd]);
    stamp_default_params(&n, params);
    stamp_default_params(&p, params);
    add_node_cap(c, &format!("{id}_out"), out, params);
}

/// `!(a & b)`. Nets: `[a, b, out, vdd, gnd]`.
pub fn add_nand2(c: &mut Circuit, nets: [NetId; 5], id: &str,
                 params: &mut HashMap<String, f32>)
{
    let (a, b, out, vdd, gnd) = (nets[0], nets[1], nets[2], nets[3], nets[4]);
    let mid = c.alloc_unknown_net();
    let p_a = Mosfet::pmos(W_PMOS,     L, format!("{id}_pa"));
    let p_b = Mosfet::pmos(W_PMOS,     L, format!("{id}_pb"));
    let n_a = Mosfet::nmos(2 * W_NMOS, L, format!("{id}_na"));   // 2× for series-stack compensation
    let n_b = Mosfet::nmos(2 * W_NMOS, L, format!("{id}_nb"));
    c.add_device(p_a.clone(), &[out, a, vdd, vdd]);
    c.add_device(p_b.clone(), &[out, b, vdd, vdd]);
    c.add_device(n_a.clone(), &[out, a, mid, gnd]);
    c.add_device(n_b.clone(), &[mid, b, gnd, gnd]);
    for m in [&p_a, &p_b, &n_a, &n_b] { stamp_default_params(m, params); }
    add_node_cap(c, &format!("{id}_out"), out, params);
    add_node_cap(c, &format!("{id}_mid"), mid, params);
}

/// `!(a & b & c)`. Nets: `[a, b, c, out, vdd, gnd]`.
pub fn add_nand3(c: &mut Circuit, nets: [NetId; 6], id: &str,
                 params: &mut HashMap<String, f32>)
{
    let (a, b, cc, out, vdd, gnd) = (nets[0], nets[1], nets[2], nets[3], nets[4], nets[5]);
    let mid1 = c.alloc_unknown_net();
    let mid2 = c.alloc_unknown_net();
    let p_a = Mosfet::pmos(W_PMOS,     L, format!("{id}_pa"));
    let p_b = Mosfet::pmos(W_PMOS,     L, format!("{id}_pb"));
    let p_c = Mosfet::pmos(W_PMOS,     L, format!("{id}_pc"));
    let n_a = Mosfet::nmos(3 * W_NMOS, L, format!("{id}_na"));   // 3× for triple-stack compensation
    let n_b = Mosfet::nmos(3 * W_NMOS, L, format!("{id}_nb"));
    let n_c = Mosfet::nmos(3 * W_NMOS, L, format!("{id}_nc"));
    c.add_device(p_a.clone(), &[out, a,  vdd,  vdd]);
    c.add_device(p_b.clone(), &[out, b,  vdd,  vdd]);
    c.add_device(p_c.clone(), &[out, cc, vdd,  vdd]);
    c.add_device(n_a.clone(), &[out,  a,  mid1, gnd]);
    c.add_device(n_b.clone(), &[mid1, b,  mid2, gnd]);
    c.add_device(n_c.clone(), &[mid2, cc, gnd,  gnd]);
    for m in [&p_a, &p_b, &p_c, &n_a, &n_b, &n_c] { stamp_default_params(m, params); }
    add_node_cap(c, &format!("{id}_out"),  out,  params);
    add_node_cap(c, &format!("{id}_mid1"), mid1, params);
    add_node_cap(c, &format!("{id}_mid2"), mid2, params);
}

/// `a & b`. Topology = Nand2 → Inverter via internal node.
/// Nets: `[a, b, out, vdd, gnd]`.
pub fn add_and2(c: &mut Circuit, nets: [NetId; 5], id: &str,
                params: &mut HashMap<String, f32>)
{
    let (a, b, out, vdd, gnd) = (nets[0], nets[1], nets[2], nets[3], nets[4]);
    let nand_out = c.alloc_unknown_net();
    add_nand2(c, [a, b, nand_out, vdd, gnd], &format!("{id}_nd"), params);
    add_inverter(c, [nand_out, out, vdd, gnd], &format!("{id}_iv"), params);
}

/// Level-sensitive D latch (NAND-based SR + input gating).
/// Transparent when `en = 1`; opaque when `en = 0`.
/// Nets: `[d, en, q, qb, vdd, gnd]`.
///
/// `ic` (optional) is populated with seeds that put the SR cross-couple
/// in a stable state at t = 0. With all-zero ICs the SR latch sits at
/// Q = Qb = 1, which is its metastable corner — Newton can fail to
/// settle. The seeds break the symmetry to Q = 0, Qb = Vdd.
pub fn add_dlatch(c: &mut Circuit, nets: [NetId; 6], id: &str,
                  params: &mut HashMap<String, f32>,
                  ic: Option<&mut HashMap<NetId, f32>>)
{
    let (d, en, q, qb, vdd, gnd) = (nets[0], nets[1], nets[2], nets[3], nets[4], nets[5]);
    let d_inv = c.alloc_unknown_net();
    let a     = c.alloc_unknown_net();
    let b     = c.alloc_unknown_net();
    add_inverter(c, [d, d_inv, vdd, gnd], &format!("{id}_dvi"), params);
    add_nand2(c, [d, en, a, vdd, gnd],     &format!("{id}_g1"), params);
    add_nand2(c, [d_inv, en, b, vdd, gnd], &format!("{id}_g2"), params);
    add_nand2(c, [a, qb, q, vdd, gnd],     &format!("{id}_g3"), params);
    add_nand2(c, [b, q,  qb, vdd, gnd],    &format!("{id}_g4"), params);
    if let Some(ic) = ic {
        // Pick Q=0, Qb=Vdd as the stable cross-couple seed. Inputs a
        // and b sit high (NAND of any 0 = 1) since en=0 initially.
        ic.insert(q, 0.0);
        ic.insert(qb, V_VDD_GUESS);
        ic.insert(a,  V_VDD_GUESS);
        ic.insert(b,  V_VDD_GUESS);
        ic.insert(d_inv, V_VDD_GUESS);
    }
}

/// Default Vdd guess used by IC seeders. Matches the demo binaries'
/// `VDD = 1.8 V`. Keeping this as an internal constant rather than a
/// per-call argument keeps the IC-aware add_* signatures simple.
const V_VDD_GUESS: f32 = 1.8;

/// Positive-edge-triggered D flip-flop (master + slave + clk inverter +
/// 2-inverter master→slave isolation buffer).
/// Nets: `[d, clk, q, qb, vdd, gnd]`.
///
/// `ic` (optional) seeds master + slave SR cross-couples to a stable
/// state at t = 0 (Q = 0, Qb = Vdd). Without these, the SR latches
/// start at the metastable corner and Newton can stall indefinitely.
pub fn add_dff(c: &mut Circuit, nets: [NetId; 6], id: &str,
               params: &mut HashMap<String, f32>,
               mut ic: Option<&mut HashMap<NetId, f32>>)
{
    let (d, clk, q, qb, vdd, gnd) = (nets[0], nets[1], nets[2], nets[3], nets[4], nets[5]);
    let clk_n  = c.alloc_unknown_net();
    let mq     = c.alloc_unknown_net();
    let mqb    = c.alloc_unknown_net();
    let buf_m  = c.alloc_unknown_net();
    let buf_o  = c.alloc_unknown_net();
    add_inverter(c, [clk, clk_n, vdd, gnd],          &format!("{id}_civ"), params);
    add_dlatch  (c, [d, clk_n, mq, mqb, vdd, gnd],   &format!("{id}_m"),   params, ic.as_deref_mut());
    add_inverter(c, [mq, buf_m, vdd, gnd],           &format!("{id}_buf1"), params);
    add_inverter(c, [buf_m, buf_o, vdd, gnd],        &format!("{id}_buf2"), params);
    add_dlatch  (c, [buf_o, clk, q, qb, vdd, gnd],   &format!("{id}_s"),   params, ic.as_deref_mut());
    if let Some(ic) = ic {
        // Clock starts low → clk_n inverter output starts high.
        ic.insert(clk_n, V_VDD_GUESS);
        // Master Q=0 → buf_m = Vdd (after first inverter), buf_o = 0
        // (after second). Slave's d input thus starts at 0.
        ic.insert(buf_m, V_VDD_GUESS);
        ic.insert(buf_o, 0.0);
    }
}

/// Level-sensitive D latch with active-low async set / reset.
/// Nets: `[d, en, set_b, reset_b, q, qb, vdd, gnd]`.
pub fn add_dlatch_sr(c: &mut Circuit, nets: [NetId; 8], id: &str,
                     params: &mut HashMap<String, f32>,
                     ic: Option<&mut HashMap<NetId, f32>>)
{
    let (d, en, set_b, reset_b, q, qb, vdd, gnd) =
        (nets[0], nets[1], nets[2], nets[3], nets[4], nets[5], nets[6], nets[7]);
    let d_inv = c.alloc_unknown_net();
    let a     = c.alloc_unknown_net();
    let b     = c.alloc_unknown_net();
    add_inverter(c, [d, d_inv, vdd, gnd],          &format!("{id}_dvi"), params);
    add_nand2(c, [d, en, a, vdd, gnd],             &format!("{id}_g1"), params);
    add_nand2(c, [d_inv, en, b, vdd, gnd],         &format!("{id}_g2"), params);
    add_nand3(c, [a, qb, set_b,   q, vdd, gnd],    &format!("{id}_g3"), params);
    add_nand3(c, [b, q,  reset_b, qb, vdd, gnd],   &format!("{id}_g4"), params);
    if let Some(ic) = ic {
        ic.insert(q, 0.0);
        ic.insert(qb, V_VDD_GUESS);
        ic.insert(a,  V_VDD_GUESS);
        ic.insert(b,  V_VDD_GUESS);
        ic.insert(d_inv, V_VDD_GUESS);
    }
}

/// Positive-edge-triggered D flip-flop with active-low async set/reset.
/// Same master-slave topology as `add_dff` but both stages honor the
/// override pins.
/// Nets: `[d, clk, set_b, reset_b, q, qb, vdd, gnd]`.
pub fn add_dff_sr(c: &mut Circuit, nets: [NetId; 8], id: &str,
                  params: &mut HashMap<String, f32>,
                  mut ic: Option<&mut HashMap<NetId, f32>>)
{
    let (d, clk, set_b, reset_b, q, qb, vdd, gnd) =
        (nets[0], nets[1], nets[2], nets[3], nets[4], nets[5], nets[6], nets[7]);
    let clk_n = c.alloc_unknown_net();
    let mq    = c.alloc_unknown_net();
    let mqb   = c.alloc_unknown_net();
    let buf_m = c.alloc_unknown_net();
    let buf_o = c.alloc_unknown_net();
    add_inverter(c, [clk, clk_n, vdd, gnd],                                   &format!("{id}_civ"),  params);
    add_dlatch_sr(c, [d, clk_n, set_b, reset_b, mq, mqb, vdd, gnd],           &format!("{id}_m"),    params, ic.as_deref_mut());
    add_inverter(c, [mq, buf_m, vdd, gnd],                                    &format!("{id}_buf1"), params);
    add_inverter(c, [buf_m, buf_o, vdd, gnd],                                 &format!("{id}_buf2"), params);
    add_dlatch_sr(c, [buf_o, clk, set_b, reset_b, q, qb, vdd, gnd],           &format!("{id}_s"),    params, ic.as_deref_mut());
    if let Some(ic) = ic {
        ic.insert(clk_n, V_VDD_GUESS);
        ic.insert(buf_m, V_VDD_GUESS);
        ic.insert(buf_o, 0.0);
    }
}
