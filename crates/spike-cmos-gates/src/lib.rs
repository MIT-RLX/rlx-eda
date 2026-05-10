//! CMOS gate-level standard cells: `Inverter`, `Nand2`, `Nand3`, `And2`.
//!
//! Each gate is a [`SpiceEmit`] block composing the `Nmos` / `Pmos`
//! primitives from `eda-spice-emit`. The CMOS topologies follow the
//! textbook (and the LTspice SAR ADC paper):
//!
//! - **Inverter**: 1 PMOS pull-up + 1 NMOS pull-down, both gates tied
//!   to the input.
//! - **Nand2** (`!(a & b)`): 2 PMOS in **parallel** between Vdd and
//!   `out`, 2 NMOS in **series** between `out` and Gnd.
//! - **Nand3** (`!(a & b & c)`): 3 PMOS parallel + 3 NMOS in series.
//! - **And2** (`a & b`): Nand2 + inverter.
//!
//! ## Net ordering convention
//!
//! Every gate puts **signal nets first, supply nets last** in the
//! `nets: &[&str]` slice passed to `emit_spice`:
//!
//! | Gate     | nets order                          |
//! | -------- | ----------------------------------- |
//! | Inverter | `[in, out, vdd, gnd]`               |
//! | Nand2    | `[a, b, out, vdd, gnd]`             |
//! | Nand3    | `[a, b, c, out, vdd, gnd]`          |
//! | And2     | `[a, b, out, vdd, gnd]`             |
//!
//! Supply nets are explicit (not hardcoded `vdd`/`0`) so the same gate
//! type drops into power-gated subcircuits, dual-rail logic, etc.
//! without modification.
//!
//! ## Internal-net naming
//!
//! Multi-stage gates (Nand2/Nand3 series-NMOS stack, And2's internal
//! Nand→Inverter node) generate internal nets by appending an `id`
//! suffix to the gate's instance designator: `<id>_int1`, `<id>_int2`,
//! etc. This keeps internal nets unique across multiple instances of
//! the same gate type in one deck without the caller having to know
//! the gate's internal structure.

use eda_spice_emit::{EmitError, Netlist, Nmos, Pmos, SpiceEmit};

pub mod schematic;

pub mod mna;

// ── Inverter ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Inverter {
    pub nmos: Nmos,
    pub pmos: Pmos,
}

impl Default for Inverter {
    fn default() -> Self {
        Self { nmos: Nmos::default(), pmos: Pmos::default() }
    }
}

impl SpiceEmit for Inverter {
    fn n_terminals(&self) -> usize { 4 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        check_arity("Inverter", self.n_terminals(), nets.len())?;
        let (in_, out, vdd, gnd) = (nets[0], nets[1], nets[2], nets[3]);
        // PMOS: drain=out, gate=in, source=vdd, bulk=vdd
        self.pmos.emit_spice(n, &[out, in_, vdd, vdd], &format!("{id}_p"))?;
        // NMOS: drain=out, gate=in, source=gnd, bulk=gnd
        self.nmos.emit_spice(n, &[out, in_, gnd, gnd], &format!("{id}_n"))?;
        Ok(())
    }
}

fn check_arity(block: &str, expected: usize, got: usize) -> Result<(), EmitError> {
    if expected != got {
        return Err(EmitError::ArityMismatch { block: block.into(), expected, got });
    }
    Ok(())
}

// ── Nand2 ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Nand2 {
    pub nmos: Nmos,
    pub pmos: Pmos,
}

impl Default for Nand2 {
    fn default() -> Self {
        // NMOS in series — each one sees half the on-conductance of a
        // standalone gate. Double the W to compensate (textbook
        // sizing rule). PMOS in parallel, so default sizing is fine.
        let nmos = Nmos { w: 2.0 * Nmos::default().w, ..Nmos::default() };
        Self { nmos, pmos: Pmos::default() }
    }
}

impl SpiceEmit for Nand2 {
    fn n_terminals(&self) -> usize { 5 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        check_arity("Nand2", self.n_terminals(), nets.len())?;
        let (a, b, out, vdd, gnd) = (nets[0], nets[1], nets[2], nets[3], nets[4]);
        // PMOS pair (parallel between vdd and out):
        self.pmos.emit_spice(n, &[out, a, vdd, vdd], &format!("{id}_pa"))?;
        self.pmos.emit_spice(n, &[out, b, vdd, vdd], &format!("{id}_pb"))?;
        // NMOS pair (series between out and gnd, via internal mid node):
        let mid = format!("{id}_int1");
        self.nmos.emit_spice(n, &[out, a, &mid, gnd], &format!("{id}_na"))?;
        self.nmos.emit_spice(n, &[&mid, b, gnd, gnd], &format!("{id}_nb"))?;
        Ok(())
    }
}

// ── Nand3 ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Nand3 {
    pub nmos: Nmos,
    pub pmos: Pmos,
}

impl Default for Nand3 {
    fn default() -> Self {
        // 3 NMOS in series ⇒ 3× width compensation.
        let nmos = Nmos { w: 3.0 * Nmos::default().w, ..Nmos::default() };
        Self { nmos, pmos: Pmos::default() }
    }
}

impl SpiceEmit for Nand3 {
    fn n_terminals(&self) -> usize { 6 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        check_arity("Nand3", self.n_terminals(), nets.len())?;
        let (a, b, c, out, vdd, gnd) =
            (nets[0], nets[1], nets[2], nets[3], nets[4], nets[5]);
        // PMOS triad (parallel):
        self.pmos.emit_spice(n, &[out, a, vdd, vdd], &format!("{id}_pa"))?;
        self.pmos.emit_spice(n, &[out, b, vdd, vdd], &format!("{id}_pb"))?;
        self.pmos.emit_spice(n, &[out, c, vdd, vdd], &format!("{id}_pc"))?;
        // NMOS triad (series via two internal nodes):
        let mid1 = format!("{id}_int1");
        let mid2 = format!("{id}_int2");
        self.nmos.emit_spice(n, &[out, a, &mid1, gnd], &format!("{id}_na"))?;
        self.nmos.emit_spice(n, &[&mid1, b, &mid2, gnd], &format!("{id}_nb"))?;
        self.nmos.emit_spice(n, &[&mid2, c, gnd, gnd], &format!("{id}_nc"))?;
        Ok(())
    }
}

// ── And2 ───────────────────────────────────────────────────────────────

/// `a & b`. Topology = Nand2 → Inverter, with the Nand's output piped
/// through an internal node to the Inverter's input.
#[derive(Debug, Clone, Copy, Default)]
pub struct And2 {
    pub nand: Nand2,
    pub inv: Inverter,
}

impl SpiceEmit for And2 {
    fn n_terminals(&self) -> usize { 5 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        check_arity("And2", self.n_terminals(), nets.len())?;
        let (a, b, out, vdd, gnd) = (nets[0], nets[1], nets[2], nets[3], nets[4]);
        let nand_out = format!("{id}_int1");
        self.nand
            .emit_spice(n, &[a, b, &nand_out, vdd, gnd], &format!("{id}_nd"))?;
        self.inv
            .emit_spice(n, &[&nand_out, out, vdd, gnd], &format!("{id}_iv"))?;
        Ok(())
    }
}

// ── Nor2 ───────────────────────────────────────────────────────────────

/// `!(a | b)`. CMOS dual of Nand2: 2 PMOS in **series** between Vdd
/// and `out`, 2 NMOS in **parallel** between `out` and Gnd.
///
/// Sizing: PMOS in series doubles the on-resistance per device, so
/// each PMOS gets 2× the default width to keep the pull-up strength
/// comparable to a standalone inverter. NMOS in parallel — default
/// sizing is fine.
#[derive(Debug, Clone, Copy)]
pub struct Nor2 {
    pub nmos: Nmos,
    pub pmos: Pmos,
}

impl Default for Nor2 {
    fn default() -> Self {
        let pmos = Pmos { w: 2.0 * Pmos::default().w, ..Pmos::default() };
        Self { nmos: Nmos::default(), pmos }
    }
}

impl SpiceEmit for Nor2 {
    fn n_terminals(&self) -> usize { 5 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        check_arity("Nor2", self.n_terminals(), nets.len())?;
        let (a, b, out, vdd, gnd) = (nets[0], nets[1], nets[2], nets[3], nets[4]);
        // PMOS pair (series between vdd and out, via internal mid node):
        let mid = format!("{id}_int1");
        self.pmos.emit_spice(n, &[&mid, a, vdd, vdd], &format!("{id}_pa"))?;
        self.pmos.emit_spice(n, &[out, b, &mid, vdd], &format!("{id}_pb"))?;
        // NMOS pair (parallel between out and gnd):
        self.nmos.emit_spice(n, &[out, a, gnd, gnd], &format!("{id}_na"))?;
        self.nmos.emit_spice(n, &[out, b, gnd, gnd], &format!("{id}_nb"))?;
        Ok(())
    }
}

// ── Or2 ────────────────────────────────────────────────────────────────

/// `a | b`. Topology = Nor2 → Inverter, mirroring And2's Nand2 →
/// Inverter pattern.
#[derive(Debug, Clone, Copy, Default)]
pub struct Or2 {
    pub nor: Nor2,
    pub inv: Inverter,
}

impl SpiceEmit for Or2 {
    fn n_terminals(&self) -> usize { 5 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        check_arity("Or2", self.n_terminals(), nets.len())?;
        let (a, b, out, vdd, gnd) = (nets[0], nets[1], nets[2], nets[3], nets[4]);
        let nor_out = format!("{id}_int1");
        self.nor
            .emit_spice(n, &[a, b, &nor_out, vdd, gnd], &format!("{id}_nr"))?;
        self.inv
            .emit_spice(n, &[&nor_out, out, vdd, gnd], &format!("{id}_iv"))?;
        Ok(())
    }
}

// ── DLatch (gated D latch, level-sensitive) ───────────────────────────

/// Level-sensitive D latch built from 4 [`Nand2`] gates + 1
/// [`Inverter`] on the data input.
///
/// Behavior:
/// - When `en = 1`: output `q` follows `d` (transparent).
/// - When `en = 0`: latch holds previous state.
///
/// ## Topology (4 NAND2 + 1 Inverter)
///
/// ```text
///        d ──┬───────G1(NAND)─── a (= S_bar) ──┐
///            │            │                    │
///            │            en                   │
///   ┌────INV(d_inv)                            │
///   │                     en                   │
///   d_inv ─────────G2(NAND)─── b (= R_bar) ─┐  │
///                                           │  │
///                              cross-coupled│  │
///                              SR latch:    │  │
///                                           │  │
///                G3: q  = NAND(a, qb) ──────┼──┘
///                G4: qb = NAND(b, q) ───────┘
/// ```
///
/// We use the canonical 5-gate structure (with explicit `INV(d)`)
/// rather than the 4-gate trick (`b = NAND(a, en)`) because the
/// 4-gate variant suffers a dynamic glitch on the falling `en` edge:
/// `a` and `b` both transition through midrail simultaneously,
/// momentarily collapsing the SR latch. The explicit inverter
/// decouples the two SR-latch inputs so they transition independently.
///
/// ## Net order
///
/// `[d, en, q, qb, vdd, gnd]`
#[derive(Debug, Clone, Copy, Default)]
pub struct DLatch {
    pub nand: Nand2,
    pub inv: Inverter,
}

impl SpiceEmit for DLatch {
    fn n_terminals(&self) -> usize { 6 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        check_arity("DLatch", self.n_terminals(), nets.len())?;
        let (d, en, q, qb, vdd, gnd) =
            (nets[0], nets[1], nets[2], nets[3], nets[4], nets[5]);
        let d_inv = format!("{id}_dinv");
        let a = format!("{id}_a");
        let b = format!("{id}_b");
        // INV: d_inv = !d
        self.inv.emit_spice(n, &[d, &d_inv, vdd, gnd], &format!("{id}_dvi"))?;
        // G1: a = NAND(d, en)        (S_bar)
        self.nand.emit_spice(n, &[d, en, &a, vdd, gnd], &format!("{id}_g1"))?;
        // G2: b = NAND(d_inv, en)    (R_bar)
        self.nand.emit_spice(n, &[&d_inv, en, &b, vdd, gnd], &format!("{id}_g2"))?;
        // G3: q  = NAND(a, qb)
        self.nand.emit_spice(n, &[&a, qb, q, vdd, gnd], &format!("{id}_g3"))?;
        // G4: qb = NAND(b, q)
        self.nand.emit_spice(n, &[&b, q, qb, vdd, gnd], &format!("{id}_g4"))?;
        Ok(())
    }
}

// ── Dff (positive-edge-triggered D flip-flop) ─────────────────────────

/// Positive-edge-triggered D flip-flop, master–slave topology.
///
/// Two [`DLatch`]es with opposite-polarity enables and one [`Inverter`]
/// to invert the clock. The master (sees `~clk`) is transparent while
/// the clock is low and captures `D` just before the rising edge; the
/// slave (sees `clk`) becomes transparent on the rising edge and
/// passes the captured value to `Q`.
///
/// ## Net order
///
/// `[d, clk, q, qb, vdd, gnd]`
#[derive(Debug, Clone, Copy, Default)]
pub struct Dff {
    pub master: DLatch,
    pub slave: DLatch,
    pub clk_inv: Inverter,
}

impl SpiceEmit for Dff {
    fn n_terminals(&self) -> usize { 6 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        check_arity("Dff", self.n_terminals(), nets.len())?;
        let (d, clk, q, qb, vdd, gnd) =
            (nets[0], nets[1], nets[2], nets[3], nets[4], nets[5]);
        let clk_n = format!("{id}_clkn");
        // Inverter on clk → clk_n
        self.clk_inv
            .emit_spice(n, &[clk, &clk_n, vdd, gnd], &format!("{id}_civ"))?;
        // Master: transparent when clk=0 (en = clk_n).
        let mq = format!("{id}_mq");
        let mqb = format!("{id}_mqb");
        self.master.emit_spice(
            n,
            &[d, &clk_n, &mq, &mqb, vdd, gnd],
            &format!("{id}_m"),
        )?;
        // Master → slave isolation buffer (2 cascaded inverters).
        // Without this, when master is transparent (clk=0) it tracks d;
        // those transitions reach slave's input gates and can kick the
        // slave's "opaque-mode" SR storage through Miller / parasitic
        // coupling. The buffer gives the slave clean rail-to-rail
        // transitions on its d input only when master itself has
        // settled, dramatically reducing the kick.
        let mq_buf_mid = format!("{id}_mqbufm");
        let mq_buf = format!("{id}_mqbuf");
        self.clk_inv.emit_spice(n, &[&mq, &mq_buf_mid, vdd, gnd], &format!("{id}_buf1"))?;
        self.clk_inv.emit_spice(n, &[&mq_buf_mid, &mq_buf, vdd, gnd], &format!("{id}_buf2"))?;
        // Slave: transparent when clk=1; D = buffered master Q.
        self.slave.emit_spice(
            n,
            &[&mq_buf, clk, q, qb, vdd, gnd],
            &format!("{id}_s"),
        )?;
        Ok(())
    }
}

// ── DLatchSR (gated D latch with async set / reset) ───────────────────

/// Level-sensitive D latch with **active-low** asynchronous set and
/// reset. Same shape as [`DLatch`] except the SR cross-couple uses
/// [`Nand3`] gates so a third input can override the latch output:
///
/// - `set_b = 0`: forces `q = 1` regardless of `d` / `en`.
/// - `reset_b = 0`: forces `q = 0` regardless of `d` / `en`.
/// - `set_b = reset_b = 1`: behaves exactly like [`DLatch`].
///
/// Driving both `set_b = 0` and `reset_b = 0` simultaneously is
/// undefined (forces `q = qb = 1` — the metastable corner of the SR
/// latch). Real designs gate this combination out.
///
/// ## Net order
///
/// `[d, en, set_b, reset_b, q, qb, vdd, gnd]` — 8 terminals.
#[derive(Debug, Clone, Copy, Default)]
pub struct DLatchSR {
    pub nand2: Nand2,
    pub nand3: Nand3,
    pub inv: Inverter,
}

impl SpiceEmit for DLatchSR {
    fn n_terminals(&self) -> usize { 8 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        check_arity("DLatchSR", self.n_terminals(), nets.len())?;
        let (d, en, set_b, reset_b, q, qb, vdd, gnd) = (
            nets[0], nets[1], nets[2], nets[3], nets[4], nets[5], nets[6], nets[7],
        );
        let d_inv = format!("{id}_dinv");
        let a = format!("{id}_a");
        let b = format!("{id}_b");
        // INV: d_inv = !d
        self.inv.emit_spice(n, &[d, &d_inv, vdd, gnd], &format!("{id}_dvi"))?;
        // G1: a = NAND(d, en) (S_bar from gating)
        self.nand2.emit_spice(n, &[d, en, &a, vdd, gnd], &format!("{id}_g1"))?;
        // G2: b = NAND(d_inv, en) (R_bar from gating)
        self.nand2.emit_spice(n, &[&d_inv, en, &b, vdd, gnd], &format!("{id}_g2"))?;
        // G3 (SR-set side): q  = NAND3(a, qb, set_b)
        //   set_b = 0 ⇒ q = 1 unconditionally.
        self.nand3
            .emit_spice(n, &[&a, qb, set_b, q, vdd, gnd], &format!("{id}_g3"))?;
        // G4 (SR-reset side): qb = NAND3(b, q, reset_b)
        //   reset_b = 0 ⇒ qb = 1 unconditionally, which forces q = NAND(_, 1) = ~_ ;
        //   combined with G3's qb=1 input → q = NAND(a, 1, set_b) = ~(a & set_b),
        //   which collapses to q=0 if a=set_b=1 (the typical reset case).
        self.nand3
            .emit_spice(n, &[&b, q, reset_b, qb, vdd, gnd], &format!("{id}_g4"))?;
        Ok(())
    }
}

// ── DffSR (positive-edge-triggered D flip-flop with async S/R) ────────

/// Positive-edge-triggered D flip-flop with **active-low** asynchronous
/// set and reset. Composes two [`DLatchSR`]es with opposite-polarity
/// enables and one [`Inverter`] on the clock — same master/slave
/// topology as [`Dff`], but both stages honor the override pins so the
/// output snaps regardless of clock state.
///
/// Use this for SAR registers and any logic that needs power-on reset
/// or asynchronous initialization. The plain [`Dff`] is fine for purely
/// clocked datapath registers.
///
/// ## Net order
///
/// `[d, clk, set_b, reset_b, q, qb, vdd, gnd]` — 8 terminals.
#[derive(Debug, Clone, Copy, Default)]
pub struct DffSR {
    pub master: DLatchSR,
    pub slave: DLatchSR,
    pub clk_inv: Inverter,
}

impl SpiceEmit for DffSR {
    fn n_terminals(&self) -> usize { 8 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        check_arity("DffSR", self.n_terminals(), nets.len())?;
        let (d, clk, set_b, reset_b, q, qb, vdd, gnd) = (
            nets[0], nets[1], nets[2], nets[3], nets[4], nets[5], nets[6], nets[7],
        );
        let clk_n = format!("{id}_clkn");
        self.clk_inv
            .emit_spice(n, &[clk, &clk_n, vdd, gnd], &format!("{id}_civ"))?;
        let mq = format!("{id}_mq");
        let mqb = format!("{id}_mqb");
        // Master: transparent when clk = 0 (en = clk_n).
        self.master.emit_spice(
            n,
            &[d, &clk_n, set_b, reset_b, &mq, &mqb, vdd, gnd],
            &format!("{id}_m"),
        )?;
        // Master → slave isolation buffer (2 cascaded inverters). See
        // the matching comment in `Dff` — necessary so master's tracking
        // of `d` while slave is opaque doesn't kick the slave's stored
        // value through input-gate Miller capacitance. Without this
        // buffer, SAR-style use cases (where d = comparator output
        // changes during subsequent phases) corrupt previously-latched
        // bits.
        let mq_buf_mid = format!("{id}_mqbufm");
        let mq_buf = format!("{id}_mqbuf");
        self.clk_inv.emit_spice(n, &[&mq, &mq_buf_mid, vdd, gnd], &format!("{id}_buf1"))?;
        self.clk_inv.emit_spice(n, &[&mq_buf_mid, &mq_buf, vdd, gnd], &format!("{id}_buf2"))?;
        // Slave: transparent when clk = 1; D = buffered master Q.
        self.slave.emit_spice(
            n,
            &[&mq_buf, clk, set_b, reset_b, q, qb, vdd, gnd],
            &format!("{id}_s"),
        )?;
        Ok(())
    }
}

// ── Truth-table helper ────────────────────────────────────────────────

/// Generic deck builder for "build supply, set input voltages, instance
/// `gate`, run `.op`, read `out`". Inputs are specified as
/// `(net_name, level)` pairs — `0` = ground, `1` = `vdd`. This is what
/// the truth-table tests call once per input combination.
pub fn deck_for_levels(
    gate: &impl SpiceEmit,
    input_nets: &[&str],
    input_levels: &[u8],
    out_net: &str,
    vdd: f64,
    gate_id: &str,
) -> Netlist {
    assert_eq!(
        input_nets.len(),
        input_levels.len(),
        "input_nets vs input_levels length mismatch"
    );
    let mut net = Netlist::new("CMOS gate truth-table probe");
    net.add_dc_source("dd", "vdd", "0", vdd);
    for (n_, &lvl) in input_nets.iter().zip(input_levels.iter()) {
        let v = if lvl == 1 { vdd } else { 0.0 };
        net.add_dc_source(&format!("in_{n_}"), n_, "0", v);
    }
    let mut nets: Vec<&str> = input_nets.to_vec();
    nets.push(out_net);
    nets.push("vdd");
    nets.push("0");
    gate.emit_spice(&mut net, &nets, gate_id).expect("emit_spice");
    net
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverter_emits_two_devices() {
        let mut n = Netlist::new("t");
        Inverter::default()
            .emit_spice(&mut n, &["in", "out", "vdd", "0"], "i1")
            .unwrap();
        assert_eq!(n.body.len(), 2);
    }

    #[test]
    fn nand2_emits_four_devices_with_internal_node() {
        let mut n = Netlist::new("t");
        Nand2::default()
            .emit_spice(&mut n, &["a", "b", "out", "vdd", "0"], "u1")
            .unwrap();
        assert_eq!(n.body.len(), 4);
        assert!(n.body.join("\n").contains("u1_int1"));
    }

    #[test]
    fn nand3_emits_six_devices_with_two_internal_nodes() {
        let mut n = Netlist::new("t");
        Nand3::default()
            .emit_spice(&mut n, &["a", "b", "c", "out", "vdd", "0"], "u1")
            .unwrap();
        assert_eq!(n.body.len(), 6);
        let body = n.body.join("\n");
        assert!(body.contains("u1_int1"));
        assert!(body.contains("u1_int2"));
    }

    #[test]
    fn and2_emits_nand_plus_inverter() {
        let mut n = Netlist::new("t");
        And2::default()
            .emit_spice(&mut n, &["a", "b", "out", "vdd", "0"], "u1")
            .unwrap();
        assert_eq!(n.body.len(), 6);
    }

    #[test]
    fn nor2_emits_four_devices_with_internal_node() {
        let mut n = Netlist::new("t");
        Nor2::default()
            .emit_spice(&mut n, &["a", "b", "out", "vdd", "0"], "u1")
            .unwrap();
        // 2 PMOS + 2 NMOS = 4 device lines.
        assert_eq!(n.body.len(), 4);
        // PMOS series mid-node
        assert!(n.body.join("\n").contains("u1_int1"));
    }

    #[test]
    fn or2_emits_nor_plus_inverter() {
        let mut n = Netlist::new("t");
        Or2::default()
            .emit_spice(&mut n, &["a", "b", "out", "vdd", "0"], "u1")
            .unwrap();
        // Nor2 (4 devices) + Inverter (2) = 6.
        assert_eq!(n.body.len(), 6);
    }

    #[test]
    fn deck_for_levels_sets_inputs_to_supply_or_ground() {
        let net = deck_for_levels(
            &Nand2::default(),
            &["a", "b"],
            &[1, 0],
            "out",
            1.8,
            "u1",
        );
        let deck = net.deck();
        assert!(deck.contains("Vdd vdd 0 DC"));
        assert!(deck.contains("Vin_a a 0 DC 1.8"));
        assert!(deck.contains("Vin_b b 0 DC 0.0"));
    }

    #[test]
    fn arity_mismatch_caught() {
        let mut n = Netlist::new("t");
        let err = Nand2::default()
            .emit_spice(&mut n, &["a", "b", "out", "vdd"], "u1")
            .unwrap_err();
        assert!(matches!(err, EmitError::ArityMismatch { expected: 5, got: 4, .. }));
    }

    #[test]
    fn dlatch_emits_inverter_plus_four_nand2() {
        let mut n = Netlist::new("t");
        DLatch::default()
            .emit_spice(&mut n, &["d", "en", "q", "qb", "vdd", "0"], "l1")
            .unwrap();
        // 4 NAND2 × 4 transistors + 1 Inverter × 2 transistors = 18.
        assert_eq!(n.body.len(), 18);
        let body = n.body.join("\n");
        assert!(body.contains("l1_dinv"));
        assert!(body.contains("l1_a"));
        assert!(body.contains("l1_b"));
    }

    #[test]
    fn dff_emits_master_slave_plus_clock_inv() {
        let mut n = Netlist::new("t");
        Dff::default()
            .emit_spice(&mut n, &["d", "clk", "q", "qb", "vdd", "0"], "f1")
            .unwrap();
        // 2 DLatch (18 ea) + 1 clock inverter + 2 buffer inverters
        // (2 ea = 4) = 18 + 18 + 2 + 4 = 42.
        assert_eq!(n.body.len(), 42);
        let body = n.body.join("\n");
        assert!(body.contains("f1_clkn"));
        assert!(body.contains("f1_mq"));
        assert!(body.contains("f1_mqbuf"));
    }

    #[test]
    fn dlatchsr_emits_inverter_two_nand2_two_nand3() {
        let mut n = Netlist::new("t");
        DLatchSR::default()
            .emit_spice(
                &mut n,
                &["d", "en", "sb", "rb", "q", "qb", "vdd", "0"],
                "l1",
            )
            .unwrap();
        // Inverter (2) + 2×NAND2 (4 each = 8) + 2×NAND3 (6 each = 12) = 22.
        assert_eq!(n.body.len(), 22);
        let body = n.body.join("\n");
        assert!(body.contains("l1_dinv"));
        assert!(body.contains("l1_a"));
        assert!(body.contains("l1_b"));
    }

    #[test]
    fn dffsr_emits_two_dlatchsr_plus_clock_inv() {
        let mut n = Netlist::new("t");
        DffSR::default()
            .emit_spice(
                &mut n,
                &["d", "clk", "sb", "rb", "q", "qb", "vdd", "0"],
                "f1",
            )
            .unwrap();
        // 2×DLatchSR (22 each = 44) + 1 clock inverter + 2 buffer
        // inverters (2 ea = 4) = 50.
        assert_eq!(n.body.len(), 50);
        let body = n.body.join("\n");
        assert!(body.contains("f1_clkn"));
        assert!(body.contains("f1_mq"));
        assert!(body.contains("f1_mqbuf"));
    }
}
