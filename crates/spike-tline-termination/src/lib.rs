//! Lossless transmission line driven by a Thevenin source into a high-Z
//! receiver — the canonical "why your 3.3 V clock rings to 5.5 V" demo.
//!
//! Topology
//! --------
//! ```text
//!   V_s(t) ──┬── R_drv ── R_term ──┬── T-line (Z0, TD) ──┬── (high-Z RX)
//!            │                     │                     │
//!           GND                   "vline_in"            "vrx"
//! ```
//! `R_term` is the series-source-termination resistor we are studying. The
//! "ringing case" sets `r_term = 0`. The "matched case" sets
//! `r_term = z0 - r_drv` so that the Thevenin impedance looking back into
//! the source equals `z0`, killing the source-end reflection coefficient.
//!
//! Validation pyramid (per workspace convention):
//! - **Tier 1 / analytic**: closed-form bounce diagram for an ideal V step,
//!   superposed for a `Pulse` (rising step at `td`, inverted falling step
//!   at `td+pw`). See `analytic_step_at` / `analytic_pulse_at`.
//! - **Tier 2 / FDTD**: discrete delay-line simulator with two queues of
//!   length `N = TD/h`. Exact for a lossless line on a uniform grid where
//!   `TD/h` is integer. See `fdtd_trace`.
//! - **Tier 3 / ngspice**: lossless `T` element via `Netlist::add_element`,
//!   compared trace-to-trace. See `tests/ngspice.rs`.

use eda_hir::SourceWaveform;
use eda_spice_emit::{Netlist, SpiceEmit, R};

/// Circuit parameters. `r_drv` is the driver output resistance (CMOS ~10 Ω);
/// `r_term` is the *additional* series resistor we are tuning. The two add
/// in series to form the Thevenin source impedance seen by the line.
#[derive(Debug, Clone, Copy)]
pub struct Topology {
    pub r_drv: f64,
    pub r_term: f64,
    pub z0: f64,
    pub td: f64,
    /// Receiver load. Use a large value (e.g. `1e9`) for "high-Z". Pure-open
    /// is `f64::INFINITY` and is supported by the analytic; the SPICE deck
    /// emits a finite shunt so ngspice has a DC path.
    pub r_load: f64,
}

impl Topology {
    /// `R_drv + R_term`, the total Thevenin source impedance.
    #[inline]
    pub fn rs_total(&self) -> f64 { self.r_drv + self.r_term }

    /// Source-end reflection coefficient `(R_s - Z0)/(R_s + Z0)`.
    #[inline]
    pub fn gamma_s(&self) -> f64 {
        let rs = self.rs_total();
        (rs - self.z0) / (rs + self.z0)
    }

    /// Load-end reflection coefficient `(R_L - Z0)/(R_L + Z0)`.
    /// `r_load = ∞` → `+1` (high-Z reflects in phase).
    #[inline]
    pub fn gamma_l(&self) -> f64 {
        if self.r_load.is_infinite() { 1.0 }
        else { (self.r_load - self.z0) / (self.r_load + self.z0) }
    }

    /// "Fresh" forward-wave amplitude the source launches given an *ideal*
    /// voltage `vs` at the V source — i.e. the response with no reflections
    /// yet present. Equals `vs · Z0/(Z0+R_s)`.
    #[inline]
    pub fn v_plus(&self, vs: f64) -> f64 {
        vs * self.z0 / (self.z0 + self.rs_total())
    }
}

// ── Analytic: bounce diagram ────────────────────────────────────────────

/// Receiver voltage as a function of time for an ideal `vs · u(t)` step
/// applied at the source. Sums all bounces that have arrived by time `t`.
///
/// Mathematically: `v_rx(t) = (1+ΓL)·V_+ · Σ_{k=0..K} (ΓL·ΓS)^k`
/// where `K = floor((t - TD) / (2·TD))` (number of completed round trips
/// since the first arrival).
pub fn analytic_step_at(topo: Topology, t: f64, vs: f64) -> f64 {
    if t < topo.td {
        return 0.0;
    }
    let vp = topo.v_plus(vs);
    let gl = topo.gamma_l();
    let gs = topo.gamma_s();
    // Number of (load-incident) wavefronts that have arrived by time t.
    // First arrival at TD; subsequent arrivals at 3·TD, 5·TD, ...
    let k_arrivals = (((t - topo.td) / (2.0 * topo.td)).floor() as i64).max(0) as u32;
    let mut sum = 0.0;
    let mut g = 1.0;
    for _ in 0..=k_arrivals {
        sum += g;
        g *= gl * gs;
    }
    (1.0 + gl) * vp * sum
}

/// Receiver voltage for a `SourceWaveform::Pulse` with `tr = tf = 0`.
/// Implemented by superposition of the step response: rising step of size
/// `(v2 − v1)` at `t = td`, falling step of size `(v1 − v2)` at `t = td+pw`.
/// The single-pulse case (`per ≤ 0` or `per` past the horizon) is exact.
pub fn analytic_pulse_at(topo: Topology, t: f64, w: &SourceWaveform) -> f64 {
    match *w {
        SourceWaveform::Pulse { v1, v2, td, tr, tf, pw, per: _ } => {
            assert!(tr == 0.0 && tf == 0.0,
                "analytic reference assumes tr=tf=0 (got tr={tr}, tf={tf})");
            let rising = analytic_step_at(topo, t - td, v2 - v1);
            let falling = if pw > 0.0 {
                analytic_step_at(topo, t - td - pw, v1 - v2)
            } else { 0.0 };
            v1 + rising + falling
        }
        _ => panic!("analytic_pulse_at: only Pulse waveforms are supported"),
    }
}

// ── FDTD: discrete delay-line simulator ─────────────────────────────────

/// Convenience: number of delay cells per direction for a given `(TD, h)`.
/// Panics if `TD/h` is not (very close to) an integer — the FDTD is exact
/// only on a uniform grid where the line transit time is an integer number
/// of timesteps.
pub fn fdtd_cells(td: f64, h: f64) -> usize {
    let n_f = td / h;
    let n = n_f.round();
    assert!((n_f - n).abs() < 1e-9,
        "fdtd_cells: TD/h must be integer (got TD/h = {n_f:.6}); pick h that divides TD");
    n as usize
}

/// Run the discrete delay-line simulator. Returns `(time, v_rx)` aligned to
/// `0..=n_steps` with `time[0] = 0` and `v_rx[0] = 0`.
///
/// The waveform is sampled at `t_k = k·h` ("end of step k"). The first
/// front reaches the receiver at `t = TD`; this matches `analytic_step_at`
/// taken with right-continuous values at the discontinuity.
pub fn fdtd_trace(
    topo: Topology,
    h: f64,
    n_steps: usize,
    waveform: &SourceWaveform,
) -> (Vec<f64>, Vec<f64>) {
    let n = fdtd_cells(topo.td, h);
    assert!(n >= 1, "TD must be >= h");
    let gs = topo.gamma_s();
    let gl = topo.gamma_l();

    let mut vf = vec![0.0_f64; n];
    let mut vb = vec![0.0_f64; n];

    // Initial condition at t=0: the source's value at t=0 launches a wave
    // into vf[0] so that, after N shifts, it arrives at the load at t=TD.
    vf[0] = topo.v_plus(waveform.value_at(0.0));

    let mut times = Vec::with_capacity(n_steps + 1);
    let mut v_rx = Vec::with_capacity(n_steps + 1);
    times.push(0.0);
    v_rx.push(0.0);

    for k in 1..=n_steps {
        let t_k = k as f64 * h;

        // Capture pre-shift values used by the boundary conditions.
        let arriving = vf[n - 1];     // wave reaching the load this step
        let returning = vb[0];        // wave reaching the source this step

        // Propagate: forward shifts right, backward shifts left.
        for i in (1..n).rev() { vf[i] = vf[i - 1]; }
        for i in 0..n - 1     { vb[i] = vb[i + 1]; }

        // Plant new boundary values.
        let leaving_load = gl * arriving;
        vb[n - 1] = leaving_load;
        vf[0] = topo.v_plus(waveform.value_at(t_k)) + gs * returning;

        // Receiver voltage = incident + reflected at the load node.
        v_rx.push(arriving + leaving_load);
        times.push(t_k);
    }
    (times, v_rx)
}

// ── SPICE deck (cross-simulator) ────────────────────────────────────────

/// Build an ngspice deck for the topology under the given stimulus.
/// Uses the lossless `T` element. `R_term = 0` is omitted from the netlist.
pub fn spice_deck(topo: Topology, waveform: &SourceWaveform) -> String {
    let mut n = Netlist::new("T-line series-termination spike (rlx-eda)");
    // BDF1 to match a possible rlx outer-loop comparison; harmless for
    // ngspice-only validation.
    n.add_preamble(".options method=gear maxord=1");

    n.add_waveform_source("src", "vsrc", "0", waveform);

    // Driver output resistance.
    R { ohms: topo.r_drv }.emit_spice(&mut n, &["vsrc", "vmid"], "drv").unwrap();

    // Optional series-termination resistor; collapse to a wire when zero.
    let line_in = if topo.r_term > 0.0 {
        R { ohms: topo.r_term }.emit_spice(&mut n, &["vmid", "vline_in"], "term").unwrap();
        "vline_in"
    } else {
        "vmid"
    };

    // Lossless transmission line. ngspice T-element syntax:
    //   T<name> A+ A- B+ B- Z0=<ohms> TD=<seconds>
    n.add_element(format!(
        "T1 {line_in} 0 vrx 0 Z0={:.6} TD={:.10e}",
        topo.z0, topo.td,
    ));

    // High-Z load. ngspice needs *some* DC path or `.tran uic` complains;
    // we use the caller-supplied `r_load` (default 1G in `Topology::high_z`).
    R { ohms: topo.r_load }.emit_spice(&mut n, &["vrx", "0"], "load").unwrap();

    // Initial conditions: everything quiescent. `uic` on `.tran` will honor
    // these and skip the DC operating point.
    n.add_element(".ic v(vmid)=0 v(vrx)=0");

    n.deck()
}

impl Topology {
    /// Convenience: 50 Ω line, 10 Ω CMOS-ish driver, no series term, 1 GΩ
    /// far-end ("high-Z"). Pick `td` for your geometry.
    pub fn unterminated(td: f64) -> Self {
        Self { r_drv: 10.0, r_term: 0.0, z0: 50.0, td, r_load: 1e9 }
    }

    /// Same as `unterminated` but with `r_term = z0 - r_drv` so the
    /// Thevenin source impedance equals `z0`.
    pub fn series_matched(td: f64) -> Self {
        let z0 = 50.0;
        let r_drv = 10.0;
        Self { r_drv, r_term: z0 - r_drv, z0, td, r_load: 1e9 }
    }
}
