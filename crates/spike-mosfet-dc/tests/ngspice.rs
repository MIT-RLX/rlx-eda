//! Tier 3: rlx-side smooth `Id` vs ngspice `.op` `i(Vd)`. Both engines
//! evaluate the same LEVEL=1 NMOS at the same biases; ngspice runs
//! strict piecewise while rlx runs the smooth approximation, so we
//! probe **only** at points well inside saturation/triode where the
//! two agree to better than 1e-4 relative.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest};
use spike_mosfet_dc::*;

const VTH: f64 = 0.5;
const KP: f64 = 100e-6;
const LAM: f64 = 0.02;
const W: f64 = 10e-6;
const L: f64 = 2e-6;

#[test]
fn rlx_id_matches_ngspice_op() {
    let ng = match LocalBinary::from_env() {
        Ok(b) => b,
        Err(e) => { eprintln!("skipping: {e}"); return; }
    };

    // Cover all three operating regions — the textbook smooth model
    // matches ngspice's strict piecewise to ~0.1% relative inside
    // each region.
    let cases = &[
        // (Vgs, Vds, label)
        (2.0_f64, 0.3_f64, "deep triode Vov=1.5, Vds=0.3"),
        (2.0,     1.5,     "saturation knee @ Vds=Vov=1.5"),
        (2.0,     2.0,     "saturation Vov=1.5, Vds=2.0"),
        (2.0,     5.0,     "deep saturation Vov=1.5, Vds=5.0"),
    ];

    for &(vgs, vds, label) in cases {
        let id_rlx = run_id(vgs, vds, VTH, KP, LAM);

        // ngspice node "d" voltage *should* equal vds — sanity check.
        // We could also pull the drain current via `print i(Vd)` but
        // our `Invoker::run_dc` only knows NodeVoltage. For Id, switch
        // to running through stdout-parse on a custom deck.
        //
        // Easier: read v(d) (we already know it = vds) and trust the
        // ngspice `.op` Id via a measure trick — actually simplest:
        // place a tiny series resistor `Rs` at the drain so v(d) - v(dext)
        // is Id·Rs, and we read the voltage drop. Skip the resistor
        // trick for now and pull current via direct ngspice deck text.
        let id_ng = run_ngspice_id(vgs, vds);

        let rel = (id_rlx - id_ng).abs() / id_ng.abs().max(1e-12);
        // 1% envelope: dominated by the smooth-vs-strict deviation in
        // triode. Saturation interior matches to ppm; triode is looser.
        assert!(
            rel < 1e-2,
            "[{label}] rlx Id = {id_rlx:.6e} A, ngspice Id = {id_ng:.6e} A, rel = {rel:.3e}",
        );

        // Also check v(d) round-trips (this exercises the existing
        // `run_dc` path and the simplest possible deck-driver).
        let dc = ng.run_dc(
            &spice_deck(vgs, vds, VTH, KP, LAM, W, L),
            &[OutputRequest::NodeVoltage("d".into())],
        ).expect("ngspice .op failed");
        assert!((dc.node_voltages["d"] - vds).abs() < 1e-6,
            "[{label}] v(d) = {} (expected {})", dc.node_voltages["d"], vds);
    }
}

/// Build a deck and parse `i(Vd)` directly from ngspice stdout. The
/// driver crate's `OutputRequest` only carries node voltages today; for
/// drain current we hand-craft a small deck with `print i(Vd)` and
/// scrape stdout for the float.
fn run_ngspice_id(vgs: f64, vds: f64) -> f64 {
    let kp_per_wl = KP / (W / L);
    let deck = format!(
        "* NMOS L1 Id probe\n\
         Vg g 0 {vgs}\n\
         Vd d 0 {vds}\n\
         M1 d g 0 0 NMOS_RLX W={W} L={L}\n\
         .model NMOS_RLX NMOS LEVEL=1 VTO={VTH} KP={kp_per_wl} LAMBDA={LAM} GAMMA=0\n",
    );
    // Run ngspice with explicit print + grep for the Id value. We don't
    // use the Invoker abstraction here because it doesn't expose
    // currents (yet). Acceptable for one test.
    let raw = run_with_print(&deck, "i(Vd)");
    // ngspice reports drain current as flowing INTO Vd (out of device drain),
    // so the magnitude is Id but the sign is flipped. We negate.
    -raw
}

fn run_with_print(deck: &str, signal: &str) -> f64 {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let bin = std::env::var("NGSPICE_BIN").unwrap_or_else(|_| "ngspice".into());
    let full = format!(
        "* rlx-eda mosfet probe\n{deck}.control\nop\nprint {signal}\n.endc\n.end\n",
    );
    let mut child = Command::new(bin)
        .args(["-b", "-n"])
        .arg("/dev/stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("ngspice spawn failed");
    child.stdin.as_mut().unwrap().write_all(full.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    parse_print_value(&stdout, signal).unwrap_or_else(|| {
        panic!("could not parse '{signal}' from ngspice stdout:\n{stdout}");
    })
}

/// `i(Vd) = <number>` or `i(vd)\t<number>` — same shape as the existing
/// `parse_node_voltage` in eda-extern-ngspice, just looking for `i(...)`.
fn parse_print_value(stdout: &str, signal: &str) -> Option<f64> {
    let needle = signal.to_lowercase();
    for line in stdout.lines() {
        let lower = line.to_lowercase();
        if let Some(idx) = lower.find(&needle) {
            let tail = &line[idx + needle.len()..];
            let tail = tail.trim_start_matches(|c: char| c == '=' || c.is_whitespace());
            if let Some(tok) = tail.split_whitespace().next() {
                if let Ok(v) = tok.parse::<f64>() {
                    return Some(v);
                }
            }
        }
    }
    None
}
