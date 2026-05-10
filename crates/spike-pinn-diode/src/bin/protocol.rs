//! Protocol-grade run: K=10 seeds × 3 ablations × {3 MNA + polynomial}
//! baselines, on the locked pre-registered configuration.
//!
//! Output: tabular results to stdout and `docs/results.md`. Acceptance
//! criteria evaluated and reported per §12.

use spike_pinn_diode::runner::{print_protocol_report, run_protocol, ProtocolDevice};

fn main() {
    let device = match std::env::var("RLX_EDA_DEVICE").ok().as_deref() {
        Some("cpu") => ProtocolDevice::Cpu,
        _           => ProtocolDevice::Mlx,
    };
    let report = run_protocol(device);
    print_protocol_report(&report);
    if let Err(e) = report.write_markdown("crates/spike-pinn-diode/docs/results.md") {
        eprintln!("warning: could not write docs/results.md: {e}");
    }
}
