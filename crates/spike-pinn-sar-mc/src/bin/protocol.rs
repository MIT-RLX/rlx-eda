use spike_pinn_sar_mc::runner::{print_report, run_protocol, ProtocolDevice};

fn main() {
    let device = match std::env::var("RLX_EDA_DEVICE").ok().as_deref() {
        Some("cpu") => ProtocolDevice::Cpu,
        _           => ProtocolDevice::Mlx,
    };
    let report = run_protocol(device);
    print_report(&report);
    if let Err(e) = report.write_markdown("crates/spike-pinn-sar-mc/docs/results.md") {
        eprintln!("warning: could not write docs/results.md: {e}");
    }
}
