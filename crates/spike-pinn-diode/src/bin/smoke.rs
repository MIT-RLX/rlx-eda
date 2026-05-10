//! Smoke binary: pick the fastest available device and run the
//! pipeline once. Validates that train + infer + race compile, run,
//! and produce sane numbers. Not the pre-registered protocol.

use rlx_runtime::Device;
use spike_pinn_diode::runner::{print_report, run_smoke};

fn main() {
    // Default to MLX on macOS, CPU elsewhere. Override with
    // `RLX_EDA_DEVICE=cpu` / `RLX_EDA_DEVICE=mlx`.
    let device = match std::env::var("RLX_EDA_DEVICE").ok().as_deref() {
        Some("cpu") => Device::Cpu,
        Some("mlx") => Device::Mlx,
        _ if cfg!(target_os = "macos") => Device::Mlx,
        _ => Device::Cpu,
    };

    let report = run_smoke(device);
    print_report(&report);
}
