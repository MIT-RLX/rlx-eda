//! Build the trait-driven divider, write GDS, print summary.

use spike_divider_block::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let r1_len = 10_000_i64;
    let r2_len = 30_000_i64;

    let (lib, _pdk, top) = make_divider_layout(r1_len, r2_len);
    let cell = lib.get(top);
    let bbox = cell.full_bbox(&lib);

    println!("RcDivider laid out via Block + Layout traits");
    println!("  R1 length: {r1_len} DBU  ({} µm)", r1_len / 1000);
    println!("  R2 length: {r2_len} DBU  ({} µm)", r2_len / 1000);
    println!("  top cell : {}", String::from(cell.name().0.clone()));
    println!("  bbox     : ({}, {}) → ({}, {}) DBU",
        bbox.min.x, bbox.min.y, bbox.max.x, bbox.max.y);
    println!("  ports    : {}", cell.ports().len());
    for p in cell.ports() {
        println!("    {:6}  ({:>6},{:>6})", p.name, p.center.x, p.center.y);
    }

    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/divider_block.gds".to_string());
    klayout_io::write_gds_path(&lib, &path)?;
    let bytes = std::fs::metadata(&path)?.len();
    println!("  wrote    : {path}  ({bytes} bytes)");

    // ── Inverse design demo: drive Vout to a target via SGD on R1, R2 ──
    println!();
    println!("Inverse design via rlx AD (target Vout = 0.4 V at V_in = 1.0 V)");
    let div = RcDivider::new(
        Resistor { length: r1_len, id: "R1".into() },
        Resistor { length: r2_len, id: "R2".into() },
    );
    let opt = DcOptimizer::default();
    let res = div.optimize_to_target(1.0, 0.4, 1_000.0, 3_000.0, &opt);
    println!("  initial  : R1=1000 Ω, R2=3000 Ω, Vout = 0.750");
    println!("  converged: R1={:.1} Ω, R2={:.1} Ω, Vout = {:.6} ({} iters, loss = {:.3e})",
        res.r1, res.r2, res.final_vout, res.iters, res.final_loss);
    Ok(())
}
