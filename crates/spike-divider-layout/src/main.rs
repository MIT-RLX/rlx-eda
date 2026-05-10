//! Build the divider layout and write it to a GDS file. Print a brief
//! summary including hierarchical bbox and per-layer shape counts.

use klayout_core::Bbox;
use spike_divider_layout::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let r1_len = 10_000_i64;   // 10 µm
    let r2_len = 30_000_i64;   // 30 µm  (V_out / V_in = R2 / (R1+R2) = 0.75)

    let (lib, pdk, top) = make_divider_layout(r1_len, r2_len);

    let top_cell = lib.get(top);
    let bbox: Bbox = top_cell.full_bbox(&lib);

    println!("RcDemo divider laid out");
    println!("  R1 length (RES): {r1_len} DBU  ({} µm)", r1_len / 1000);
    println!("  R2 length (RES): {r2_len} DBU  ({} µm)", r2_len / 1000);
    println!("  full bbox      : ({}, {}) → ({}, {}) DBU",
        bbox.min.x, bbox.min.y, bbox.max.x, bbox.max.y);
    println!("  ports          : {}", top_cell.ports().len());
    for p in top_cell.ports() {
        println!("    {:6}  ({:>6},{:>6})  layer={:?}", p.name, p.center.x, p.center.y, p.layer);
    }

    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/divider.gds".to_string());
    klayout_io::write_gds_path(&lib, &path)?;
    let bytes = std::fs::metadata(&path)?.len();
    println!("  wrote          : {path}  ({bytes} bytes)");

    let _ = pdk;
    Ok(())
}
