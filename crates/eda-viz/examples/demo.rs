//! Render the LVS-verified divider layout (from `spike-divider-block`)
//! AND its derived symbolic schematic from the *same* `RcDivider`
//! value. Emit SVG and PNG into `target/eda-viz-demo/`.
//!
//! What this demo proves:
//!
//! - Layout side calls `make_divider_layout`, which goes through
//!   `RcDivider::layout(...)` — the same code path
//!   `spike-divider-block/tests/lvs.rs` verifies has 3 METAL1 nets.
//! - Schematic side calls `RcDivider::schematic(&pdk)` — the new
//!   `Schematic<P>` HIR trait impl — and feeds the resulting
//!   `SchematicIr` to `eda_viz::schematic::from_ir`. So the symbols
//!   and wires you see are derived from the same Rust fields
//!   (`r1.length`, `r2.length`) that drove the layout.
//!
//! Layout and schematic cannot drift: change the divider once, both
//! views update together.

use eda_hir::Schematic as _;
use eda_viz::{layout, png::svg_to_png, schematic, Style};
use spike_divider_block::{Resistor, RcDemo, RcDivider};
use eda_hir::Layout as _;

fn main() {
    let out = std::path::PathBuf::from("target/eda-viz-demo");
    std::fs::create_dir_all(&out).unwrap();

    // Single source of truth: one RcDivider value drives both views.
    let divider = RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );

    // Layout — via Layout<RcDemo>::layout, then render.
    let lib = RcDemo::new_library("rc_demo");
    let pdk = RcDemo::register(&lib);
    let top = divider.layout(&lib, &pdk);
    let layout_svg = layout::render_to_svg(&lib, top, &Style::default());
    std::fs::write(out.join("divider_layout.svg"), &layout_svg).unwrap();
    std::fs::write(out.join("divider_layout.png"), svg_to_png(&layout_svg, 3.0).unwrap()).unwrap();

    // Schematic — derived from the SAME RcDivider via Schematic<RcDemo>::schematic.
    let ir = divider.schematic(&pdk);
    let schem = schematic::from_ir(&ir);
    let schem_svg = schematic::render_to_svg(&schem, &schematic::SchemStyle::default());
    std::fs::write(out.join("divider_schematic.svg"), &schem_svg).unwrap();
    std::fs::write(out.join("divider_schematic.png"), svg_to_png(&schem_svg, 3.0).unwrap()).unwrap();

    for f in ["divider_layout.svg", "divider_layout.png",
              "divider_schematic.svg", "divider_schematic.png"] {
        println!("wrote {}", out.join(f).display());
    }
}
