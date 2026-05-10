//! Byte-equality snapshot tests for SVG output.
//!
//! Locks the current rendered SVG bytes for the canonical demo
//! schematic + layout (driven by `RcDivider::{layout, schematic}`).
//! Any future renderer change that perturbs the output — geometry,
//! colors, fonts, attribute order — surfaces as a `git diff` of the
//! golden file, so we choose to accept it (regenerate the golden) or
//! reject it (revert the renderer change).
//!
//! ## Updating goldens
//!
//! When a renderer change is intentional:
//!
//!     cargo run -p eda-viz --example demo --features png
//!     cp target/eda-viz-demo/divider_layout.svg \
//!        target/eda-viz-demo/divider_schematic.svg \
//!        crates/eda-viz/tests/golden/
//!
//! Then commit. The diff in the PR shows exactly what changed.

use eda_hir::{Layout as _, Schematic as _};
use eda_viz::{layout, schematic, Style};
use spike_divider_block::{RcDemo, RcDivider, Resistor};

fn render_divider_layout_svg() -> String {
    let lib = RcDemo::new_library("rc_demo");
    let pdk = RcDemo::register(&lib);
    let div = RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );
    let top = div.layout(&lib, &pdk);
    layout::render_to_svg(&lib, top, &Style::default())
}

fn render_divider_schematic_svg() -> String {
    let lib = RcDemo::new_library("rc_demo");
    let pdk = RcDemo::register(&lib);
    let div = RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );
    let ir = div.schematic(&pdk);
    let s = schematic::from_ir(&ir);
    schematic::render_to_svg(&s, &schematic::SchemStyle::default())
}

#[test]
fn divider_layout_matches_golden() {
    let actual = render_divider_layout_svg();
    let golden = include_str!("golden/divider_layout.svg");
    if actual != golden {
        let path = std::env::temp_dir().join("divider_layout_actual.svg");
        let _ = std::fs::write(&path, &actual);
        panic!(
            "layout SVG drift detected.\n\
             actual written to {}\n\
             golden at tests/golden/divider_layout.svg\n\
             diff with: diff <(cat tests/golden/divider_layout.svg) {}\n\
             if intended, regenerate goldens (see test docs).",
            path.display(), path.display(),
        );
    }
}

#[test]
fn divider_schematic_matches_golden() {
    let actual = render_divider_schematic_svg();
    let golden = include_str!("golden/divider_schematic.svg");
    if actual != golden {
        let path = std::env::temp_dir().join("divider_schematic_actual.svg");
        let _ = std::fs::write(&path, &actual);
        panic!(
            "schematic SVG drift detected.\n\
             actual written to {}\n\
             golden at tests/golden/divider_schematic.svg\n\
             diff with: diff <(cat tests/golden/divider_schematic.svg) {}\n\
             if intended, regenerate goldens (see test docs).",
            path.display(), path.display(),
        );
    }
}
