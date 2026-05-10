//! Schematic SVG for the 8-bit R-2R DAC under a perturbed `Design`.
//!
//! Symbolic counterpart to `layout.rs`: 16 resistors arranged in the
//! R-2R ladder topology, each annotated with its actual resistance
//! (the deviation indices in `design` resolved to ohms via `r_value`).
//! Output is a self-contained SVG document.
//!
//! Coordinates are in `eda_viz::schematic::SchemStyle` units, which
//! the renderer scales to pixels via `pixels_per_unit`. Resistor pin
//! offsets are ±2 schematic units around the anchor; spine resistors
//! sit on `y = 0` with anchors 4 units apart so each spine node lands
//! on integer multiples of 4.

use eda_viz::schematic::{render_to_svg, Orient, SchemStyle, Schematic, Symbol};

use crate::{r_in_idx, r_sp_idx, r_term_idx, r_value, Design, N_BITS, N_NODES};

const SPINE_PITCH: f64 = 6.0; // x-distance between adjacent spine nodes
const FEEDER_Y: f64    = 6.0; // anchor y for vertical input feeders
const TERM_Y: f64      = -6.0; // anchor y for the vertical termination

fn fmt_ohms(ohms: f64) -> String {
    if ohms >= 1_000.0 {
        format!("{:.2} kΩ", ohms / 1_000.0)
    } else {
        format!("{:.0} Ω", ohms)
    }
}

/// Spine node x-coordinate (n_0 = 0, n_7 = (N_NODES-1) * pitch).
fn spine_node_x(i: usize) -> f64 {
    (i as f64) * SPINE_PITCH
}

/// Build the R-2R schematic for the given design and render it to SVG.
pub fn render_design_svg(design: &Design, title: Option<&str>) -> String {
    let mut sch = Schematic::new();
    sch.title = title.map(str::to_owned);

    // -----------------------------------------------------------------
    // Spine row: 7 horizontal R resistors connecting n_i to n_{i+1}.
    // Symbol pin offsets are (-2, 0) and (2, 0); spine pitch is 6 so
    // anchor lands at the midpoint and a 1-unit wire stub bridges
    // each gap to the next resistor's pin.
    // -----------------------------------------------------------------
    for s in 0..(N_NODES - 1) {
        let x_left = spine_node_x(s);
        let x_right = spine_node_x(s + 1);
        let anchor = ((x_left + x_right) / 2.0, 0.0);
        let ohms = r_value(design, r_sp_idx(s));
        sch.place(
            Symbol::Resistor {
                label: format!("R_sp{s}"),
                value: Some(fmt_ohms(ohms)),
            },
            anchor,
            Orient::Horizontal,
        );
        // Stubs from spine node out to the symbol's W pin and from the
        // symbol's E pin to the next spine node.
        sch.wire([(x_left, 0.0), (anchor.0 - 2.0, 0.0)]);
        sch.wire([(anchor.0 + 2.0, 0.0), (x_right, 0.0)]);
    }

    // -----------------------------------------------------------------
    // Input feeders: 8 vertical 2R resistors at each spine node.
    // After Orient::Vertical, pin 0 is at (anchor.x, anchor.y + 2)
    // (top, "in" side) and pin 1 is at (anchor.x, anchor.y - 2)
    // (bottom, spine side). We connect pin 1 to the spine node and
    // place an `in_b` pin label at the top.
    // -----------------------------------------------------------------
    for b in 0..N_BITS {
        let x = spine_node_x(b);
        let anchor = (x, FEEDER_Y);
        let ohms = r_value(design, r_in_idx(b));
        sch.place(
            Symbol::Resistor {
                label: format!("R_in{b}"),
                value: Some(fmt_ohms(ohms)),
            },
            anchor,
            Orient::Vertical,
        );
        // Bottom pin → spine node.
        sch.wire([(x, FEEDER_Y - 2.0), (x, 0.0)]);
        // Top pin → in_b label.
        sch.wire([(x, FEEDER_Y + 2.0), (x, FEEDER_Y + 3.5)]);
        sch.pin_label((x, FEEDER_Y + 4.0), format!("in_{b}"));
    }

    // -----------------------------------------------------------------
    // Termination resistor: vertical 2R hanging below n_0 to vlow.
    // -----------------------------------------------------------------
    let term_x = spine_node_x(0);
    let term_anchor = (term_x, TERM_Y);
    let term_ohms = r_value(design, r_term_idx());
    sch.place(
        Symbol::Resistor {
            label: "R_term".to_owned(),
            value: Some(fmt_ohms(term_ohms)),
        },
        term_anchor,
        Orient::Vertical,
    );
    sch.wire([(term_x, 0.0), (term_x, TERM_Y + 2.0)]);
    sch.wire([(term_x, TERM_Y - 2.0), (term_x, TERM_Y - 3.5)]);
    sch.pin_label((term_x, TERM_Y - 4.0), "vlow");

    // Vout pin label at the MSB end of the spine.
    let vout_x = spine_node_x(N_NODES - 1);
    sch.wire([(vout_x, 0.0), (vout_x + 2.5, 0.0)]);
    sch.pin_label((vout_x + 3.0, 0.0), "vout");

    let mut style = SchemStyle::default();
    style.pad = 5.0;          // a bit more breathing room for the long ladder
    render_to_svg(&sch, &style)
}

/// Convenience: write the SVG to `path`.
pub fn write_svg_for_design(
    design: &Design,
    title: Option<&str>,
    path: impl AsRef<std::path::Path>,
) -> std::io::Result<()> {
    let svg = render_design_svg(design, title);
    std::fs::write(path, svg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_design_renders_a_well_formed_svg() {
        let design: Design = [2u8; 16];
        let svg = render_design_svg(&design, Some("nominal"));
        assert!(svg.starts_with("<svg") || svg.contains("<svg"));
        assert!(svg.contains("R_sp0"));
        assert!(svg.contains("R_in7"));
        assert!(svg.contains("R_term"));
        assert!(svg.contains("vlow"));
        assert!(svg.contains("vout"));
        // Resistance label appears for at least one resistor.
        assert!(svg.contains("kΩ"));
    }

    #[test]
    fn perturbed_design_changes_value_labels() {
        let nom: Design = [2u8; 16];
        let mut perturbed = nom;
        perturbed[r_sp_idx(3)] = 4; // +5%
        let a = render_design_svg(&nom, None);
        let b = render_design_svg(&perturbed, None);
        assert_ne!(a, b);
    }
}
