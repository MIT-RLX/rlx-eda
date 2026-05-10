//! GDS layout for the 8-bit R-2R DAC under a perturbed `Design`.
//!
//! 16 rectangular resistors on the `RES` layer, with body lengths
//! encoding the deviation alphabet (so a 0% resistor is 10 µm long, a
//! +5% resistor is 10.5 µm, etc.). METAL1 contact pads + VIA1 squares
//! at each end. Three rows on a single horizontal grid:
//!
//! ```text
//!                      r_in[0]   r_in[1]  ...   r_in[7]   <- input feeders
//!                       │         │              │
//!                      ─■─       ─■─    ...     ─■─       <- horizontal R bodies
//!                       │         │              │
//!     n_0 ─■─ n_1 ─■─ n_2 ─■─ ... ─■─ n_7 = vout         <- spine row
//!     │
//!    ─■─                                                  <- termination (vlow)
//!     │
//!    vlow
//! ```
//!
//! All resistors are placed *horizontally* (no rotation) for
//! geometric simplicity. The "vertical-looking" feeder/term wires are
//! METAL1 rectangles routed by hand from the spine node up/down to the
//! corresponding feeder's port_a pad. This gives a real, GDS-correct
//! layout that respects the R-2R electrical topology — it's not a
//! styled diagram.
//!
//! The PDK is intentionally minimal (RES, METAL1, VIA1 — same shape as
//! `spike-divider-layout::RcDemo`) so this spike doesn't drag in a
//! foundry. A real flow would replace it with `Sky130` or `Gf180mcu`
//! from `eda-pdks` and parameterise resistor cells through a primitive
//! library; that's a larger lift than this experiment.

use klayout_core::{
    Angle90, Bbox, CellBuilder, CellId, Instance, Library, Point, Port, Rect, Trans, Vec2,
};
use klayout_pdk::pdk;

use crate::{r_in_idx, r_sp_idx, r_term_idx, r_value, Design, N_BITS, N_NODES};

pdk! {
    pub R2RPdk {
        dbu: 1000,
        layers: {
            RES    = (50, 0),
            METAL1 = (10, 0),
            VIA1   = (20, 0),
        },
        ports: { Electrical },
    }
}

// Geometry constants (DBU, with dbu=1000 → 1 nm/DBU, so 1000 DBU = 1 µm).
const RES_WIDTH: i64 = 1_000;       // 1 µm body width
const PAD: i64       = 2_000;       // 2 µm pad square
const VIA: i64       = 500;         // 0.5 µm via square

// One DBU per ohm gives 10 kΩ → 10 000 DBU (10 µm) — convenient for
// a 1 µm-wide RES sheet. We don't actually compute resistance here;
// the length is just a visual encoding of the design's deviation.
const DBU_PER_OHM: f64 = 1.0;

const Y_SPINE: i64   = 0;
const Y_FEEDER: i64  = 30_000;       // 30 µm above spine
const Y_TERM: i64    = -30_000;      // 30 µm below spine

const WIRE_HALF_W: i64 = PAD / 2;    // routing wire half-width (matches pad)

/// Convert an ohm value to body length in DBU.
fn ohms_to_len(ohms: f64) -> i64 {
    (ohms * DBU_PER_OHM).round() as i64
}

/// Build a single rectangular-resistor primitive cell of the given body
/// length in DBU. Same recipe as `spike-divider-layout::build_resistor_cell`,
/// duplicated locally so this crate doesn't have to depend on that spike.
fn build_resistor_cell(lib: &Library, pdk: &R2RPdk, length: i64, name: &str) -> CellId {
    let mut b = CellBuilder::new(name);

    b.add_shape(pdk.RES, Rect::new(Bbox::new(
        Point::new(0, 0),
        Point::new(length, RES_WIDTH),
    )));

    let cy        = RES_WIDTH / 2;
    let half_pad  = PAD / 2;
    let half_via  = VIA / 2;
    for &x in &[0_i64, length] {
        b.add_shape(pdk.METAL1, Rect::new(Bbox::new(
            Point::new(x - half_pad, cy - half_pad),
            Point::new(x + half_pad, cy + half_pad),
        )));
        b.add_shape(pdk.VIA1, Rect::new(Bbox::new(
            Point::new(x - half_via, cy - half_via),
            Point::new(x + half_via, cy + half_via),
        )));
    }

    b.add_port(
        Port::new("a", pdk.METAL1, Point::new(-half_pad, cy), Angle90::W, PAD)
            .with_kind(R2RPdk::Electrical),
    );
    b.add_port(
        Port::new("b", pdk.METAL1, Point::new(length + half_pad, cy), Angle90::E, PAD)
            .with_kind(R2RPdk::Electrical),
    );

    lib.insert(b)
}

/// METAL1 rectangle bridging two points along the y axis at a fixed x.
/// Used for the (visually) vertical wires from a spine node up to a
/// feeder's port_a pad and from n_0 down to the termination.
fn vertical_wire(top: &mut CellBuilder, layer: klayout_core::LayerIndex, x: i64, y0: i64, y1: i64) {
    let (lo, hi) = if y0 < y1 { (y0, y1) } else { (y1, y0) };
    top.add_shape(layer, Rect::new(Bbox::new(
        Point::new(x - WIRE_HALF_W, lo),
        Point::new(x + WIRE_HALF_W, hi),
    )));
}

/// Build the full R-2R DAC layout for `design` and return
/// `(Library, top cell id)`. The library has the PDK registered and
/// the top cell exposes ports `in_0..in_7`, `vlow`, and `vout`.
pub fn build_r2r_layout(design: &Design) -> (Library, CellId) {
    let lib = R2RPdk::new_library("dado_r2r");
    let pdk = R2RPdk::register(&lib);

    // Resistor lengths from the design's deviation indices.
    let term_len   = ohms_to_len(r_value(design, r_term_idx()));
    let feeder_len: [i64; N_BITS] = std::array::from_fn(|b| ohms_to_len(r_value(design, r_in_idx(b))));
    let spine_len:  [i64; N_NODES - 1] = std::array::from_fn(|s| ohms_to_len(r_value(design, r_sp_idx(s))));

    // Build the 16 unique resistor cells (one per resistor; lengths
    // generally differ even within a row because of independent
    // deviations).
    let term_cell   = build_resistor_cell(&lib, &pdk, term_len, "R_term");
    let feeder_cells: [CellId; N_BITS] = std::array::from_fn(|b| {
        build_resistor_cell(&lib, &pdk, feeder_len[b], &format!("R_in{b}"))
    });
    let spine_cells: [CellId; N_NODES - 1] = std::array::from_fn(|s| {
        build_resistor_cell(&lib, &pdk, spine_len[s], &format!("R_sp{s}"))
    });

    // Spine layout: place 7 spine resistors abutting end-to-end. Each
    // resistor cell has its left pad centred at x=0 and right pad at
    // x=length. Setting `dx_{i+1} = dx_i + length_i` makes the right
    // pad of [i] coincide exactly with the left pad of [i+1] — both
    // are 2 µm METAL1 squares overlapping perfectly, so the shared
    // node is a single electrical net at zero routing cost.
    let mut spine_trans = Vec::with_capacity(N_NODES - 1);
    let mut spine_node_x = Vec::with_capacity(N_NODES); // x of n_0..n_7
    let mut x = 0_i64;
    spine_node_x.push(x); // n_0
    for s in 0..(N_NODES - 1) {
        spine_trans.push(Trans::translate(Vec2::new(x, Y_SPINE)));
        x += spine_len[s];
        spine_node_x.push(x); // n_{s+1}
    }
    // n_7 (= vout) is at x = sum of spine lengths.
    let _vout_x = *spine_node_x.last().unwrap();

    // Input feeders: place each above its corresponding spine node so
    // a single vertical METAL1 wire from spine to feeder port_a is
    // sufficient. Cell-local port_a is at (-half_pad, cy), so we place
    // the cell at (spine_node_x + half_pad, Y_FEEDER) → port_a absolute
    // is at (spine_node_x, Y_FEEDER + cy). The wire then runs from
    // (spine_node_x, Y_SPINE + cy) up to (spine_node_x, Y_FEEDER + cy).
    let half_pad = PAD / 2;
    let cy       = RES_WIDTH / 2;
    let feeder_trans: [Trans; N_BITS] = std::array::from_fn(|b| {
        Trans::translate(Vec2::new(spine_node_x[b] + half_pad, Y_FEEDER))
    });
    let term_trans = Trans::translate(Vec2::new(spine_node_x[0] + half_pad, Y_TERM));

    // Top cell.
    let mut top = CellBuilder::new("r2r_dac");
    for s in 0..(N_NODES - 1) {
        top.add_instance(Instance::new(spine_cells[s], spine_trans[s]));
    }
    for b in 0..N_BITS {
        top.add_instance(Instance::new(feeder_cells[b], feeder_trans[b]));
    }
    top.add_instance(Instance::new(term_cell, term_trans));

    // Vertical METAL1 wires:
    //   spine node n_i → feeder[i].port_a   (for each bit)
    //   spine node n_0 → term.port_a        (down to vlow)
    for b in 0..N_BITS {
        vertical_wire(&mut top, pdk.METAL1, spine_node_x[b], Y_SPINE + cy, Y_FEEDER + cy);
    }
    vertical_wire(&mut top, pdk.METAL1, spine_node_x[0], Y_TERM + cy, Y_SPINE + cy);

    // Top-level ports — one per external net.
    //   in_b: feeder[b].port_b                  (right end of feeder)
    //   vlow: term.port_b                        (right end of term)
    //   vout: spine[N_NODES-2].port_b           (right end of last spine)
    for b in 0..N_BITS {
        let cell = lib.get(feeder_cells[b]);
        let port_b = cell.port("b").expect("feeder.b").transform(feeder_trans[b]);
        top.add_port(
            Port::new(&format!("in_{b}"), pdk.METAL1, port_b.center, port_b.angle, port_b.width)
                .with_kind(R2RPdk::Electrical),
        );
    }
    {
        let cell = lib.get(term_cell);
        let port_b = cell.port("b").expect("term.b").transform(term_trans);
        top.add_port(
            Port::new("vlow", pdk.METAL1, port_b.center, port_b.angle, port_b.width)
                .with_kind(R2RPdk::Electrical),
        );
    }
    {
        let last = N_NODES - 2;
        let cell = lib.get(spine_cells[last]);
        let port_b = cell.port("b").expect("spine.b").transform(spine_trans[last]);
        top.add_port(
            Port::new("vout", pdk.METAL1, port_b.center, port_b.angle, port_b.width)
                .with_kind(R2RPdk::Electrical),
        );
    }

    let top_id = lib.insert(top);
    (lib, top_id)
}

/// Convenience: build the layout for `design` and write it to GDS at
/// `path`. Returns the path for chaining/logging.
pub fn write_gds_for_design(
    design: &Design,
    path: impl AsRef<std::path::Path>,
) -> Result<std::path::PathBuf, klayout_io::IoError> {
    let (lib, _top) = build_r2r_layout(design);
    klayout_io::write_gds_path(&lib, path.as_ref())?;
    Ok(path.as_ref().to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use klayout_io::write_gds_bytes;

    #[test]
    fn nominal_layout_round_trips() {
        let design: Design = [2u8; 16];
        let (lib, _top) = build_r2r_layout(&design);
        let bytes = write_gds_bytes(&lib).expect("gds write");
        // GDS header magic — file should be non-empty and start with a
        // valid HEADER record (record-len 6, record-type 0x0002).
        assert!(bytes.len() > 100);
        assert_eq!(bytes[2], 0x00);
        assert_eq!(bytes[3], 0x02);
    }

    #[test]
    fn perturbed_layout_lengths_differ_from_nominal() {
        // Two designs producing different total spine lengths must
        // produce different GDS byte streams (sanity that deviations
        // actually change geometry).
        let nom: Design = [2u8; 16];
        let mut perturbed = nom;
        perturbed[r_sp_idx(0)] = 0; // -5%
        perturbed[r_in_idx(7)] = 4; // +5%
        let (lib_a, _) = build_r2r_layout(&nom);
        let (lib_b, _) = build_r2r_layout(&perturbed);
        let a = write_gds_bytes(&lib_a).expect("a");
        let b = write_gds_bytes(&lib_b).expect("b");
        assert_ne!(a, b, "perturbations didn't change the GDS output");
    }
}
