//! `Layout<P>` + `Tile<P>` for `Mac8x8Tile`, generic over any
//! `MosfetPdk`. Sky130 is the v1 default by convention but the impls
//! work for `Gf180mcu` (and any future `MosfetPdk`) for free —
//! "pick PDK later" lives entirely in the caller's choice of `pdk`.
//!
//! Topology dispatch is internal: each variant of [`MacTopology`]
//! has its own layout body. Only `Digital` has a real implementation
//! target in v1; analog variants are `unimplemented!()` until the
//! design work lands.

use eda_hir::{Block, Layout};
use eda_tile::{EdgePort, RailSpec, Side, Tile};
use klayout_core::{Bbox, CellBuilder, CellId, Library, Point, Rect, Trans, Vec2};
use spike_divider_block::MosfetPdk;

use crate::tile::Mac8x8Tile;
use crate::topology::MacTopology;

// ── Digital MAC tile geometry constants ────────────────────────────
//
// Per `eda-bench-tinyconv/PLAN.md` "Digital MAC tile floorplan":
// 4-row sc_hd floorplan, sky130 conventions. `dbu_per_um = 1000` so
// 1 µm = 1000 DBU and 1 nm = 1 DBU.

/// sky130_fd_sc_hd cell row height = 2.72 µm.
const SC_HD_ROW_DBU: i64 = 2_720;
/// sky130_fd_sc_hd power-rail width = 480 nm.
const SC_HD_RAIL_WIDTH_DBU: i64 = 480;
/// sky130 DBU/µm.
const SKY130_DBU_PER_UM: i64 = 1_000;

/// Tile pitch X. The busiest row is row 1 (multiplier upper half +
/// accumulator low) at ~196 µm of cell width; 220 µm gives ~12 %
/// routing margin on it. PLAN.md "Digital MAC tile floorplan"
/// originally guessed 24 µm — corrected as cell placement landed.
const DIGITAL_PITCH_X_DBU: i64 = 220_000;
/// Tile pitch Y = 4 sc_hd rows.
const DIGITAL_PITCH_Y_DBU: i64 = 4 * SC_HD_ROW_DBU; // 10_880

// ── Cell widths (sky130_fd_sc_hd subset) ───────────────────────────
//
// Match the mock sc_hd subset in `eda-stdcells::mock`; real foundry
// widths swap in when `ScHdLibrary::load` runs against actual GDS.

const W_DFXTP1_DBU: i64 = 2_300;
const W_AND2_DBU: i64 = 1_290;
const W_FA1_DBU: i64 = 3_680;
const W_INV1_DBU: i64 = 1_840;

// ── Per-row cell counts ────────────────────────────────────────────
//
// Mirrors the PLAN.md floorplan, with the multiplier (64 AND2 +
// 56 FA) and accumulator (32 DFF + 32 FA) split across rows so
// total widths stay within pitch_x. Row 3 control is a placeholder
// 10 × `inv_1` until the FSM lowering lands.

// Row 0: weight register + multiplier rows 0-3.
const ROW0_WEIGHT_DFFS: usize = 8;
const ROW0_PP_AND2: usize = 32; // PP rows 0-3 × 8 bits
const ROW0_SUM_FAS: usize = 24; // sum rows 0-2 × 8 bits

// Row 1: multiplier rows 4-7 + accumulator low half.
const ROW1_PP_AND2: usize = 32; // PP rows 4-7 × 8 bits
const ROW1_SUM_FAS: usize = 32; // sum rows 3-6 × 8 bits
const ROW1_ACCUM_LOW_DFFS: usize = 16;

// Row 2: accumulator high half + final-adder low.
const ROW2_ACCUM_HIGH_DFFS: usize = 16;
const ROW2_FINAL_ADD_LOW_FAS: usize = 16;

// Row 3: final-adder high + control + output mux.
const ROW3_FINAL_ADD_HIGH_FAS: usize = 16;
const ROW3_CONTROL_INV: usize = 10;

impl Mac8x8Tile {
    /// Per-cell-type inventory of placed cells across all rows.
    /// Returns `[(cell_name, count), ...]` aggregated across the
    /// 4-row floorplan. Cell names match the foundry-canonical
    /// names (e.g. `"sky130_fd_sc_hd__dfxtp_1"`) so the inventory
    /// is `ScHdLibrary`-keyable.
    ///
    /// Used by the bench harness to:
    ///   - compute baseline cell area from real Liberty data
    ///     (instead of the placeholder constant in `behavioral.rs`),
    ///   - report per-cell-type cell counts,
    ///   - sanity-check that `place_row` actually placed what it
    ///     claimed (via `tests/cell_inventory.rs`).
    ///
    /// Topology-dispatched: Digital returns the v1 floorplan; CR
    /// and CM are stubs.
    pub fn cell_inventory(&self) -> Vec<(&'static str, usize)> {
        match self.topology {
            MacTopology::Digital => digital_cell_inventory(),
            MacTopology::ChargeRedistribution => {
                unimplemented!("CR inventory — deferred")
            }
            MacTopology::CurrentMode => {
                unimplemented!("CM inventory — deferred")
            }
        }
    }
}

fn digital_cell_inventory() -> Vec<(&'static str, usize)> {
    let dffs = ROW0_WEIGHT_DFFS + ROW1_ACCUM_LOW_DFFS + ROW2_ACCUM_HIGH_DFFS;
    let and2s = ROW0_PP_AND2 + ROW1_PP_AND2;
    let fas = ROW0_SUM_FAS
        + ROW1_SUM_FAS
        + ROW2_FINAL_ADD_LOW_FAS
        + ROW3_FINAL_ADD_HIGH_FAS;
    let invs = ROW3_CONTROL_INV;
    vec![
        ("sky130_fd_sc_hd__dfxtp_1", dffs),
        ("sky130_fd_sc_hd__and2_1", and2s),
        ("sky130_fd_sc_hd__fa_1", fas),
        ("sky130_fd_sc_hd__inv_1", invs),
    ]
}

impl<P> Layout<P> for Mac8x8Tile
where
    P: MosfetPdk,
{
    fn layout(&self, lib: &Library, pdk: &P) -> CellId {
        match self.topology {
            MacTopology::Digital => digital_layout_skeleton(self, lib, pdk),
            MacTopology::ChargeRedistribution => {
                unimplemented!("Charge-redistribution analog MAC — deferred")
            }
            MacTopology::CurrentMode => {
                unimplemented!("Current-mode analog MAC — deferred")
            }
        }
    }
}

impl<P> Tile<P> for Mac8x8Tile
where
    P: MosfetPdk,
{
    fn pitch(&self) -> Vec2 {
        match self.topology {
            MacTopology::Digital => Vec2::new(DIGITAL_PITCH_X_DBU, DIGITAL_PITCH_Y_DBU),
            MacTopology::ChargeRedistribution => unimplemented!("CR pitch — deferred"),
            MacTopology::CurrentMode => unimplemented!("CM pitch — deferred"),
        }
    }

    fn rails(&self, pdk: &P) -> RailSpec {
        match self.topology {
            MacTopology::Digital => digital_rails(pdk),
            MacTopology::ChargeRedistribution => unimplemented!("CR rails — deferred"),
            MacTopology::CurrentMode => unimplemented!("CM rails — deferred"),
        }
    }

    fn edge_ports(&self, side: Side, pdk: &P) -> Vec<EdgePort> {
        match self.topology {
            MacTopology::Digital => digital_edge_ports(side, pdk),
            MacTopology::ChargeRedistribution => {
                unimplemented!("CR edge ports — deferred")
            }
            MacTopology::CurrentMode => {
                unimplemented!("CM edge ports — deferred")
            }
        }
    }
}

// ── Digital topology bodies ────────────────────────────────────────

/// **v1 Digital MAC tile layout** — rails + row 0 cell placement.
///
/// Row 0 = 8 × `dfxtp_1` (weight register) + 32 × `and2_1`
/// (multiplier partial products, rows 0-3) + 24 × `fa_1`
/// (multiplier summing rows 0-2). Total 64 cells.
///
/// **Caller responsibility:** populate `lib` with the foundry
/// (or mock) sc_hd cell library before calling — either via
/// `ScHdLibrary::load(...)` for production or
/// `populate_mock_sc_hd(...)` for tests. Missing cells panic with
/// the helpful message from `StdCell::layout`.
///
/// Rows 1-3 are NOT placed yet (PLAN.md "Implementation order"
/// steps 5+). Once they land, this comment + the row 0 constants
/// generalize.
fn digital_layout_skeleton<P: MosfetPdk>(tile: &Mac8x8Tile, lib: &Library, pdk: &P) -> CellId {
    let mut b = CellBuilder::new(tile.name());
    let pitch_x = DIGITAL_PITCH_X_DBU;
    let pitch_y = DIGITAL_PITCH_Y_DBU;
    let half_rail = SC_HD_RAIL_WIDTH_DBU / 2;

    // Power rails on met1 — match `rails()` exactly so any future
    // drift between the trait and the geometry is visible.
    let rails = <Mac8x8Tile as Tile<P>>::rails(tile, pdk);
    for &y in rails.vdd_tracks.iter().chain(rails.gnd_tracks.iter()) {
        b.add_shape(
            pdk.metal1(),
            Rect::new(Bbox::new(
                Point::new(0, y - half_rail),
                Point::new(pitch_x, y + half_rail),
            )),
        );
    }

    // Tile-boundary marker on met1 along the top + bottom of the
    // pitch, 1 nm thick — visible in GDS viewers, ignored by extraction.
    for &y in &[0_i64, pitch_y] {
        b.add_shape(
            pdk.metal1(),
            Rect::new(Bbox::new(
                Point::new(0, y),
                Point::new(pitch_x, y + 1),
            )),
        );
    }

    place_row(&mut b, lib, 0, &[
        ("sky130_fd_sc_hd__dfxtp_1", ROW0_WEIGHT_DFFS, W_DFXTP1_DBU),
        ("sky130_fd_sc_hd__and2_1",  ROW0_PP_AND2,     W_AND2_DBU),
        ("sky130_fd_sc_hd__fa_1",    ROW0_SUM_FAS,     W_FA1_DBU),
    ]);
    place_row(&mut b, lib, 1, &[
        ("sky130_fd_sc_hd__and2_1",  ROW1_PP_AND2,        W_AND2_DBU),
        ("sky130_fd_sc_hd__fa_1",    ROW1_SUM_FAS,        W_FA1_DBU),
        ("sky130_fd_sc_hd__dfxtp_1", ROW1_ACCUM_LOW_DFFS, W_DFXTP1_DBU),
    ]);
    place_row(&mut b, lib, 2, &[
        ("sky130_fd_sc_hd__dfxtp_1", ROW2_ACCUM_HIGH_DFFS,    W_DFXTP1_DBU),
        ("sky130_fd_sc_hd__fa_1",    ROW2_FINAL_ADD_LOW_FAS,  W_FA1_DBU),
    ]);
    place_row(&mut b, lib, 3, &[
        ("sky130_fd_sc_hd__fa_1",    ROW3_FINAL_ADD_HIGH_FAS, W_FA1_DBU),
        ("sky130_fd_sc_hd__inv_1",   ROW3_CONTROL_INV,        W_INV1_DBU),
    ]);

    lib.insert(b)
}

/// Walk one row left-to-right, placing each `(cell, count, width)`
/// segment in order. y origin = `row_idx * SC_HD_ROW_DBU`.
///
/// **TODO (PLAN.md step 6+ routing):** odd rows should Y-flip so the
/// shared power rails between abutted rows line up (sc_hd cells have
/// VDD at the bottom of their footprint; row 1 needs GND at its
/// bottom instead). Skipped in v1 — mock cells are unmarked
/// rectangles, so flip orientation doesn't yet matter. Wire up
/// when the foundry library lands and rails carry directional info.
fn place_row(
    b: &mut CellBuilder,
    lib: &Library,
    row_idx: usize,
    segments: &[(&str, usize, i64)],
) {
    let mut x: i64 = 0;
    let y: i64 = (row_idx as i64) * SC_HD_ROW_DBU;
    for (name, count, width) in segments {
        place_n(b, lib, name, *count, &mut x, y, *width);
    }
}

fn place_n(
    b: &mut CellBuilder,
    lib: &Library,
    cell_name: &str,
    count: usize,
    x: &mut i64,
    y: i64,
    width: i64,
) {
    let cid = lib.by_name(cell_name).unwrap_or_else(|| {
        panic!(
            "Mac8x8Tile::layout: foundry cell {cell_name:?} not found in library — \
             load it via `ScHdLibrary::load(...)` (production) or \
             `populate_mock_sc_hd(...)` (tests) before calling tile.layout()"
        )
    });
    for _ in 0..count {
        b.instantiate(cid, Trans::translate(Vec2::new(*x, y)));
        *x += width;
    }
}

/// Sky130_fd_sc_hd convention: VDD and GND rails alternate every
/// `SC_HD_ROW_DBU`, with shared rails between abutting cell rows.
/// For a 4-row tile, that's 5 rails total (3 VDD + 2 GND, or vice
/// versa depending on row 0 polarity — sc_hd convention is VDD at
/// row bottom).
///
/// v1 places both VDD and GND on `met1` (sc_hd's native power layer
/// — the only metal layer guaranteed by the `MosfetPdk` trait). If
/// later we want separate rails on different layers, extend
/// `MosfetPdk` with `met2`/`met3` and split here.
fn digital_rails<P: MosfetPdk>(pdk: &P) -> RailSpec {
    let m1 = pdk.metal1();
    let mut vdd_tracks = Vec::with_capacity(3);
    let mut gnd_tracks = Vec::with_capacity(2);
    for row in 0..=4 {
        let y = row * SC_HD_ROW_DBU;
        if row % 2 == 0 {
            vdd_tracks.push(y);
        } else {
            gnd_tracks.push(y);
        }
    }
    RailSpec {
        vdd_layer: m1,
        gnd_layer: m1,
        width_dbu: SC_HD_RAIL_WIDTH_DBU,
        dbu_per_um: SKY130_DBU_PER_UM,
        vdd_tracks,
        gnd_tracks,
    }
}

/// Edge-port table from the floorplan section. v1 places all ports
/// on `met1` for the same reason `rails` does — the `MosfetPdk`
/// trait only guarantees `metal1`. Bus ports get one `EdgePort` per
/// bit so abutment between neighbours is bit-exact.
///
/// Offsets are evenly distributed across the relevant edge length,
/// inside the rails (VDD at y=0, GND at y=2720 etc.). Real offsets
/// refine once row-level cell placement lands.
fn digital_edge_ports<P: MosfetPdk>(side: Side, pdk: &P) -> Vec<EdgePort> {
    let m1 = pdk.metal1();
    match side {
        // Activations stream W → E across each row.
        // 8 bits, evenly distributed across the tile height.
        Side::West => activation_ports("act_in", side, m1),
        Side::East => activation_ports("act_pass", side, m1),
        // Weights load N → S down each column.
        Side::North => weight_ports("weight_in", side, m1),
        Side::South => weight_ports("weight_pass", side, m1),
    }
}

fn activation_ports(prefix: &str, side: Side, layer: klayout_core::LayerIndex) -> Vec<EdgePort> {
    // Step the 8 bit-ports across the tile height, inset away from
    // the top/bottom rails. Rails sit at y ∈ {0, 2720, 5440, 8160,
    // 10880}; ports go at the 8 inter-rail mid-points / quarter
    // points so they don't collide with met1 power lines.
    let inset = SC_HD_ROW_DBU / 2; // 1360 DBU
    let usable = DIGITAL_PITCH_Y_DBU - 2 * inset;
    (0..8)
        .map(|bit| EdgePort {
            name: format!("{prefix}[{bit}]"),
            side,
            offset_dbu: inset + (usable * bit) / 7,
            layer,
        })
        .collect()
}

fn weight_ports(prefix: &str, side: Side, layer: klayout_core::LayerIndex) -> Vec<EdgePort> {
    // 8 bit-ports across the tile width, evenly spaced and inset
    // from the L/R edges so they don't collide with column-routing.
    let inset = DIGITAL_PITCH_X_DBU / 16;
    let usable = DIGITAL_PITCH_X_DBU - 2 * inset;
    (0..8)
        .map(|bit| EdgePort {
            name: format!("{prefix}[{bit}]"),
            side,
            offset_dbu: inset + (usable * bit) / 7,
            layer,
        })
        .collect()
}
