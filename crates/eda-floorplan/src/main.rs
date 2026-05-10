//! Generate floor plans (PNR-driven where applicable) for every
//! layout-bearing component in the workspace, all targeting one PDK
//! (`Sky130Lite` — sky130 layer numbers, exposed via the `RcLikePdk`
//! and `MosfetPdk` traits the spike Layouts are written against).
//!
//! Outputs land under `target/floorplans/<component>/` as:
//!
//! ```text
//! target/floorplans/
//!   <component>/
//!     floorplan.svg          # rendered layout, includes legend + labels
//!     floorplan.gds          # GDS-II for hand-off to klayout / drc
//!     summary.txt            # bbox, instance count, port list
//!   index.md                 # one-line entry per component
//! ```

mod svg_import;

use std::fs;
use std::path::PathBuf;

use eda_hir::{Layout, PinDirection};
use eda_pdks::GdsfactoryGeneric;
use eda_pnr::{ManhattanRouter, Netlist, PnrFlow};
use eda_stdcells::mock::populate_mock_sc_hd;
use eda_viz::schematic::{
    LabelAlign, Orient as SchemOrient, SchemStyle, Schematic as SchemDoc, Symbol,
};
use eda_viz::Style;
use klayout_core::{
    Angle90, Bbox, CellBuilder, CellId, Instance, Library, Point, Port, Rect, Text, Trans, Vec2,
};
use spike_divider_block::pdks::Sky130Lite;
use spike_divider_block::{
    Capacitor, Diode, MosPolarity, Mosfet, RcDivider, RcLikePdk, Resistor, VoltageSource,
};
use spike_lna::{Lna, RfDemo, SpiralInductor};
// `spike-tinyconv-tile` / `-array` are in the dep tree but no longer
// referenced — `build_mac_tile` / `build_array` now compose the same
// 4-row sc_hd floorplan + 2×2 grid via `eda_pnr::PnrFlow` instead of
// going through `Mac8x8Tile::layout` / `ArrayBlock::layout`.
use spike_waveguide_block::{Mzi, Waveguide};

fn out_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .join("target/floorplans")
}

fn write_floorplan(
    name: &str,
    summary: &str,
    lib: &Library,
    top: CellId,
) -> std::io::Result<PathBuf> {
    let dir = out_root().join(name);
    fs::create_dir_all(&dir)?;

    let svg_path = dir.join("floorplan.svg");
    let gds_path = dir.join("floorplan.gds");
    let txt_path = dir.join("summary.txt");

    let mut style = Style::default();
    style.show_instance_labels = true;
    style.show_ports = true;
    style.show_legend = true;
    style.units_per_dbu = 0.01;

    eda_viz::layout::write_svg(lib, top, &style, &svg_path)?;
    klayout_io::write_gds_path(lib, gds_path.to_str().expect("utf-8 path"))
        .map_err(|e| std::io::Error::other(format!("gds write: {e}")))?;
    fs::write(&txt_path, summary)?;

    Ok(dir)
}

fn cell_summary(name: &str, lib: &Library, top: CellId) -> String {
    let cell = lib.get(top);
    let bbox = cell.full_bbox(lib);
    let w_um = (bbox.max.x - bbox.min.x) as f64 / 1000.0;
    let h_um = (bbox.max.y - bbox.min.y) as f64 / 1000.0;
    let mut s = String::new();
    s.push_str(&format!("component   : {name}\n"));
    s.push_str(&format!("top cell    : {}\n", cell.name().as_str()));
    s.push_str(&format!(
        "bbox (DBU)  : ({},{}) → ({},{})\n",
        bbox.min.x, bbox.min.y, bbox.max.x, bbox.max.y
    ));
    s.push_str(&format!(
        "size        : {:.2} µm × {:.2} µm  ({:.2} µm²)\n",
        w_um,
        h_um,
        w_um * h_um
    ));
    s.push_str(&format!("instances   : {}\n", cell.instances().len()));
    s.push_str(&format!("ports       : {}\n", cell.ports().len()));
    for p in cell.ports() {
        s.push_str(&format!(
            "  - {:>10}  ({:>7},{:>7})  layer={:?}\n",
            p.name, p.center.x, p.center.y, p.layer
        ));
    }
    s
}

struct Generated {
    name: &'static str,
    description: &'static str,
    summary: String,
    dir: PathBuf,
}

fn run_one<F>(
    name: &'static str,
    description: &'static str,
    build: F,
) -> Generated
where
    F: FnOnce() -> (Library, CellId),
{
    let (lib, top) = build();
    let summary = cell_summary(name, &lib, top);
    let dir = write_floorplan(name, &summary, &lib, top)
        .unwrap_or_else(|e| panic!("floorplan {name}: {e}"));
    println!("[{name:<24}] {} → {}", description, dir.display());
    println!("{summary}");
    Generated { name, description, summary, dir }
}

fn with_sky130lite(build: impl FnOnce(&Library, &Sky130Lite) -> CellId) -> (Library, CellId) {
    let lib = Sky130Lite::new_library("sky130lite");
    let pdk = Sky130Lite::register(&lib);
    let top = build(&lib, &pdk);
    (lib, top)
}

fn with_rfdemo(build: impl FnOnce(&Library, &RfDemo) -> CellId) -> (Library, CellId) {
    let lib = RfDemo::new_library("rfdemo");
    let pdk = RfDemo::register(&lib);
    let top = build(&lib, &pdk);
    (lib, top)
}

fn with_photonic(build: impl FnOnce(&Library, &GdsfactoryGeneric) -> CellId) -> (Library, CellId) {
    let lib = GdsfactoryGeneric::new_library("gdsfactory_generic");
    let pdk = GdsfactoryGeneric::register(&lib);
    let top = build(&lib, &pdk);
    (lib, top)
}

fn build_resistor(lib: &Library, pdk: &Sky130Lite) -> CellId {
    Resistor { length: 10_000, id: "R0".into() }.layout(lib, pdk)
}

fn build_mosfet(lib: &Library, pdk: &Sky130Lite) -> CellId {
    Mosfet {
        polarity: MosPolarity::Nmos,
        model: Default::default(),
        w: 2_000,
        l: 500,
        id: "M0".into(),
    }
    .layout(lib, pdk)
}

fn build_rc_divider(lib: &Library, pdk: &Sky130Lite) -> CellId {
    // Composite — `RcDivider::layout` runs through `eda_pnr::pnr_layout`,
    // which exercises `Connectivity` + `ManualPlacer` + `ManhattanRouter`
    // end-to-end. The router materializes the `vmid` net between the
    // two resistor pads as actual rectangles on metal1.
    let div = RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );
    div.layout(lib, pdk)
}

fn build_rc_divider_explicit_pnr(lib: &Library, pdk: &Sky130Lite) -> CellId {
    // Same RcDivider but driven explicitly through `PnrFlow::run` so the
    // generated cell name advertises that this came from the explicit
    // place-and-route flow (the SVG renderer will still pick up the
    // routed wires because they were stamped at the top-cell level).
    use eda_pnr::Connectivity;
    let div = RcDivider::new(
        Resistor { length: 10_000, id: "R1pnr".into() },
        Resistor { length: 30_000, id: "R2pnr".into() },
    );
    let netlist = <RcDivider as Connectivity<Sky130Lite>>::connectivity(&div, lib, pdk);
    let placement = <RcDivider as Connectivity<Sky130Lite>>::transforms(&div, &netlist, lib);
    let placer = eda_pnr::ManualPlacer::new(placement);
    let router = ManhattanRouter::default();
    PnrFlow::new(placer, router).run(&netlist, lib).top
}

// ── TinyConv tile + array via PnrFlow ─────────────────────────────
//
// Cell counts and row positions match `Mac8x8Tile::layout`'s 4-row
// digital floorplan (8 weight DFFs + 32 PP AND2 + 24 sum FA, then
// 32 AND2 + 32 FA + 16 accum_low DFF, then 16 accum_high DFF + 16
// final_add_low FA, then 16 final_add_high FA + 10 control INV — 202
// cells total). What's new: every cell goes into a `Netlist` rather
// than being stamped at a manual `Trans`, and `ManhattanRouter` draws
// real metal1 wires for representative connectivity (clk + reset
// fan-out across all DFFs, per-row carry chains across the FA arrays,
// and per-row weight-bus broadcast).

const SC_HD_ROW_DBU:  i64 = 2_720;
const W_DFXTP1_DBU:   i64 = 2_300;
const W_AND2_DBU:     i64 = 1_290;
const W_FA1_DBU:      i64 = 3_680;
const W_INV1_DBU:     i64 = 1_840;
// Per-row cell counts — matched to spike-tinyconv-tile/src/layout.rs.
const ROW0_DFFS:  usize = 8;
const ROW0_AND2S: usize = 32;
const ROW0_FAS:   usize = 24;
const ROW1_AND2S: usize = 32;
const ROW1_FAS:   usize = 32;
const ROW1_DFFS:  usize = 16;
const ROW2_DFFS:  usize = 16;
const ROW2_FAS:   usize = 16;
const ROW3_FAS:   usize = 16;
const ROW3_INVS:  usize = 10;

struct WrappedStdcells {
    dff:  CellId,
    and2: CellId,
    fa:   CellId,
    inv:  CellId,
}

fn wrap_tile_stdcells(lib: &Library, pdk: &Sky130Lite) -> WrappedStdcells {
    populate_mock_sc_hd(lib, pdk);
    let dff_base  = lib.by_name("sky130_fd_sc_hd__dfxtp_1").expect("mock dfxtp_1");
    let and2_base = lib.by_name("sky130_fd_sc_hd__and2_1").expect("mock and2_1");
    let fa_base   = lib.by_name("sky130_fd_sc_hd__fa_1").expect("mock fa_1");
    let inv_base  = lib.by_name("sky130_fd_sc_hd__inv_1").expect("mock inv_1");
    WrappedStdcells {
        dff: wrap_stdcell(lib, pdk, dff_base, "TC_DFF_wrap", &[
            ("d",   'W', 0.3),
            ("clk", 'W', 0.7),
            ("q",   'E', 0.5),
        ]),
        and2: wrap_stdcell(lib, pdk, and2_base, "TC_AND2_wrap", &[
            ("a",   'W', 0.3),
            ("b",   'W', 0.7),
            ("y",   'E', 0.5),
        ]),
        fa: wrap_stdcell(lib, pdk, fa_base, "TC_FA_wrap", &[
            ("a",   'W', 0.8),
            ("b",   'W', 0.5),
            ("cin", 'W', 0.2),
            ("sum", 'E', 0.7),
            ("cout",'E', 0.3),
        ]),
        inv: wrap_stdcell(lib, pdk, inv_base, "TC_INV_wrap", &[
            ("in",  'W', 0.5),
            ("out", 'E', 0.5),
        ]),
    }
}

/// Place `count` instances of `cell` in a row at `(start_x, y)` with
/// the given x-pitch, returning the indices added (parallel to the
/// caller's per-instance net-naming) and the next x cursor.
fn add_row_instances(
    nl: &mut Netlist,
    transforms: &mut Vec<Trans>,
    cell: CellId,
    name_prefix: &str,
    count: usize,
    start_x: i64,
    y: i64,
    width: i64,
) -> (Vec<usize>, i64) {
    let mut idxs = Vec::with_capacity(count);
    let mut x = start_x;
    for i in 0..count {
        let inst = nl.add_instance(format!("{name_prefix}_{i}"), cell);
        transforms.push(Trans::translate(Vec2::new(x, y)));
        idxs.push(inst);
        x += width;
    }
    (idxs, x)
}

fn build_mac_tile(lib: &Library, pdk: &Sky130Lite) -> CellId {
    let cells = wrap_tile_stdcells(lib, pdk);
    let mut nl = Netlist::new("Block_TinyConvTile".to_string())
        .with_default_signal_layer(pdk.metal1());
    let mut transforms: Vec<Trans> = Vec::with_capacity(202);

    // Row Y-coords (sc_hd convention: rows abut, no gaps).
    let row_y = |r: usize| (r as i64) * SC_HD_ROW_DBU;

    // Row 0: weight DFFs ‖ PP AND2 ‖ sum FA.
    let (r0_dff,  x) = add_row_instances(&mut nl, &mut transforms, cells.dff,
        "r0_wreg",  ROW0_DFFS,  0, row_y(0), W_DFXTP1_DBU);
    let (r0_and2, x) = add_row_instances(&mut nl, &mut transforms, cells.and2,
        "r0_pp",    ROW0_AND2S, x, row_y(0), W_AND2_DBU);
    let (r0_fa,   _) = add_row_instances(&mut nl, &mut transforms, cells.fa,
        "r0_sum",   ROW0_FAS,   x, row_y(0), W_FA1_DBU);
    // Row 1.
    let (r1_and2, x) = add_row_instances(&mut nl, &mut transforms, cells.and2,
        "r1_pp",    ROW1_AND2S, 0, row_y(1), W_AND2_DBU);
    let (r1_fa,   x) = add_row_instances(&mut nl, &mut transforms, cells.fa,
        "r1_sum",   ROW1_FAS,   x, row_y(1), W_FA1_DBU);
    let (r1_dff,  _) = add_row_instances(&mut nl, &mut transforms, cells.dff,
        "r1_acc",   ROW1_DFFS,  x, row_y(1), W_DFXTP1_DBU);
    // Row 2.
    let (r2_dff,  x) = add_row_instances(&mut nl, &mut transforms, cells.dff,
        "r2_acc",   ROW2_DFFS,  0, row_y(2), W_DFXTP1_DBU);
    let (r2_fa,   _) = add_row_instances(&mut nl, &mut transforms, cells.fa,
        "r2_fadd",  ROW2_FAS,   x, row_y(2), W_FA1_DBU);
    // Row 3.
    let (r3_fa,   x) = add_row_instances(&mut nl, &mut transforms, cells.fa,
        "r3_fadd",  ROW3_FAS,   0, row_y(3), W_FA1_DBU);
    let (_r3_inv, _) = add_row_instances(&mut nl, &mut transforms, cells.inv,
        "r3_ctrl",  ROW3_INVS,  x, row_y(3), W_INV1_DBU);

    // Connectivity:
    //
    //   * Per-row carry chains (2-pin segments — `FA_i.cout →
    //     FA_{i+1}.cin`) which the 1-bend planner draws as small
    //     abutment-adjacent jogs, since cout/cin sit on opposite
    //     edges of abutting cells.
    //   * Single-pin `clk` + `rst_b` nets so external ports can be
    //     exposed at sensible coordinates without the router
    //     stamping a row-spanning metal1 mat on top of the DFF row
    //     (which is what an earlier rev did — 8-bit wires at the
    //     same y as the cell bodies, creating visual overlap on
    //     this single-layer PDK).
    let _ = (r0_dff, r1_dff, r2_dff, r0_and2, r1_and2);
    for chain in [&r0_fa[..], &r1_fa[..], &r2_fa[..], &r3_fa[..]] {
        for w in chain.windows(2) {
            let net = format!("carry_{}", w[0]);
            nl.connect(net.clone(), w[0], "cout");
            nl.connect(net,         w[1], "cin");
        }
    }

    // External clk / rst_b — single-pin nets (router emits no wire,
    // but `expose` lifts them to top-level ports so the array level
    // can fan out from the tile boundary).
    if let Some(&w0) = r0_fa.first() {
        nl.connect("clk",   w0, "a");
        nl.connect("rst_b", w0, "b");
    }
    nl.expose("clk",   "clk",   Some(PinDirection::Input));
    nl.expose("rst_b", "rst_b", Some(PinDirection::Input));

    let placer = eda_pnr::ManualPlacer::new(transforms);
    PnrFlow::new(placer, ManhattanRouter::default()).run(&nl, lib).top
}

fn build_array(lib: &Library, pdk: &Sky130Lite) -> CellId {
    // Wrap the PNR-driven tile cell with named edge ports so the
    // array-level router has terminals to hit.
    let tile_inner = build_mac_tile(lib, pdk);
    let tile_w = lib.get(tile_inner).full_bbox(lib).max.x;
    let tile_h = lib.get(tile_inner).full_bbox(lib).max.y;
    let mut tile_wrap_b = CellBuilder::new("Tile_wrap".to_string());
    tile_wrap_b.add_instance(Instance::new(tile_inner, Trans::IDENTITY));
    let kind = pdk.electrical_kind();
    let pad = 1_500_i64;
    let port = |b: &mut CellBuilder, name: &str, x: i64, y: i64, ang: Angle90| {
        b.add_port(
            Port::new(name, pdk.metal1(), Point::new(x, y), ang, pad).with_kind(kind),
        );
    };
    port(&mut tile_wrap_b, "act_w",  0,      tile_h / 2,     Angle90::W);
    port(&mut tile_wrap_b, "act_e",  tile_w, tile_h / 2,     Angle90::E);
    port(&mut tile_wrap_b, "psum_n", tile_w / 2, tile_h,     Angle90::N);
    port(&mut tile_wrap_b, "psum_s", tile_w / 2, 0,          Angle90::S);
    port(&mut tile_wrap_b, "clk",    0,      tile_h * 3 / 4, Angle90::W);
    port(&mut tile_wrap_b, "rst_b",  0,      tile_h * 1 / 4, Angle90::W);
    add_bbox_marker(&mut tile_wrap_b, pdk, tile_w, tile_h);
    let tile = lib.insert(tile_wrap_b);

    // 2×2 grid: tile[r][c]. Tile names follow `tile_r{r}_c{c}`.
    let mut nl = Netlist::new("TinyConvArray2x2".to_string())
        .with_default_signal_layer(pdk.metal1());
    let mut transforms = Vec::with_capacity(4);
    let mut id = vec![vec![0_usize; 2]; 2];
    let chan_x = 20_000_i64;
    let chan_y = 5_000_i64;
    for r in 0..2 {
        for c in 0..2 {
            id[r][c] = nl.add_instance(format!("tile_r{r}_c{c}"), tile);
            let x = (c as i64) * (tile_w + chan_x);
            let y = (r as i64) * (tile_h + chan_y);
            transforms.push(Trans::translate(Vec2::new(x, y)));
        }
    }

    // Inter-tile abutment nets:
    //   row r: act_e(tile[r][0]) → act_w(tile[r][1])
    //   col c: psum_n(tile[0][c]) → psum_s(tile[1][c])
    // Plus global clk + rst_b fan-out to every tile.
    for r in 0..2 {
        nl.connect(format!("act_row{r}"), id[r][0], "act_e");
        nl.connect(format!("act_row{r}"), id[r][1], "act_w");
    }
    for c in 0..2 {
        nl.connect(format!("psum_col{c}"), id[0][c], "psum_n");
        nl.connect(format!("psum_col{c}"), id[1][c], "psum_s");
    }
    for r in 0..2 {
        for c in 0..2 {
            nl.connect("clk",   id[r][c], "clk");
            nl.connect("rst_b", id[r][c], "rst_b");
        }
    }

    nl.expose("clk",   "clk",   Some(PinDirection::Input));
    nl.expose("rst_b", "rst_b", Some(PinDirection::Input));
    nl.expose("act_in",  "act_row0", Some(PinDirection::Input));
    nl.expose("psum_top", "psum_col0", Some(PinDirection::Output));

    let placer = eda_pnr::ManualPlacer::new(transforms);
    // Steiner for the 4-pin clk/rst nets so they don't show up as
    // overlapping star segments at the array's left edge.
    let router = ManhattanRouter::default()
        .with_multi_pin(eda_pnr::MultiPinStrategy::Steiner);
    PnrFlow::new(placer, router).run(&nl, lib).top
}

// ── Additional sky130 primitives ───────────────────────────────────

fn build_diode(lib: &Library, pdk: &Sky130Lite) -> CellId {
    Diode { size: 4_000, is_value: 1.0e-15, id: "D0".into() }.layout(lib, pdk)
}

fn build_capacitor(lib: &Library, pdk: &Sky130Lite) -> CellId {
    Capacitor { plate_size: 8_000, id: "C0".into() }.layout(lib, pdk)
}

fn build_voltage_source(lib: &Library, pdk: &Sky130Lite) -> CellId {
    VoltageSource::from_volts(1.8, "VDD").layout(lib, pdk)
}

fn build_mosfet_pmos(lib: &Library, pdk: &Sky130Lite) -> CellId {
    Mosfet {
        polarity: MosPolarity::Pmos,
        model: Default::default(),
        w: 4_000,
        l: 500,
        id: "MP0".into(),
    }
    .layout(lib, pdk)
}

// ── Standard cells (eda-stdcells mock library) ─────────────────────

/// Render a single mock sc_hd standard cell as its own top cell. The
/// mock library populates the lib once, then `lib.by_name(cell)`
/// resolves to the foundry-canonical CellId.
fn build_stdcell(name: &'static str) -> impl FnOnce(&Library, &Sky130Lite) -> CellId {
    move |lib, pdk| {
        populate_mock_sc_hd(lib, pdk);
        lib.by_name(name).unwrap_or_else(|| panic!("stdcell {name} not in mock library"))
    }
}

// ── Block-level SAR ADC floorplan ──────────────────────────────────

// ── SAR ADC sub-block helpers ─────────────────────────────────────
//
// One helper per sub-block type that lives in the workspace:
// `SampleHold`, `SarRegister<N>`, `R2RDac<N>`, `Comparator`. Each
// helper takes the *actual* struct (not an invented placeholder) and
// builds a composite cell out of **real primitive instances** —
// `Resistor`/`Mosfet`/`Capacitor` from `spike-divider-block` and the
// mock `dfxtp_1` / `inv_1` cells from `eda-stdcells::mock`. Sizes,
// port positions, and the netlist that wires them together all derive
// from the same struct fields `SarAdc::emit_spice` reads, so the
// floor plan's geometry matches the circuit being simulated.
//
// Cell naming convention `Block_<Tag>` makes `eda-viz`'s
// `short_instance_label` heuristic (which returns the second
// underscore-separated field) display "SH" / "DAC" / "SAR" / "CMP"
// instead of collapsing to a shared parent prefix.

/// Add four 1-DBU-thick perimeter stripes on metal1 so a composite
/// cell with no direct shapes still has a non-empty `local_bbox`.
/// Without this, `eda-viz`'s recursive flatten walks composite
/// instances with `local_bbox == EMPTY` (sentinel `i64::MIN/MAX`) and
/// panics when negating the y-coord for label placement.
/// Doesn't paint a solid background — the four stripes are 1 nm wide
/// each, invisible at typical zoom but enough to define the bbox.
fn add_bbox_marker(cb: &mut CellBuilder, pdk: &Sky130Lite, w: i64, h: i64) {
    let l = pdk.metal1();
    cb.add_shape(l, Rect::new(Bbox::new(Point::new(0, 0), Point::new(w, 1))));
    cb.add_shape(l, Rect::new(Bbox::new(Point::new(0, h - 1), Point::new(w, h))));
    cb.add_shape(l, Rect::new(Bbox::new(Point::new(0, 0), Point::new(1, h))));
    cb.add_shape(l, Rect::new(Bbox::new(Point::new(w - 1, 0), Point::new(w, h))));
}

/// Helper: edge port on a cell of the given size. Used after
/// composite child placement to surface the named connection point on
/// the bbox edge so PNR can route to it.
fn add_edge_port(
    cb: &mut CellBuilder,
    pdk: &Sky130Lite,
    name: &str,
    w: i64,
    h: i64,
    side: char,
    frac: f32,
) {
    let kind = pdk.electrical_kind();
    let pad = 800_i64;
    let along = |extent: i64, f: f32| ((extent as f32) * f.clamp(0.0, 1.0)) as i64;
    let (px, py, ang) = match side {
        'W' => (0,              along(h, frac), Angle90::W),
        'E' => (w,              along(h, frac), Angle90::E),
        'S' => (along(w, frac), 0,              Angle90::S),
        'N' => (along(w, frac), h,              Angle90::N),
        _ => unreachable!("side must be W/E/S/N"),
    };
    cb.add_port(
        Port::new(name, pdk.metal1(), Point::new(px, py), ang, pad).with_kind(kind),
    );
}

/// `SampleHold` → composite cell built through `PnrFlow`. Three real
/// primitives (pass-gate NMOS switch, metal1 hold cap, NMOS source-
/// follower buffer) wired by an explicit netlist; the router emits the
/// internal `vsamp` net (switch source ↔ cap top-plate ↔ buffer gate)
/// so the metal1 trace shows up as actual routed geometry.
fn make_sh_cell(
    lib: &Library,
    pdk: &Sky130Lite,
    sh: &spike_sample_hold::SampleHold,
    _id: &str,
) -> CellId {
    let cap_side = ((sh.c_hold * 30.0).sqrt().max(8.0) * 1_000.0) as i64;

    let m_switch = Mosfet::nmos(2_000, 500, "M_switch").layout(lib, pdk);
    let c_hold   = Capacitor { plate_size: cap_side, id: "C_hold".into() }.layout(lib, pdk);
    let m_buf    = Mosfet::nmos(4_000, 500, "M_buf").layout(lib, pdk);

    let switch_w = lib.get(m_switch).local_bbox().max.x
        - lib.get(m_switch).local_bbox().min.x;
    let chan = 4_000_i64;

    let mut nl = Netlist::new("Block_SH".to_string())
        .with_default_signal_layer(pdk.metal1());
    let i_switch = nl.add_instance("M_switch", m_switch);
    let i_cap    = nl.add_instance("C_hold",   c_hold);
    let i_buf    = nl.add_instance("M_buf",    m_buf);

    // Topology: `vsamp` is the held node (switch.s ↔ cap.a ↔ buf.g).
    nl.connect("vin",    i_switch, "d");
    nl.connect("clk_sh", i_switch, "g");
    nl.connect("vsamp",  i_switch, "s");
    nl.connect("vsamp",  i_cap,    "a");
    nl.connect("vsamp",  i_buf,    "g");
    nl.connect("vdd",    i_buf,    "d");
    nl.connect("vhold",  i_buf,    "s");
    nl.connect("gnd",    i_cap,    "b");
    nl.connect("gnd",    i_switch, "b");
    nl.connect("gnd",    i_buf,    "b");

    nl.expose("vin",    "vin",    Some(PinDirection::Input));
    nl.expose("clk_sh", "clk_sh", Some(PinDirection::Input));
    nl.expose("vhold",  "vhold",  Some(PinDirection::Output));
    nl.expose("vdd",    "vdd",    Some(PinDirection::Power));
    nl.expose("gnd",    "gnd",    Some(PinDirection::Ground));

    // Place: switch on left, cap in the middle, buffer on right.
    let transforms = vec![
        Trans::translate(Vec2::new(0,                       0)),
        Trans::translate(Vec2::new(switch_w + chan,         0)),
        Trans::translate(Vec2::new(switch_w + chan + cap_side + chan, 0)),
    ];
    let placer = eda_pnr::ManualPlacer::new(transforms);
    PnrFlow::new(placer, ManhattanRouter::default()).run(&nl, lib).top
}

/// Wrap a port-less mock stdcell with named edge ports so `PnrFlow`'s
/// router has somewhere to terminate. The base cell is instantiated
/// once at identity; named ports get added on the bbox edges. Returns
/// a fresh `CellId` for the wrapped cell.
fn wrap_stdcell(
    lib: &Library,
    pdk: &Sky130Lite,
    base: CellId,
    wrap_name: &str,
    ports: &[(&str, char, f32)],
) -> CellId {
    let bb = lib.get(base).local_bbox();
    let w = bb.max.x;
    let h = bb.max.y;
    let mut cb = CellBuilder::new(wrap_name.to_string());
    cb.add_instance(Instance::new(base, Trans::IDENTITY));
    for &(name, side, frac) in ports {
        add_edge_port(&mut cb, pdk, name, w, h, side, frac);
    }
    // No bbox marker rect — `eda-viz`'s collect now falls back to
    // `full_bbox(lib)` for cells with empty `local_bbox`, so the
    // wrapper renders correctly without painting four 1-DBU marker
    // stripes per instance (which were 800+ extra rects per tile).
    lib.insert(cb)
}

/// `SarRegister<N>` → composite cell driven by `PnrFlow`. The mock
/// `dfxtp_1` and `inv_1` stdcells from `eda-stdcells::mock` are
/// port-less rectangles, so each base cell gets wrapped once with
/// named edge ports (`d`/`clk`/`q`/`vdd`/`vss` for the DFF;
/// `in`/`out`/`vdd`/`vss` for the inverter); then the SAR-register
/// netlist places `N` DFFs + `N` phase-buffer inverters + 1 reset
/// buffer + 1 cmp buffer and `ManhattanRouter` wires them. Topology
/// (simplified, sufficient for floor plan PNR):
///
/// ```text
///   phase_i  ──► inv_phase_i ──► (sink, drives no DFF — keeps the
///                                  external port real but the
///                                  per-bit clock comes from `capture_i`)
///   capture_i ──► dff_i.clk
///   cmp       ──► inv_cmp.in ──► cmp_int ──► dff_i.d   (∀ i)
///   dff_i.q   ──► bit_i
///   reset_b   ──► inv_reset.in
/// ```
fn make_sar_cell<const N: usize>(
    lib: &Library,
    pdk: &Sky130Lite,
    _sar: &spike_sar_register::SarRegister<N>,
    _id: &str,
) -> CellId {
    populate_mock_sc_hd(lib, pdk);
    let dff_base = lib.by_name("sky130_fd_sc_hd__dfxtp_1").expect("mock dfxtp_1");
    let inv_base = lib.by_name("sky130_fd_sc_hd__inv_1").expect("mock inv_1");

    // Wrap once: every dff / inv instance below resolves to the same
    // wrapped CellId, keeping the GDS dedup-friendly.
    let dff_wrap = wrap_stdcell(lib, pdk, dff_base, "DFF_wrap", &[
        ("d",   'W', 0.3),
        ("clk", 'W', 0.7),
        ("q",   'E', 0.5),
        ("vdd", 'N', 0.5),
        ("vss", 'S', 0.5),
    ]);
    let inv_wrap = wrap_stdcell(lib, pdk, inv_base, "INV_wrap", &[
        ("in",  'W', 0.5),
        ("out", 'E', 0.5),
        ("vdd", 'N', 0.5),
        ("vss", 'S', 0.5),
    ]);

    let dff_w = lib.get(dff_base).local_bbox().max.x;
    let dff_h = lib.get(dff_base).local_bbox().max.y;
    let inv_w = lib.get(inv_base).local_bbox().max.x;
    let chan  = 1_500_i64;

    let mut nl = Netlist::new("Block_SAR".to_string())
        .with_default_signal_layer(pdk.metal1());

    // Per-bit: 1 DFF + 1 phase-buffer inverter.
    let i_dffs:    Vec<usize> = (0..N).map(|i| nl.add_instance(format!("dff_{i}"),    dff_wrap)).collect();
    let i_phbufs:  Vec<usize> = (0..N).map(|i| nl.add_instance(format!("inv_ph_{i}"), inv_wrap)).collect();
    // Two extra inverters: one buffers `cmp` (its output drives every
    // dff.d), one swallows `reset_b` (sink so that net has a real pin).
    let i_cmp_inv   = nl.add_instance("inv_cmp",   inv_wrap);
    let i_reset_inv = nl.add_instance("inv_reset", inv_wrap);

    for i in 0..N {
        nl.connect(format!("phase_{i}"),   i_phbufs[i], "in");
        // phaseb_i is purely internal — sink for the phase buffer
        // (mirrors a real SAR FSM where the per-bit phase passes
        // through clock-generation logic before reaching the DFF).
        nl.connect(format!("phaseb_{i}"),  i_phbufs[i], "out");
        nl.connect(format!("capture_{i}"), i_dffs[i],   "clk");
        nl.connect("cmp_int",              i_dffs[i],   "d");
        nl.connect(format!("bit_{i}"),     i_dffs[i],   "q");
        nl.connect("vdd",                  i_dffs[i],   "vdd");
        nl.connect("gnd",                  i_dffs[i],   "vss");
        nl.connect("vdd",                  i_phbufs[i], "vdd");
        nl.connect("gnd",                  i_phbufs[i], "vss");
    }
    nl.connect("cmp",     i_cmp_inv,   "in");
    nl.connect("cmp_int", i_cmp_inv,   "out");
    nl.connect("vdd",     i_cmp_inv,   "vdd");
    nl.connect("gnd",     i_cmp_inv,   "vss");
    nl.connect("reset_b", i_reset_inv, "in");
    nl.connect("vdd",     i_reset_inv, "vdd");
    nl.connect("gnd",     i_reset_inv, "vss");

    // External pins — every port the SAR ADC top expects on this cell.
    nl.expose("cmp",     "cmp",     Some(PinDirection::Input));
    nl.expose("reset_b", "reset_b", Some(PinDirection::Input));
    nl.expose("vdd",     "vdd",     Some(PinDirection::Power));
    nl.expose("gnd",     "gnd",     Some(PinDirection::Ground));
    for i in 0..N {
        nl.expose(format!("phase_{i}"),   format!("phase_{i}"),   Some(PinDirection::Input));
        nl.expose(format!("capture_{i}"), format!("capture_{i}"), Some(PinDirection::Input));
        nl.expose(format!("bit_{i}"),     format!("bit_{i}"),     Some(PinDirection::Output));
    }

    // Placement: row 0 = N DFFs, row 1 = N phase inverters above,
    // then the cmp + reset buffers tucked at the right.
    let row1_y = dff_h + chan;
    let mut transforms = Vec::with_capacity(2 * N + 2);
    for i in 0..N {
        transforms.push(Trans::translate(Vec2::new(i as i64 * dff_w, 0)));
    }
    for i in 0..N {
        transforms.push(Trans::translate(Vec2::new(i as i64 * inv_w, row1_y)));
    }
    let extras_x = (N as i64) * dff_w.max(inv_w) + chan;
    transforms.push(Trans::translate(Vec2::new(extras_x,            0)));     // inv_cmp
    transforms.push(Trans::translate(Vec2::new(extras_x + inv_w + chan, 0))); // inv_reset

    let placer = eda_pnr::ManualPlacer::new(transforms);
    PnrFlow::new(placer, ManhattanRouter::default()).run(&nl, lib).top
}

/// `R2RDac<N>` → composite cell: an honest R-2R resistor ladder built
/// from `Resistor` primitives. N "R" series resistors form the top
/// spine, N "2R" parallel resistors hang downward, plus one 2R
/// termination — `2N + 1` real `Resistor` instances total. Length per
/// resistor scales with `r_ohms` via `length_to_resistance` (sky130
/// poly: 100 Ω/sq, 1 µm width).
fn make_dac_cell<const N: usize>(
    lib: &Library,
    pdk: &Sky130Lite,
    dac: &spike_dac_r2r::R2RDac<N>,
    _id: &str,
) -> CellId {
    use spike_divider_block::resistance_to_length;
    let r_len  = resistance_to_length(dac.r_ohms as f32);
    let r2_len = resistance_to_length((2.0 * dac.r_ohms) as f32);
    let pitch  = r_len + 4_000; // spine pitch incl. tap

    // Tile each bit as its own horizontal row `[2R-leg | gap | R-spine
    // segment]`, rows stacked top-to-bottom (MSB on top, LSB just above
    // the termination row). Inter-resistor connectivity (tap nodes,
    // vlow / vout / bit_i nets) flows through `PnrFlow` so the router
    // emits real metal1 wires between resistor terminals — no
    // hand-stamped boundary; the routed shapes define the cell extent.
    let row_h = RES_HEIGHT + 6_000;
    let gap = 4_000_i64;
    let _ = pitch; // legacy horizontal-spine pitch, no longer used

    // 1. Lay out child Resistor primitives (real `Layout` impls).
    let r_term_id = Resistor { length: r2_len, id: "Rterm".into() }.layout(lib, pdk);
    let r_ids: Vec<CellId> = (0..N)
        .map(|i| Resistor { length: r_len, id: format!("Rs{i}") }.layout(lib, pdk))
        .collect();
    let r2_ids: Vec<CellId> = (0..N)
        .map(|i| Resistor { length: r2_len, id: format!("R2_{i}") }.layout(lib, pdk))
        .collect();

    // 2. Netlist: tap_0 ties Rterm.b + 2R_0.a + Rs_0.a; each subsequent
    //    tap_i ties Rs_{i-1}.b + 2R_i.a + Rs_i.a; vout = Rs_{N-1}.b;
    //    vlow = Rterm.a; bit_i = 2R_i.b.
    let mut nl = Netlist::new("Block_DAC".to_string())
        .with_default_signal_layer(pdk.metal1());
    let i_term = nl.add_instance("Rterm", r_term_id);
    let i_rs:  Vec<usize> = (0..N).map(|i| nl.add_instance(format!("Rs{i}"), r_ids[i])).collect();
    let i_2r:  Vec<usize> = (0..N).map(|i| nl.add_instance(format!("R2_{i}"), r2_ids[i])).collect();

    nl.connect("vlow",  i_term, "a");
    nl.connect("tap_0", i_term, "b");
    nl.connect("tap_0", i_2r[0], "a");
    nl.connect("tap_0", i_rs[0], "a");
    for i in 1..N {
        let tn = format!("tap_{i}");
        nl.connect(&tn, i_rs[i - 1], "b");
        nl.connect(&tn, i_2r[i],     "a");
        nl.connect(&tn, i_rs[i],     "a");
    }
    nl.connect("vout", i_rs[N - 1], "b");
    for i in 0..N {
        let bn: &'static str = Box::leak(format!("bit_{i}").into_boxed_str());
        nl.connect(bn, i_2r[i], "b");
    }

    nl.expose("vlow", "vlow", Some(PinDirection::Ground));
    nl.expose("vout", "vout", Some(PinDirection::Output));
    for i in 0..N {
        let bn: &'static str = Box::leak(format!("bit_{i}").into_boxed_str());
        nl.expose(bn, bn, Some(PinDirection::Input));
    }

    // 3. Placement (matches `add_instance` order: Rterm, then Rs_0..Rs_{N-1},
    //    then 2R_0..2R_{N-1}).
    let mut transforms = Vec::with_capacity(2 * N + 1);
    transforms.push(Trans::translate(Vec2::new(0, 0))); // Rterm at bottom
    for i in 0..N {
        let row_y = (i as i64 + 1) * row_h;
        transforms.push(Trans::translate(Vec2::new(r2_len + gap, row_y))); // Rs_i (spine)
    }
    for i in 0..N {
        let row_y = (i as i64 + 1) * row_h;
        transforms.push(Trans::translate(Vec2::new(0, row_y))); // 2R_i (leg)
    }

    // 4. PnrFlow stamps the placed instances + routes every net.
    let placer = eda_pnr::ManualPlacer::new(transforms);
    PnrFlow::new(placer, ManhattanRouter::default()).run(&nl, lib).top
}
const RES_HEIGHT: i64 = 1_500;

/// `Comparator` → 7-MOSFET regenerative latch driven by `PnrFlow`.
/// Real circuit topology: NMOS tail at the bottom, NMOS input pair
/// above, NMOS cross-coupled latch above that, PMOS load pair on top.
/// The router draws the dense `intl` / `intr` nets (5-pin each — the
/// cross-coupled gates + drains + PMOS load drains/gates) so the
/// regenerative cross-coupling is visible as actual routed metal1.
fn make_cmp_cell(
    lib: &Library,
    pdk: &Sky130Lite,
    cmp: &spike_comparator::Comparator,
    _id: &str,
) -> CellId {
    let m_tail  = Mosfet::nmos(8_000, 500, "M_tail").layout(lib, pdk);
    let m_inp_l = Mosfet::nmos(4_000, 500, "M_in_p").layout(lib, pdk);
    let m_inp_r = Mosfet::nmos(4_000, 500, "M_in_n").layout(lib, pdk);
    let m_lat_l = Mosfet::nmos(4_000, 500, "M_lat_l").layout(lib, pdk);
    let m_lat_r = Mosfet::nmos(4_000, 500, "M_lat_r").layout(lib, pdk);
    let m_pld_l = Mosfet { polarity: MosPolarity::Pmos, model: Default::default(),
        w: 6_000, l: 500, id: "M_pld_l".into() }.layout(lib, pdk);
    let m_pld_r = Mosfet { polarity: MosPolarity::Pmos, model: Default::default(),
        w: 6_000, l: 500, id: "M_pld_r".into() }.layout(lib, pdk);

    let mut nl = Netlist::new(format!("Block_CMP_k{}", cmp.k as i64))
        .with_default_signal_layer(pdk.metal1());
    let i_tail  = nl.add_instance("M_tail",  m_tail);
    let i_inp_l = nl.add_instance("M_in_p",  m_inp_l);
    let i_inp_r = nl.add_instance("M_in_n",  m_inp_r);
    let i_lat_l = nl.add_instance("M_lat_l", m_lat_l);
    let i_lat_r = nl.add_instance("M_lat_r", m_lat_r);
    let i_pld_l = nl.add_instance("M_pld_l", m_pld_l);
    let i_pld_r = nl.add_instance("M_pld_r", m_pld_r);

    // Tail bias: g = vdd (always-on for the floorplan; a real latch
    // would gate it with clk).
    nl.connect("vdd",  i_tail,  "g");
    nl.connect("gnd",  i_tail,  "s");
    nl.connect("gnd",  i_tail,  "b");
    nl.connect("tail", i_tail,  "d");

    // Input pair sources tied to tail; gates take vp/vn.
    nl.connect("tail", i_inp_l, "s");
    nl.connect("tail", i_inp_r, "s");
    nl.connect("vp",   i_inp_l, "g");
    nl.connect("vn",   i_inp_r, "g");
    nl.connect("intl", i_inp_l, "d");
    nl.connect("intr", i_inp_r, "d");
    nl.connect("gnd",  i_inp_l, "b");
    nl.connect("gnd",  i_inp_r, "b");

    // NMOS cross-coupled latch — same drains as input pair, gates
    // crossed.
    nl.connect("tail", i_lat_l, "s");
    nl.connect("tail", i_lat_r, "s");
    nl.connect("intl", i_lat_l, "d");
    nl.connect("intr", i_lat_r, "d");
    nl.connect("intr", i_lat_l, "g");
    nl.connect("intl", i_lat_r, "g");
    nl.connect("gnd",  i_lat_l, "b");
    nl.connect("gnd",  i_lat_r, "b");

    // PMOS cross-coupled load — drains tie to intl/intr, gates crossed,
    // sources to vdd, body to vdd.
    nl.connect("intl", i_pld_l, "d");
    nl.connect("intr", i_pld_r, "d");
    nl.connect("intr", i_pld_l, "g");
    nl.connect("intl", i_pld_r, "g");
    nl.connect("vdd",  i_pld_l, "s");
    nl.connect("vdd",  i_pld_r, "s");
    nl.connect("vdd",  i_pld_l, "b");
    nl.connect("vdd",  i_pld_r, "b");

    nl.expose("vp",   "vp",   Some(PinDirection::Input));
    nl.expose("vn",   "vn",   Some(PinDirection::Input));
    nl.expose("vout", "intr", Some(PinDirection::Output));
    nl.expose("vdd",  "vdd",  Some(PinDirection::Power));
    nl.expose("gnd",  "gnd",  Some(PinDirection::Ground));

    // Placement: 4 rows, 2 columns. Tail spans both columns at the
    // bottom (placed in the left column for simplicity).
    let row_h = 9_000_i64;
    let col_w = 7_000_i64;
    let transforms = vec![
        Trans::translate(Vec2::new(col_w / 2,   0)),         // M_tail (centered)
        Trans::translate(Vec2::new(0,           row_h)),     // M_in_p
        Trans::translate(Vec2::new(col_w,       row_h)),     // M_in_n
        Trans::translate(Vec2::new(0,           2 * row_h)), // M_lat_l
        Trans::translate(Vec2::new(col_w,       2 * row_h)), // M_lat_r
        Trans::translate(Vec2::new(0,           3 * row_h)), // M_pld_l
        Trans::translate(Vec2::new(col_w,       3 * row_h)), // M_pld_r
    ];
    let placer = eda_pnr::ManualPlacer::new(transforms);
    PnrFlow::new(placer, ManhattanRouter::default()).run(&nl, lib).top
}

// ── SAR ADC schematic ─────────────────────────────────────────────
//
// Symbolic block-level schematic of the same `SarAdc<N>` struct: four
// `Subcircuit` symbols (one per sub-block) wired by named nets that
// match the layout-side netlist 1:1, plus pin labels for the external
// terminals. Renders via `eda_viz::schematic::render_to_svg`.

fn build_sar_schematic<const N: usize>(_adc: &spike_sar_adc::SarAdc<N>) -> SchemDoc {
    let mut s = SchemDoc::new();
    s.title = Some(format!("SarAdc<{N}> — block-level schematic"));

    // Pin counts kept compact per Subcircuit so each symbol stays
    // small and labels never crash into the body. Per-bit `phase_i` /
    // `capture_i` / `bit_i` pins collapse to single bus pins
    // (`phase[N-1:0]`, `capture[N-1:0]`, `code[N-1:0]`); vdd/gnd are
    // implicit and not drawn as Subcircuit pins.
    let bus_phase = format!("phase[{}:0]",   N - 1);
    let bus_cap   = format!("capture[{}:0]", N - 1);
    let bus_code  = format!("code[{}:0]",    N - 1);

    // Symbols spaced wide enough that label runs (longest = ~10 chars)
    // sit cleanly between adjacent bodies.
    let sh_pins: Vec<String> = ["vin", "vhold", "clk_sh"]
        .iter().map(|s| s.to_string()).collect();
    let sh = s.place(
        Symbol::Subcircuit { label: "S/H".into(), pin_names: sh_pins.clone() },
        (-22.0, 8.0),
        SchemOrient::Horizontal,
    );
    let cmp_pins: Vec<String> = ["vp", "vn", "vout"]
        .iter().map(|s| s.to_string()).collect();
    let cmp = s.place(
        Symbol::Subcircuit { label: "Cmp".into(), pin_names: cmp_pins.clone() },
        (0.0, 0.0),
        SchemOrient::Horizontal,
    );
    let sar_pins: Vec<String> = vec![
        "cmp".into(), bus_phase.clone(), bus_cap.clone(),
        "reset_b".into(), bus_code.clone(),
    ];
    let sar = s.place(
        Symbol::Subcircuit { label: format!("SAR<{N}>"), pin_names: sar_pins.clone() },
        (22.0, 0.0),
        SchemOrient::Horizontal,
    );
    let dac_pins: Vec<String> = vec![bus_code.clone(), "vlow".into(), "vout".into()];
    let dac = s.place(
        Symbol::Subcircuit { label: format!("R2R_DAC<{N}>"), pin_names: dac_pins.clone() },
        (0.0, -10.0),
        SchemOrient::Horizontal,
    );

    let pin_at = |placed_id: usize, name: &str, names: &[String]| -> (f64, f64) {
        let i = names.iter().position(|n| n == name).expect("pin name");
        s.symbols[placed_id].pin(i)
    };
    let p_sh_vin    = pin_at(sh,  "vin",     &sh_pins);
    let p_sh_vhold  = pin_at(sh,  "vhold",   &sh_pins);
    let p_sh_clk    = pin_at(sh,  "clk_sh",  &sh_pins);
    let p_cmp_vp    = pin_at(cmp, "vp",      &cmp_pins);
    let p_cmp_vn    = pin_at(cmp, "vn",      &cmp_pins);
    let p_cmp_vout  = pin_at(cmp, "vout",    &cmp_pins);
    let p_dac_vout  = pin_at(dac, "vout",    &dac_pins);
    let p_dac_vlow  = pin_at(dac, "vlow",    &dac_pins);
    let p_dac_code  = pin_at(dac, &bus_code, &dac_pins);
    let p_sar_cmp   = pin_at(sar, "cmp",     &sar_pins);
    let p_sar_reset = pin_at(sar, "reset_b", &sar_pins);
    let p_sar_phase = pin_at(sar, &bus_phase,&sar_pins);
    let p_sar_cap   = pin_at(sar, &bus_cap,  &sar_pins);
    let p_sar_code  = pin_at(sar, &bus_code, &sar_pins);

    // Inter-block wires — strict Manhattan routes through reserved
    // channels so no segment crosses a symbol body and no two
    // verticals share an x-coordinate.
    //
    // Symbol bodies (schematic units): S/H x∈[-23.4,-20.6], y∈[6.2,9.8];
    //   Cmp x∈[-1.4,1.4], y∈[-1.8,1.8]; SAR x∈[20.6,23.4], y∈[-3,3];
    //   DAC x∈[-1.4,1.4], y∈[-11.8,-8.2].
    // Routing channels: west=x≈-7 / -9, east=x≈12 / 18, north=y≈5,
    // south=y≈-5 / -7. Each wire owns its column / row.
    s.wire_named("vhold", [
        p_sh_vhold,                         // (-20.4, 8.0)
        (-7.0, p_sh_vhold.1),               // → west channel
        (-7.0, p_cmp_vp.1),                 // → down to Cmp.vp y
        p_cmp_vp,                           // → (-1.6, 0.6)
    ]);
    s.wire_named("vdac", [
        p_dac_vout,                         // (-1.6, -10.6)
        (-9.0, p_dac_vout.1),               // → outer-west channel
        (-9.0, -5.0),                       // → up
        (12.0, -5.0),                       // → south channel right past Cmp
        (12.0, p_cmp_vn.1),                 // → up to Cmp.vn y (= 0)
        p_cmp_vn,                           // → (1.6, 0)
    ]);
    s.wire_named("cmp", [
        p_cmp_vout,                         // (-1.6, -0.6)
        (-5.0, p_cmp_vout.1),               // → inner-west channel
        (-5.0, 5.0),                        // → up to north channel
        (18.0, 5.0),                        // → north channel over Cmp & SAR
        (18.0, p_sar_cmp.1),                // → down to SAR.cmp y
        p_sar_cmp,                          // → (20.4, 1.2)
    ]);
    s.wire_named(bus_code.clone(), [
        p_sar_code,                         // (20.4, -1.2)
        (18.0, p_sar_code.1),               // → east channel left of SAR
        (18.0, -7.0),                       // → down to outer-south channel
        (p_dac_code.0, -7.0),               // → left to DAC code x
        p_dac_code,                         // → (-1.6, -9.4)
    ]);

    // External pin labels — `End` for left-side pins (text grows
    // leftward away from the symbol body), `Start` for right-side
    // (default rightward), `Middle` for top/bottom.
    s.pin_label_aligned(p_sh_vin,    "vin",        LabelAlign::End);
    s.pin_label_aligned(p_sh_clk,    "clk_sh",     LabelAlign::End);
    s.pin_label_aligned(p_dac_vlow,  "vlow / gnd", LabelAlign::Middle);
    s.pin_label_aligned(p_sar_reset, "reset_b",    LabelAlign::Start);
    // SAR pin order alternates left/right: idx 0=cmp(L), 1=phase(R),
    // 2=capture(L), 3=reset_b(R), 4=code(L). So phase / reset_b take
    // `Start` (rightward), capture takes `End` (leftward into the gap).
    s.pin_label_aligned(p_sar_phase, &bus_phase,   LabelAlign::Start);
    s.pin_label_aligned(p_sar_cap,   &bus_cap,     LabelAlign::End);
    s
}

/// SAR ADC top-level floorplan from the *actual* `SarAdc<N>` struct
/// in `spike-sar-adc`. Sub-block sizes derive from struct fields
/// (`SampleHold::c_hold`, `R2RDac::r_ohms`, the const generic `N`),
/// instance names mirror `SarAdc::emit_spice`'s `id` scheme
/// (`{id}_sh`, `{id}_sar`, `{id}_dac`, `{id}_cmp`), and the netlist
/// is a layout-domain transcription of the same internal nets that
/// emission writes — `vhold`, `v_dac`, `cmp` — plus the 3N+5
/// external terminals it expects.
fn build_sar_adc_floorplan<const N: usize>(
    lib: &Library,
    pdk: &Sky130Lite,
    adc: &spike_sar_adc::SarAdc<N>,
    id: &str,
) -> CellId {
    let sh_id  = make_sh_cell(lib, pdk, &adc.sh,   id);
    let sar_id = make_sar_cell::<N>(lib, pdk, &adc.sar, id);
    let dac_id = make_dac_cell::<N>(lib, pdk, &adc.dac, id);
    let cmp_id = make_cmp_cell(lib, pdk, &adc.comp, id);

    let top_name = format!("SarAdc_{id}_N{N}");
    let mut nl = Netlist::new(top_name).with_default_signal_layer(pdk.metal1());
    let i_sh  = nl.add_instance(format!("{id}_sh"),  sh_id);
    let i_sar = nl.add_instance(format!("{id}_sar"), sar_id);
    let i_dac = nl.add_instance(format!("{id}_dac"), dac_id);
    let i_cmp = nl.add_instance(format!("{id}_cmp"), cmp_id);

    // Internal nets — exactly the ones `SarAdc::emit_spice` declares.
    let vhold = format!("{id}_vhold");
    let vdac  = format!("{id}_vdac");
    let cmp   = format!("{id}_cmp");
    nl.connect(&vhold, i_sh,  "vhold");
    nl.connect(&vhold, i_cmp, "vp");
    nl.connect(&vdac,  i_dac, "vout");
    nl.connect(&vdac,  i_cmp, "vn");
    nl.connect(&cmp,   i_cmp, "vout");
    nl.connect(&cmp,   i_sar, "cmp");

    // Per-bit feedback: SAR `bit_i` → DAC `bit_i`.
    for i in 0..N {
        let bn = format!("bit_{i}");
        nl.connect(format!("dcode_{i}"), i_sar, &bn);
        nl.connect(format!("dcode_{i}"), i_dac, &bn);
    }

    // External terminals — same set + order as `SarAdc::emit_spice`'s
    // `nets` slice (3N + 5 nets total).
    nl.connect("vin",     i_sh, "vin");
    nl.connect("clk_sh",  i_sh, "clk_sh");
    nl.connect("vdd",     i_sh, "vdd");
    nl.connect("gnd",     i_sh, "gnd");
    nl.connect("vdd",     i_sar, "vdd");
    nl.connect("gnd",     i_sar, "gnd");
    nl.connect("gnd",     i_dac, "vlow");
    nl.connect("reset_b", i_sar, "reset_b");
    for i in 0..N {
        nl.connect(format!("phase_{i}"),   i_sar, &format!("phase_{i}"));
        nl.connect(format!("capture_{i}"), i_sar, &format!("capture_{i}"));
    }

    nl.expose("vin",     "vin",     Some(PinDirection::Input));
    nl.expose("clk_sh",  "clk_sh",  Some(PinDirection::Input));
    nl.expose("reset_b", "reset_b", Some(PinDirection::Input));
    nl.expose("vdd",     "vdd",     Some(PinDirection::Power));
    nl.expose("gnd",     "gnd",     Some(PinDirection::Ground));
    for i in 0..N {
        nl.expose(format!("phase_{i}"),   format!("phase_{i}"),   Some(PinDirection::Input));
        nl.expose(format!("capture_{i}"), format!("capture_{i}"), Some(PinDirection::Input));
        nl.expose(format!("bit_{i}"),     format!("dcode_{i}"),   Some(PinDirection::Output));
    }

    // Floorplan: top row holds the analog strip (S/H over DAC | comp);
    // the SAR register sits below it — the SAR register is a row of
    // 4 DffSRs and would otherwise stretch the top-level aspect to
    // ~5.5:1. Stacking it below brings the aspect closer to 1.5:1
    // so the figure renders cleanly inside markdown reports.
    let chan = 30_000_i64;
    // Use the actual cell heights via lib lookup so placement reflects
    // any size changes the helpers compute from struct fields.
    // Use full_bbox so geometry inside child instances counts; the
    // composite cells (Block_SH/DAC/SAR/CMP) hold zero direct shapes,
    // only instances of real primitives, so local_bbox would return
    // the empty bbox (i64::MIN/MAX) and overflow on arithmetic.
    let sh_bb  = lib.get(sh_id).full_bbox(lib);
    let dac_bb = lib.get(dac_id).full_bbox(lib);
    let cmp_bb = lib.get(cmp_id).full_bbox(lib);
    let sar_bb = lib.get(sar_id).full_bbox(lib);
    let sh_h  = (sh_bb.max.y  - sh_bb.min.y).max(10_000);
    let dac_h = (dac_bb.max.y - dac_bb.min.y).max(10_000);
    let sh_w  = (sh_bb.max.x  - sh_bb.min.x).max(10_000);
    let dac_w = (dac_bb.max.x - dac_bb.min.x).max(10_000);
    let cmp_w = (cmp_bb.max.x - cmp_bb.min.x).max(10_000);
    let sar_h = (sar_bb.max.y - sar_bb.min.y).max(10_000);
    let analog_w = sh_w.max(dac_w);
    let analog_strip_h = sh_h + dac_h + chan;
    // Top strip: DAC at bottom-left, S/H above DAC, comparator to the right.
    let p_dac = Trans::translate(Vec2::new(0, sar_h + chan));
    let p_sh  = Trans::translate(Vec2::new(0, sar_h + chan + dac_h + chan));
    let p_cmp = Trans::translate(Vec2::new(analog_w + chan,
                                            sar_h + chan + (sh_h + dac_h) / 2));
    // SAR register row sits below the analog strip, spanning the width.
    let _ = analog_strip_h;
    let p_sar = Trans::translate(Vec2::new(0, 0));
    let placer = eda_pnr::ManualPlacer::new(vec![p_sh, p_sar, p_dac, p_cmp]);

    PnrFlow::new(placer, ManhattanRouter::default()).run(&nl, lib).top
}

/// Top-level entry — instantiates the default 4-bit `SarAdc` from
/// `spike-sar-adc` and runs it through the floorplan flow.
fn build_sar_adc(lib: &Library, pdk: &Sky130Lite) -> CellId {
    let adc = spike_sar_adc::SarAdc::<4>::default();
    build_sar_adc_floorplan(lib, pdk, &adc, "u_sar_adc")
}

// ── Logo / project banner ──────────────────────────────────────────

/// Pure-art floor plan: MIT logo (stacked stripes), the project name
/// "rlx-eda", and a stylized beaver mascot — all built from boxes
/// + text shapes on the existing PDK layers, so the cell flows
/// through the same layout/render/GDS-export path as a real circuit.
fn build_logo(lib: &Library, pdk: &Sky130Lite) -> CellId {
    let mut cb = CellBuilder::new("RlxEda_Logo".to_string());
    let met = pdk.metal1();
    let res = pdk.res();
    let via = pdk.via1();

    let rect = |cb: &mut CellBuilder, layer, x0, y0, x1, y1| {
        cb.add_shape(
            layer,
            Rect::new(Bbox::new(Point::new(x0, y0), Point::new(x1, y1))),
        );
    };
    let text = |cb: &mut CellBuilder, layer, s: &str, anchor, size, halign| {
        cb.add_shape(
            layer,
            klayout_core::Shape::Text(Text {
                string: s.into(),
                anchor,
                size,
                halign,
                valign: klayout_core::VAlign::Middle,
            }),
        );
    };

    // ── MIT logo, schematic-style (stacked horizontal bars + tall I) ──
    // Per MIT's "stack of stripes" identity: M / I / T as a pixel grid.
    // Each glyph is a 7-tall × 5-wide grid in 2 µm cells.
    let cell = 2_000_i64;
    let on = |grid: &[(i64, i64)]| grid.to_vec();
    let m_pix = on(&[
        (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
        (1,5),(2,4),(3,5),
        (4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),
    ]);
    let i_pix: Vec<(i64,i64)> = (0..7).map(|y| (0_i64, y)).collect();
    let t_pix = on(&[
        (0,6),(1,6),(2,6),(3,6),(4,6),
        (2,0),(2,1),(2,2),(2,3),(2,4),(2,5),
    ]);
    let mut x_origin = 0_i64;
    let logo_y = 80_000_i64;
    for (glyph, w_cells) in [(&m_pix, 5), (&i_pix, 1), (&t_pix, 5)] {
        for &(gx, gy) in glyph {
            let x = x_origin + gx * cell;
            let y = logo_y + gy * cell;
            rect(&mut cb, met, x, y, x + cell, y + cell);
        }
        x_origin += (w_cells + 2) * cell;
    }

    // ── Project name "rlx-eda" — text on metal1 ────────────────────
    text(
        &mut cb,
        met,
        "rlx-eda",
        Point::new(35_000, 70_000),
        8_000,
        klayout_core::HAlign::Center,
    );
    text(
        &mut cb,
        met,
        "Massachusetts Institute of Technology",
        Point::new(35_000, 60_000),
        2_000,
        klayout_core::HAlign::Center,
    );

    // ── Beaver mascot — built from boxes ───────────────────────────
    // Origin at center-bottom of the logo, beaver faces right.
    // All coordinates in DBU (1 µm = 1000 DBU).
    let bx = 0_i64;       // left edge of beaver bbox
    let by = -10_000_i64; // bottom edge

    // Body: rounded-rectangle approximation = central rect + corner trims.
    rect(&mut cb, res, bx + 5_000,  by + 5_000,  bx + 45_000, by + 25_000); // body trunk
    rect(&mut cb, res, bx + 8_000,  by + 3_000,  bx + 42_000, by + 5_000);  // belly bottom
    rect(&mut cb, res, bx + 8_000,  by + 25_000, bx + 42_000, by + 27_000); // back top

    // Head — rightward, slightly higher than body.
    rect(&mut cb, res, bx + 38_000, by + 18_000, bx + 60_000, by + 33_000); // head box
    rect(&mut cb, res, bx + 40_000, by + 33_000, bx + 58_000, by + 35_000); // forehead curve

    // Ears (two small bumps on top of head).
    rect(&mut cb, res, bx + 42_000, by + 35_000, bx + 46_000, by + 38_000);
    rect(&mut cb, res, bx + 52_000, by + 35_000, bx + 56_000, by + 38_000);

    // Eye (single dot on metal1 for contrast).
    rect(&mut cb, met, bx + 51_000, by + 28_000, bx + 53_000, by + 30_000);

    // Nose / muzzle (small box past the head).
    rect(&mut cb, res, bx + 60_000, by + 22_000, bx + 64_000, by + 27_000);

    // Two iconic beaver front teeth — vertical bars on metal1.
    rect(&mut cb, met, bx + 60_000, by + 19_000, bx + 61_500, by + 22_000);
    rect(&mut cb, met, bx + 62_000, by + 19_000, bx + 63_500, by + 22_000);

    // Tail: flat paddle off the left side of the body.
    rect(&mut cb, res, bx - 12_000, by + 8_000, bx + 5_000, by + 22_000);
    // Tail crosshatch — small vias for texture.
    for i in 0..4 {
        for j in 0..3 {
            let cx = bx - 10_000 + i * 4_000;
            let cy = by + 10_000 + j * 4_000;
            rect(&mut cb, via, cx, cy, cx + 1_500, cy + 1_500);
        }
    }

    // Four little legs poking out the bottom of the body.
    for &lx in &[bx + 10_000, bx + 18_000, bx + 30_000, bx + 38_000] {
        rect(&mut cb, res, lx, by, lx + 4_000, by + 5_000);
    }

    // Caption under the beaver.
    text(
        &mut cb,
        met,
        "MIT Beaver — Castor canadensis",
        Point::new(25_000, by - 4_000),
        2_000,
        klayout_core::HAlign::Center,
    );

    lib.insert(cb)
}

// ── SVG import (generic vector → polygons-on-PDK-layer) ───────────

fn build_svg_imported(path: &'static str) -> impl FnOnce(&Library, &Sky130Lite) -> CellId {
    move |lib, pdk| {
        // No frame rect — earlier override stamped a full-bbox box on
        // the same layer as the imported polygons, which painted the
        // whole cell solid and obliterated the path geometry. Letting
        // the polygons stand on their own keeps the negative space
        // transparent (i.e. white in the renderer's background).
        let mut opts = svg_import::SvgImportOptions::default();
        opts.port_kind = Some(pdk.electrical_kind());
        let cell_name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("svg_import")
            .to_string();
        svg_import::import_svg(
            lib,
            std::path::Path::new(path),
            cell_name,
            pdk.metal1(),
            &opts,
        )
        .unwrap_or_else(|e| panic!("svg import {path}: {e}"))
    }
}

// ── RF (spike-lna, RfDemo PDK) ─────────────────────────────────────

fn build_spiral_inductor(lib: &Library, pdk: &RfDemo) -> CellId {
    // 5-turn 60 µm spiral on the dedicated METAL_TOP layer — the
    // canonical inductor primitive used by the LNA composite.
    SpiralInductor {
        outer_dbu: 60_000,
        n_turns: 5,
        width_dbu: 4_000,
        spacing_dbu: 2_000,
        id: "L0".into(),
    }
    .layout(lib, pdk)
}

fn build_lna(lib: &Library, pdk: &RfDemo) -> CellId {
    // 2.4 GHz inductively-degenerated cascode — composite block
    // built from two RF MOSFETs, three spiral inductors, five pads,
    // and the matching network. `Lna::layout` runs through eda_pnr
    // (Connectivity → ManualPlacer → ManhattanRouter) just like
    // RcDivider, so the rendered floorplan shows real routed wires
    // between cascode, source-degen / gate / drain inductors, and
    // the I/O pads.
    Lna::lna_24ghz("u_lna").layout(lib, pdk)
}

// ── Photonic (spike-waveguide-block, GdsfactoryGeneric PDK) ────────

fn build_waveguide(lib: &Library, pdk: &GdsfactoryGeneric) -> CellId {
    // 500 nm-wide × 100 µm-long SOI strip — the canonical photonic
    // primitive (single Box on `pdk.wg()`, in/out optical ports).
    Waveguide { width: 500, length: 100_000, id: "WG0".into() }.layout(lib, pdk)
}

fn build_mzi(lib: &Library, pdk: &GdsfactoryGeneric) -> CellId {
    // Two-arm MZI: 500 nm width, 100 µm vs 110 µm arms (10 µm
    // mismatch puts the operating point near a useful Δφ).
    Mzi::new(500, 100_000, 110_000, "u_mzi").layout(lib, pdk)
}

fn write_index(generated: &[Generated]) -> std::io::Result<()> {
    let path = out_root().join("index.md");
    let mut s = String::new();
    s.push_str("# Floor plans (sky130 / Sky130Lite)\n\n");
    s.push_str(
        "Generated by `cargo run -p eda-floorplan --bin floorplan-all`. \
         All components share one PDK (sky130 layer numbers via \
         `spike_divider_block::pdks::Sky130Lite`) so layer colors and \
         GDS pairs are comparable across designs.\n\n",
    );
    s.push_str("| Component | Description | Floor plan | GDS | Summary |\n");
    s.push_str("|---|---|---|---|---|\n");
    for g in generated {
        s.push_str(&format!(
            "| `{}` | {} | [svg](./{name}/floorplan.svg) | [gds](./{name}/floorplan.gds) | [txt](./{name}/summary.txt) |\n",
            g.name,
            g.description,
            name = g.name,
        ));
    }
    s.push('\n');
    for g in generated {
        s.push_str(&format!("## `{}`\n\n```\n{}```\n\n", g.name, g.summary));
    }
    fs::write(&path, s)?;
    println!("\nindex written: {}", path.display());
    Ok(())
}

fn main() {
    let root = out_root();
    fs::create_dir_all(&root).expect("create floorplans dir");
    println!("Writing floor plans under {}\n", root.display());

    let stdcell_names: &[&'static str] = &[
        "sky130_fd_sc_hd__inv_1",
        "sky130_fd_sc_hd__buf_1",
        "sky130_fd_sc_hd__nand2_1",
        "sky130_fd_sc_hd__nor2_1",
        "sky130_fd_sc_hd__and2_1",
        "sky130_fd_sc_hd__fa_1",
        "sky130_fd_sc_hd__dfxtp_1",
        "sky130_fd_sc_hd__mux2_1",
    ];

    let mut runs: Vec<Generated> = vec![
        // ── CMOS primitives — Sky130Lite ──────────────────────────
        run_one("resistor",
            "poly resistor primitive [sky130]",
            || with_sky130lite(build_resistor)),
        run_one("diode",
            "Shockley diode primitive (RES square + 2 metal1 pads) [sky130]",
            || with_sky130lite(build_diode)),
        run_one("capacitor",
            "MIM-style capacitor primitive (single metal1 plate) [sky130]",
            || with_sky130lite(build_capacitor)),
        run_one("voltage_source",
            "Ideal voltage source primitive (1.8 V VDD) [sky130]",
            || with_sky130lite(build_voltage_source)),
        run_one("mosfet_nmos",
            "NMOS device primitive W=2 µm / L=0.5 µm [sky130]",
            || with_sky130lite(build_mosfet)),
        run_one("mosfet_pmos",
            "PMOS device primitive W=4 µm / L=0.5 µm with n-well [sky130]",
            || with_sky130lite(build_mosfet_pmos)),
        // ── CMOS composites — Sky130Lite + PNR ────────────────────
        run_one("rc_divider",
            "2-resistor divider — PNR via Layout::layout [sky130]",
            || with_sky130lite(build_rc_divider)),
        run_one("rc_divider_pnr",
            "same divider driven explicitly through PnrFlow::run [sky130]",
            || with_sky130lite(build_rc_divider_explicit_pnr)),
        run_one("tinyconv_tile_digital",
            "Mac8x8Tile digital topology — 4-row sc_hd floorplan [sky130]",
            || with_sky130lite(build_mac_tile)),
        run_one("tinyconv_array_2x2",
            "ArrayBlock 2×2 of digital MAC tiles [sky130]",
            || with_sky130lite(build_array)),
        run_one("sar_adc",
            "SAR ADC floorplan — sub-block sizes derived from spike_sar_adc::SarAdc<4>::default() fields, PNR-routed [sky130]",
            || with_sky130lite(build_sar_adc)),
        // ── RF — RfDemo (top-metal layer for spiral inductors) ────
        run_one("spiral_inductor",
            "RF spiral inductor primitive on METAL_TOP [RfDemo]",
            || with_rfdemo(build_spiral_inductor)),
        run_one("lna_24ghz",
            "Inductively-degenerated cascode LNA at 2.4 GHz [RfDemo]",
            || with_rfdemo(build_lna)),
        // ── Photonic — gdsfactory generic SOI ─────────────────────
        run_one("waveguide",
            "500 nm × 100 µm SOI strip waveguide [gdsfactory-generic]",
            || with_photonic(build_waveguide)),
        run_one("mzi",
            "Two-arm Mach-Zehnder interferometer [gdsfactory-generic]",
            || with_photonic(build_mzi)),
    ];

    // ── Logo / project banner (pure art) ──────────────────────────
    runs.push(run_one("rlx_eda_logo",
        "MIT logo + rlx-eda name + beaver mascot — pure-art floor plan [sky130]",
        || with_sky130lite(build_logo)));

    // ── SVG-imported decals ───────────────────────────────────────
    let svg_imports: &[(&'static str, &'static str, &'static str)] = &[
        (
            "svg_imported_medialab",
            "logos/MIT_Media_Lab_logo.svg",
            "SVG import demo — vector paths flattened to polygons on metal1 [sky130]",
        ),
    ];
    for &(out_name, src_rel, descr) in svg_imports {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let abs = PathBuf::from(manifest_dir)
            .ancestors()
            .nth(2)
            .expect("workspace root")
            .join(src_rel);
        let abs_str: &'static str = Box::leak(abs.to_string_lossy().into_owned().into_boxed_str());
        if !std::path::Path::new(abs_str).exists() {
            eprintln!("[skip {out_name}] source SVG not found at {abs_str}");
            continue;
        }
        runs.push(run_one(out_name, descr, || with_sky130lite(build_svg_imported(abs_str))));
    }

    // ── Standard cells — eda-stdcells mock library ───────────────
    for &cell in stdcell_names {
        // Strip the "sky130_fd_sc_hd__" prefix for the on-disk dir so
        // file paths stay sensible. Using `Box::leak` is the lazy way
        // to satisfy `&'static str`; the binary lives for the process
        // lifetime so the leak is bounded by the run.
        let short: &'static str = Box::leak(format!(
            "stdcell_{}",
            cell.trim_start_matches("sky130_fd_sc_hd__"),
        ).into_boxed_str());
        let descr: &'static str = Box::leak(format!(
            "{cell} (mock sc_hd) — single foundry cell laid out standalone"
        ).into_boxed_str());
        runs.push(run_one(short, descr, || with_sky130lite(build_stdcell(cell))));
    }

    if let Err(e) = write_index(&runs) {
        eprintln!("warn: failed to write index: {e}");
    }

    // Schematic: render the SAR ADC's symbolic block-level diagram
    // alongside its floor plan so reviewers can cross-reference
    // sub-block placement with the circuit topology.
    let sar = spike_sar_adc::SarAdc::<4>::default();
    let schem = build_sar_schematic(&sar);
    let style = SchemStyle::default();
    let schem_svg = eda_viz::schematic::render_to_svg(&schem, &style);
    let schem_path = root.join("sar_adc").join("schematic.svg");
    if let Err(e) = fs::write(&schem_path, &schem_svg) {
        eprintln!("warn: failed to write SAR schematic: {e}");
    } else {
        println!("SAR schematic written: {}", schem_path.display());
    }

    let abs_root = root.canonicalize().unwrap_or(root);
    println!(
        "\n{} floor plans generated under {}",
        runs.len(),
        abs_root.display()
    );
}

#[cfg(test)]
mod smoke {
    //! Minimal smoke tests so the binary keeps building when the
    //! upstream component constructors drift. Each test just builds
    //! the layout and confirms the top cell has a non-empty bbox.

    use super::*;

    fn check(name: &str, build: impl FnOnce(&Library, &Sky130Lite) -> CellId) {
        let lib = Sky130Lite::new_library(name);
        let pdk = Sky130Lite::register(&lib);
        let top = build(&lib, &pdk);
        let bbox = lib.get(top).full_bbox(&lib);
        assert!(!bbox.is_empty(), "{name}: empty bbox");
    }

    fn check_rf(name: &str, build: impl FnOnce(&Library, &RfDemo) -> CellId) {
        let lib = RfDemo::new_library(name);
        let pdk = RfDemo::register(&lib);
        let top = build(&lib, &pdk);
        assert!(!lib.get(top).full_bbox(&lib).is_empty(), "{name}: empty bbox");
    }

    fn check_photonic(name: &str, build: impl FnOnce(&Library, &GdsfactoryGeneric) -> CellId) {
        let lib = GdsfactoryGeneric::new_library(name);
        let pdk = GdsfactoryGeneric::register(&lib);
        let top = build(&lib, &pdk);
        assert!(!lib.get(top).full_bbox(&lib).is_empty(), "{name}: empty bbox");
    }

    #[test] fn resistor() { check("r", build_resistor); }
    #[test] fn mosfet()   { check("m", build_mosfet); }
    #[test] fn divider()  { check("d", build_rc_divider); }
    #[test] fn tile()     { check("t", build_mac_tile); }
    #[test] fn array()    { check("a", build_array); }
    #[test] fn spiral()   { check_rf("s", build_spiral_inductor); }
    #[test] fn lna()      { check_rf("l", build_lna); }
    #[test] fn waveguide(){ check_photonic("w", build_waveguide); }
    #[test] fn mzi()      { check_photonic("m", build_mzi); }
}
