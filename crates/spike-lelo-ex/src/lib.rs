//! sky130A 1:4 NMOS current mirror — verbatim repro of the AIC2026
//! sky130nm tutorial's LELO_EX testbench, driven through `eda-sim-harness`.
//!
//! Topology:
//!
//! ```text
//!     iref (5 µA)
//!        │
//!     ───┴───
//!     │     │
//!  ┌──┘     └──┐
//!  │           │
//!  D=G          D ── (ammeter Vmeas) ── vbias
//!  │           │
//!  M1          M2  (W = 4 × W_M1)
//!  │           │
//!  S,B = 0     S,B = 0
//! ```
//!
//! M1 is diode-connected (gate tied to drain), set by a 5 µA current
//! source `Iref`. M1's gate also drives M2's gate; M2's drain is held
//! at `vbias` (vdd/2) through a zero-volt source `Vmeas` so we can
//! pick the drain current off as `i(vmeas)` — or, equivalently, via
//! `meas tran ibn find i(vmeas)`.
//!
//! Measurements ([`measurements`]):
//!  - `ibn` — M2 drain current at `t_stop`. Should be ≈ 4 × 5 µA = 20 µA.
//!  - `vgs_m1` — M1 gate-source at `t_stop`. Sky130 nfet_01v8 typical
//!    Vth0 ≈ 450 mV; with 5 µA in `W=2u L=0.5u`, expect ~600–700 mV.

use std::path::PathBuf;

use eda_extract::{
    extract, lvs_compare, sky130_recognizer, DeviceKind, SchematicDevice,
};
use eda_pex::{
    extract as pex_extract, extract_resistance, emit_spice_caps, emit_spice_resistors,
    NetOnLayer, PexLayer, Stack as PexStack,
};
use eda_sim_harness::{
    Analysis, Corner, CornerKind, MeasureLog, Measurement, Testbench, VerifierResult,
    VerifyReport, View,
};
use eda_spice_emit::Netlist;
use klayout_connect::{extract_hierarchical, Conductor, ExtractConfig};
use klayout_core::{
    Angle90, Bbox, CellBuilder, CellId, Instance, LayerIndex, LayerInfo, Library, Path, Point,
    Port, Rect, Shape, Trans, Vec2,
};
use klayout_route::{ManhattanPlanner, Obstacles, Planner};

/// How `LeloEx` should produce the synthetic verification layout.
///
/// - `Static` — hand-coded Manhattan rectangles. Deterministic by
///   construction; the routes are the geometry the spike's author
///   wrote.
/// - `Pnr` — the nets are *declared*, then routed by
///   `klayout-route::ManhattanPlanner` at build time. Same instance
///   placement, same final connectivity, but the wire geometry
///   comes from the planner so swapping the planner (A* obstacle
///   avoidance, pathfinder, …) changes the layout without touching
///   spike code.
///
/// Both configs flow through the same `verify` / `pex_spice_lines`
/// paths — they're alternate ways to populate one
/// [`SyntheticLayout`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LayoutConfig {
    #[default]
    Static,
    Pnr,
}

#[doc(hidden)]
pub mod __test_helpers {
    //! Hidden API exposing internals so integration tests can poke
    //! at the synthetic layout and PEX path without spinning up the
    //! whole sky130A ngspice harness. Stable for use within
    //! `spike-lelo-ex`'s own tests; not a public API.

    pub fn synthetic_layout(tb: &super::LeloEx) -> super::SyntheticLayout {
        tb.build_synthetic_layout()
    }

    pub fn pex_lines(tb: &super::LeloEx) -> Vec<String> {
        tb.pex_spice_lines()
    }
}

/// Sky130 1:4 NMOS current mirror testbench.
#[derive(Debug, Clone)]
pub struct LeloEx {
    /// Top-level SPICE library. Resolved via
    /// `rlx_eda_cli::resolve_pdk("sky130A").lib_path` in the integration
    /// test, but the field is configurable so unit tests can point at a
    /// stub.
    pub lib_path: PathBuf,
    /// Reference current injected into M1, amperes (default 5 µA).
    pub iref_a: f64,
    /// Drain bias on M2 (V). Default vdd/2 = 0.9 V; override per corner
    /// via [`LeloEx::with_drain_bias`].
    pub drain_bias_v: f64,
    /// M1 width, length (microns, sky130 `.option scale=1u` already
    /// baked into the lib include).
    pub w_um: f64,
    pub l_um: f64,

    /// Stand-in routing parasitics injected when `corner.view ==
    /// View::Layout`. We don't have a real GDS-extracted `.lpe.spi`
    /// for LELO_EX, so we simulate "layout-extracted" with a small
    /// series resistance in M2's drain path (`layout_drain_r`) and a
    /// parasitic drain capacitance (`layout_drain_c`). The resistor
    /// shifts the mirror current at DC by IR-drop in the drain bias;
    /// the cap shifts settling time during the transient ramp.
    /// Calibrated to produce ~1–2 % steady-state Δ between Sch and Lay
    /// — realistic for a 2 µm-routed analog cell. Set either to 0 to
    /// disable that contribution.
    pub layout_drain_r: f64,
    pub layout_drain_c: f64,
    /// How to produce the verification layout. See [`LayoutConfig`].
    pub layout_config: LayoutConfig,
}

impl LeloEx {
    pub fn new(lib_path: impl Into<PathBuf>) -> Self {
        Self {
            lib_path: lib_path.into(),
            iref_a: 5e-6,
            drain_bias_v: 0.9,
            // L = 2 µm — long-channel regime where λ stays small enough
            // that the mirror ratio tracks W-ratio cleanly. Short L
            // (≤ 0.5 µm) bumps the ratio by ~30 % because M2 sees a
            // higher Vds than diode-connected M1.
            w_um: 2.0,
            l_um: 2.0,
            // 4 kΩ in series at M2's drain ⇒ ~ 4 kΩ × 25 µA = 100 mV
            // IR drop. With M2's Vds dropping by 100 mV (out of 0.9 V
            // at the bias node), λ-modulated current shifts by ~1.5 %.
            layout_drain_r: 4_000.0,
            layout_drain_c: 100e-15,
            layout_config: LayoutConfig::Static,
        }
    }

    /// Builder: switch the testbench to the PnR-routed layout
    /// config. Same instance placement, routes computed by
    /// `klayout-route::ManhattanPlanner`. Only affects the Layout
    /// view; Schematic corners are unchanged.
    pub fn with_layout_config(mut self, c: LayoutConfig) -> Self {
        self.layout_config = c;
        self
    }
}

/// `(name, point)` pairs for the three top-level Ports the LELO_EX
/// verification layout exposes — gate, mout, gnd. Same in both
/// `Static` and `Pnr` configs so net-name resolution is identical.
fn lelo_top_port_anchors() -> [(&'static str, Point); 3] {
    [
        ("gate", Point::new(4_000, 2_000)),
        ("mout", Point::new(12_500, 4_000)),
        ("gnd",  Point::new(7_500, 250)),
    ]
}

/// Build a synthetic external Port the planner can route towards.
/// Used for routes that need to terminate at a "this is mout"-shaped
/// anchor that isn't itself a cell-instance pin.
fn synthetic_external_port(layer: LayerIndex, at: Point, angle: Angle90) -> Port {
    Port::new("ext", layer, at, angle, 500)
}

/// Manhattan path → axis-aligned rectangles. `klayout-connect`'s
/// hierarchical extractor recognises `Box` and `Polygon` shapes
/// but skips `Path`, so a planner-emitted Path gets converted to
/// per-segment rectangles before insertion. Each segment is the
/// path's two endpoints expanded by `width/2` perpendicular to
/// the segment direction. Caps are Flat (planner default).
fn path_to_rects(path: &Path) -> Vec<Rect> {
    let mut out: Vec<Rect> = Vec::new();
    let half = path.width / 2;
    if path.points.len() < 2 || half <= 0 {
        return out;
    }
    for i in 0..path.points.len() - 1 {
        let a = path.points[i];
        let b = path.points[i + 1];
        let (x0, x1) = (a.x.min(b.x), a.x.max(b.x));
        let (y0, y1) = (a.y.min(b.y), a.y.max(b.y));
        let bb = if a.x == b.x {
            // Vertical segment.
            Bbox::new(
                Point::new(a.x - half, y0),
                Point::new(a.x + half, y1),
            )
        } else if a.y == b.y {
            // Horizontal segment.
            Bbox::new(
                Point::new(x0, a.y - half),
                Point::new(x1, a.y + half),
            )
        } else {
            // Diagonal — punt; planner shouldn't emit these.
            continue;
        };
        out.push(Rect::new(bb));
    }
    out
}

/// Build the `nfet_01v8` placeholder cell used by both layout
/// configs. 4 met1 port pads (d/g/s/b) at carefully-chosen
/// positions so single-layer routing can connect them without a
/// physical short. A poly stripe through the body gives DRC poly
/// to chew on.
fn nfet_placeholder_cell(lib: &Library, met1: LayerIndex, poly: LayerIndex) -> CellId {
    let mut cb = CellBuilder::new("nfet_01v8");
    let pads: [(&str, Point); 4] = [
        ("d", Point::new(2_000, 4_000)),
        ("g", Point::new(    0, 2_000)),
        ("s", Point::new(2_000,    0)),
        ("b", Point::new(4_000,   500)),
    ];
    for (name, c) in &pads {
        cb.add_shape(met1, Rect::new(Bbox::new(
            Point::new(c.x - 250, c.y - 250),
            Point::new(c.x + 250, c.y + 250),
        )));
        cb.add_port(Port::new(*name, met1, *c, Angle90::E, 500));
    }
    cb.add_shape(poly, Rect::new(Bbox::new(
        Point::new(1_900, 0),
        Point::new(2_100, 4_000),
    )));
    lib.insert(cb)
}

/// Layer + cell handles produced by [`LeloEx::build_synthetic_layout`].
/// Carried alongside the `Library` so verify / PEX paths can iterate
/// nets per layer without re-deriving handles.
#[doc(hidden)]
pub struct SyntheticLayout {
    pub lib: Library,
    pub top: CellId,
    pub met1: LayerIndex,
    pub met1_label: LayerIndex,
    pub poly: LayerIndex,
}

impl LeloEx {
    /// Build the LELO_EX synthetic verification layout (two
    /// `nfet_01v8` placeholder instances + met1 routing for gate /
    /// mout / gnd). Dispatches on [`Self::layout_config`].
    fn build_synthetic_layout(&self) -> SyntheticLayout {
        match self.layout_config {
            LayoutConfig::Static => self.build_synthetic_layout_static(),
            LayoutConfig::Pnr    => self.build_synthetic_layout_pnr(),
        }
    }

    /// Hand-coded Manhattan rectangles. The original verification
    /// layout: routes are explicit geometry chosen so the gate L
    /// doesn't accidentally short to the body riser, etc.
    fn build_synthetic_layout_static(&self) -> SyntheticLayout {
        let lib = Library::new("lelo_ex_layout", 1000);
        let met1 = lib.layer(LayerInfo::named("met1", 68, 20));
        let met1_label = lib.layer(LayerInfo::named("met1.label", 68, 5));
        let poly = lib.layer(LayerInfo::named("poly", 66, 20));

        let nfet = nfet_placeholder_cell(&lib, met1, poly);

        let mut tb_cell = CellBuilder::new("lelo_top");
        tb_cell.add_instance(Instance::new(nfet, Trans::IDENTITY));
        tb_cell.add_instance(Instance::new(nfet, Trans::translate(Vec2::new(8_000, 0))));
        // Gate L-shape.
        tb_cell.add_shape(met1, Rect::new(Bbox::new(
            Point::new(1_750, 1_750), Point::new(2_250, 4_250),
        )));
        tb_cell.add_shape(met1, Rect::new(Bbox::new(
            Point::new(  -250, 1_750), Point::new(8_250, 2_250),
        )));
        // mout horizontal stripe.
        tb_cell.add_shape(met1, Rect::new(Bbox::new(
            Point::new( 9_750, 3_750), Point::new(15_000, 4_250),
        )));
        // gnd bottom bus (covers s and b ports).
        tb_cell.add_shape(met1, Rect::new(Bbox::new(
            Point::new(  -250,  -250), Point::new(15_000,    750),
        )));
        for (name, pt) in lelo_top_port_anchors() {
            tb_cell.add_port(Port::new(name, met1_label, pt, Angle90::E, 500));
        }
        let top = lib.insert(tb_cell);
        SyntheticLayout { lib, top, met1, met1_label, poly }
    }

    /// PnR-routed: same instances, routes computed by
    /// `klayout-route::ManhattanPlanner`. Each net is declared as a
    /// pin list; the planner picks the wire geometry. The result
    /// is functionally equivalent to the static config — same nets,
    /// same DRC/LVS/EM/PEX outputs — but the wire geometry differs.
    ///
    /// Routes emit as `Shape::Path` rather than `Shape::Box`, so
    /// `klayout-connect` and `klayout-drc` see the same metal but
    /// at the planner's chosen shape.
    fn build_synthetic_layout_pnr(&self) -> SyntheticLayout {
        let lib = Library::new("lelo_ex_layout_pnr", 1000);
        let met1 = lib.layer(LayerInfo::named("met1", 68, 20));
        let met1_label = lib.layer(LayerInfo::named("met1.label", 68, 5));
        let poly = lib.layer(LayerInfo::named("poly", 66, 20));

        let nfet = nfet_placeholder_cell(&lib, met1, poly);

        let mut tb_cell = CellBuilder::new("lelo_top");
        let m1_t = Trans::IDENTITY;
        let m2_t = Trans::translate(Vec2::new(8_000, 0));
        tb_cell.add_instance(Instance::new(nfet, m1_t));
        tb_cell.add_instance(Instance::new(nfet, m2_t));

        // Resolve each port by composing the cell port with the
        // instance's transform. nfet ports live in cell-local coords;
        // we need top-frame ports for the planner.
        let port_at = |trans: Trans, name: &str| -> Port {
            let cell = lib.get(nfet);
            cell.port(name).expect("port present").transform(trans)
        };
        let m1_d = port_at(m1_t, "d");
        let m1_g = port_at(m1_t, "g");
        let m1_s = port_at(m1_t, "s");
        let m1_b = port_at(m1_t, "b");
        let m2_d = port_at(m2_t, "d");
        let m2_g = port_at(m2_t, "g");
        let m2_s = port_at(m2_t, "s");
        let m2_b = port_at(m2_t, "b");

        // Net declarations: list of pin sequences. Two-pin pairs
        // are passed straight to the planner; multi-pin nets are
        // chained pairwise (M1.d → M1.g → M2.g for the gate).
        let planner = ManhattanPlanner;
        let env = Obstacles::default();
        let mut planned: Vec<Path> = Vec::new();
        // gate
        planned.push(planner.plan(&m1_d, &m1_g, &env));
        planned.push(planner.plan(&m1_g, &m2_g, &env));
        // mout: M2.d → an external "mout" anchor far east.
        let mout_target = synthetic_external_port(met1_label, lelo_top_port_anchors()[1].1, Angle90::W);
        planned.push(planner.plan(&m2_d, &mout_target, &env));
        // gnd: M1.s → M2.s, then M1.b → M2.b, plus a stitch from
        // M1.s → M1.b so source and body merge into one net.
        planned.push(planner.plan(&m1_s, &m2_s, &env));
        planned.push(planner.plan(&m1_b, &m2_b, &env));
        planned.push(planner.plan(&m1_s, &m1_b, &env));

        // Stylize each path as axis-aligned rectangles on met1.
        // klayout-connect's hierarchical extractor processes Box +
        // Polygon shapes but skips Path; converting per-segment
        // means the planner output is fully visible to the
        // connectivity analyzer. Real flows would use
        // `klayout-route::WirePathStylizer` (emits Path) and rely
        // on a klayout-connect path-handler — that's a follow-up.
        for path in planned {
            for rect in path_to_rects(&path) {
                tb_cell.add_shape(met1, rect);
            }
        }

        for (name, pt) in lelo_top_port_anchors() {
            tb_cell.add_port(Port::new(name, met1_label, pt, Angle90::E, 500));
        }
        let top = lib.insert(tb_cell);
        SyntheticLayout { lib, top, met1, met1_label, poly }
    }

    /// Run capacitive PEX over the synthetic layout and return the
    /// SPICE element lines ready to append to a Layout-view deck.
    /// All nets sit on met1 in this synthetic geometry, so coupling
    /// to other metals is zero — the only parasitics are
    /// per-net-to-substrate caps.
    fn pex_spice_lines(&self) -> Vec<String> {
        let layout = self.build_synthetic_layout();
        let cfg = ExtractConfig {
            conductors: vec![Conductor { layer: layout.met1, label_layer: layout.met1_label }],
            vias: vec![],
        };
        let netlist = extract_hierarchical(&layout.lib, layout.top, &cfg);
        let nets_on_met1: Vec<NetOnLayer<'_>> = netlist
            .nets()
            .iter()
            .map(|n| NetOnLayer {
                name: n.name.as_str(),
                layer: PexLayer::Metal1,
                polygons: &n.polygons,
                bbox: n.bbox,
            })
            .collect();
        let stack = PexStack::sky130a_default();
        let mut caps = pex_extract(&nets_on_met1, layout.lib.dbu(), &stack, "0");
        // Drop the substrate cap on `gnd` — it would short ground to
        // itself in the deck.
        caps.retain(|p| {
            let a = if p.net_a == "gnd" { "0" } else { p.net_a.as_str() };
            let b = if p.net_b == "gnd" { "0" } else { p.net_b.as_str() };
            a != b
        });
        let mut out: Vec<String> = Vec::new();
        out.push(format!(
            "* PEX (eda-pex tier-1, sky130a defaults): {} caps", caps.len(),
        ));
        out.extend(emit_spice_caps(&caps));

        // R-side: keep emission as a comment for now since plumbing
        // each net's far-node rewrite into the deck is a topology
        // change we don't want to do per-corner. Surfaces the
        // computed values in the deck for inspection.
        let resistors = extract_resistance(&nets_on_met1, layout.lib.dbu(), &stack);
        out.push(format!(
            "* PEX R (informational; not topology-rewritten): {} per-net Rs",
            resistors.len(),
        ));
        for (_, _, line) in emit_spice_resistors(&resistors) {
            out.push(format!("* {line}"));
        }
        out
    }
}

impl Testbench for LeloEx {
    fn name(&self) -> &str { "lelo_ex" }

    fn build_netlist(&self, corner: &Corner) -> Netlist {
        let mut nl = Netlist::new("sky130A LELO_EX 1:4 NMOS current mirror");

        // sky130's ngspice corner library uses `.option scale=1.0u`, so
        // the `.lib <path> <section>` include sets that for us. All
        // dimensions in this deck are then in microns.
        //
        // sky130 Monte Carlo composition: trying to chain
        // `.lib '<>' tt` + `.lib '<>' mc` re-parses the model files
        // and trips ngspice on the `l=$;w=$;…` placeholder lines in
        // the duplicated `.subckt` headers (sky130's mismatch corner
        // files use `$` as a build-time slot marker — not HSPICE-A
        // syntax). Workaround: include the `tt` corner once (the
        // `mismatch.corner.spice` file ships there and gates its
        // `agauss(...)` terms behind `mc_mm_switch`/`mc_pr_switch`),
        // then flip those switches via `.param`. Combined with the
        // harness's `set rndseed=<n>` per corner, this gives
        // reproducible per-run draws.
        let lib = self.lib_path.display();
        nl.add_preamble(format!(
            ".lib '{lib}' {}",
            if corner.kind == CornerKind::Mc { "tt" } else { corner.lib_section.as_str() },
        ));
        if corner.kind == CornerKind::Mc {
            nl.add_preamble(".param mc_mm_switch=1 mc_pr_switch=1".to_string());
        }

        // Bias rails.
        nl.add_dc_source("vbias", "vbias", "0", self.drain_bias_v);

        // Reference current pump: ngspice's `I N+ N- val` convention is
        // that positive current flows N+ → N- *inside* the source, so
        // externally the source draws current AT N+ and supplies it at
        // N-. To inject `iref` into the `gate` node we put `gate` on
        // the N- side: `Iref 0 gate iref` ⇒ +iref current arrives at
        // `gate`, which M1 (diode-connected) sinks to source = gnd.
        nl.add_element(format!("Iref 0 gate DC {:.10e}", self.iref_a));

        // M1: drain=gate (diode-connected), source=0, body=0.
        nl.add_element(format!(
            "XM1 gate gate 0 0 sky130_fd_pr__nfet_01v8 W={} L={}",
            self.w_um, self.l_um,
        ));
        // M2: gate shared with M1, drain = mout. Layout view inserts
        // a series resistor between M2.drain and the ammeter so M2 sees
        // a lower Vds (= V(vbias) − I·Rdrain), modeling routing IR drop.
        let m2_drain_node = if corner.view == View::Layout && self.layout_drain_r > 0.0 {
            nl.add_element(format!("Rdrain m2_drain mout {:.6e}", self.layout_drain_r));
            "m2_drain"
        } else { "mout" };
        nl.add_element(format!(
            "XM2 {m2_drain_node} gate 0 0 sky130_fd_pr__nfet_01v8 W={} L={}",
            self.w_um * 4.0, self.l_um,
        ));
        // Parasitic drain cap (transient settling effect, no DC shift).
        if corner.view == View::Layout && self.layout_drain_c > 0.0 {
            nl.add_element(format!("Cdrain {m2_drain_node} 0 {:.6e}", self.layout_drain_c));
        }
        // Zero-volt ammeter so we can pick off i(vmeas) cleanly.
        nl.add_element("Vmeas vbias mout DC 0".to_string());

        // Initial conditions: discharge the gate at t=0; the harness's
        // tran `uic` flag will start from here. Without this, ngspice's
        // pre-DC OP would already have biased the mirror, and `tran uic`
        // would skip it but the IC default for an undeclared net is 0.
        nl.add_element(".ic v(gate)=0 v(mout)=0".to_string());

        // Layout-view: append computed PEX. The hand-decorated
        // `Rdrain` / `Cdrain` stubs above remain because the synthetic
        // layout is too small to produce the kΩ-class series R that
        // an actual LELO_EX route would have. The `Cpex` lines below
        // are physically computed from the same synthetic geometry
        // `verify()` runs DRC/LVS/EM on — exercises the
        // `eda-pex → eda-spice-emit` end-to-end flow even when the
        // numbers themselves are small. When the layout is replaced
        // with real foundry-cell geometry, the hand-decorated stubs
        // can drop out and PEX takes over.
        if corner.view == View::Layout {
            for line in self.pex_spice_lines() {
                nl.add_element(line);
            }
        }

        nl
    }

    fn measurements(&self) -> Vec<Measurement> {
        vec![
            // Drain current of M2: i(vmeas). The current convention in
            // ngspice for a Vsource "Vmeas p n" is positive = current
            // flowing into p. We placed Vmeas as `Vmeas vbias mout 0`
            // so i(vmeas) is the current flowing out of vbias into M2's
            // drain — exactly the mirrored output current.
            Measurement::tran("ibn", "find i(vmeas) at=20u", Some("A")),
            // Gate-source voltage of M1 (gate-to-0).
            Measurement::tran("vgs_m1", "find v(gate) at=20u", Some("V")),
            // Average current drawn from the bias rail over the
            // transient window. Power is derived from this in
            // `derive()` below as `i_avg × V_bias` — see the helper
            // `Measurement::power_from_const_supply`.
            Measurement::supply_current("i_vbias", "vbias"),
        ]
    }

    fn derive(&self, m: &eda_sim_harness::MeasureLog) -> Vec<(String, f64)> {
        // p_vbias = |i_vbias| × V_bias. Watts. Constant-supply path is
        // exact for this testbench — vbias is a DC source.
        Measurement::power_from_const_supply(m, "i_vbias", "p_vbias", self.drain_bias_v)
            .into_iter()
            .collect()
    }

    fn analysis(&self) -> Analysis {
        // 20 µs is plenty: the gate node settles in a few hundred ns
        // through M1's gm and the parasitic gate cap.
        Analysis::Tran { t_step: 100e-9, t_stop: 20e-6, uic: true }
    }

    fn plot_signals(&self) -> Vec<String> {
        // The mirror current `i(vmeas)` doesn't survive the harness's
        // default filter (it's an `i(...)` of an internal ammeter
        // source — ngspice writes it under the synthesized name
        // `vmeas#branch`). The voltage taps `v(gate)` and `v(mout)`
        // pass automatically.
        vec!["i(vmeas)".into(), "vmeas#branch".into()]
    }

    /// Layout-side verification: DRC + LVS + EM on a synthetic
    /// hierarchical layout representing the LELO_EX topology. Two
    /// instances of an `nfet_01v8` placeholder cell (M1, M2),
    /// routed by met1 stripes carrying the gate, mout, vbias, and
    /// gnd nets. DRC runs on the geometry, LVS extracts the
    /// connectivity and diffs against a schematic-declared netlist,
    /// EM runs against the peak supply current measured by ngspice.
    ///
    /// This is *not* a real GDS extraction of LELO_EX — the
    /// `nfet_01v8` cells here are placeholders with d/g/s/b ports
    /// on small met1 dots, not the full foundry transistor GDS. The
    /// point is end-to-end *flow*: the same `extract → recognize →
    /// lvs_compare` path that a real foundry-cell layout would
    /// drive, exercised on synthetic geometry the test can
    /// construct deterministically.
    ///
    /// What lands in the per-corner report:
    ///
    /// - **DRC**: sky130A min-width / min-space on met1 + poly.
    ///   Geometry constructed clean by design.
    /// - **LVS**: extract the routed connectivity, recognize each
    ///   instance as a MOSFET, compare positionally against the
    ///   schematic-declared `[M1, M2]` topology with the same gate /
    ///   mout / vbias / gnd nets.
    /// - **EM**: peak `|i_vbias|` from the parsed measurement log
    ///   driving the 1 µm-wide met1 routing stripes against
    ///   sky130A's met1 Jmax.
    ///
    /// Schematic corners get `VerifyReport::empty()` — the panel
    /// hides itself.
    fn verify(&self, corner: &Corner, m: &MeasureLog) -> VerifyReport {
        if corner.view != View::Layout {
            return VerifyReport::empty();
        }

        let SyntheticLayout { lib, top, met1, met1_label, poly } =
            self.build_synthetic_layout();

        // ── DRC ───────────────────────────────────────────────────
        let drc_v = eda_drc::check_sky130a(
            &lib, top, poly, met1, None, None, None, None,
        );
        let drc_first = drc_v.first().map_or(String::new(), |v| v.rule.to_string());

        // ── LVS ───────────────────────────────────────────────────
        // Extract connectivity. The recognizer matches both `M1` /
        // `M2` instances since their cell name is `nfet_01v8`.
        let cfg = ExtractConfig {
            conductors: vec![Conductor { layer: met1, label_layer: met1_label }],
            vias: vec![],
        };
        let recog = sky130_recognizer();
        let (lvs_count, lvs_first) = match extract(&lib, top, &cfg, &recog) {
            Ok(design) => {
                // Schematic-declared topology. Net names match the
                // top-level port names so LVS can compare positionally.
                let schem = vec![
                    SchematicDevice {
                        instance_index: 0,
                        kind: DeviceKind::Other("M".into()),
                        value: 0.0,
                        terminals: vec![
                            ("d".into(), "gate".into()),
                            ("g".into(), "gate".into()),
                            ("s".into(), "gnd".into()),
                            ("b".into(), "gnd".into()),
                        ],
                    },
                    SchematicDevice {
                        instance_index: 1,
                        kind: DeviceKind::Other("M".into()),
                        value: 0.0,
                        terminals: vec![
                            ("d".into(), "mout".into()),
                            ("g".into(), "gate".into()),
                            ("s".into(), "gnd".into()),
                            ("b".into(), "gnd".into()),
                        ],
                    },
                ];
                let mm = lvs_compare(&schem, &design.devices, 0.0);
                let first = mm.first().map_or(String::new(), |x| format!("{x:?}"));
                (mm.len(), first)
            }
            // Extraction itself failed (unwired pin / dangling top
            // port). Surface as one violation with the error message.
            Err(e) => (1, format!("extract: {e}")),
        };

        // ── EM ────────────────────────────────────────────────────
        let i_peak = m.get("i_vbias")
            .and_then(|v| v.as_number())
            .map(|x| x.abs())
            .unwrap_or(0.0);
        let segments = vec![
            eda_em::Segment {
                net: "vbias".into(),
                layer: eda_em::Layer::Metal1,
                width_um: 1.0,
                current_a: i_peak,
            },
            eda_em::Segment {
                net: "mout".into(),
                layer: eda_em::Layer::Metal1,
                width_um: 1.0,
                current_a: i_peak,
            },
        ];
        let em_v = eda_em::check(
            &segments,
            &eda_em::Jmax::sky130_metal(),
            &eda_em::LayerThickness::sky130_metal(),
        ).unwrap_or_default();
        let em_first = em_v.first().map_or(String::new(), |v| {
            format!("{}: {:.2}× over Jmax", v.net, v.margin_ratio)
        });

        VerifyReport::empty()
            .set_drc(VerifierResult::from_count(drc_v.len(), drc_first))
            .set_lvs(VerifierResult::from_count(lvs_count, lvs_first))
            .set_em(VerifierResult::from_count(em_v.len(),  em_first))
    }
}
