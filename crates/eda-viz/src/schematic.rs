//! Symbolic schematic renderer.
//!
//! No `Schematic<P>` HIR yet — this module exposes a small data
//! structure ([`Schematic`]) the caller fills in directly. When a
//! `Schematic<P>` trait lands in `eda-hir`, a thin adapter on top of
//! this renderer will turn that IR into a [`Schematic`].
//!
//! ## Coordinate system
//!
//! Schematic coordinates are float, y-up, in arbitrary "schematic
//! units" — typically use a grid (e.g. 1 unit = 10 px). The renderer
//! flips y for SVG and applies a single `pixels_per_unit` scale.
//!
//! ## Symbols
//!
//! [`Symbol`] enumerates the canonical glyphs (resistor, capacitor,
//! diode, voltage source, ground). Each symbol declares its terminal
//! anchors so wires connect cleanly. Adding a new glyph: extend
//! `Symbol`, give it an entry in [`Symbol::draw`] and [`Symbol::pins`].

use crate::svg::SvgDoc;

/// Visual style for the schematic. Independent of [`crate::Style`]
/// (which targets layout rendering) because the parameters are
/// different — schematic rendering is unitless.
#[derive(Clone, Debug)]
pub struct SchemStyle {
    /// SVG user units per schematic unit.
    pub pixels_per_unit: f64,
    /// Schematic-unit padding around the bbox of all symbols+wires.
    pub pad: f64,
    /// Stroke width (user units).
    pub stroke_width: f64,
    /// Background color, or `None` for transparent.
    pub background: Option<String>,
    /// Wire / symbol stroke color.
    pub ink: String,
    /// Label color.
    pub label: String,
    /// Label font size (user units).
    pub font_size: f64,
}

impl Default for SchemStyle {
    fn default() -> Self {
        Self {
            pixels_per_unit: 14.0,
            pad: 4.5, // wide enough to fit value labels rendered to the right of vertical symbols without an explicit text-extent pass
            stroke_width: 0.12,
            background: Some("white".into()),
            ink: "#222".into(),
            label: "#000".into(),
            font_size: 0.85,
        }
    }
}

/// Canonical symbol glyphs. Each is drawn into a 4×2 schematic-unit box
/// centered on the symbol's anchor, with terminals on the left/right
/// (or top/bottom for [`Symbol::Ground`]).
#[derive(Clone, Debug)]
pub enum Symbol {
    /// Zigzag resistor. `value` is the optional label (e.g. `"1kΩ"`).
    Resistor { label: String, value: Option<String> },
    /// Two-plate capacitor.
    Capacitor { label: String, value: Option<String> },
    /// Triangle-and-bar diode.
    Diode { label: String },
    /// Independent DC voltage source — circle with `+`/`-`.
    Vsource { label: String, value: Option<String> },
    /// Ground reference. Single terminal at the symbol's top.
    Ground,
    /// N-channel MOSFET — 4 terminals D / G / S / B with arrow on
    /// source pointing *into* the channel (NMOS convention).
    Nmos { label: String, value: Option<String> },
    /// P-channel MOSFET — same shape as NMOS with arrow on source
    /// pointing *out of* the channel.
    Pmos { label: String, value: Option<String> },
    /// Hierarchical subcircuit — a labeled rectangle with named pins
    /// on its sides. Renderers draw a box; netlist emitters emit a
    /// `.subckt` reference.
    Subcircuit {
        label: String,
        /// Pin names in order — distributed alternately on the left
        /// and right sides of the box.
        pin_names: Vec<String>,
    },
}

impl Symbol {
    /// Pin offsets (schematic units, relative to the symbol's anchor),
    /// in the symbol's *horizontal-default* frame. The `Orient::Vertical`
    /// transform rotates these into screen space, so a vertical NMOS
    /// places D on top, S on bottom, G on the left, B on the right.
    ///
    /// 2-terminal symbols return `[left, right]`; `Ground` returns one
    /// pin at the top of the body; MOSFETs return 4 pins in
    /// **`[D, G, S, B]`** order (matches `eda_hir::Mosfet::currents()`).
    pub fn pins(&self) -> Vec<(f64, f64)> {
        match self {
            Symbol::Ground => vec![(0.0, 1.0)],
            // MOSFET pins: D west, S east (channel runs horizontally
            // in default frame); G below, B above. Distance 1.5 keeps
            // pins outside the 0.7-radius body.
            Symbol::Nmos { .. } | Symbol::Pmos { .. } => vec![
                (-1.5,  0.0),   // D
                ( 0.0, -1.5),   // G
                ( 1.5,  0.0),   // S
                ( 0.0,  1.5),   // B
            ],
            Symbol::Subcircuit { pin_names, .. } => {
                // Even pins on left, odd pins on right; spaced
                // vertically so the body height grows with pin count.
                let n = pin_names.len();
                let mut pts = Vec::with_capacity(n);
                let dy = 1.2;
                let n_left  = (n + 1) / 2;
                let n_right = n / 2;
                let h_left  = (n_left.saturating_sub(1)) as f64 * dy;
                let h_right = (n_right.saturating_sub(1)) as f64 * dy;
                let mut left_i = 0;
                let mut right_i = 0;
                for i in 0..n {
                    if i % 2 == 0 {
                        let y = h_left * 0.5 - left_i as f64 * dy;
                        pts.push((-1.6, y));
                        left_i += 1;
                    } else {
                        let y = h_right * 0.5 - right_i as f64 * dy;
                        pts.push(( 1.6, y));
                        right_i += 1;
                    }
                }
                pts
            }
            _ => vec![(-2.0, 0.0), (2.0, 0.0)],
        }
    }
}

/// Orientation of a placed symbol. Affects pin positions.
#[derive(Copy, Clone, Debug, Default)]
pub enum Orient {
    #[default]
    Horizontal,
    Vertical,
}

/// One placed symbol in the schematic.
#[derive(Clone, Debug)]
pub struct Placed {
    pub anchor: (f64, f64),
    pub orient: Orient,
    pub symbol: Symbol,
}

impl Placed {
    /// Absolute coordinate of the i-th pin.
    pub fn pin(&self, i: usize) -> (f64, f64) {
        let p = self.symbol.pins()[i];
        let (x, y) = match self.orient {
            Orient::Horizontal => p,
            Orient::Vertical   => (p.1, -p.0),
        };
        (self.anchor.0 + x, self.anchor.1 + y)
    }
}

/// Polyline wire connecting two or more points. Net label optional.
#[derive(Clone, Debug, Default)]
pub struct Wire {
    pub points: Vec<(f64, f64)>,
    pub net: Option<String>,
}

/// Horizontal text alignment for a [`PinLabel`]. Picks the SVG
/// `text-anchor` so the rendered label sits to the right of the pin
/// (`Start` — default, text grows rightward), to the left of the pin
/// (`End` — text grows leftward, useful for labels that would
/// otherwise crash into a symbol body), or centered on the pin
/// (`Middle` — useful for top/bottom-of-symbol pins).
#[derive(Copy, Clone, Debug, Default)]
pub enum LabelAlign {
    #[default]
    Start,
    End,
    Middle,
}

/// One labeled "external" point — typically a port or test node.
#[derive(Clone, Debug)]
pub struct PinLabel {
    pub at: (f64, f64),
    pub text: String,
    pub align: LabelAlign,
}

/// A schematic to render. Build it by pushing symbols, wires, and pin
/// labels; [`render_to_svg`] turns it into an SVG document.
#[derive(Clone, Debug, Default)]
pub struct Schematic {
    pub title: Option<String>,
    pub symbols: Vec<Placed>,
    pub wires: Vec<Wire>,
    pub pins: Vec<PinLabel>,
}

impl Schematic {
    pub fn new() -> Self { Self::default() }

    pub fn place(&mut self, symbol: Symbol, anchor: (f64, f64), orient: Orient) -> usize {
        let id = self.symbols.len();
        self.symbols.push(Placed { anchor, orient, symbol });
        id
    }

    pub fn wire(&mut self, points: impl IntoIterator<Item = (f64, f64)>) {
        self.wires.push(Wire { points: points.into_iter().collect(), net: None });
    }

    pub fn wire_named(
        &mut self,
        net: impl Into<String>,
        points: impl IntoIterator<Item = (f64, f64)>,
    ) {
        self.wires.push(Wire { points: points.into_iter().collect(), net: Some(net.into()) });
    }

    pub fn pin_label(&mut self, at: (f64, f64), text: impl Into<String>) {
        self.pins.push(PinLabel { at, text: text.into(), align: LabelAlign::Start });
    }

    /// Like [`pin_label`] but with explicit horizontal alignment so the
    /// label can grow leftward (`End`), centered (`Middle`), or
    /// rightward (`Start`) from the pin location. Use `End` for labels
    /// on the left side of a symbol so the text doesn't extend into the
    /// symbol body.
    pub fn pin_label_aligned(&mut self, at: (f64, f64), text: impl Into<String>, align: LabelAlign) {
        self.pins.push(PinLabel { at, text: text.into(), align });
    }
}

/// Render `schem` to a complete SVG document.
pub fn render_to_svg(schem: &Schematic, style: &SchemStyle) -> String {
    let bbox = bbox_of(schem);
    let (min_x, min_y, max_x, max_y) = match bbox {
        Some(b) => b,
        None => return SvgDoc::new(0.0, 0.0, 1.0, 1.0).finish(),
    };
    let pad = style.pad;
    let upu = style.pixels_per_unit;

    // y flip: schematic y-up → svg y-down.
    let vx = (min_x - pad) * upu;
    let vy = -(max_y + pad) * upu;
    let vw = (max_x - min_x + 2.0 * pad) * upu;
    let vh = (max_y - min_y + 2.0 * pad) * upu;

    let mut doc = SvgDoc::new(vx, vy, vw, vh);

    if let Some(bg) = &style.background {
        doc.rect(vx, vy, vw, vh, &format!("fill=\"{bg}\" stroke=\"none\""));
    }

    let stroke_w = style.stroke_width * upu;
    let group_attrs = format!(
        "fill=\"none\" stroke=\"{}\" stroke-width=\"{stroke_w:.3}\" \
         stroke-linecap=\"round\" stroke-linejoin=\"round\"",
        style.ink,
    );
    doc.group_open(&group_attrs);

    for w in &schem.wires {
        if w.points.len() < 2 { continue; }
        let pts: Vec<(f64, f64)> = w.points.iter().map(|p| map(*p, upu)).collect();
        doc.polyline(&pts, "");
    }

    for placed in &schem.symbols {
        draw_symbol(&mut doc, placed, upu, style);
    }

    doc.group_close();

    // Junction dots — schematic convention says wires that *cross*
    // don't connect, but wires that *meet* (T or +) do. The dot is the
    // visible disambiguator: any point where ≥3 wire endpoints / symbol
    // pins converge gets a small filled circle. Without this the vout
    // tap is technically ambiguous.
    let dots = junction_points(schem);
    if !dots.is_empty() {
        let dot_attrs = format!("fill=\"{}\" stroke=\"none\"", style.ink);
        doc.group_open(&dot_attrs);
        let r = (style.stroke_width * 2.0).max(0.18) * upu;
        for (px, py) in dots {
            let (mx, my) = map((px, py), upu);
            doc.circle(mx, my, r, "");
        }
        doc.group_close();
    }

    let label_attrs = format!(
        "font-family=\"sans-serif\" font-size=\"{:.2}\" fill=\"{}\"",
        style.font_size * upu,
        style.label,
    );
    doc.group_open(&label_attrs);
    for placed in &schem.symbols {
        draw_symbol_labels(&mut doc, placed, upu, style);
    }
    // Net names are stored on Wire::net but not auto-rendered: when
    // multiple wires share a net, multiple labels at every midpoint
    // become noise. Use explicit `pin_label`s for the labels you want.
    for pin in &schem.pins {
        let (x, y) = map(pin.at, upu);
        let (offset_x, anchor_attr) = match pin.align {
            LabelAlign::Start  => ( 0.3 * upu, "text-anchor=\"start\""),
            LabelAlign::End    => (-0.3 * upu, "text-anchor=\"end\""),
            LabelAlign::Middle => (0.0,        "text-anchor=\"middle\""),
        };
        doc.text(x + offset_x, y - 0.3 * upu, anchor_attr, &pin.text);
    }
    doc.group_close();

    if let Some(t) = &schem.title {
        // Title sits inside the top padding band, x-centered on the
        // bbox so it never escapes the viewBox.
        let cx = (min_x + max_x) * 0.5;
        let ty = max_y + pad * 0.6;
        let (x, y) = map((cx, ty), upu);
        doc.text(
            x, y,
            &format!(
                "font-family=\"sans-serif\" font-size=\"{:.2}\" font-weight=\"bold\" \
                 fill=\"{}\" text-anchor=\"middle\"",
                style.font_size * 1.5 * upu,
                style.label,
            ),
            t,
        );
    }

    doc.finish()
}

// ── drawing ─────────────────────────────────────────────────────────────

fn draw_symbol(doc: &mut SvgDoc, placed: &Placed, upu: f64, style: &SchemStyle) {
    // Lead lines from anchor to each pin (drawn before the glyph so the
    // glyph overpaints the meeting point cleanly).
    for i in 0..placed.symbol.pins().len() {
        let pin = placed.pin(i);
        let lead = lead_endpoint(placed, i);
        let pts = [map(pin, upu), map(lead, upu)];
        doc.line(pts[0].0, pts[0].1, pts[1].0, pts[1].1, "");
    }
    match &placed.symbol {
        Symbol::Resistor { .. } => draw_resistor(doc, placed, upu),
        Symbol::Capacitor { .. } => draw_capacitor(doc, placed, upu),
        Symbol::Diode { .. } => draw_diode(doc, placed, upu),
        Symbol::Vsource { .. } => draw_vsource(doc, placed, upu, style),
        Symbol::Ground => draw_ground(doc, placed, upu),
        Symbol::Nmos { .. } => draw_mosfet(doc, placed, upu, /*nmos*/ true),
        Symbol::Pmos { .. } => draw_mosfet(doc, placed, upu, /*nmos*/ false),
        Symbol::Subcircuit { pin_names, .. } => draw_subcircuit(doc, placed, upu, pin_names.len()),
    }
}

/// Endpoint of the symbol body on the side of pin `i`, in the
/// schematic's global frame. The lead is drawn from the pin to this
/// point; the glyph fills the space between the two leads.
///
/// Computed as `body_half * unit(pin)` — i.e. lead_endpoint sits exactly
/// on the body edge along the pin direction, so lead and body always
/// meet. (Previous formula `pin * (|pin| - body_half) / |pin|`
/// accidentally worked for the resistor — body_half=1, |pin|=2 — and
/// left a gap for every other symbol.)
fn lead_endpoint(placed: &Placed, i: usize) -> (f64, f64) {
    let pin_local = placed.symbol.pins()[i];
    let body_half = match &placed.symbol {
        Symbol::Resistor { .. }  => 1.0,
        Symbol::Capacitor { .. } => 0.3,
        Symbol::Diode { .. }     => 0.7,
        Symbol::Vsource { .. }   => 0.8,
        Symbol::Ground           => 0.0, // pin is the body
        // MOSFET body is roughly square — 0.7-unit half-extent on both
        // axes works since pins are placed at distance 1.5.
        Symbol::Nmos { .. } | Symbol::Pmos { .. } => 0.7,
        // Subcircuit: body edge is at x=±1.4, so leads from pins at
        // x=±1.6 are short (0.2 unit) — we want the rectangle
        // boundary, not anchor.
        Symbol::Subcircuit { .. } => 1.4,
    };
    let len = (pin_local.0 * pin_local.0 + pin_local.1 * pin_local.1).sqrt();
    let local = if len > 1e-9 {
        let s = body_half / len;
        (pin_local.0 * s, pin_local.1 * s)
    } else {
        pin_local
    };
    let oriented = match placed.orient {
        Orient::Horizontal => local,
        Orient::Vertical   => (local.1, -local.0),
    };
    (placed.anchor.0 + oriented.0, placed.anchor.1 + oriented.1)
}

fn draw_resistor(doc: &mut SvgDoc, placed: &Placed, upu: f64) {
    // Zigzag along the symbol's primary axis. 6 segments fit a 2-unit body.
    let zig = [
        (-1.0,  0.0),
        (-0.8,  0.4),
        (-0.4, -0.4),
        ( 0.0,  0.4),
        ( 0.4, -0.4),
        ( 0.8,  0.4),
        ( 1.0,  0.0),
    ];
    let pts: Vec<(f64, f64)> = zig.iter()
        .map(|p| transform_local(*p, placed))
        .map(|p| map(p, upu))
        .collect();
    doc.polyline(&pts, "");
}

fn draw_capacitor(doc: &mut SvgDoc, placed: &Placed, upu: f64) {
    let plate_h = 0.8;
    let gap = 0.3;
    let left = [(-gap, -plate_h), (-gap, plate_h)];
    let right = [( gap, -plate_h), ( gap, plate_h)];
    let l: Vec<(f64, f64)> = left.iter().map(|p| transform_local(*p, placed)).map(|p| map(p, upu)).collect();
    let r: Vec<(f64, f64)> = right.iter().map(|p| transform_local(*p, placed)).map(|p| map(p, upu)).collect();
    doc.line(l[0].0, l[0].1, l[1].0, l[1].1, "");
    doc.line(r[0].0, r[0].1, r[1].0, r[1].1, "");
}

fn draw_diode(doc: &mut SvgDoc, placed: &Placed, upu: f64) {
    // Triangle (anode) pointing right toward bar (cathode).
    let tri = [(-0.7, -0.5), ( 0.5, 0.0), (-0.7, 0.5)];
    let bar = [( 0.5, -0.5), ( 0.5, 0.5)];
    let t: Vec<(f64, f64)> = tri.iter().map(|p| transform_local(*p, placed)).map(|p| map(p, upu)).collect();
    let b: Vec<(f64, f64)> = bar.iter().map(|p| transform_local(*p, placed)).map(|p| map(p, upu)).collect();
    doc.polygon(&t, "");
    doc.line(b[0].0, b[0].1, b[1].0, b[1].1, "");
}

fn draw_vsource(doc: &mut SvgDoc, placed: &Placed, upu: f64, style: &SchemStyle) {
    let center = map(placed.anchor, upu);
    let r = 0.8 * upu;
    doc.circle(center.0, center.1, r, "");
    // Pin 0 is the positive terminal. Place `+` near pin 0 and `−` near
    // pin 1, both inside the circle. The pins() offsets give us the
    // correct direction in the orient-local frame; transform_local
    // then handles the rotation for vertical placement.
    let pins = placed.symbol.pins();
    let p0 = pins.get(0).copied().unwrap_or((0.0, 0.0));
    let p1 = pins.get(1).copied().unwrap_or((0.0, 0.0));
    let inset = 0.42;
    let unit = |p: (f64, f64)| {
        let l = (p.0 * p.0 + p.1 * p.1).sqrt().max(1e-9);
        (p.0 / l, p.1 / l)
    };
    let u0 = unit(p0);
    let u1 = unit(p1);
    let plus  = transform_local((u0.0 * inset, u0.1 * inset), placed);
    let minus = transform_local((u1.0 * inset, u1.1 * inset), placed);
    let pp = map(plus, upu);
    let mm = map(minus, upu);
    let attrs = format!(
        "font-family=\"sans-serif\" font-size=\"{:.2}\" fill=\"{}\" stroke=\"none\" \
         text-anchor=\"middle\" dominant-baseline=\"central\"",
        style.font_size * upu, style.ink,
    );
    doc.text(pp.0, pp.1, &attrs, "+");
    doc.text(mm.0, mm.1, &attrs, "−");
}

/// Draw a 4-terminal MOSFET. `nmos = true` puts the source arrow
/// pointing into the channel (NMOS); false puts it pointing outward
/// (PMOS). Pin order is `[D, G, S, B]`, matching `Symbol::pins()`.
///
/// Body is a 1.4×1.4 unit square centered on the anchor, with the gate
/// stripe on the south face, drain on the west, source on the east,
/// bulk on the north — in the *horizontal-default* frame. Vertical
/// orientation rotates the whole glyph 90° via `transform_local`.
fn draw_mosfet(doc: &mut SvgDoc, placed: &Placed, upu: f64, nmos: bool) {
    // Channel bar (between drain and source pins) — horizontal line.
    let chan_l = transform_local((-0.7, 0.0), placed);
    let chan_r = transform_local(( 0.7, 0.0), placed);
    let cl = map(chan_l, upu);
    let cr = map(chan_r, upu);
    doc.line(cl.0, cl.1, cr.0, cr.1, "");

    // Gate: short stripe parallel to the channel, slightly south,
    // separated by a small "gate-oxide" gap.
    let gate_l = transform_local((-0.6, -0.25), placed);
    let gate_r = transform_local(( 0.6, -0.25), placed);
    let gl = map(gate_l, upu);
    let gr = map(gate_r, upu);
    doc.line(gl.0, gl.1, gr.0, gr.1, "");
    // Gate stub down to the gate pin (which is at (0, -1.5) in local).
    let gate_stub_top = transform_local((0.0, -0.25), placed);
    let gate_stub_bot = transform_local((0.0, -0.7), placed);
    let gst = map(gate_stub_top, upu);
    let gsb = map(gate_stub_bot, upu);
    doc.line(gst.0, gst.1, gsb.0, gsb.1, "");

    // Bulk tie — short stub up to the bulk pin (0, 1.5).
    let bulk_top = transform_local((0.0, 0.7), placed);
    let bulk_bot = transform_local((0.0, 0.0), placed);
    let bt = map(bulk_top, upu);
    let bb = map(bulk_bot, upu);
    doc.line(bb.0, bb.1, bt.0, bt.1, "");

    // Source arrow — small triangle on the source side of the channel.
    // NMOS: arrow points INTO the channel (right-to-left for east-side
    // source means arrow tip on left, base on right). PMOS: arrow
    // points OUT of channel (tip on right, base on left).
    let arrow = if nmos {
        // tip at (0.5, 0), base around (0.7, ±0.12)
        [(0.5, 0.0), (0.7, -0.12), (0.7, 0.12)]
    } else {
        [(0.7, 0.0), (0.5, -0.12), (0.5, 0.12)]
    };
    let pts: Vec<(f64, f64)> = arrow.iter()
        .map(|p| transform_local(*p, placed))
        .map(|p| map(p, upu))
        .collect();
    doc.polygon(&pts, "fill=\"currentColor\"");
}

/// Draw a `Subcircuit` symbol — a labeled rectangle with pin stubs on
/// each side. Sized to fit the pin count (1.4 × max(2, n_pins)).
fn draw_subcircuit(doc: &mut SvgDoc, placed: &Placed, upu: f64, n_pins: usize) {
    // Rectangle covering the local frame ±half-extent on each axis.
    let half_w = 1.4;
    let half_h = (n_pins.max(2) as f64) * 0.6;
    let corners = [
        (-half_w, -half_h),
        ( half_w, -half_h),
        ( half_w,  half_h),
        (-half_w,  half_h),
    ];
    let pts: Vec<(f64, f64)> = corners.iter()
        .map(|p| transform_local(*p, placed))
        .map(|p| map(p, upu))
        .collect();
    doc.polygon(&pts, "fill=\"none\"");
    let _ = doc; // suppress unused-binding lint variants
}

fn draw_ground(doc: &mut SvgDoc, placed: &Placed, upu: f64) {
    // Three horizontal stripes of decreasing length, anchor is the
    // single pin location.
    let widths = [1.0, 0.6, 0.3];
    for (i, w) in widths.iter().enumerate() {
        let y = -(0.0 + i as f64 * 0.3);
        let a = transform_local((-w * 0.5, y), placed);
        let b = transform_local(( w * 0.5, y), placed);
        let pa = map(a, upu);
        let pb = map(b, upu);
        doc.line(pa.0, pa.1, pb.0, pb.1, "");
    }
}

fn draw_symbol_labels(doc: &mut SvgDoc, placed: &Placed, upu: f64, style: &SchemStyle) {
    let (label, value) = match &placed.symbol {
        Symbol::Resistor { label, value }  => (Some(label), value.as_ref()),
        Symbol::Capacitor { label, value } => (Some(label), value.as_ref()),
        Symbol::Diode { label }            => (Some(label), None),
        Symbol::Vsource { label, value }   => (Some(label), value.as_ref()),
        Symbol::Ground                     => (None, None),
        Symbol::Nmos { label, value }      => (Some(label), value.as_ref()),
        Symbol::Pmos { label, value }      => (Some(label), value.as_ref()),
        Symbol::Subcircuit { label, .. }   => (Some(label), None),
    };
    let Some(name) = label else { return };
    // Body half-height in the symbol's local frame (y for horizontal,
    // x for vertical). Most symbols fit in the default ±1.0 box;
    // Subcircuit grows with pin count (`half_h = n_pins * 0.6`), so
    // the label offset has to follow or it sits inside the body.
    let body_half = match &placed.symbol {
        Symbol::Subcircuit { pin_names, .. } => (pin_names.len().max(2) as f64) * 0.6,
        _ => 1.0,
    };
    let name_dy  = body_half + 0.6;
    let value_dy = body_half + 0.6;
    // Place name above the body, value below — text-anchor=start so the
    // labels sit cleanly to the right of vertical symbols and don't
    // collide with the symbol body.
    let (name_offset, value_offset, anchor) = match placed.orient {
        Orient::Horizontal => ((0.0,  name_dy), (0.0, -value_dy), "middle"),
        Orient::Vertical   => ((name_dy, 0.7),  (value_dy, -0.7), "start"),
    };
    let attrs = format!(
        "font-family=\"sans-serif\" font-size=\"{:.2}\" fill=\"{}\" \
         text-anchor=\"{anchor}\"",
        style.font_size * upu, style.label,
    );
    let (nx, ny) = map(
        (placed.anchor.0 + name_offset.0, placed.anchor.1 + name_offset.1),
        upu,
    );
    doc.text(nx, ny, &attrs, name);
    if let Some(v) = value {
        let (vx, vy) = map(
            (placed.anchor.0 + value_offset.0, placed.anchor.1 + value_offset.1),
            upu,
        );
        doc.text(vx, vy, &attrs, v);
    }
}

// ── geometry helpers ────────────────────────────────────────────────────

fn transform_local(local: (f64, f64), placed: &Placed) -> (f64, f64) {
    let rotated = match placed.orient {
        Orient::Horizontal => local,
        Orient::Vertical   => (local.1, -local.0),
    };
    (placed.anchor.0 + rotated.0, placed.anchor.1 + rotated.1)
}

fn map(p: (f64, f64), upu: f64) -> (f64, f64) {
    (p.0 * upu, -p.1 * upu)
}

fn bbox_of(schem: &Schematic) -> Option<(f64, f64, f64, f64)> {
    let mut have = false;
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let grow = |x: f64, y: f64,
                have: &mut bool,
                mn_x: &mut f64, mn_y: &mut f64,
                mx_x: &mut f64, mx_y: &mut f64| {
        *have = true;
        if x < *mn_x { *mn_x = x; }
        if y < *mn_y { *mn_y = y; }
        if x > *mx_x { *mx_x = x; }
        if y > *mx_y { *mx_y = y; }
    };
    for placed in &schem.symbols {
        for i in 0..placed.symbol.pins().len() {
            let (x, y) = placed.pin(i);
            grow(x, y, &mut have, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
        }
        grow(placed.anchor.0, placed.anchor.1,
             &mut have, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
    }
    for w in &schem.wires {
        for &(x, y) in &w.points {
            grow(x, y, &mut have, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
        }
    }
    for p in &schem.pins {
        grow(p.at.0, p.at.1,
             &mut have, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
    }
    if have { Some((min_x, min_y, max_x, max_y)) } else { None }
}

/// Find points where ≥3 wire endpoints / symbol pins converge.
/// Returns the deduplicated set of intersection coordinates.
///
/// Coordinates are bucketed at 0.1-unit resolution (well below any
/// schematic-grid spacing we use) so floating-point noise from
/// translate/merge doesn't desync points that *should* be the same.
///
/// "Endpoints" of a wire are its first and last point AND every
/// interior corner (since polylines bend at each interior point —
/// a corner is also a place wires "meet"). Symbol pins also count.
fn junction_points(schem: &Schematic) -> Vec<(f64, f64)> {
    use std::collections::HashMap;
    let key = |x: f64, y: f64| ((x * 10.0).round() as i64, (y * 10.0).round() as i64);
    let mut counts: HashMap<(i64, i64), (f64, f64, u32)> = HashMap::new();
    let bump = |x: f64, y: f64, counts: &mut HashMap<(i64, i64), (f64, f64, u32)>| {
        let k = key(x, y);
        let entry = counts.entry(k).or_insert((x, y, 0));
        entry.2 += 1;
    };
    for w in &schem.wires {
        for &(x, y) in &w.points {
            bump(x, y, &mut counts);
        }
    }
    for placed in &schem.symbols {
        for i in 0..placed.symbol.pins().len() {
            let (x, y) = placed.pin(i);
            bump(x, y, &mut counts);
        }
    }
    let mut out: Vec<(f64, f64)> = counts.values()
        .filter(|(_, _, n)| *n >= 3)
        .map(|(x, y, _)| (*x, *y))
        .collect();
    // Deterministic order so SVG bytes are stable.
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
    out
}

// ── adapter from eda-hir::SchematicIr ──────────────────────────────────

/// Build a renderable [`Schematic`] from the framework-neutral
/// [`eda_hir::SchematicIr`] that a `Schematic<P>::schematic(pdk)`
/// returns.
///
/// The IR is the boundary: any block crate produces it, eda-viz
/// consumes it. Anything you can express in the IR is renderable
/// without further coupling.
pub fn from_ir(ir: &eda_hir::SchematicIr) -> Schematic {
    let mut s = Schematic::new();
    s.title = ir.title.clone();
    for sym in &ir.symbols {
        let kind = match &sym.kind {
            eda_hir::SymbolKind::Resistor => Symbol::Resistor {
                label: sym.label.clone(), value: sym.value.clone(),
            },
            eda_hir::SymbolKind::Capacitor => Symbol::Capacitor {
                label: sym.label.clone(), value: sym.value.clone(),
            },
            eda_hir::SymbolKind::Diode => Symbol::Diode {
                label: sym.label.clone(),
            },
            eda_hir::SymbolKind::Vsource => Symbol::Vsource {
                label: sym.label.clone(), value: sym.value.clone(),
            },
            eda_hir::SymbolKind::Ground => Symbol::Ground,
            eda_hir::SymbolKind::Nmos => Symbol::Nmos {
                label: sym.label.clone(), value: sym.value.clone(),
            },
            eda_hir::SymbolKind::Pmos => Symbol::Pmos {
                label: sym.label.clone(), value: sym.value.clone(),
            },
            eda_hir::SymbolKind::Subcircuit { pin_names } => Symbol::Subcircuit {
                label: sym.label.clone(),
                pin_names: pin_names.clone(),
            },
        };
        let orient = match sym.orient {
            eda_hir::SchemOrient::Horizontal => Orient::Horizontal,
            eda_hir::SchemOrient::Vertical   => Orient::Vertical,
        };
        s.place(kind, sym.anchor, orient);
    }
    for w in &ir.wires {
        match &w.net {
            Some(name) => s.wire_named(name.clone(), w.points.iter().copied()),
            None       => s.wire(w.points.iter().copied()),
        }
    }
    for p in &ir.ports {
        s.pin_label(p.at, p.name.clone());
    }
    s
}

// ── canned schematics for common circuits ──────────────────────────────

/// Build a symbolic voltage-divider schematic (V → R1 → vout → R2 → GND).
/// The labels and values are taken from the caller, so this works for any
/// resistor pair without depending on `eda-hir` types.
pub fn voltage_divider(
    r1_label: &str, r1_value: Option<&str>,
    r2_label: &str, r2_value: Option<&str>,
    v_label: &str,  v_value:  Option<&str>,
) -> Schematic {
    let mut s = Schematic::new();
    s.title = Some("Voltage divider".into());

    // Layout (schematic units, y up):
    //   V at x=0, vertical, anchor (0, 6)  → pins (0, 8) top and (0, 4) bot
    //   R1 at x=6, vertical, anchor (6, 6) → pins (6, 8) top and (6, 4) bot
    //   R2 at x=6, vertical, anchor (6, 2) → pins (6, 4) top and (6, 0) bot
    //   GND_R below R2 at (6, -2): pin (6, -1)
    //   GND_V below V  at (0,  2): pin (0,  3)
    //
    // Aligning V's anchor with R1's lets `vin` be a single straight
    // horizontal wire at y=8, never crossing a body. Same for the
    // grounds at y=2: a short vertical drop from each component's
    // bottom pin to its ground pin.
    s.place(
        Symbol::Vsource { label: v_label.into(), value: v_value.map(Into::into) },
        (0.0, 6.0), Orient::Vertical,
    );
    s.place(
        Symbol::Resistor { label: r1_label.into(), value: r1_value.map(Into::into) },
        (6.0, 6.0), Orient::Vertical,
    );
    s.place(
        Symbol::Resistor { label: r2_label.into(), value: r2_value.map(Into::into) },
        (6.0, 2.0), Orient::Vertical,
    );
    s.place(Symbol::Ground, (6.0, -2.0), Orient::default());
    s.place(Symbol::Ground, (0.0,  2.0), Orient::default());

    s.wire_named("vin",  [(0.0, 8.0), (6.0, 8.0)]);
    s.wire_named("vout", [(6.0, 4.0), (7.6, 4.0)]);
    s.wire([(6.0, 0.0), (6.0, -1.0)]);
    s.wire([(0.0, 4.0), (0.0,  3.0)]);

    s.pin_label((7.7, 4.0), "vout");

    s
}
