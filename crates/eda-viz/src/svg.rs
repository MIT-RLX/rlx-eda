//! Minimal SVG-string builder.
//!
//! No external SVG crate: the output is small, the structure is shallow,
//! and writing strings keeps the dep tree at one (`klayout-core`).

use std::fmt::Write;

/// Builder that accumulates an SVG document. The viewBox is applied as
/// `min_x min_y width height` in user units.
pub struct SvgDoc {
    pub min_x: f64,
    pub min_y: f64,
    pub width: f64,
    pub height: f64,
    body: String,
}

impl SvgDoc {
    pub fn new(min_x: f64, min_y: f64, width: f64, height: f64) -> Self {
        Self { min_x, min_y, width, height, body: String::new() }
    }

    pub fn raw(&mut self, s: &str) -> &mut Self {
        self.body.push_str(s);
        self
    }

    pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, attrs: &str) -> &mut Self {
        let _ = write!(
            self.body,
            "<rect x=\"{x:.3}\" y=\"{y:.3}\" width=\"{w:.3}\" height=\"{h:.3}\" {attrs}/>",
        );
        self
    }

    pub fn polygon(&mut self, points: &[(f64, f64)], attrs: &str) -> &mut Self {
        self.body.push_str("<polygon points=\"");
        for (i, (x, y)) in points.iter().enumerate() {
            if i > 0 { self.body.push(' '); }
            let _ = write!(self.body, "{x:.3},{y:.3}");
        }
        let _ = write!(self.body, "\" {attrs}/>");
        self
    }

    pub fn polyline(&mut self, points: &[(f64, f64)], attrs: &str) -> &mut Self {
        self.body.push_str("<polyline points=\"");
        for (i, (x, y)) in points.iter().enumerate() {
            if i > 0 { self.body.push(' '); }
            let _ = write!(self.body, "{x:.3},{y:.3}");
        }
        let _ = write!(self.body, "\" {attrs}/>");
        self
    }

    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, attrs: &str) -> &mut Self {
        let _ = write!(
            self.body,
            "<line x1=\"{x1:.3}\" y1=\"{y1:.3}\" x2=\"{x2:.3}\" y2=\"{y2:.3}\" {attrs}/>",
        );
        self
    }

    pub fn circle(&mut self, cx: f64, cy: f64, r: f64, attrs: &str) -> &mut Self {
        let _ = write!(
            self.body,
            "<circle cx=\"{cx:.3}\" cy=\"{cy:.3}\" r=\"{r:.3}\" {attrs}/>",
        );
        self
    }

    pub fn text(&mut self, x: f64, y: f64, attrs: &str, body: &str) -> &mut Self {
        let _ = write!(
            self.body,
            "<text x=\"{x:.3}\" y=\"{y:.3}\" {attrs}>{}</text>",
            xml_escape(body),
        );
        self
    }

    pub fn group_open(&mut self, attrs: &str) -> &mut Self {
        let _ = write!(self.body, "<g {attrs}>");
        self
    }

    pub fn group_close(&mut self) -> &mut Self {
        self.body.push_str("</g>");
        self
    }

    pub fn finish(self) -> String {
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
             <svg xmlns=\"http://www.w3.org/2000/svg\" \
             viewBox=\"{:.3} {:.3} {:.3} {:.3}\" \
             width=\"{:.0}\" height=\"{:.0}\">{}</svg>",
            self.min_x, self.min_y, self.width, self.height,
            self.width.max(1.0), self.height.max(1.0),
            self.body,
        )
    }
}

/// Escape the five XML metacharacters. SVG text nodes are XML.
pub fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<'  => out.push_str("&lt;"),
            '>'  => out.push_str("&gt;"),
            '&'  => out.push_str("&amp;"),
            '"'  => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _    => out.push(c),
        }
    }
    out
}

/// Deterministic palette: pick a hex color from a fixed list keyed by an
/// integer hash. The Knuth-multiplicative scramble ensures adjacent
/// `(layer, datatype)` pairs (e.g. 10/0, 20/0, 50/0) hit different
/// palette slots — without the scramble, raw `key % N` collides on
/// nearby integer keys.
pub fn palette_color(key: u32) -> &'static str {
    // 12 high-contrast hues, deliberately picked to look distinct at
    // moderate fill-opacity. Light/grays excluded so layer regions stay
    // visible against a white background.
    const PALETTE: &[&str] = &[
        "#e74c3c", // red
        "#3498db", // blue
        "#f39c12", // orange
        "#2ecc71", // green
        "#9b59b6", // purple
        "#1abc9c", // teal
        "#e67e22", // pumpkin
        "#34495e", // navy
        "#d35400", // dark orange
        "#27ae60", // dark green
        "#2980b9", // dark blue
        "#c0392b", // dark red
    ];
    let scrambled = key.wrapping_mul(2_654_435_769);
    PALETTE[(scrambled as usize) % PALETTE.len()]
}
