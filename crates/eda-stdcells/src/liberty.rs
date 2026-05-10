//! Minimal Liberty (.lib) metadata extractor.
//!
//! v1 only pulls what the bench harness needs: cell area (for
//! physical-metric attribution), pin list + direction, and the
//! Boolean function. Timing arcs, leakage tables, and power groups
//! are deferred — ORFS via OpenSTA is the source of truth for
//! timing, and the in-house flow doesn't yet do per-cell timing
//! analysis.
//!
//! Parser is intentionally narrow: not a full Liberty 2007
//! frontend. Strategy:
//!   1. Tokenize into `{`, `}`, `(`, `)`, `:`, `;`, `,`, idents,
//!      numbers, quoted strings.
//!   2. Walk with a `Scope` stack so we know whether the current
//!      `area : ...` line lives inside a `cell` (we want it) or
//!      inside e.g. `cell_footprint` (we don't).
//!   3. Recognized attributes: `area`, `direction`, `function`.
//!      Everything else is silently skipped — the parser is
//!      tolerant of unknown groups + attributes by design so adding
//!      a new attribute is one match arm, not a re-architecture.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum PinDirection {
    Input,
    Output,
    Inout,
    Power,
    Ground,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct PinInfo {
    pub name: String,
    pub direction: PinDirection,
    /// Boolean function for output pins, e.g. `"!(A & B)"` for nand2.
    /// `None` for inputs and rails.
    pub function: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct LibertyMetadata {
    pub cell_name: String,
    /// Cell area in µm². Liberty reports it in PDK-defined units
    /// (typically square microns); we store the raw number ×1000 as
    /// an integer to keep `Hash + Eq` honest.
    pub area_um2_x1000: u64,
    pub pins: Vec<PinInfo>,
}

#[derive(Debug, thiserror::Error)]
pub enum LibertyError {
    #[error("Liberty parse error: {0}")]
    Parse(String),
}

/// Parse a `.lib` file's worth of cell metadata. Returns one
/// `LibertyMetadata` per `cell ("...") { ... }` block.
pub fn parse_lib(text: &str) -> Result<Vec<LibertyMetadata>, LibertyError> {
    let tokens = tokenize(text)?;
    walk(&tokens)
}

// ── tokenizer ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    LBrace,
    RBrace,
    LParen,
    RParen,
    Colon,
    Semi,
    Comma,
    Ident(String),
    Str(String),
    Num(f64),
}

fn tokenize(text: &str) -> Result<Vec<Token>, LibertyError> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];

        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Block comment /* ... */
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            continue;
        }

        // Line comment // ...
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Backslash line continuation (some Liberty exporters wrap)
        if b == b'\\' {
            i += 1;
            continue;
        }

        match b {
            b'{' => { out.push(Token::LBrace); i += 1; continue; }
            b'}' => { out.push(Token::RBrace); i += 1; continue; }
            b'(' => { out.push(Token::LParen); i += 1; continue; }
            b')' => { out.push(Token::RParen); i += 1; continue; }
            b':' => { out.push(Token::Colon);  i += 1; continue; }
            b';' => { out.push(Token::Semi);   i += 1; continue; }
            b',' => { out.push(Token::Comma);  i += 1; continue; }
            _ => {}
        }

        // Quoted string. Liberty quotes can span multiple tokens
        // separated by `\` continuation, but we treat backslash as
        // already-stripped above; inside a quote it's literal.
        if b == b'"' {
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            if i >= bytes.len() {
                return Err(LibertyError::Parse("unterminated string".into()));
            }
            let s = std::str::from_utf8(&bytes[start..i])
                .map_err(|e| LibertyError::Parse(e.to_string()))?
                .to_string();
            out.push(Token::Str(s));
            i += 1;
            continue;
        }

        // Number — leading digit, sign, or dot. Float-shaped.
        if b.is_ascii_digit() || (b == b'-' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit())
            || (b == b'.' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit())
        {
            let start = i;
            if b == b'-' || b == b'+' {
                i += 1;
            }
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || matches!(bytes[i], b'.' | b'e' | b'E' | b'+' | b'-'))
            {
                // Stop at sign that isn't part of an exponent.
                if matches!(bytes[i], b'+' | b'-')
                    && !matches!(bytes.get(i.wrapping_sub(1)).copied(), Some(b'e') | Some(b'E'))
                {
                    break;
                }
                i += 1;
            }
            let s = std::str::from_utf8(&bytes[start..i])
                .map_err(|e| LibertyError::Parse(e.to_string()))?;
            let n: f64 = s
                .parse()
                .map_err(|e: std::num::ParseFloatError| LibertyError::Parse(e.to_string()))?;
            out.push(Token::Num(n));
            continue;
        }

        // Identifier — letters, digits, underscore, dot, slash.
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric()
                    || matches!(bytes[i], b'_' | b'.' | b'/' | b'-' | b'$'))
            {
                i += 1;
            }
            let s = std::str::from_utf8(&bytes[start..i])
                .map_err(|e| LibertyError::Parse(e.to_string()))?
                .to_string();
            out.push(Token::Ident(s));
            continue;
        }

        return Err(LibertyError::Parse(format!(
            "unexpected byte {:?} at offset {}",
            b as char, i
        )));
    }

    Ok(out)
}

// ── walker ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    Other,
    Library,
    Cell,
    Pin,
}

fn walk(tokens: &[Token]) -> Result<Vec<LibertyMetadata>, LibertyError> {
    let mut cells: Vec<LibertyMetadata> = Vec::new();
    let mut scope: Vec<Scope> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        match &tokens[i] {
            // group head: <ident> ( <name> ) {
            Token::Ident(name) => {
                let kind = name.as_str();
                if let Some((group_name, after)) = read_group_head(tokens, i) {
                    let s = match kind {
                        "library" => Scope::Library,
                        "cell" => {
                            cells.push(LibertyMetadata {
                                cell_name: group_name.clone(),
                                area_um2_x1000: 0,
                                pins: Vec::new(),
                            });
                            Scope::Cell
                        }
                        "pin" => {
                            if let Some(c) = cells.last_mut() {
                                c.pins.push(PinInfo {
                                    name: group_name.clone(),
                                    direction: PinDirection::Input, // default; overridden by attr
                                    function: None,
                                });
                            }
                            Scope::Pin
                        }
                        _ => Scope::Other,
                    };
                    scope.push(s);
                    i = after; // positioned right after the `{`
                    continue;
                }

                // attribute: <ident> : <value> ;
                if let Some((attr, value, after)) = read_attribute(tokens, i) {
                    apply_attribute(&scope, &mut cells, attr, value);
                    i = after;
                    continue;
                }

                i += 1;
            }
            Token::RBrace => {
                scope.pop();
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    Ok(cells)
}

/// Recognize `<ident> ( <name> ) {` starting at index `i`. `<name>`
/// may be a quoted string or a bare ident; some Liberty groups omit
/// the `(...)` entirely (e.g. `timing { ... }`), which we accept and
/// return an empty group_name.
///
/// On match, returns (group_name, index-after-`{`).
fn read_group_head(t: &[Token], i: usize) -> Option<(String, usize)> {
    // <ident> at i
    if !matches!(t.get(i), Some(Token::Ident(_))) {
        return None;
    }
    let mut j = i + 1;

    // Optional `( <args> )` — args may be empty (`leakage_power ()`),
    // single name (`cell ("inv_1")`), or comma-separated values
    // (`capacitive_load_unit(1.0, "pf")`). We capture the first
    // string-shaped token as the group name; empty args → "".
    let group_name = if matches!(t.get(j), Some(Token::LParen)) {
        j += 1;
        let name = match t.get(j) {
            Some(Token::Str(s)) => {
                j += 1;
                s.clone()
            }
            Some(Token::Ident(s)) => {
                j += 1;
                s.clone()
            }
            Some(Token::Num(n)) => {
                let s = format!("{n}");
                j += 1;
                s
            }
            // Empty args `()` — accept and yield "" as the group name.
            Some(Token::RParen) => String::new(),
            _ => return None,
        };
        // skip any extra args (Liberty allows comma-separated lists)
        while j < t.len() && !matches!(t[j], Token::RParen) {
            j += 1;
        }
        if !matches!(t.get(j), Some(Token::RParen)) {
            return None;
        }
        j += 1;
        name
    } else {
        String::new()
    };

    if matches!(t.get(j), Some(Token::LBrace)) {
        Some((group_name, j + 1))
    } else {
        None
    }
}

enum AttrValue {
    Str(String),
    Num(f64),
    Ident(String),
}

/// Recognize `<ident> : <value> ;`. Value may be a number, ident, or
/// quoted string. Multi-value attributes (lists, complex types) are
/// not recognized — caller skips them by falling through to the next
/// token.
fn read_attribute(t: &[Token], i: usize) -> Option<(String, AttrValue, usize)> {
    let attr = match t.get(i) {
        Some(Token::Ident(s)) => s.clone(),
        _ => return None,
    };
    if !matches!(t.get(i + 1), Some(Token::Colon)) {
        return None;
    }
    let val = match t.get(i + 2) {
        Some(Token::Str(s)) => AttrValue::Str(s.clone()),
        Some(Token::Num(n)) => AttrValue::Num(*n),
        Some(Token::Ident(s)) => AttrValue::Ident(s.clone()),
        _ => return None,
    };
    if !matches!(t.get(i + 3), Some(Token::Semi)) {
        return None;
    }
    Some((attr, val, i + 4))
}

fn apply_attribute(
    scope: &[Scope],
    cells: &mut Vec<LibertyMetadata>,
    attr: String,
    val: AttrValue,
) {
    let top = scope.last().copied().unwrap_or(Scope::Other);
    match (top, attr.as_str()) {
        (Scope::Cell, "area") => {
            if let AttrValue::Num(n) = val {
                if let Some(c) = cells.last_mut() {
                    c.area_um2_x1000 = (n * 1000.0).round().max(0.0) as u64;
                }
            }
        }
        (Scope::Pin, "direction") => {
            let dir_str = match val {
                AttrValue::Ident(s) | AttrValue::Str(s) => s,
                _ => return,
            };
            let dir = match dir_str.as_str() {
                "input" => PinDirection::Input,
                "output" => PinDirection::Output,
                "inout" => PinDirection::Inout,
                "internal" => return,
                _ => return,
            };
            if let Some(c) = cells.last_mut() {
                if let Some(p) = c.pins.last_mut() {
                    p.direction = dir;
                }
            }
        }
        (Scope::Pin, "function") => {
            if let AttrValue::Str(s) = val {
                if let Some(c) = cells.last_mut() {
                    if let Some(p) = c.pins.last_mut() {
                        p.function = Some(s);
                    }
                }
            }
        }
        // pg_pin direction lives in `pg_pin (...) { pg_type : ... ; }`;
        // for the v1 bench harness we don't need it. If a contributor
        // wants power-pin annotations later, extend `Scope` and add
        // a `(Scope::PgPin, "pg_type")` arm.
        _ => {}
    }
}
