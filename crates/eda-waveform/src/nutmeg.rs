//! Parser for the Nutmeg "raw" format used by ngspice **and** LTspice
//! (ASCII + binary).
//!
//! ngspice emits this from a `.control` block via `write <path> <signals>`
//! (default binary; `set filetype=ascii` switches to ASCII). LTspice
//! emits the same format as `<deck>.raw` next to the input file when
//! invoked with `-b`; pass `-ascii` to force ASCII payload. Both land
//! here.
//!
//! We support both ASCII and binary: binary because that's what
//! ngspice/LTspice write by default and what the `cicwave` viewer
//! (`/Users/Shared/mtl/cicwave/src/cicwave/ngraw.py`) consumes — so test
//! fixtures stay openable in cicwave for visual debug. ASCII because
//! it's trivially embeddable as a string fixture in unit tests, and
//! because the LTspice driver pins ASCII for portability.
//!
//! ## Cross-simulator quirks
//!
//! - `Flags:` is `real` / `complex` for ngspice, but LTspice may append
//!   modifiers like `real forward` or `complex stepped`. We split on
//!   whitespace and accept the first token.
//! - LTspice may emit extra header lines (`Offset:`,
//!   `Backannotation:`, …) ngspice doesn't. Unknown headers are ignored.
//! - LTspice's *binary* payload uses single-precision for dependent
//!   values by default and double-precision for the time axis. The
//!   LTspice driver pins ASCII to sidestep this; full binary
//!   cross-sim support is a follow-on.
//!
//! Format layout is identical between the two modes up through the header;
//! only the values block differs (`Values:` vs `Binary:`).
//!
//! ## File layout
//!
//! ```text
//! Title: <...>
//! Date: <...>
//! Plotname: <Transient Analysis | AC Analysis | DC transfer characteristic | ...>
//! Flags: <real | complex>
//! No. Variables: <N>
//! No. Points:    <P>
//! Variables:
//!     0  time          time            (or "frequency", "v-sweep", ...)
//!     1  v(vout)       voltage
//!     ...
//! Values:
//!  0   <var0_value_at_point_0>
//!      <var1_value_at_point_0>
//!      ...
//!  1   <var0_value_at_point_1>
//!      <var1_value_at_point_1>
//!      ...
//! ```
//!
//! For `Flags: complex`, every value field is `<re>,<im>` (the independent
//! variable still gets a zero imaginary part). For `Flags: real`, values are
//! plain `f64` scalars.
//!
//! ## What this parser is not
//!
//! - No support for multiple plots concatenated in one file. ngspice only
//!   produces that when `write` is called more than once; we always issue
//!   one `write` per analysis.

use std::io::Write;

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NutmegFlavor {
    Real,
    Complex,
}

/// Parsed Nutmeg file.
///
/// Variable names appear in `var_names` in file order; index 0 is always the
/// independent variable (time, frequency, …). For real plots, `values[i]`
/// has length `n_points` of `f64`. For complex plots, `complex_values[i]`
/// has length `n_points` of `(re, im)`. Exactly one of `values` /
/// `complex_values` is populated according to `flavor`.
#[derive(Debug, Clone)]
pub struct NutmegPlot {
    pub plotname: String,
    pub flavor: NutmegFlavor,
    pub var_names: Vec<String>,
    pub values: Vec<Vec<f64>>,
    pub complex_values: Vec<Vec<(f64, f64)>>,
}

impl NutmegPlot {
    /// Lookup a variable by name. Names from ngspice are typically
    /// case-insensitive (`v(vout)` vs `V(VOUT)`); match accordingly.
    pub fn var_index(&self, name: &str) -> Option<usize> {
        let needle = name.to_lowercase();
        self.var_names.iter().position(|n| n.to_lowercase() == needle)
    }

    pub fn real_trace(&self, name: &str) -> Option<&[f64]> {
        let i = self.var_index(name)?;
        self.values.get(i).map(|v| v.as_slice())
    }

    pub fn complex_trace(&self, name: &str) -> Option<&[(f64, f64)]> {
        let i = self.var_index(name)?;
        self.complex_values.get(i).map(|v| v.as_slice())
    }
}

#[derive(Debug, Error)]
pub enum NutmegError {
    #[error("missing required header field: {0}")]
    MissingHeader(&'static str),
    #[error("unrecognized Flags value: {0}")]
    BadFlags(String),
    #[error("expected {expected} variables, found {found}")]
    VariableCountMismatch { expected: usize, found: usize },
    #[error("expected {expected} points, parsed {found}")]
    PointCountMismatch { expected: usize, found: usize },
    #[error("could not parse number {0:?}")]
    BadNumber(String),
    #[error("unexpected line in Values block: {0:?}")]
    UnexpectedValuesLine(String),
}

/// Parse an ASCII Nutmeg dump.
pub fn parse(input: &str) -> Result<NutmegPlot, NutmegError> {
    parse_bytes(input.as_bytes())
}

/// Parse an ASCII or binary Nutmeg dump from raw bytes. The header is read
/// as text up through either `Values:\n` (ASCII) or `Binary:\n` (binary);
/// the trailing payload is parsed accordingly.
pub fn parse_bytes(input: &[u8]) -> Result<NutmegPlot, NutmegError> {
    let (header_end, kind) = find_payload_marker(input)?;
    let header = std::str::from_utf8(&input[..header_end])
        .map_err(|e| NutmegError::BadNumber(format!("non-UTF8 header: {e}")))?;
    let payload = &input[header_end..];

    let parsed_hdr = parse_header(header)?;
    let HeaderFields { plotname, flavor, n_vars, n_points, var_names } = parsed_hdr;

    let (values, complex_values) = match kind {
        PayloadKind::Ascii => parse_ascii_values(payload, flavor, n_vars, n_points)?,
        PayloadKind::Binary => parse_binary_values(payload, flavor, n_vars, n_points)?,
    };

    Ok(NutmegPlot {
        plotname,
        flavor,
        var_names,
        values: if flavor == NutmegFlavor::Real { values } else { Vec::new() },
        complex_values: if flavor == NutmegFlavor::Complex { complex_values } else { Vec::new() },
    })
}

/// Write a NutmegPlot in ASCII format. Round-trips through [`parse`].
///
/// Closes the loop: rlx-eda's native solver can dump a `.raw` that
/// LTspice / ngspice / cicwave can ingest, the same way this crate
/// reads their output. The independent variable is whichever entry of
/// `var_names` is at index 0 (per Nutmeg convention) — `time`,
/// `frequency`, `v-sweep`, etc. The plot is auto-typed (`time` →
/// "Transient Analysis", `frequency` → "AC Analysis", anything else →
/// "DC transfer characteristic") if `plot.plotname` is empty.
///
/// Binary writer is a follow-on; ASCII is the safe interchange format
/// because it's simulator-version-independent (binary RAW has variants
/// across LTspice IV vs XVII and across ngspice flavors).
pub fn write_ascii<W: Write>(plot: &NutmegPlot, mut w: W) -> Result<(), NutmegError> {
    let n_vars = plot.var_names.len();
    let n_points = match plot.flavor {
        NutmegFlavor::Real => plot.values.first().map(|v| v.len()).unwrap_or(0),
        NutmegFlavor::Complex => plot.complex_values.first().map(|v| v.len()).unwrap_or(0),
    };

    // Sanity check: every series matches n_points so we can iterate by point.
    match plot.flavor {
        NutmegFlavor::Real => {
            if plot.values.len() != n_vars {
                return Err(NutmegError::VariableCountMismatch {
                    expected: n_vars,
                    found: plot.values.len(),
                });
            }
            for v in &plot.values {
                if v.len() != n_points {
                    return Err(NutmegError::PointCountMismatch {
                        expected: n_points,
                        found: v.len(),
                    });
                }
            }
        }
        NutmegFlavor::Complex => {
            if plot.complex_values.len() != n_vars {
                return Err(NutmegError::VariableCountMismatch {
                    expected: n_vars,
                    found: plot.complex_values.len(),
                });
            }
            for v in &plot.complex_values {
                if v.len() != n_points {
                    return Err(NutmegError::PointCountMismatch {
                        expected: n_points,
                        found: v.len(),
                    });
                }
            }
        }
    }

    let plotname = if plot.plotname.is_empty() {
        infer_plotname(&plot.var_names)
    } else {
        plot.plotname.clone()
    };
    let flags = match plot.flavor {
        NutmegFlavor::Real => "real",
        NutmegFlavor::Complex => "complex",
    };

    write_io(&mut w, format_args!("Title: rlx-eda waveform export\n"))?;
    write_io(&mut w, format_args!("Plotname: {plotname}\n"))?;
    write_io(&mut w, format_args!("Flags: {flags}\n"))?;
    write_io(&mut w, format_args!("No. Variables: {n_vars}\n"))?;
    write_io(&mut w, format_args!("No. Points: {n_points}\n"))?;
    write_io(&mut w, format_args!("Variables:\n"))?;
    for (i, name) in plot.var_names.iter().enumerate() {
        let unit = unit_for(name, i == 0, &plotname);
        write_io(&mut w, format_args!("\t{i}\t{name}\t{unit}\n"))?;
    }
    write_io(&mut w, format_args!("Values:\n"))?;
    match plot.flavor {
        NutmegFlavor::Real => {
            for pt in 0..n_points {
                for var in 0..n_vars {
                    if var == 0 {
                        write_io(&mut w, format_args!(" {pt}\t{:.15e}\n", plot.values[var][pt]))?;
                    } else {
                        write_io(&mut w, format_args!("\t{:.15e}\n", plot.values[var][pt]))?;
                    }
                }
                write_io(&mut w, format_args!("\n"))?;
            }
        }
        NutmegFlavor::Complex => {
            for pt in 0..n_points {
                for var in 0..n_vars {
                    let (re, im) = plot.complex_values[var][pt];
                    if var == 0 {
                        write_io(&mut w, format_args!(" {pt}\t{:.15e},{:.15e}\n", re, im))?;
                    } else {
                        write_io(&mut w, format_args!("\t{:.15e},{:.15e}\n", re, im))?;
                    }
                }
                write_io(&mut w, format_args!("\n"))?;
            }
        }
    }
    Ok(())
}

fn write_io<W: Write>(w: &mut W, args: std::fmt::Arguments) -> Result<(), NutmegError> {
    w.write_fmt(args).map_err(|e| NutmegError::BadNumber(format!("io: {e}")))
}

/// Heuristic Plotname when the caller didn't set one. Matches what
/// ngspice / LTspice put in their own headers.
fn infer_plotname(var_names: &[String]) -> String {
    match var_names.first().map(|s| s.to_lowercase()).as_deref() {
        Some("time") => "Transient Analysis".into(),
        Some("frequency") => "AC Analysis".into(),
        _ => "DC transfer characteristic".into(),
    }
}

/// Unit string in the `Variables:` section. Cosmetic — parsers ignore
/// it (we read only `toks[1]` which is the name) but cicwave / GTKWave
/// honor it for axis labels.
fn unit_for(name: &str, is_independent: bool, plotname: &str) -> &'static str {
    if is_independent {
        if plotname.contains("Transient") { "time" }
        else if plotname.contains("AC") { "frequency" }
        else { "voltage" }
    } else if name.starts_with("v(") || name.starts_with("V(") {
        "voltage"
    } else if name.starts_with("i(") || name.starts_with("I(") {
        "current"
    } else {
        "voltage"
    }
}

#[derive(Debug, Clone, Copy)]
enum PayloadKind { Ascii, Binary }

/// Return `(byte_offset_just_past_marker_line, kind)`.
fn find_payload_marker(input: &[u8]) -> Result<(usize, PayloadKind), NutmegError> {
    // Marker is one of `Values:\n` / `Binary:\n` at the start of a line.
    // ngspice always writes them lowercase-leading-uppercase; we accept
    // either case to be safe (the python ref lowercases for comparison).
    let mut start = 0;
    while start < input.len() {
        let nl = input[start..].iter().position(|&b| b == b'\n').map(|p| start + p).unwrap_or(input.len());
        let line = &input[start..nl];
        let trimmed = trim_ascii(line);
        if eq_ignore_ascii_case(trimmed, b"Values:") {
            return Ok((nl + 1, PayloadKind::Ascii));
        }
        if eq_ignore_ascii_case(trimmed, b"Binary:") {
            return Ok((nl + 1, PayloadKind::Binary));
        }
        start = nl + 1;
    }
    Err(NutmegError::MissingHeader("Values/Binary"))
}

fn trim_ascii(s: &[u8]) -> &[u8] {
    let mut a = 0;
    let mut b = s.len();
    while a < b && s[a].is_ascii_whitespace() { a += 1; }
    while b > a && s[b - 1].is_ascii_whitespace() { b -= 1; }
    &s[a..b]
}

fn eq_ignore_ascii_case(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.eq_ignore_ascii_case(y))
}

struct HeaderFields {
    plotname: String,
    flavor: NutmegFlavor,
    n_vars: usize,
    n_points: usize,
    var_names: Vec<String>,
}

fn parse_header(header: &str) -> Result<HeaderFields, NutmegError> {
    let mut plotname = None;
    let mut flavor = None;
    let mut n_vars: Option<usize> = None;
    let mut n_points: Option<usize> = None;
    let mut var_names: Vec<String> = Vec::new();

    let mut lines = header.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_end();
        if let Some(rest) = strip_prefix_ci(trimmed, "Plotname:") {
            plotname = Some(rest.trim().to_string());
        } else if let Some(rest) = strip_prefix_ci(trimmed, "Flags:") {
            // LTspice may append modifiers (`real forward`, `complex
            // stepped`); the flavor is always the first whitespace token.
            let first = rest.split_whitespace().next().unwrap_or("");
            flavor = Some(match first.to_ascii_lowercase().as_str() {
                "real" => NutmegFlavor::Real,
                "complex" => NutmegFlavor::Complex,
                _ => return Err(NutmegError::BadFlags(rest.trim().to_string())),
            });
        } else if let Some(rest) = strip_prefix_ci(trimmed, "No. Variables:") {
            n_vars = rest.trim().parse::<usize>().ok();
        } else if let Some(rest) = strip_prefix_ci(trimmed, "No. Points:") {
            n_points = rest.trim().parse::<usize>().ok();
        } else if trimmed.eq_ignore_ascii_case("Variables:") {
            let nv = n_vars.ok_or(NutmegError::MissingHeader("No. Variables"))?;
            for _ in 0..nv {
                let v = lines.next().ok_or(NutmegError::VariableCountMismatch {
                    expected: nv,
                    found: var_names.len(),
                })?;
                let toks: Vec<&str> = v.split_whitespace().collect();
                if toks.len() < 3 {
                    return Err(NutmegError::VariableCountMismatch {
                        expected: nv,
                        found: var_names.len(),
                    });
                }
                var_names.push(toks[1].to_string());
            }
        }
    }

    let flavor = flavor.ok_or(NutmegError::MissingHeader("Flags"))?;
    let n_vars = n_vars.ok_or(NutmegError::MissingHeader("No. Variables"))?;
    let n_points = n_points.ok_or(NutmegError::MissingHeader("No. Points"))?;
    if var_names.len() != n_vars {
        return Err(NutmegError::VariableCountMismatch { expected: n_vars, found: var_names.len() });
    }
    Ok(HeaderFields {
        plotname: plotname.unwrap_or_default(),
        flavor,
        n_vars,
        n_points,
        var_names,
    })
}

fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() < prefix.len() { return None; }
    let head = &s[..prefix.len()];
    if head.eq_ignore_ascii_case(prefix) { Some(&s[prefix.len()..]) } else { None }
}

fn parse_ascii_values(
    payload: &[u8],
    flavor: NutmegFlavor,
    n_vars: usize,
    n_points: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<(f64, f64)>>), NutmegError> {
    let s = std::str::from_utf8(payload)
        .map_err(|e| NutmegError::BadNumber(format!("non-UTF8 ASCII payload: {e}")))?;

    let mut values: Vec<Vec<f64>> = vec![Vec::with_capacity(n_points); n_vars];
    let mut complex_values: Vec<Vec<(f64, f64)>> = vec![Vec::with_capacity(n_points); n_vars];

    let mut lines = s.lines();
    let mut points_seen = 0usize;
    let mut buf: Vec<&str> = Vec::with_capacity(n_vars);
    while points_seen < n_points {
        buf.clear();
        while buf.len() < n_vars {
            let line = lines.next().ok_or(NutmegError::PointCountMismatch {
                expected: n_points,
                found: points_seen,
            })?;
            if line.trim().is_empty() {
                continue;
            }
            buf.push(line);
        }
        let head = buf[0];
        let head_val = head
            .split_whitespace()
            .nth(1)
            .ok_or_else(|| NutmegError::UnexpectedValuesLine(head.to_string()))?;
        push_value(flavor, head_val, &mut values[0], &mut complex_values[0])?;
        for var_idx in 1..n_vars {
            let tok = buf[var_idx]
                .split_whitespace()
                .next()
                .ok_or_else(|| NutmegError::UnexpectedValuesLine(buf[var_idx].to_string()))?;
            push_value(flavor, tok, &mut values[var_idx], &mut complex_values[var_idx])?;
        }
        points_seen += 1;
    }
    Ok((values, complex_values))
}

fn parse_binary_values(
    payload: &[u8],
    flavor: NutmegFlavor,
    n_vars: usize,
    n_points: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<(f64, f64)>>), NutmegError> {
    let scalars_per_value = match flavor {
        NutmegFlavor::Real => 1,
        NutmegFlavor::Complex => 2,
    };
    let need = n_vars * n_points * scalars_per_value * 8;
    if payload.len() < need {
        return Err(NutmegError::PointCountMismatch {
            expected: n_points,
            found: payload.len() / (n_vars * scalars_per_value * 8),
        });
    }

    let mut values: Vec<Vec<f64>> = vec![Vec::with_capacity(n_points); n_vars];
    let mut complex_values: Vec<Vec<(f64, f64)>> = vec![Vec::with_capacity(n_points); n_vars];

    // ngspice writes point-major: for each point, n_vars values back-to-back.
    let mut cursor = 0usize;
    for _pt in 0..n_points {
        for var_idx in 0..n_vars {
            match flavor {
                NutmegFlavor::Real => {
                    let v = read_f64_le(&payload[cursor..cursor + 8]);
                    values[var_idx].push(v);
                    cursor += 8;
                }
                NutmegFlavor::Complex => {
                    let re = read_f64_le(&payload[cursor..cursor + 8]);
                    let im = read_f64_le(&payload[cursor + 8..cursor + 16]);
                    complex_values[var_idx].push((re, im));
                    cursor += 16;
                }
            }
        }
    }
    Ok((values, complex_values))
}

fn read_f64_le(bytes: &[u8]) -> f64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    f64::from_le_bytes(buf)
}

fn push_value(
    flavor: NutmegFlavor,
    tok: &str,
    real_buf: &mut Vec<f64>,
    complex_buf: &mut Vec<(f64, f64)>,
) -> Result<(), NutmegError> {
    match flavor {
        NutmegFlavor::Real => {
            let v: f64 = tok.parse().map_err(|_| NutmegError::BadNumber(tok.to_string()))?;
            real_buf.push(v);
        }
        NutmegFlavor::Complex => {
            let mut parts = tok.split(',');
            let re = parts
                .next()
                .ok_or_else(|| NutmegError::BadNumber(tok.to_string()))?
                .parse::<f64>()
                .map_err(|_| NutmegError::BadNumber(tok.to_string()))?;
            let im = parts
                .next()
                .ok_or_else(|| NutmegError::BadNumber(tok.to_string()))?
                .parse::<f64>()
                .map_err(|_| NutmegError::BadNumber(tok.to_string()))?;
            complex_buf.push((re, im));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_FIXTURE: &str = "\
Title: * probe
Date: ignored
Plotname: Transient Analysis
Flags: real
No. Variables: 3
No. Points: 3
Variables:
\t0\ttime\ttime
\t1\tv(vin)\tvoltage
\t2\tv(vout)\tvoltage
Values:
 0\t0.000000000000000e+00
\t1.000000000000000e+00
\t0.000000000000000e+00

 1\t1.000000000000000e-09
\t1.000000000000000e+00
\t5.000000000000000e-01

 2\t2.000000000000000e-09
\t1.000000000000000e+00
\t7.500000000000000e-01
";

    const COMPLEX_FIXTURE: &str = "\
Title: * probe ac
Plotname: AC Analysis
Flags: complex
No. Variables: 2
No. Points: 2
Variables:
\t0\tfrequency\tfrequency\tgrid=3
\t1\tv(vout)\tvoltage
Values:
 0\t1.000000000000000e+03,0.000000000000000e+00
\t9.000000000000000e-01,-1.000000000000000e-01

 1\t1.000000000000000e+04,0.000000000000000e+00
\t5.000000000000000e-01,-5.000000000000000e-01
";

    #[test]
    fn parses_real_transient() {
        let plot = parse(REAL_FIXTURE).unwrap();
        assert_eq!(plot.flavor, NutmegFlavor::Real);
        assert_eq!(plot.var_names, vec!["time", "v(vin)", "v(vout)"]);
        assert_eq!(plot.values.len(), 3);
        assert_eq!(plot.values[0], vec![0.0, 1e-9, 2e-9]);
        assert_eq!(plot.values[2], vec![0.0, 0.5, 0.75]);
    }

    #[test]
    fn parses_complex_ac() {
        let plot = parse(COMPLEX_FIXTURE).unwrap();
        assert_eq!(plot.flavor, NutmegFlavor::Complex);
        assert_eq!(plot.var_names, vec!["frequency", "v(vout)"]);
        let f = plot.complex_trace("frequency").unwrap();
        assert_eq!(f, &[(1e3, 0.0), (1e4, 0.0)]);
        let v = plot.complex_trace("v(vout)").unwrap();
        assert_eq!(v, &[(0.9, -0.1), (0.5, -0.5)]);
    }

    #[test]
    fn case_insensitive_lookup() {
        let plot = parse(REAL_FIXTURE).unwrap();
        let v = plot.real_trace("V(VOUT)").unwrap();
        assert_eq!(v, &[0.0, 0.5, 0.75]);
    }

    #[test]
    fn missing_flags_errors() {
        let s = "Plotname: x\nNo. Variables: 1\nNo. Points: 0\nVariables:\n\t0\tt\ttime\nValues:\n";
        assert!(matches!(parse(s), Err(NutmegError::MissingHeader("Flags"))));
    }

    #[test]
    fn parses_binary_real() {
        // Hand-built binary plot: 2 vars (time, v), 3 points, real f64s.
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(
            b"Plotname: Transient Analysis\n\
              Flags: real\n\
              No. Variables: 2\n\
              No. Points: 3\n\
              Variables:\n\
              \t0\ttime\ttime\n\
              \t1\tv(out)\tvoltage\n\
              Binary:\n",
        );
        for &(t, v) in &[(0.0_f64, 0.0_f64), (1e-9, 0.5), (2e-9, 0.75)] {
            bytes.extend_from_slice(&t.to_le_bytes());
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let plot = parse_bytes(&bytes).unwrap();
        assert_eq!(plot.flavor, NutmegFlavor::Real);
        assert_eq!(plot.real_trace("time").unwrap(), &[0.0, 1e-9, 2e-9]);
        assert_eq!(plot.real_trace("v(out)").unwrap(), &[0.0, 0.5, 0.75]);
    }

    #[test]
    fn parses_ltspice_multi_word_flags() {
        // LTspice stamps `Flags: real forward` on transients. We only
        // care about the first token.
        let s = "\
Title: * test
Plotname: Transient Analysis
Flags: real forward
No. Variables: 2
No. Points: 1
Variables:
\t0\ttime\ttime
\t1\tv(out)\tvoltage
Values:
 0\t0.000000000000000e+00
\t1.500000000000000e+00
";
        let plot = parse(s).unwrap();
        assert_eq!(plot.flavor, NutmegFlavor::Real);
        assert_eq!(plot.real_trace("v(out)").unwrap(), &[1.5]);
    }

    #[test]
    fn parses_ltspice_extra_headers() {
        // LTspice often emits headers ngspice doesn't (`Offset:`,
        // `Backannotation:`). They should be silently ignored.
        let s = "\
Title: * test
Date: Fri Jan 01 12:00:00 2026
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 1
Offset: 0.0000000000000000e+000
Command: Linear Technology Corporation LTspice XVII
Backannotation:
Variables:
\t0\ttime\ttime
\t1\tv(out)\tvoltage
Values:
 0\t0.000000000000000e+00
\t2.000000000000000e+00
";
        let plot = parse(s).unwrap();
        assert_eq!(plot.real_trace("v(out)").unwrap(), &[2.0]);
    }

    #[test]
    fn parses_binary_complex() {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(
            b"Plotname: AC Analysis\n\
              Flags: complex\n\
              No. Variables: 2\n\
              No. Points: 2\n\
              Variables:\n\
              \t0\tfrequency\tfrequency\n\
              \t1\tv(out)\tvoltage\n\
              Binary:\n",
        );
        for &((fr, fi), (vr, vi)) in &[
            ((1e3_f64, 0.0_f64), (0.9_f64, -0.1_f64)),
            ((1e4, 0.0), (0.5, -0.5)),
        ] {
            bytes.extend_from_slice(&fr.to_le_bytes());
            bytes.extend_from_slice(&fi.to_le_bytes());
            bytes.extend_from_slice(&vr.to_le_bytes());
            bytes.extend_from_slice(&vi.to_le_bytes());
        }
        let plot = parse_bytes(&bytes).unwrap();
        assert_eq!(plot.flavor, NutmegFlavor::Complex);
        assert_eq!(plot.complex_trace("v(out)").unwrap(), &[(0.9, -0.1), (0.5, -0.5)]);
    }

    #[test]
    fn write_ascii_round_trips_real() {
        let original = NutmegPlot {
            plotname: String::new(), // exercise plotname inference
            flavor: NutmegFlavor::Real,
            var_names: vec!["time".into(), "v(out)".into(), "v(mid)".into()],
            values: vec![
                vec![0.0, 1e-9, 2e-9, 3e-9],
                vec![0.0, 0.5, 0.75, 0.9],
                vec![0.0, 0.25, 0.5, 0.6],
            ],
            complex_values: vec![],
        };
        let mut buf = Vec::new();
        write_ascii(&original, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("Plotname: Transient Analysis"));
        assert!(s.contains("Flags: real"));

        let back = parse(&s).unwrap();
        assert_eq!(back.flavor, NutmegFlavor::Real);
        assert_eq!(back.var_names, original.var_names);
        for (a, b) in back.values.iter().zip(original.values.iter()) {
            assert_eq!(a.len(), b.len());
            for (av, bv) in a.iter().zip(b.iter()) {
                assert!((av - bv).abs() < 1e-12, "{av} vs {bv}");
            }
        }
    }

    #[test]
    fn write_ascii_round_trips_complex() {
        let original = NutmegPlot {
            plotname: String::new(),
            flavor: NutmegFlavor::Complex,
            var_names: vec!["frequency".into(), "v(out)".into()],
            values: vec![],
            complex_values: vec![
                vec![(1e3, 0.0), (1e4, 0.0), (1e5, 0.0)],
                vec![(0.9, -0.1), (0.5, -0.5), (0.1, -0.9)],
            ],
        };
        let mut buf = Vec::new();
        write_ascii(&original, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("Plotname: AC Analysis"));
        assert!(s.contains("Flags: complex"));

        let back = parse(&s).unwrap();
        assert_eq!(back.flavor, NutmegFlavor::Complex);
        let v = back.complex_trace("v(out)").unwrap();
        assert!((v[0].0 - 0.9).abs() < 1e-12 && (v[0].1 + 0.1).abs() < 1e-12);
        assert!((v[2].0 - 0.1).abs() < 1e-12 && (v[2].1 + 0.9).abs() < 1e-12);
    }

    #[test]
    fn write_ascii_rejects_inconsistent_lengths() {
        let bad = NutmegPlot {
            plotname: String::new(),
            flavor: NutmegFlavor::Real,
            var_names: vec!["time".into(), "v(out)".into()],
            values: vec![vec![0.0, 1.0], vec![0.0, 1.0, 2.0]], // mismatch
            complex_values: vec![],
        };
        let mut buf = Vec::new();
        let res = write_ascii(&bad, &mut buf);
        assert!(matches!(res, Err(NutmegError::PointCountMismatch { .. })));
    }
}
