//! Value Change Dump (VCD) writer + reader for digital traces.
//!
//! Why VCD? Once we add SAR-logic blocks (DFFs, SAR registers, output
//! door), the natural debug view is bit-level: digital waveforms in
//! GTKWave / Surfer / cicwave's digital pane. Analog `.raw` shows
//! voltages; VCD shows bits. Same waveform, two views.
//!
//! ## Writer
//!
//! [`write_thresholded`] turns a `Waveform::Real` into a 1-bit-per-signal
//! VCD by comparing each sample against a threshold. Used to lift rlx-eda
//! transient outputs to a digital view.
//!
//! ### How thresholding works
//!
//! Analog → digital is `v >= threshold ? 1 : 0`. For 1V rail-to-rail
//! the natural threshold is 0.5V. This is what the LTspice paper's SAR
//! ADC uses (figure waveforms colored as 0/1 per bit).
//!
//! ## Reader
//!
//! [`read`] parses an external VCD (cocotb / Verilator / Surfer dumps,
//! plus our own writer's output) back into a `Waveform::Real`. Single
//! bits become 0.0/1.0; multi-bit busses become integer-valued reals;
//! `x`/`z` and friends become `NaN`. Scopes are flattened into dotted
//! signal names.
//!
//! Round-tripping our own writer is lossy on signal *names* — the
//! writer scrubs SPICE node names like `v(out)` into VCD-legal ids
//! like `v_out_` — but the values and time axis round-trip exactly.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use thiserror::Error;

use crate::Waveform;

#[derive(Debug, Error)]
pub enum VcdError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("waveform must be real-valued for VCD export")]
    NotReal,
    #[error("waveform has no samples")]
    Empty,
    #[error("malformed VCD: {0}")]
    Parse(String),
    #[error("value change references unknown identifier {0:?}")]
    UnknownId(String),
}

/// Convert a real waveform to a 1-bit-per-signal VCD by thresholding.
///
/// Time axis is rescaled into the VCD `$timescale` (we always use `1ps`
/// — picosecond resolution covers SAR ADC clocks well into the GHz).
pub fn write_thresholded<W: Write>(
    wave: &Waveform,
    threshold: f64,
    mut w: W,
) -> Result<(), VcdError> {
    let (axis_name, axis, signals) = match wave {
        Waveform::Real { axis_name, axis, signals } => (axis_name, axis, signals),
        Waveform::Complex { .. } => return Err(VcdError::NotReal),
    };
    if axis.is_empty() {
        return Err(VcdError::Empty);
    }

    writeln!(w, "$date rlx-eda generated $end")?;
    writeln!(w, "$version eda-waveform 0.0.1 $end")?;
    writeln!(w, "$timescale 1ps $end")?;
    writeln!(w, "$scope module {} $end", sanitize_id(axis_name))?;

    // Assign a single-character VCD identifier per signal. ASCII 33..127
    // is the legal range; we have at most ~94 signals before needing
    // multi-char ids — fine for SAR-scale designs.
    let mut id_for: BTreeMap<&str, char> = BTreeMap::new();
    for (i, name) in signals.keys().enumerate() {
        let c = char::from_u32(33 + i as u32).unwrap_or('!');
        id_for.insert(name.as_str(), c);
        writeln!(w, "$var wire 1 {} {} $end", c, sanitize_id(name))?;
    }
    writeln!(w, "$upscope $end")?;
    writeln!(w, "$enddefinitions $end")?;

    // Initial dump at t=0.
    writeln!(w, "$dumpvars")?;
    let mut last: BTreeMap<&str, u8> = BTreeMap::new();
    for (name, samples) in signals {
        let bit = if samples[0] >= threshold { 1 } else { 0 };
        last.insert(name.as_str(), bit);
        writeln!(w, "{}{}", bit, id_for[name.as_str()])?;
    }
    writeln!(w, "$end")?;

    // Subsequent ticks: only emit deltas.
    for i in 1..axis.len() {
        let t_ps = (axis[i] * 1e12).round() as i64;
        let mut header_written = false;
        for (name, samples) in signals {
            let bit = if samples[i] >= threshold { 1 } else { 0 };
            if last[name.as_str()] != bit {
                if !header_written {
                    writeln!(w, "#{}", t_ps)?;
                    header_written = true;
                }
                writeln!(w, "{}{}", bit, id_for[name.as_str()])?;
                *last.get_mut(name.as_str()).unwrap() = bit;
            }
        }
    }
    Ok(())
}

/// One multi-bit bus to emit. The bit signals must already exist in the
/// waveform; each is thresholded independently.
#[derive(Debug, Clone, Copy)]
pub struct BusSpec<'a> {
    /// Name of the bus signal in the VCD output (e.g. `code[3:0]`).
    pub name: &'a str,
    /// Analog signal names making up the bus, **MSB first**.
    pub bits: &'a [&'a str],
    /// Per-bus threshold (`v >= threshold ? 1 : 0`).
    pub threshold: f64,
}

/// Write a VCD with one or more multi-bit busses plus optional extra
/// single-bit signals.
///
/// Why this matters: a SAR ADC's natural output is one N-bit code, not N
/// independent bits. GTKWave / Surfer / cicwave render `b1010 #` as a
/// single bus row, which is what verification engineers want when they
/// scope through a conversion. Use [`write_thresholded`] when you want
/// every analog signal as its own bit; use this when you have a
/// known-grouped bus.
///
/// `extras` is `(signal_name, threshold)` for single-bit emissions
/// (clocks, control bits) alongside the busses.
///
/// Bus bit-signal names must exist in the waveform. The waveform's
/// time axis is reused as VCD ticks (1ps timescale).
pub fn write_buses<W: Write>(
    wave: &Waveform,
    buses: &[BusSpec],
    extras: &[(&str, f64)],
    mut w: W,
) -> Result<(), VcdError> {
    let (axis_name, axis, signals) = match wave {
        Waveform::Real {
            axis_name,
            axis,
            signals,
        } => (axis_name, axis, signals),
        Waveform::Complex { .. } => return Err(VcdError::NotReal),
    };
    if axis.is_empty() {
        return Err(VcdError::Empty);
    }

    // Validate bus bit names exist.
    for bus in buses {
        if bus.bits.is_empty() {
            return Err(VcdError::Parse(format!("bus {:?} has zero bits", bus.name)));
        }
        for bit in bus.bits {
            if !signals.contains_key(*bit) {
                return Err(VcdError::Parse(format!(
                    "bus {:?} references missing signal {:?}",
                    bus.name, bit
                )));
            }
        }
    }
    for (name, _) in extras {
        if !signals.contains_key(*name) {
            return Err(VcdError::Parse(format!("extra references missing signal {:?}", name)));
        }
    }

    writeln!(w, "$date rlx-eda generated $end")?;
    writeln!(w, "$version eda-waveform 0.0.1 $end")?;
    writeln!(w, "$timescale 1ps $end")?;
    writeln!(w, "$scope module {} $end", sanitize_id(axis_name))?;

    // Assign ids: busses first, then extras. Each gets a unique VCD id.
    let mut next_id: usize = 0;
    let mut bus_ids: Vec<String> = Vec::with_capacity(buses.len());
    for bus in buses {
        let id = make_id(next_id);
        next_id += 1;
        writeln!(
            w,
            "$var wire {} {} {} $end",
            bus.bits.len(),
            id,
            sanitize_id(bus.name)
        )?;
        bus_ids.push(id);
    }
    let mut extra_ids: Vec<String> = Vec::with_capacity(extras.len());
    for (name, _) in extras {
        let id = make_id(next_id);
        next_id += 1;
        writeln!(w, "$var wire 1 {} {} $end", id, sanitize_id(name))?;
        extra_ids.push(id);
    }
    writeln!(w, "$upscope $end")?;
    writeln!(w, "$enddefinitions $end")?;

    // Helper: bus value at sample i as MSB-first 0/1 string.
    let bus_at = |bus: &BusSpec, i: usize| -> String {
        let mut s = String::with_capacity(bus.bits.len());
        for bit in bus.bits {
            let v = signals[*bit][i];
            s.push(if v >= bus.threshold { '1' } else { '0' });
        }
        s
    };

    // Initial dump at t=0.
    writeln!(w, "$dumpvars")?;
    let mut last_bus: Vec<String> = Vec::with_capacity(buses.len());
    for (j, bus) in buses.iter().enumerate() {
        let bits = bus_at(bus, 0);
        writeln!(w, "b{} {}", bits, bus_ids[j])?;
        last_bus.push(bits);
    }
    let mut last_extra: Vec<u8> = Vec::with_capacity(extras.len());
    for (j, (name, threshold)) in extras.iter().enumerate() {
        let bit = if signals[*name][0] >= *threshold { 1 } else { 0 };
        writeln!(w, "{}{}", bit, extra_ids[j])?;
        last_extra.push(bit);
    }
    writeln!(w, "$end")?;

    // Deltas.
    for i in 1..axis.len() {
        let t_ps = (axis[i] * 1e12).round() as i64;
        let mut header_written = false;
        let emit_header = |w: &mut W, header_written: &mut bool| -> std::io::Result<()> {
            if !*header_written {
                writeln!(w, "#{}", t_ps)?;
                *header_written = true;
            }
            Ok(())
        };

        for (j, bus) in buses.iter().enumerate() {
            let bits = bus_at(bus, i);
            if bits != last_bus[j] {
                emit_header(&mut w, &mut header_written)?;
                writeln!(w, "b{} {}", bits, bus_ids[j])?;
                last_bus[j] = bits;
            }
        }
        for (j, (name, threshold)) in extras.iter().enumerate() {
            let bit = if signals[*name][i] >= *threshold { 1 } else { 0 };
            if bit != last_extra[j] {
                emit_header(&mut w, &mut header_written)?;
                writeln!(w, "{}{}", bit, extra_ids[j])?;
                last_extra[j] = bit;
            }
        }
    }
    Ok(())
}

/// Generate a printable VCD identifier for the n-th signal. Single-char
/// (ASCII 33..126, 94 codepoints) up to 93; then two-char, etc. We never
/// emit `$` so the id can't collide with directives.
fn make_id(mut n: usize) -> String {
    const LO: u32 = 33;  // '!'
    const HI: u32 = 126; // '~'
    const RANGE: u32 = HI - LO + 1; // 94
    let mut chars: Vec<char> = Vec::new();
    loop {
        let mut c = char::from_u32(LO + (n as u32) % RANGE).unwrap();
        if c == '$' {
            // Skip '$' to avoid colliding with VCD directives. Bump the
            // remainder by one (and carry up) if we land on it.
            c = char::from_u32(LO + (n as u32 + 1) % RANGE).unwrap();
        }
        chars.push(c);
        n /= RANGE as usize;
        if n == 0 {
            break;
        }
        n -= 1; // standard "spreadsheet column" base-N: A, B, .. Z, AA, AB
    }
    chars.iter().rev().collect()
}

/// Parse a VCD into a `Waveform::Real`.
///
/// - Single-bit values (`0`/`1`) become `0.0` / `1.0`.
/// - Multi-bit busses (`b1010 id`) become the integer value as `f64`.
/// - Real-valued changes (`r1.5 id`) pass through.
/// - `x`, `z`, `u`, `?`, `-` (and the `h`/`l`/`H`/`L` strong-drive variants
///   for high/low) all become `NaN` for busses with any unknown bits, and
///   for single bits anything outside `{0,1,h,H,l,L}` is `NaN`.
///
/// The axis is built from the union of timestamps observed in the dump;
/// each signal's value at axis[i] is its most recent value at or before
/// that timestamp (NaN if it was never assigned). Scopes are flattened
/// into dotted names — e.g. `top.dut.clk`.
pub fn read<R: Read>(mut r: R) -> Result<Waveform, VcdError> {
    let mut buf = String::new();
    r.read_to_string(&mut buf)?;
    let mut tokens = buf.split_ascii_whitespace();

    let mut timescale_seconds: f64 = 1e-12;
    let mut vars: Vec<VcdVar> = Vec::new();
    let mut scope_stack: Vec<String> = Vec::new();

    // Phase 1: declarations.
    loop {
        let tok = tokens
            .next()
            .ok_or_else(|| VcdError::Parse("unexpected EOF in header".into()))?;
        match tok {
            "$timescale" => {
                let mut s = String::new();
                for t in tokens.by_ref() {
                    if t == "$end" {
                        break;
                    }
                    s.push_str(t);
                }
                timescale_seconds = parse_timescale(&s)?;
            }
            "$scope" => {
                // $scope <kind> <name> $end
                let _kind = tokens
                    .next()
                    .ok_or_else(|| VcdError::Parse("$scope missing kind".into()))?;
                let name = tokens
                    .next()
                    .ok_or_else(|| VcdError::Parse("$scope missing name".into()))?;
                scope_stack.push(name.to_string());
                for t in tokens.by_ref() {
                    if t == "$end" {
                        break;
                    }
                }
            }
            "$upscope" => {
                scope_stack.pop();
                for t in tokens.by_ref() {
                    if t == "$end" {
                        break;
                    }
                }
            }
            "$var" => {
                // $var <kind> <width> <id> <name> [bit-select] $end
                let _kind = tokens
                    .next()
                    .ok_or_else(|| VcdError::Parse("$var missing kind".into()))?;
                let width: usize = tokens
                    .next()
                    .ok_or_else(|| VcdError::Parse("$var missing width".into()))?
                    .parse()
                    .map_err(|_| VcdError::Parse("$var bad width".into()))?;
                let id = tokens
                    .next()
                    .ok_or_else(|| VcdError::Parse("$var missing id".into()))?
                    .to_string();
                let name = tokens
                    .next()
                    .ok_or_else(|| VcdError::Parse("$var missing name".into()))?
                    .to_string();
                let mut suffix = String::new();
                for t in tokens.by_ref() {
                    if t == "$end" {
                        break;
                    }
                    suffix.push_str(t);
                }
                let leaf = if suffix.is_empty() {
                    name
                } else {
                    format!("{}{}", name, suffix)
                };
                let full = if scope_stack.is_empty() {
                    leaf
                } else {
                    format!("{}.{}", scope_stack.join("."), leaf)
                };
                vars.push(VcdVar { id, width, name: full });
            }
            "$enddefinitions" => {
                for t in tokens.by_ref() {
                    if t == "$end" {
                        break;
                    }
                }
                break;
            }
            // Header sections we don't care about — skip to $end.
            "$date" | "$version" | "$comment" => {
                for t in tokens.by_ref() {
                    if t == "$end" {
                        break;
                    }
                }
            }
            other if other.starts_with('$') => {
                for t in tokens.by_ref() {
                    if t == "$end" {
                        break;
                    }
                }
            }
            other => {
                return Err(VcdError::Parse(format!(
                    "unexpected token {:?} in header",
                    other
                )));
            }
        }
    }

    let mut id_to_idx: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, v) in vars.iter().enumerate() {
        id_to_idx.insert(v.id.as_str(), i);
    }

    // Phase 2: value changes. Snapshot strategy — accumulate value
    // changes into `current`, flush on each new `#timestamp` (and at
    // EOF) so each axis entry holds the post-update state.
    let n = vars.len();
    let mut current: Vec<f64> = vec![f64::NAN; n];
    let mut axis: Vec<f64> = Vec::new();
    let mut series: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut step_t_ticks: i64 = 0;
    let mut step_started = false;

    while let Some(tok) = tokens.next() {
        if let Some(rest) = tok.strip_prefix('#') {
            if step_started {
                axis.push(step_t_ticks as f64 * timescale_seconds);
                for (i, v) in current.iter().enumerate() {
                    series[i].push(*v);
                }
            }
            step_t_ticks = rest
                .parse::<i64>()
                .map_err(|_| VcdError::Parse(format!("bad timestamp #{}", rest)))?;
            step_started = true;
        } else if tok.starts_with('$') {
            match tok {
                // Open a value-change block at the current step.
                "$dumpvars" | "$dumpall" | "$dumpon" | "$dumpoff" => {
                    step_started = true;
                }
                "$end" => {}
                "$comment" => {
                    for t in tokens.by_ref() {
                        if t == "$end" {
                            break;
                        }
                    }
                }
                _ => {
                    for t in tokens.by_ref() {
                        if t == "$end" {
                            break;
                        }
                    }
                }
            }
        } else {
            apply_change(tok, &mut tokens, &mut current, &id_to_idx)?;
            step_started = true;
        }
    }
    if step_started {
        axis.push(step_t_ticks as f64 * timescale_seconds);
        for (i, v) in current.iter().enumerate() {
            series[i].push(*v);
        }
    }

    let mut signals = BTreeMap::new();
    for (v, samples) in vars.into_iter().zip(series.into_iter()) {
        signals.insert(v.name, samples);
    }
    Ok(Waveform::Real {
        axis_name: "time".into(),
        axis,
        signals,
    })
}

struct VcdVar {
    id: String,
    #[allow(dead_code)]
    width: usize,
    name: String,
}

fn apply_change<'a, I: Iterator<Item = &'a str>>(
    tok: &'a str,
    tokens: &mut I,
    current: &mut [f64],
    id_to_idx: &BTreeMap<&str, usize>,
) -> Result<(), VcdError> {
    let first = tok
        .chars()
        .next()
        .ok_or_else(|| VcdError::Parse("empty value-change token".into()))?;
    match first {
        '0' | '1' | 'x' | 'X' | 'z' | 'Z' | 'h' | 'H' | 'l' | 'L' | 'u' | 'U' | '-' | '?' => {
            let id = &tok[1..];
            let idx = *id_to_idx
                .get(id)
                .ok_or_else(|| VcdError::UnknownId(id.to_string()))?;
            current[idx] = bit_value(first);
        }
        'b' | 'B' => {
            let bits = &tok[1..];
            let id = tokens
                .next()
                .ok_or_else(|| VcdError::Parse("bus value missing identifier".into()))?;
            let idx = *id_to_idx
                .get(id)
                .ok_or_else(|| VcdError::UnknownId(id.to_string()))?;
            current[idx] = bus_value(bits);
        }
        'r' | 'R' => {
            let val = &tok[1..];
            let id = tokens
                .next()
                .ok_or_else(|| VcdError::Parse("real value missing identifier".into()))?;
            let idx = *id_to_idx
                .get(id)
                .ok_or_else(|| VcdError::UnknownId(id.to_string()))?;
            current[idx] = val
                .parse::<f64>()
                .map_err(|_| VcdError::Parse(format!("bad real value {:?}", val)))?;
        }
        _ => return Err(VcdError::Parse(format!("unexpected value token {:?}", tok))),
    }
    Ok(())
}

fn bit_value(c: char) -> f64 {
    match c {
        '0' | 'l' | 'L' => 0.0,
        '1' | 'h' | 'H' => 1.0,
        _ => f64::NAN,
    }
}

fn bus_value(bits: &str) -> f64 {
    if bits.is_empty() {
        return f64::NAN;
    }
    let mut acc: u64 = 0;
    for b in bits.chars() {
        acc = acc.wrapping_shl(1);
        match b {
            '0' => {}
            '1' => acc |= 1,
            // Any unknown bit poisons the whole bus.
            _ => return f64::NAN,
        }
    }
    acc as f64
}

/// Parse a `$timescale` body like `1ps`, `10 ns`, or `100  us`.
fn parse_timescale(s: &str) -> Result<f64, VcdError> {
    let s = s.trim();
    let split = s
        .find(|c: char| c.is_ascii_alphabetic())
        .ok_or_else(|| VcdError::Parse(format!("bad timescale {:?}", s)))?;
    let (num, unit) = s.split_at(split);
    let num: f64 = num
        .trim()
        .parse()
        .map_err(|_| VcdError::Parse(format!("bad timescale number {:?}", num)))?;
    let mul = match unit.trim() {
        "s" => 1.0,
        "ms" => 1e-3,
        "us" => 1e-6,
        "ns" => 1e-9,
        "ps" => 1e-12,
        "fs" => 1e-15,
        other => return Err(VcdError::Parse(format!("unknown timescale unit {:?}", other))),
    };
    Ok(num * mul)
}

/// VCD identifiers are restricted (no whitespace, parens, brackets,
/// equals). SPICE node names like `v(out)` need scrubbing.
fn sanitize_id(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '(' | ')' | '[' | ']' | ' ' | '\t' | '=' => '_',
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_header_and_initial_dump() {
        let mut signals = BTreeMap::new();
        signals.insert("v(out)".to_string(), vec![0.0, 1.0, 0.0]);
        let w = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1e-9, 2e-9],
            signals,
        };
        let mut buf = Vec::new();
        write_thresholded(&w, 0.5, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("$timescale 1ps $end"));
        assert!(s.contains("$var wire 1 ! v_out_ $end"));
        assert!(s.contains("$dumpvars\n0!\n$end"));
        // Transition at 1ns = 1000ps.
        assert!(s.contains("#1000\n1!"));
        // Transition back at 2ns.
        assert!(s.contains("#2000\n0!"));
    }

    #[test]
    fn rejects_complex() {
        let w = Waveform::Complex {
            axis_name: "frequency".into(),
            axis: vec![1e3],
            signals: BTreeMap::new(),
        };
        let mut buf = Vec::new();
        assert!(matches!(
            write_thresholded(&w, 0.5, &mut buf),
            Err(VcdError::NotReal)
        ));
    }

    #[test]
    fn round_trip_writer_to_reader() {
        let mut signals = BTreeMap::new();
        signals.insert("v(out)".to_string(), vec![0.0, 1.0, 0.0]);
        let w = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1e-9, 2e-9],
            signals,
        };
        let mut buf = Vec::new();
        write_thresholded(&w, 0.5, &mut buf).unwrap();
        let back = read(buf.as_slice()).unwrap();
        assert_eq!(back.axis(), &[0.0, 1e-9, 2e-9]);
        // Writer flattens "v(out)" → "v_out_" and prepends the axis-name
        // scope ("time"), so the readback name is "time.v_out_".
        let samples = back.real("time.v_out_").expect("readback signal missing");
        assert_eq!(samples, &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn reads_bus_and_real_values() {
        let src = "\
$timescale 10 ns $end
$scope module top $end
$var wire 4 # data[3:0] $end
$var real 64 % vref $end
$upscope $end
$enddefinitions $end
#0
b0000 #
r1.25 %
#1
b1010 #
r2.5 %
#2
bxxxx #
";
        let w = read(src.as_bytes()).unwrap();
        // 10ns timescale: tick i is i*10e-9 seconds.
        assert_eq!(w.axis(), &[0.0, 10e-9, 20e-9]);
        let data = w.real("top.data[3:0]").unwrap();
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 10.0); // 0b1010
        assert!(data[2].is_nan()); // unknown bus poisons the value
        let vref = w.real("top.vref").unwrap();
        assert_eq!(vref, &[1.25, 2.5, 2.5]); // last value carries forward
    }

    #[test]
    fn reads_single_bit_unknowns_and_strong_drive() {
        // x on a single bit → NaN; H/L → 1.0/0.0.
        let src = "\
$timescale 1ps $end
$var wire 1 ! a $end
$var wire 1 \" b $end
$enddefinitions $end
#0
x!
H\"
#100
1!
0\"
";
        let w = read(src.as_bytes()).unwrap();
        assert_eq!(w.axis(), &[0.0, 100e-12]);
        assert!(w.real("a").unwrap()[0].is_nan());
        assert_eq!(w.real("a").unwrap()[1], 1.0);
        assert_eq!(w.real("b").unwrap(), &[1.0, 0.0]);
    }

    #[test]
    fn reads_dumpvars_block() {
        let src = "\
$timescale 1ps $end
$var wire 1 ! clk $end
$enddefinitions $end
$dumpvars
0!
$end
#1000
1!
";
        let w = read(src.as_bytes()).unwrap();
        assert_eq!(w.axis(), &[0.0, 1e-9]);
        assert_eq!(w.real("clk").unwrap(), &[0.0, 1.0]);
    }

    #[test]
    fn bus_writer_round_trips_through_reader() {
        // 3-bit code: b2=MSB, b1, b0=LSB. Two transitions:
        //   t=0:    000
        //   t=1ns:  101 = 5
        //   t=2ns:  111 = 7
        let mut signals = BTreeMap::new();
        signals.insert("b0".to_string(), vec![0.0, 1.0, 1.0]);
        signals.insert("b1".to_string(), vec![0.0, 0.0, 1.0]);
        signals.insert("b2".to_string(), vec![0.0, 1.0, 1.0]);
        signals.insert("clk".to_string(), vec![0.0, 1.0, 0.0]);
        let wave = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1e-9, 2e-9],
            signals,
        };
        let buses = [BusSpec {
            name: "code[2:0]",
            bits: &["b2", "b1", "b0"],
            threshold: 0.5,
        }];
        let extras = [("clk", 0.5_f64)];
        let mut buf = Vec::new();
        write_buses(&wave, &buses, &extras, &mut buf).unwrap();

        // Output must declare a 3-bit wire for the bus.
        let s = std::str::from_utf8(&buf).unwrap();
        assert!(s.contains("$var wire 3 "));
        assert!(s.contains("$var wire 1 "));
        assert!(s.contains("b000 "));
        assert!(s.contains("b101 "));
        assert!(s.contains("b111 "));

        // Read back and check the bus came through as integer codes.
        // sanitize_id turns "code[2:0]" into "code_2:0_" since `[` and `]`
        // aren't VCD-legal.
        let back = read(buf.as_slice()).unwrap();
        let code = back.real("time.code_2:0_").expect("bus signal missing");
        assert_eq!(code, &[0.0, 5.0, 7.0]);
        let clk = back.real("time.clk").expect("clk missing");
        assert_eq!(clk, &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn bus_writer_rejects_missing_bit() {
        let mut signals = BTreeMap::new();
        signals.insert("b0".to_string(), vec![0.0]);
        let wave = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0],
            signals,
        };
        let buses = [BusSpec {
            name: "code",
            bits: &["b0", "b1"], // b1 doesn't exist
            threshold: 0.5,
        }];
        let mut buf = Vec::new();
        assert!(matches!(
            write_buses(&wave, &buses, &[], &mut buf),
            Err(VcdError::Parse(_))
        ));
    }

    #[test]
    fn make_id_avoids_dollar() {
        // None of the first 100 ids should equal "$".
        for n in 0..100 {
            assert_ne!(make_id(n), "$");
            // Also: ids must be non-empty and printable.
            let id = make_id(n);
            assert!(!id.is_empty());
            for c in id.chars() {
                assert!(c.is_ascii_graphic());
            }
        }
    }

    #[test]
    fn parses_timescales() {
        assert!((parse_timescale("1ps").unwrap() - 1e-12).abs() < 1e-24);
        assert!((parse_timescale("10 ns").unwrap() - 10e-9).abs() < 1e-18);
        assert!((parse_timescale("100  us").unwrap() - 100e-6).abs() < 1e-12);
        assert!(parse_timescale("1xs").is_err());
    }
}
