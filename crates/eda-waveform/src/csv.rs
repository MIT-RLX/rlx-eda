//! Minimal CSV writer / reader for `Waveform`.
//!
//! Keep it tiny — one column per signal, header row, no quoting (signal
//! names are SPICE node names, never contain commas). Real waveforms
//! get one column per signal; complex waveforms get `<name>_re` and
//! `<name>_im` columns. cicwave reads these directly via pandas.
//!
//! No dependency on a CSV crate: this is a few hundred bytes of code
//! and the format is constrained. If we ever need quoting / escaping,
//! pull in `csv` then.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};

use thiserror::Error;

use crate::Waveform;

#[derive(Debug, Error)]
pub enum CsvError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("empty file")]
    Empty,
    #[error("malformed row {row} (expected {expected} columns, got {got})")]
    MalformedRow { row: usize, expected: usize, got: usize },
    #[error("could not parse number {tok:?} at row {row}, column {col}")]
    BadNumber { tok: String, row: usize, col: usize },
}

/// Write a waveform as CSV. First column is the axis; remaining columns
/// are signals in the iteration order of the underlying `BTreeMap`
/// (alphabetical, deterministic).
pub fn write<W: Write>(wave: &Waveform, mut w: W) -> Result<(), CsvError> {
    match wave {
        Waveform::Real { axis_name, axis, signals } => {
            write!(w, "{}", axis_name)?;
            for name in signals.keys() {
                write!(w, ",{}", name)?;
            }
            writeln!(w)?;
            for (i, t) in axis.iter().enumerate() {
                write!(w, "{:.10e}", t)?;
                for samples in signals.values() {
                    write!(w, ",{:.10e}", samples[i])?;
                }
                writeln!(w)?;
            }
        }
        Waveform::Complex { axis_name, axis, signals } => {
            write!(w, "{}", axis_name)?;
            for name in signals.keys() {
                write!(w, ",{}_re,{}_im", name, name)?;
            }
            writeln!(w)?;
            for (i, f) in axis.iter().enumerate() {
                write!(w, "{:.10e}", f)?;
                for samples in signals.values() {
                    let (re, im) = samples[i];
                    write!(w, ",{:.10e},{:.10e}", re, im)?;
                }
                writeln!(w)?;
            }
        }
    }
    Ok(())
}

/// Read a real-valued waveform from CSV. Complex round-trip via
/// `_re`/`_im` columns is not implemented — we have one consumer
/// (cicwave-style import for transient traces) and CSV is a side-show
/// for tabular debugging, not the primary interchange.
pub fn read_real<R: Read>(r: R) -> Result<Waveform, CsvError> {
    let mut lines = BufReader::new(r).lines();
    let header = lines.next().ok_or(CsvError::Empty)??;
    let cols: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
    if cols.is_empty() {
        return Err(CsvError::Empty);
    }
    let axis_name = cols[0].clone();
    let signal_names: Vec<String> = cols[1..].to_vec();

    let mut axis: Vec<f64> = Vec::new();
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); signal_names.len()];
    for (row_idx, line) in lines.enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let toks: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if toks.len() != cols.len() {
            return Err(CsvError::MalformedRow {
                row: row_idx + 1,
                expected: cols.len(),
                got: toks.len(),
            });
        }
        let parse =
            |idx: usize, tok: &str| tok.parse::<f64>().map_err(|_| CsvError::BadNumber {
                tok: tok.to_string(),
                row: row_idx + 1,
                col: idx,
            });
        axis.push(parse(0, toks[0])?);
        for (j, &t) in toks[1..].iter().enumerate() {
            columns[j].push(parse(j + 1, t)?);
        }
    }
    let mut signals = BTreeMap::new();
    for (name, col) in signal_names.into_iter().zip(columns.into_iter()) {
        signals.insert(name, col);
    }
    Ok(Waveform::Real { axis_name, axis, signals })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Waveform {
        let mut signals = BTreeMap::new();
        signals.insert("v(out)".to_string(), vec![0.0, 0.5, 0.75]);
        Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1e-9, 2e-9],
            signals,
        }
    }

    #[test]
    fn round_trip_real() {
        let w = fixture();
        let mut buf = Vec::new();
        write(&w, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let back = read_real(s.as_bytes()).unwrap();
        assert_eq!(back.axis(), w.axis());
        assert_eq!(back.real("v(out)").unwrap(), w.real("v(out)").unwrap());
    }

    #[test]
    fn header_row_is_first() {
        let w = fixture();
        let mut buf = Vec::new();
        write(&w, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.starts_with("time,v(out)\n"));
    }
}
