//! Parse `.lib <section>` headers out of a SPICE corner library.
//!
//! Format (sky130 example):
//!
//! ```text
//! * Typical corner (tt)
//! .lib tt
//! .param mc_mm_switch=0
//! .include "corners/tt.spice"
//! .endl tt
//! ```
//!
//! We only need the section names, not their contents. This is one cheap
//! scan; we ignore SPICE's case rules (they don't matter at this layer)
//! and bail at the first parse hiccup.

use std::path::Path;

/// Read `path` and return the ordered list of unique `.lib <name>`
/// section names. Names normalize to lower-case to match the convention
/// `cicsim` uses.
pub fn sections_from_lib(path: &Path) -> std::io::Result<Vec<String>> {
    let s = std::fs::read_to_string(path)?;
    Ok(parse_sections(&s))
}

pub fn parse_sections(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in text.lines() {
        let t = line.trim_start();
        // Two-arg `.lib <section>` is a section header. Single-arg
        // `.lib <path> <section>` (an *include*) is a different beast
        // and we want to skip it.
        let lower = t.to_lowercase();
        let after = if let Some(r) = lower.strip_prefix(".lib ") {
            r
        } else if let Some(r) = lower.strip_prefix(".lib\t") {
            r
        } else {
            continue;
        };
        let toks: Vec<&str> = after.split_whitespace().collect();
        if toks.len() != 1 {
            continue; // include form, skip
        }
        let name = toks[0].to_string();
        if !out.iter().any(|x| x == &name) {
            out.push(name);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sky130_style_corners() {
        let s = r#"
* preamble
.lib tt
.param mc_mm_switch=0
.endl tt

.lib ff
.endl ff

.lib ss
.endl ss
"#;
        assert_eq!(parse_sections(s), vec!["tt", "ff", "ss"]);
    }

    #[test]
    fn ignores_lib_include_form() {
        // `.lib <path> <section>` — caller is including a section.
        // We must not capture that as a header.
        let s = ".lib ../models/sky130.lib tt\n.lib tt\n.endl tt\n";
        assert_eq!(parse_sections(s), vec!["tt"]);
    }

    #[test]
    fn dedupes_repeated_sections() {
        let s = ".lib tt\n.endl tt\n.lib tt\n.endl tt\n";
        assert_eq!(parse_sections(s), vec!["tt"]);
    }

    #[test]
    fn empty_file_returns_empty() {
        assert!(parse_sections("").is_empty());
        assert!(parse_sections("* just a comment\n").is_empty());
    }
}
