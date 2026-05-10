//! TOML registry at `~/.config/rlx-eda/pdks.toml`.
//!
//! Schema:
//!
//! ```toml
//! [[pdk]]
//! name = "sky130A"
//! lib_path = "/Users/me/.ciel/sky130/sky130A/libs.tech/ngspice/sky130.lib.spice"
//! sections = ["tt", "ff", "ss", "fs", "sf"]
//! vdd_nom = 1.8
//! source = "ciel"          # "ciel" or "user"
//! ```
//!
//! `source = "ciel"` is informational — re-running `pdk list` re-derives
//! ciel-managed entries from the ciel root anyway. `source = "user"`
//! entries are *only* in the registry, so manually-registered PDKs
//! survive across upgrades.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::Error;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Ciel,
    User,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdkEntry {
    pub name: String,
    pub lib_path: PathBuf,
    pub sections: Vec<String>,
    #[serde(default = "default_vdd")]
    pub vdd_nom: f64,
    #[serde(default = "default_source")]
    pub source: Source,
}

fn default_vdd() -> f64 { 1.8 }
fn default_source() -> Source { Source::User }

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Registry {
    #[serde(default, rename = "pdk")]
    pub entries: Vec<PdkEntry>,
}

impl Registry {
    pub fn config_path() -> Result<PathBuf, Error> {
        if let Ok(p) = std::env::var("RLX_EDA_PDK_CONFIG") {
            return Ok(PathBuf::from(p));
        }
        let base = if let Ok(p) = std::env::var("XDG_CONFIG_HOME") {
            PathBuf::from(p)
        } else if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(home).join(".config")
        } else {
            return Err(Error::NoConfigHome);
        };
        Ok(base.join("rlx-eda").join("pdks.toml"))
    }

    /// Load the registry; return an empty one when the file is missing.
    pub fn load_or_default() -> Result<Self, Error> {
        let p = Self::config_path()?;
        if !p.exists() { return Ok(Self::default()); }
        let s = std::fs::read_to_string(&p)?;
        Ok(toml::from_str(&s)?)
    }

    pub fn save(&self) -> Result<(), Error> {
        let p = Self::config_path()?;
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let s = toml::to_string_pretty(self)?;
        std::fs::write(&p, s)?;
        Ok(())
    }

    pub fn find(&self, name: &str) -> Option<&PdkEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Insert or update. Match key is the entry name.
    pub fn upsert(&mut self, entry: PdkEntry) {
        if let Some(slot) = self.entries.iter_mut().find(|e| e.name == entry.name) {
            *slot = entry;
        } else {
            self.entries.push(entry);
        }
    }

    pub fn remove(&mut self, name: &str) -> bool {
        let n = self.entries.len();
        self.entries.retain(|e| e.name != name);
        n != self.entries.len()
    }
}

/// Helper for tests: load from a specific path bypassing the env lookup.
#[cfg(test)]
pub fn load_from(p: &std::path::Path) -> Result<Registry, Error> {
    if !p.exists() { return Ok(Registry::default()); }
    let s = std::fs::read_to_string(p)?;
    Ok(toml::from_str(&s)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_one_entry() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("pdks.toml");
        let mut reg = Registry::default();
        reg.upsert(PdkEntry {
            name: "sky130A".into(),
            lib_path: PathBuf::from("/x/sky130.lib.spice"),
            sections: vec!["tt".into(), "ff".into()],
            vdd_nom: 1.8,
            source: Source::Ciel,
        });
        let s = toml::to_string_pretty(&reg).unwrap();
        std::fs::write(&cfg, &s).unwrap();
        let loaded = load_from(&cfg).unwrap();
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].name, "sky130A");
        assert_eq!(loaded.entries[0].sections, vec!["tt", "ff"]);
    }

    #[test]
    fn upsert_replaces_same_name() {
        let mut reg = Registry::default();
        reg.upsert(PdkEntry {
            name: "x".into(), lib_path: PathBuf::from("/a"),
            sections: vec!["tt".into()], vdd_nom: 1.0, source: Source::User,
        });
        reg.upsert(PdkEntry {
            name: "x".into(), lib_path: PathBuf::from("/b"),
            sections: vec!["ff".into()], vdd_nom: 2.0, source: Source::User,
        });
        assert_eq!(reg.entries.len(), 1);
        assert_eq!(reg.entries[0].lib_path, PathBuf::from("/b"));
    }

    #[test]
    fn remove_is_idempotent() {
        let mut reg = Registry::default();
        reg.upsert(PdkEntry {
            name: "x".into(), lib_path: PathBuf::from("/a"),
            sections: vec!["tt".into()], vdd_nom: 1.0, source: Source::User,
        });
        assert!(reg.remove("x"));
        assert!(!reg.remove("x"));
        assert!(reg.entries.is_empty());
    }
}
