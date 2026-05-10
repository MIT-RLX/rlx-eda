//! SHA-of-deck cache for skip-rerun.
//!
//! cicsim hashes (deck + model timestamps + ngspice options) and skips
//! re-running when the hash matches the previous run. Two flags govern
//! behavior:
//!
//! - **`--no-sha`** (Force) — recompute even if the hash is unchanged.
//! - **`--no-run`** (ReuseOnly) — never re-run; load from the cache or
//!   error.
//!
//! Cache layout on disk:
//!
//! ```text
//! <cache_dir>/<sha>.deck          # the exact deck submitted to ngspice
//! <cache_dir>/<sha>.stdout        # ngspice stdout (for .meas parsing)
//! <cache_dir>/<sha>.raw           # nutmeg binary, when transient/ac
//! <cache_dir>/<sha>.meta.json     # corner label, analysis kind, etc.
//! ```
//!
//! Hashing inputs: the deck text + every `.lib` / `.include` target's
//! mtime as `(path, mtime_ns)` pairs. Mtime change ⇒ hash change ⇒
//! re-run. We don't read the lib files themselves; mtime is a strict
//! superset (any content edit bumps mtime).

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    /// Reuse hash matches; re-run on miss. Default.
    Auto,
    /// Recompute regardless of hash (cicsim `--no-sha`).
    Force,
    /// Never re-run; if the cache is empty for this hash, error
    /// (cicsim `--no-run`).
    ReuseOnly,
    /// Don't read or write the cache at all.
    Off,
}

#[derive(Debug, Clone)]
pub struct Cache {
    pub dir: PathBuf,
    pub mode: CacheMode,
}

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub sha: String,
    pub stdout_path: PathBuf,
    pub raw_path: PathBuf,
    pub deck_path: PathBuf,
    pub meta_path: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("ReuseOnly mode: no cache entry for sha {0}")]
    NoEntry(String),
}

impl Cache {
    pub fn new(dir: impl Into<PathBuf>, mode: CacheMode) -> Self {
        Self { dir: dir.into(), mode }
    }

    pub fn off() -> Self {
        Self { dir: PathBuf::from("."), mode: CacheMode::Off }
    }

    /// Compute the cache SHA for `(deck, lib_paths)`. Lib *contents*
    /// are folded in as `<basename>:<sha256>` lines after the deck so
    /// the cache key:
    ///
    /// - survives `~/.volare → /opt/pdks/sky130A` path moves (we don't
    ///   commit the absolute path, only the basename)
    /// - invalidates on actual content changes (sky130 PDK upgrade)
    /// - tolerates mtime drift (rsync, git checkout) without a re-run
    ///
    /// Reading multi-megabyte SPICE libs every call would dominate
    /// runtime, so we memoize content hashes by `(path, mtime)` in a
    /// process-wide cache. The mtime is part of the *cache key only* —
    /// the SHA itself is over file bytes.
    pub fn compute_sha(deck: &str, lib_paths: &[&Path]) -> String {
        let mut h = Sha256::new();
        h.update(deck.as_bytes());
        h.update(b"\n--libs--\n");
        for p in lib_paths {
            let basename = p.file_name().and_then(|s| s.to_str()).unwrap_or("?");
            let content_sha = lib_content_sha(p);
            h.update(basename.as_bytes());
            h.update(b":");
            h.update(content_sha.as_bytes());
            h.update(b"\n");
        }
        let digest = h.finalize();
        let mut s = String::with_capacity(64);
        for b in digest.iter() {
            use std::fmt::Write;
            let _ = write!(s, "{:02x}", b);
        }
        s
    }

    pub fn entry(&self, sha: &str) -> CacheEntry {
        Self::entry_for_sha(&self.dir, sha)
    }

    fn entry_for_sha(dir: &Path, sha: &str) -> CacheEntry {
        CacheEntry {
            sha: sha.to_string(),
            stdout_path: dir.join(format!("{sha}.stdout")),
            raw_path: dir.join(format!("{sha}.raw")),
            deck_path: dir.join(format!("{sha}.deck")),
            meta_path: dir.join(format!("{sha}.meta.json")),
        }
    }

    /// True when a cached stdout exists *and* the mode permits reuse.
    pub fn can_reuse(&self, entry: &CacheEntry) -> bool {
        match self.mode {
            CacheMode::Off | CacheMode::Force => false,
            CacheMode::Auto | CacheMode::ReuseOnly => entry.stdout_path.exists(),
        }
    }

    /// Persist the run artifacts. No-op when mode is `Off`.
    pub fn store(
        &self,
        entry: &CacheEntry,
        deck: &str,
        stdout: &str,
        raw_bytes: Option<&[u8]>,
        meta_json: &str,
    ) -> Result<(), CacheError> {
        if self.mode == CacheMode::Off {
            return Ok(());
        }
        std::fs::create_dir_all(&self.dir)?;
        std::fs::write(&entry.deck_path, deck)?;
        std::fs::write(&entry.stdout_path, stdout)?;
        std::fs::write(&entry.meta_path, meta_json)?;
        if let Some(b) = raw_bytes {
            std::fs::write(&entry.raw_path, b)?;
        }
        Ok(())
    }

    pub fn load_stdout(&self, entry: &CacheEntry) -> Result<String, CacheError> {
        Ok(std::fs::read_to_string(&entry.stdout_path)?)
    }

    pub fn load_raw(&self, entry: &CacheEntry) -> Option<Vec<u8>> {
        std::fs::read(&entry.raw_path).ok()
    }
}

/// Process-wide memo for `(path, mtime) → sha256(content)`. SPICE libs
/// are megabytes; hashing them every cache lookup is wasteful when 99%
/// of runs see the exact same library file. Mtime is the cheap probe;
/// the actual hash is over file bytes.
static LIB_HASH_CACHE: Mutex<Option<HashMap<(PathBuf, u128), String>>> = Mutex::new(None);

fn lib_content_sha(path: &Path) -> String {
    let mtime_key: u128 = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let key = (path.to_path_buf(), mtime_key);

    if let Ok(mut guard) = LIB_HASH_CACHE.lock() {
        let map = guard.get_or_insert_with(HashMap::new);
        if let Some(s) = map.get(&key) { return s.clone(); }
        // Cache miss: read + hash. We read in 1 MiB chunks to avoid
        // gobbling RAM on a >100 MB lib. SPICE libs are usually <30 MB
        // but PDKs that ship resistor/cap families can balloon.
        let sha = match std::fs::File::open(path) {
            Ok(mut f) => {
                use std::io::Read;
                let mut h = Sha256::new();
                let mut buf = [0u8; 1 << 20];
                loop {
                    match f.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => h.update(&buf[..n]),
                        Err(_) => return "io-error".into(),
                    }
                }
                let digest = h.finalize();
                let mut s = String::with_capacity(64);
                for b in digest.iter() {
                    use std::fmt::Write;
                    let _ = write!(s, "{:02x}", b);
                }
                s
            }
            Err(_) => "missing".into(),
        };
        map.insert(key, sha.clone());
        sha
    } else {
        "lock-poisoned".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn sha_is_stable_for_same_inputs() {
        let s1 = Cache::compute_sha("Vdd vdd 0 1.8\n", &[]);
        let s2 = Cache::compute_sha("Vdd vdd 0 1.8\n", &[]);
        assert_eq!(s1, s2);
        assert_eq!(s1.len(), 64);
    }

    #[test]
    fn sha_changes_with_deck() {
        let s1 = Cache::compute_sha("Vdd vdd 0 1.8\n", &[]);
        let s2 = Cache::compute_sha("Vdd vdd 0 3.3\n", &[]);
        assert_ne!(s1, s2);
    }

    #[test]
    fn sha_changes_with_lib_content() {
        // Distinct files (different mtime keys) with different bytes
        // ⇒ different cache key — verifies content hashing without
        // depending on mtime granularity.
        let dir = tempfile::tempdir().unwrap();
        let lib_a = dir.path().join("a.lib");
        let lib_b = dir.path().join("b.lib");
        std::fs::write(&lib_a, "version-1-content").unwrap();
        std::fs::write(&lib_b, "completely-different-content").unwrap();
        let sa = Cache::compute_sha("deck", &[&lib_a]);
        let sb = Cache::compute_sha("deck", &[&lib_b]);
        assert_ne!(sa, sb);
    }

    #[test]
    fn sha_survives_path_move_with_same_content() {
        // Two libs with the SAME content but different paths/basenames
        // — still get different keys (basename participates in the
        // hash). This is intentional: a `tt` section in one PDK must
        // not collide with a `tt` section in another PDK with the same
        // bytes by coincidence.
        let dir = tempfile::tempdir().unwrap();
        let lib_a = dir.path().join("vendor_a/sky130.lib.spice");
        let lib_b = dir.path().join("vendor_b/sky130.lib.spice");
        std::fs::create_dir_all(lib_a.parent().unwrap()).unwrap();
        std::fs::create_dir_all(lib_b.parent().unwrap()).unwrap();
        std::fs::write(&lib_a, ".lib tt\nM1 d g s b nmos\n.endl tt").unwrap();
        std::fs::write(&lib_b, ".lib tt\nM1 d g s b nmos\n.endl tt").unwrap();
        let sa = Cache::compute_sha("deck", &[&lib_a]);
        let sb = Cache::compute_sha("deck", &[&lib_b]);
        // Same basename + same content ⇒ same hash (path-portable).
        assert_eq!(sa, sb);
    }

    #[test]
    fn store_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path(), CacheMode::Auto);
        let sha = Cache::compute_sha("hello", &[]);
        let entry = cache.entry(&sha);
        cache.store(&entry, "hello", "stdout-text", None, "{}").unwrap();
        assert!(cache.can_reuse(&entry));
        assert_eq!(cache.load_stdout(&entry).unwrap(), "stdout-text");
    }

    #[test]
    fn force_mode_skips_reuse() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path(), CacheMode::Force);
        let sha = Cache::compute_sha("hello", &[]);
        let entry = cache.entry(&sha);
        cache.store(&entry, "hello", "out", None, "{}").unwrap();
        assert!(!cache.can_reuse(&entry));
    }

    #[test]
    fn off_mode_does_not_write() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path(), CacheMode::Off);
        let sha = Cache::compute_sha("hello", &[]);
        let entry = cache.entry(&sha);
        cache.store(&entry, "hello", "out", None, "{}").unwrap();
        assert!(!entry.stdout_path.exists());
    }
}
