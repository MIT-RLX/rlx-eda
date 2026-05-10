//! Failure bisection — when accuracy regresses on a PVT corner,
//! localize the divergence so the contributor knows which backend
//! lied, at which layer, and at which validation level.
//!
//! Cross-cutting requirement #5 in PLAN.md.
//!
//! v1 granularity = per-backend, first-divergent-layer (the
//! `divergence_first_layer` field on `Functional`). Image- and
//! tile-level localization arrives when per-image activation
//! checkpoints get added to `Functional` (or to a sibling type) —
//! the `Divergence` struct already holds places for those.

use crate::metrics::functional::{Functional, Level};

/// One divergence event. Lower-cardinality fields are populated in
/// v1; per-image / per-tile / max_abs_err fields stay `None` /
/// `0.0` until activation checkpoints land.
#[derive(Debug, Clone, PartialEq)]
pub struct Divergence {
    pub backend: &'static str,
    pub level: Level,
    pub layer: usize,
    pub image: Option<u32>,
    pub tile: Option<(usize, usize)>,
    pub max_abs_err: f64,
}

/// For each `(backend, Functional)` pair whose run reports
/// `divergence_first_layer = Some(_)`, emit a `Divergence` record.
/// Backends that match the reference (no divergence) produce no
/// records. Output order follows input order so the report stays
/// deterministic.
pub fn bisect(runs: &[(&'static str, Functional)]) -> Vec<Divergence> {
    runs.iter()
        .filter_map(|(backend, run)| {
            run.divergence_first_layer.map(|layer| Divergence {
                backend,
                level: run.level,
                layer,
                image: None,
                tile: None,
                max_abs_err: 0.0,
            })
        })
        .collect()
}
