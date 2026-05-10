//! `tile_grid` — compose an `nx × ny` array of one tile type into a
//! parent cell with rail continuity and edge-port abutment enforced.
//!
//! Self-composition (one tile type tiled into a uniform grid) makes
//! the abutment checks trivial: every neighbour is the same tile, so
//! "do edges line up?" reduces to "is `edge_ports(N) == edge_ports(S)`
//! and `edge_ports(E) == edge_ports(W)`?". A future heterogeneous
//! composer would need pair-wise checks; we don't have a use case
//! yet.
//!
//! Inter-tile bus routing (signals that cross more than one tile) is
//! delegated to `klayout_route::ManhattanPlanner`. Not done by this
//! function — it's the consumer's job once the grid is laid out.

use klayout_core::{CellBuilder, CellId, Library, Trans, Vec2};

use crate::pdn::current_density_check;
use crate::tile::{Side, Tile};

#[derive(Debug, thiserror::Error)]
pub enum GridError {
    #[error("rail mismatch: {0}")]
    RailMismatch(String),
    #[error("edge port mismatch on {side:?}: {detail}")]
    EdgePortMismatch { side: Side, detail: String },
    #[error("PDN: {0}")]
    Pdn(#[from] crate::pdn::PdnError),
}

/// Composer config for the per-strap PDN check. Caller passes the
/// per-tile peak current and the PDK's Jmax so we can run
/// `current_density_check` on the rail straps that result from
/// abutting `nx` tiles in a row (or `ny` in a column).
#[derive(Debug, Clone, Copy)]
pub struct PdnCheck {
    pub per_tile_peak_ma: f64,
    pub jmax_ma_per_um: f64,
}

/// Build an `nx × ny` grid of `tile` into `lib`. Validates abutment
/// + (optionally) PDN at compose time, then instantiates the tile
/// at every `(i*pitch.x, j*pitch.y)` position into a fresh parent
/// cell named `<tile.name()>__grid_<nx>x<ny>`.
pub fn tile_grid<T, P>(
    tile: &T,
    nx: usize,
    ny: usize,
    lib: &Library,
    pdk: &P,
    pdn: Option<PdnCheck>,
) -> Result<CellId, GridError>
where
    T: Tile<P>,
{
    // ── Abutment contract ────────────────────────────────────────
    // Self-composition: vertical neighbours need N == S, horizontal
    // neighbours need E == W (same offsets, same names, same layers).
    if nx > 1 {
        check_pair(tile, pdk, Side::West, Side::East)?;
    }
    if ny > 1 {
        check_pair(tile, pdk, Side::North, Side::South)?;
    }

    // ── Rail self-consistency ────────────────────────────────────
    // Same-tile composition trivially satisfies "rails align across
    // edges" because all rails are identical. Surface as an
    // explicit check so any future per-row variation surfaces here.
    let rails = tile.rails(pdk);
    if rails.vdd_tracks.is_empty() && rails.gnd_tracks.is_empty() {
        return Err(GridError::RailMismatch(
            "tile declares no rails — uniform grid would have no power".into(),
        ));
    }

    // ── PDN check (optional) ─────────────────────────────────────
    if let Some(p) = pdn {
        // Worst case: longest strap = max(nx, ny) tiles per power
        // line. Caller can split the check by orientation if rails
        // are direction-specific later.
        let strap_len = nx.max(ny);
        current_density_check(&rails, p.per_tile_peak_ma, strap_len, p.jmax_ma_per_um)?;
    }

    // ── Compose ──────────────────────────────────────────────────
    let pitch = tile.pitch();
    let child_id = tile.layout(lib, pdk);

    let mut parent = CellBuilder::new(format!("{}__grid_{}x{}", tile.name(), nx, ny));
    for j in 0..ny {
        for i in 0..nx {
            let dx = (i as i64) * pitch.x;
            let dy = (j as i64) * pitch.y;
            parent.instantiate(child_id, Trans::translate(Vec2::new(dx, dy)));
        }
    }
    Ok(lib.insert(parent))
}

fn check_pair<T, P>(tile: &T, pdk: &P, lhs: Side, rhs: Side) -> Result<(), GridError>
where
    T: Tile<P>,
{
    let l = tile.edge_ports(lhs, pdk);
    let r = tile.edge_ports(rhs, pdk);
    if l.len() != r.len() {
        return Err(GridError::EdgePortMismatch {
            side: lhs,
            detail: format!("{lhs:?} has {} ports, {rhs:?} has {}", l.len(), r.len()),
        });
    }
    for (a, b) in l.iter().zip(r.iter()) {
        if a.offset_dbu != b.offset_dbu || a.layer != b.layer {
            return Err(GridError::EdgePortMismatch {
                side: lhs,
                detail: format!(
                    "{} ({}) ↔ {} ({}): offset {} vs {}, layer mismatch={}",
                    a.name,
                    format!("{lhs:?}"),
                    b.name,
                    format!("{rhs:?}"),
                    a.offset_dbu,
                    b.offset_dbu,
                    a.layer != b.layer,
                ),
            });
        }
    }
    Ok(())
}
