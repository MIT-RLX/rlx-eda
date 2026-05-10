//! Placement — turn a [`Netlist`] into a per-instance `Trans`.
//!
//! The trait is one method, no associated types: hand a netlist
//! plus the library (so the placer can read each instance's bbox)
//! and get back a [`Placement`] holding one `Trans` per instance.
//! Two built-in placers ship: [`ManualPlacer`] (caller supplies
//! the transforms) and [`GridPlacer`] (auto-arrange in rows by
//! bbox). AD-driven placement lives in `crate::ad`.

use klayout_core::{Bbox, Library, Trans, Vec2};

use crate::netlist::Netlist;

/// Resulting placement: one `Trans` per instance, plus the
/// bounding box that contains every placed child's footprint
/// (excluding routing, which the router is allowed to extend
/// past the placement bbox).
#[derive(Clone, Debug)]
pub struct Placement {
    pub transforms: Vec<Trans>,
    pub bbox: Bbox,
}

pub trait Placer {
    fn place(&self, netlist: &Netlist, lib: &Library) -> Placement;
}

/// Caller hands in one `Trans` per instance. Mostly a way to lift
/// the existing per-spike `Layout::layout` "place children at
/// hand-picked offsets" pattern into the PNR pipeline without
/// changing the placement decisions.
#[derive(Clone, Debug)]
pub struct ManualPlacer {
    pub transforms: Vec<Trans>,
}

impl ManualPlacer {
    pub fn new(transforms: Vec<Trans>) -> Self { Self { transforms } }
}

impl Placer for ManualPlacer {
    fn place(&self, netlist: &Netlist, lib: &Library) -> Placement {
        assert_eq!(
            self.transforms.len(),
            netlist.instances.len(),
            "ManualPlacer: transforms must be 1:1 with netlist instances",
        );
        Placement {
            transforms: self.transforms.clone(),
            bbox: union_placed_bbox(netlist, lib, &self.transforms),
        }
    }
}

/// Lay each instance left-to-right, wrap to a new row after `cols`
/// columns, with `gap_x` / `gap_y` of empty space between bounding
/// boxes. Useful when the caller doesn't have a hand-picked
/// floorplan and just wants placement to "do something sensible"
/// while routing still works.
#[derive(Clone, Debug)]
pub struct GridPlacer {
    pub gap_x: i64,
    pub gap_y: i64,
    pub cols: usize,
}

impl Default for GridPlacer {
    fn default() -> Self { Self { gap_x: 5_000, gap_y: 5_000, cols: 4 } }
}

impl GridPlacer {
    pub fn new(gap_x: i64, gap_y: i64, cols: usize) -> Self {
        Self { gap_x, gap_y, cols: cols.max(1) }
    }
}

impl Placer for GridPlacer {
    fn place(&self, netlist: &Netlist, lib: &Library) -> Placement {
        let mut transforms = Vec::with_capacity(netlist.instances.len());
        let mut row_height: i64 = 0;
        let mut cur_x: i64 = 0;
        let mut cur_y: i64 = 0;
        let mut col_idx = 0usize;

        for inst in &netlist.instances {
            let cell = lib.get(inst.cell);
            let bbox = cell.local_bbox();
            // Translate so the cell's bbox.min lands at (cur_x, cur_y).
            let dx = cur_x - bbox.min.x;
            let dy = cur_y - bbox.min.y;
            transforms.push(Trans::translate(Vec2::new(dx, dy)));

            let w = bbox.width();
            let h = bbox.height();
            row_height = row_height.max(h);

            col_idx += 1;
            if col_idx >= self.cols {
                col_idx = 0;
                cur_x = 0;
                cur_y += row_height + self.gap_y;
                row_height = 0;
            } else {
                cur_x += w + self.gap_x;
            }
        }

        Placement {
            bbox: union_placed_bbox(netlist, lib, &transforms),
            transforms,
        }
    }
}

/// Compute the union bbox of every instance after the supplied
/// transforms are applied. Used by both built-in placers (and
/// available to custom placers via `pub`).
pub fn union_placed_bbox(netlist: &Netlist, lib: &Library, transforms: &[Trans]) -> Bbox {
    let mut acc: Option<Bbox> = None;
    for (inst, t) in netlist.instances.iter().zip(transforms.iter()) {
        let cell = lib.get(inst.cell);
        let local = cell.local_bbox();
        let placed = transform_bbox(local, *t);
        acc = Some(match acc {
            Some(a) => union(a, placed),
            None => placed,
        });
    }
    acc.unwrap_or_else(empty_bbox)
}

fn transform_bbox(b: Bbox, t: Trans) -> Bbox {
    t.apply_bbox(b)
}

fn union(a: Bbox, b: Bbox) -> Bbox {
    use klayout_core::Point;
    Bbox::new(
        Point::new(a.min.x.min(b.min.x), a.min.y.min(b.min.y)),
        Point::new(a.max.x.max(b.max.x), a.max.y.max(b.max.y)),
    )
}

fn empty_bbox() -> Bbox {
    use klayout_core::Point;
    Bbox::new(Point::new(0, 0), Point::new(0, 0))
}
