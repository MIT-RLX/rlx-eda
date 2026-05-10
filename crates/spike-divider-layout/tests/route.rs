//! Tier 2: routing produces the expected centerline path. Reuse the same
//! `ManhattanPlanner` the spike uses; assert points + widths match the
//! geometry we set up.

use klayout_core::{Angle90, LayerIndex, LayerInfo, Library, Point, Port};
use klayout_route::{ManhattanPlanner, Obstacles, Planner};

fn metal1(lib: &Library) -> LayerIndex {
    lib.layer(LayerInfo::named("METAL1", 10, 0))
}

#[test]
fn east_west_facing_ports_with_y_offset_yield_l_bend() {
    let lib = Library::new("rt", 1000);
    let m1 = metal1(&lib);

    let src = Port::new("s", m1, Point::new(11_000, 500),  Angle90::E, 2_000);
    let dst = Port::new("d", m1, Point::new(14_000, -2_500), Angle90::W, 2_000);

    let path = ManhattanPlanner.plan(&src, &dst, &Obstacles::default());

    // Three points: src, elbow at (dst.x, src.y), dst. Width = max of port widths.
    assert_eq!(path.points.len(), 3, "expected L-bend, got {:?}", path.points);
    assert_eq!(path.points[0], Point::new(11_000, 500));
    assert_eq!(path.points[1], Point::new(14_000, 500));
    assert_eq!(path.points[2], Point::new(14_000, -2_500));
    assert_eq!(path.width, 2_000);
}

#[test]
fn straight_line_when_aligned() {
    let lib = Library::new("rt", 1000);
    let m1 = metal1(&lib);

    let src = Port::new("s", m1, Point::new(0, 0),     Angle90::E, 100);
    let dst = Port::new("d", m1, Point::new(1_000, 0), Angle90::W, 100);

    let path = ManhattanPlanner.plan(&src, &dst, &Obstacles::default());

    // Elbow == dst.center, so collapses to two points.
    assert_eq!(path.points.len(), 2, "expected straight line, got {:?}", path.points);
    assert_eq!(path.points[0], Point::new(0, 0));
    assert_eq!(path.points[1], Point::new(1_000, 0));
}
