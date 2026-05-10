# eda-tile

Pitch-matched abutment + power-rail helpers. Lets a `Block` declare
itself a `Tile` (fixed pitch, declared power rails, declared edge
ports), then composes a grid of tiles with guaranteed rail
continuity. Inter-tile bus routing delegates to
`klayout_route::ManhattanPlanner`.

Also carries a coarse current-density check on power straps —
PLAN.md cross-cutting requirement #4: single-tile sim won't catch
supply collapse when 256 MACs fire in lockstep.

Build-order step 2 in [`eda-bench-tinyconv/PLAN.md`](../eda-bench-tinyconv/PLAN.md).
