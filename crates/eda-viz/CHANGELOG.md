# Changelog — eda-viz

## 0.0.1 (in development)

Initial crate. Two renderers, plus a CLI binary, plus an integration
test that binds rendering to LVS verification so a connectivity-broken
layout can never silently slip into a demo.

### Layout renderer (`layout::render_to_svg`)
- Recursive cell-tree flattening with `Trans` composition.
- Per-layer `<g>` groups with palette-driven colors. Optional
  `LayerPalette` override (e.g., from a foundry `.lyp` file).
- `Shape::Box`, `Shape::Polygon`, `Shape::Path`, and `Shape::Text` all
  rendered.
- `Repetition::Regular` and `Repetition::Irregular` instance arrays
  flatten correctly (covered by `repetition_array_stamps_correctly`).
- Cycle guard in `collect` — defensive against self-referential cell
  graphs from buggy LIR transforms.
- `Style` knobs: `pad_dbu`, `units_per_dbu`, `stroke_width`,
  `fill_opacity`, `background`, `show_ports`, `show_legend`,
  `layer_palette`, `highlights`, `tooltips`, `hidden_layers`,
  `show_instance_labels`, `emit_css_classes`.
- Layer-color legend in a strip above the cell.
- DRC/LVS overlay via `Style::highlights: Vec<Highlight>` (with
  `highlights_from_drc` adapter under the `drc` feature).
- Hover tooltips via `<title>` children (opt-in `Style::tooltips`).
- CSS classes on layer groups (opt-in `Style::emit_css_classes`)
  for external stylesheet overrides.

### Schematic renderer (`schematic::render_to_svg`)
- `eda_hir::SchematicIr` is the canonical input. `from_ir` adapter
  translates IR symbols/wires/ports to the renderer's data shape.
- Symbols: Resistor, Capacitor, Diode, Vsource (with orientation-aware
  `+`/`−` markers), Ground, Nmos, Pmos, Subcircuit (labeled box with
  pin stubs).
- Junction dots at any point where ≥3 wire endpoints/pins converge —
  resolves T-vs-crossing schematic ambiguity.
- Per-symbol pin→net assignment via `SchemSymbol::pin_nets`. Used by
  netlist emitters and (in the future) for visual net highlighting.
- Lead-endpoint math (`body_half * unit(pin)`) so leads always meet
  body edges regardless of pin distance / body size.

### Waveform renderer (`waveform::render_to_svg`)
- Multi-trace XY plotter with auto-scaling axes, "nice" round-number
  tick spacing (1, 2, 2.5, 5 × 10^k), gridlines, legend, optional
  title, and clip-path so out-of-range traces don't bleed onto axes.

### `Schematic<P>` HIR trait
- New trait in `eda-hir` mirroring `Layout<P>`. A block declares once
  via Rust struct fields; both renderer paths read from the same
  source. `RcDivider::schematic` consumes `r1.length` / `r2.length`
  via `length_to_resistance` so resistor values stay in sync with the
  layout-side resistor lengths.
- Implementations in `spike-divider-block` for `Resistor`, `Capacitor`,
  `Diode`, `Mosfet`, `RcDivider`.

### CLI (`eda-viz` binary, `cli` feature)
- `eda-viz <input.gds | input.oas> [-o out.svg | out.png] [--cell <name>]`
- Auto-detects format by extension.
- Defaults output to `<input>.svg` next to the source.

### Test pyramid
- Unit-level: per-symbol pin-count smoke tests, repetition stamping.
- Integration: `renderer_runs_only_on_lvs_verified_layout` re-runs
  spike-divider-block's LVS check before rendering — a broken layout
  fails verification, not visualization.
- Foundry: `renders_under_sky130lite_pdk` exercises a real-PDK
  flow.
- Golden SVG snapshot tests for the canonical divider layout +
  schematic; renderer changes surface as a `git diff` of the golden.

### Bundled assets
- `assets/DejaVuSans.ttf` (public-domain DejaVu changes over Bitstream
  Vera) for deterministic PNG text rasterization.
- `assets/divider_layout.{svg,png}` and `divider_schematic.{svg,png}`
  for the README.

### Optional features
- `png` — `resvg` + `tiny-skia` + `usvg` for PNG rasterization.
- `cli` — `klayout-io` for GDS/OASIS reading; pulls `png`.
- `drc` — `klayout-drc` adapter for DRC violations → highlights.
- `svgz` — `flate2` for gzip-compressed SVG output.
