//! `eda-trace` — domain-agnostic optimization-trace + report harness.
//!
//! Every spike crate's "ML trace" / "characterization" bin used to
//! re-implement the same skeleton: an Adam loop with per-step
//! logging, a hand-rolled `line_chart_svg` (because `eda-viz` lacks
//! log-y), an SVG-to-PNG rasterization step, a CSV writer, and a
//! markdown templater that embeds the resulting images. Three such
//! bins (`spike-waveguide-block::mzi_ml_trace`,
//! `spike-divider-block::ml_trace`, `spike-sar-adc::conversion_trace`)
//! convergently grew the same ~600-800 lines of glue. This crate
//! lifts that glue once.
//!
//! ## What it knows
//!
//! Three things, none of them domain-specific:
//!
//! 1. **A row-oriented trace** ([`Trace`], [`TraceRow`], [`OptStep`]) —
//!    a step counter plus a `BTreeMap<String, f64>` of named scalars
//!    per step. Loss, parameters, gradients, derived metrics
//!    (notch wavelength, return loss, gate fidelity, INL, …) are all
//!    just keys in the map. The "optimizer" is anything implementing
//!    [`OptStep`] — gradient descent, Adam, DADO, RL search, a
//!    deterministic device characterization pass, anything that
//!    produces one row per step.
//!
//! 2. **Declarative charts** ([`ChartSpec`], [`ChartKind`], [`YSeries`]) —
//!    pick which series go on the X axis, which go on Y, whether
//!    Y is log-scaled, line vs. scatter vs. before/after overlay. The
//!    harness renders to SVG (and PNG if the `png` feature is on)
//!    using its own line-chart implementation — so log-y, multi-series
//!    overlays, and custom titles all just work.
//!
//! 3. **A markdown report** ([`ReportMeta`], [`Reference`], [`Report::write`]) —
//!    the harness fills in the title, optimization outcome, a step-by-step
//!    table, and embeds every chart + the floorplan; the caller
//!    supplies the prose ("what is a Mach-Zehnder?", "why
//!    inductively-degenerated cascode?"), the [`Reference`] rows for
//!    literature validation, and any `extra_panels` the harness
//!    doesn't know how to make (Mermaid diagrams, FFT bin plots,
//!    Bloch-sphere snapshots).
//!
//! ## What it doesn't know
//!
//! - **Layouts** — floorplans arrive as SVG strings the caller
//!   pre-rendered (with `eda-viz::layout::render_to_svg` for
//!   klayout-backed designs, or anything else for non-layout
//!   domains). The harness embeds the SVG and rasterizes it; it
//!   never imports klayout.
//! - **Simulators** — no SPICE, no ngspice, no rlx. The optimizer
//!   closure does whatever it does and returns a row.
//! - **Domains** — [`Domain`] is just a tag rendered in the
//!   markdown header. Nothing in the harness branches on it.
//!
//! ## Minimal use
//!
//! ```ignore
//! use eda_trace::{Trace, TraceCfg, TraceRow, ChartSpec, ChartKind,
//!                 ReportMeta, Domain, Report, FloorplanSource};
//!
//! let mut state = MyOptimizer::init();
//! let cfg = TraceCfg { name: "my_run".into(), steps: 200, log_at: Default::default() };
//! let trace = Trace::run(&cfg, |step| {
//!     let (loss, grad) = state.step();
//!     TraceRow::new(step)
//!         .with("loss", loss)
//!         .with("grad", grad)
//!         .with("param", state.param)
//! });
//!
//! let charts = vec![
//!     ChartSpec::line("loss", "Loss", "step", "|loss|").with_y_log(true)
//!         .add_series("loss", "loss"),
//!     ChartSpec::line("param", "Param trajectory", "step", "param")
//!         .add_series("param", "param"),
//! ];
//! let meta = ReportMeta::new("My run", "MyCircuit", Domain::Rf);
//! Report { trace: &trace, meta: &meta, charts: &charts,
//!          floorplan: FloorplanSource::None,
//!          extra_panels: vec![] }
//!     .write("crates/my-spike/docs/my_run").unwrap();
//! ```

pub mod charts;
pub mod optim;
pub mod report;
pub mod trace;

pub use charts::{render_chart_svg, ChartKind, ChartSpec, YSeries};
pub use optim::{default_device, default_device_or_env, AdamState, BetaSchedule, LrSchedule};
pub use report::{
    Domain, FloorplanSource, Panel, PanelBody, Reference, Report, ReportMeta,
};
pub use trace::{LogSchedule, OptStep, Trace, TraceCfg, TraceRow};
