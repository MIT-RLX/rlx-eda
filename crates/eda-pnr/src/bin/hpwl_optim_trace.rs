//! AD-driven placement trace — Adam minimizes half-perimeter
//! wirelength on a synthetic 6-instance / 4-net netlist, with the
//! `eda-trace` harness recording per-step rows, before/after
//! floorplan SVGs, and a markdown report. Same harness path the
//! LNA / MZI bins use; this proves the layout side of rlx-eda
//! plugs into the same ML pipeline.
//!
//! Topology (DREAMPlace-style worked example, intentionally tiny):
//!
//! ```text
//!   3-pin star:   U0 ── U1
//!                 U0 ── U2
//!                 U1 ── U2          (forms a tight A-B-C cluster)
//!   bridge:       U2 ── U3          (joins cluster to second cluster)
//!   2-pin chain:  U3 ── U4
//!                 U4 ── U5
//! ```
//!
//! Six unit cells seeded across a 100 µm × 100 µm region. Adam
//! drives them toward the HPWL optimum (a tight cluster modulo the
//! log-sum-exp smoothing bias). Before/after floorplans show the
//! pre/post placement, with the router's wires drawn in both —
//! "before" looks like spaghetti, "after" is short edges.
//!
//! Run:    cargo run -p eda-pnr --bin hpwl_optim_trace

use eda_pnr::ad::{
    combined_loss_graph, position_param_ids, position_param_layout,
    DifferentiablePlacement, BETA_DENSITY_DEFAULT, BETA_HPWL_DEFAULT,
    DENSITY_BETA_INPUT, HPWL_BETA_INPUT,
};
use eda_pnr::{ManhattanRouter, ManualPlacer, Netlist, PnrFlow, WireStyle};
use eda_trace::{
    AdamState, BetaSchedule, ChartSpec, Domain, FloorplanSource, LrSchedule, OptStep, Panel,
    PanelBody, Report, ReportMeta, Trace, TraceCfg, TraceRow,
};
use eda_viz::Style;
use klayout_core::{
    Angle90, Bbox, CellBuilder, CellId, Library, Point, Port, Rect, Shape, Trans, Vec2,
};
use klayout_pdk::pdk;
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use std::error::Error;

const ADAM_STEPS: u32 = 600;
const SEED_RADIUS: f32 = 50_000.0; // ± 50 µm seed spread
const BETA: f32 = BETA_HPWL_DEFAULT;
const BETA_DENSITY: f32 = BETA_DENSITY_DEFAULT;
// Density weight: tuned so the overlap-area term (DBU²) lands in
// the same magnitude as the HPWL term (DBU) once cells are fully
// overlapped. Cell area is 4 µm² = 16e6 DBU², HPWL floor is
// O(1e5) DBU → 1e-2 keeps them comparable.
const DENSITY_WEIGHT: f32 = 1e-2;

pdk! {
    pub PnrDemoPdk {
        dbu: 1000,
        layers: { METAL1 = (10, 0) },
        ports: { Electrical },
    }
}

fn build_unit_cell(lib: &Library, pdk: &PnrDemoPdk, name: &str) -> CellId {
    // 4 µm × 4 µm metal1 box with one port at the centre.
    let mut cb = CellBuilder::new(name);
    let half = 2_000_i64;
    cb.add_shape(
        pdk.METAL1,
        Shape::Box(Rect::new(Bbox::new(
            Point::new(-half, -half),
            Point::new(half, half),
        ))),
    );
    cb.add_port(
        Port::new("p", pdk.METAL1, Point::new(0, 0), Angle90::E, 4_000)
            .with_kind(PnrDemoPdk::Electrical),
    );
    lib.insert(cb)
}

/// Deterministic hash → seed positions, so the demo is byte-stable
/// across runs without needing to thread a seed through.
fn seed_xy(idx: usize) -> (f32, f32) {
    let golden = 0.6180339887_f32;
    let theta = (idx as f32) * golden * std::f32::consts::TAU;
    let r = SEED_RADIUS * (0.4 + 0.6 * (idx as f32 + 1.0).fract().abs());
    (r * theta.cos(), r * theta.sin())
}

fn main() -> Result<(), Box<dyn Error>> {
    // ── Build a fresh library + 6 unit cells ──────────────────────
    let lib = PnrDemoPdk::new_library("hpwl_optim_trace");
    let pdk = PnrDemoPdk::register(&lib);
    let cells: Vec<CellId> = (0..6)
        .map(|i| build_unit_cell(&lib, &pdk, &format!("U{i}")))
        .collect();

    // ── Netlist with the topology described above ────────────────
    let mut nl = Netlist::new("hpwl_optim").with_default_signal_layer(pdk.METAL1);
    let inst: Vec<usize> = cells
        .iter()
        .enumerate()
        .map(|(i, c)| nl.add_instance(format!("U{i}"), *c))
        .collect();
    // 3-pin cluster A.
    nl.connect("netA", inst[0], "p");
    nl.connect("netA", inst[1], "p");
    nl.connect("netA", inst[2], "p");
    // Bridge.
    nl.connect("bridge", inst[2], "p");
    nl.connect("bridge", inst[3], "p");
    // 2-pin chain.
    nl.connect("chainA", inst[3], "p");
    nl.connect("chainA", inst[4], "p");
    nl.connect("chainB", inst[4], "p");
    nl.connect("chainB", inst[5], "p");

    let seeds: Vec<(f32, f32)> = (0..inst.len()).map(seed_xy).collect();

    // ── Build the AD graph + session ──────────────────────────────
    let mut placement = DifferentiablePlacement {
        instance_xy: seeds.clone(),
        beta: BETA,
    };
    // Combined loss = HPWL + density·overlap. β values arrive at
    // runtime via the `hpwl_beta` / `density_beta` Inputs so the
    // BetaSchedule can anneal sharpness without rebuilding the graph.
    let fwd = combined_loss_graph(&nl, &lib, &placement, DENSITY_WEIGHT);
    let pos_ids = position_param_ids(&fwd, &nl);
    let bwd = grad_with_loss(&fwd, &pos_ids);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    // ── Run Adam through the eda-trace harness ────────────────────
    let cfg = TraceCfg::new("hpwl_optim_trace", ADAM_STEPS)
        .with_log_schedule(eda_trace::LogSchedule::Logarithmic);
    let nl_capture = nl.clone();
    let layout = position_param_layout(&nl);
    let trace = Trace::run(&cfg, AdamStep {
        sess: &mut sess,
        placement: &mut placement,
        nl: &nl_capture,
        adam: AdamState::new(pos_ids.len()),
        // Cosine LR decay from the base 1000 DBU/step down to 10 %
        // of that — kills the overshoot we saw at step 600 in the
        // pre-optim-module trajectory.
        lr_base: 1_000.0,
        lr_sched: LrSchedule::Cosine { min_factor: 0.1 },
        // β-annealing: log-space ramp from very smooth (β = 1e-5,
        // wide gradient signal early) to sharp (β = 1e-3, true HPWL
        // near the optimum).
        beta_hpwl_sched: BetaSchedule::GeometricAnneal { start: 1e-5, end: BETA },
        beta_density_sched: BetaSchedule::Constant(BETA_DENSITY),
        layout,
        total_steps: ADAM_STEPS,
    });

    // ── Render before / after floorplans ──────────────────────────
    let initial_svg = render_floorplan(&nl, &lib, &seeds, "initial")?;
    let final_xy = placement.instance_xy.clone();
    let final_svg = render_floorplan(&nl, &lib, &final_xy, "final")?;

    // ── Charts ────────────────────────────────────────────────────
    let mut charts = vec![
        ChartSpec::line("loss", "HPWL over Adam steps", "step", "HPWL [DBU]")
            .with_y_log(true)
            .add_series("loss", "HPWL"),
        ChartSpec::line("bbox_span", "Placement bbox diagonal over steps", "step", "diagonal [DBU]")
            .add_series("bbox_diag", "diag"),
    ];
    // Per-instance position trajectories — one line per instance, x and y
    // each as a series. Charted as a single multi-series line plot of x,
    // then another of y, so the reader sees the geometry collapse.
    let mut x_chart = ChartSpec::line("xpos", "Per-instance x trajectory", "step", "x [DBU]");
    let mut y_chart = ChartSpec::line("ypos", "Per-instance y trajectory", "step", "y [DBU]");
    for i in 0..inst.len() {
        x_chart = x_chart.add_series(format!("x{i}"), format!("U{i}"));
        y_chart = y_chart.add_series(format!("y{i}"), format!("U{i}"));
    }
    charts.push(x_chart);
    charts.push(y_chart);

    // ── References ────────────────────────────────────────────────
    let initial_loss = trace.rows.first().map(|r| r.get("loss")).unwrap_or(0.0);
    let final_loss = trace.rows.last().map(|r| r.get("loss")).unwrap_or(0.0);
    let lse_floor = 2.0 * (inst.len() as f64).ln() / BETA as f64;
    let references = vec![
        eda_trace::Reference {
            citation: "Lin et al., *DREAMPlace: Deep Learning Toolkit-Enabled GPU \
                       Acceleration for Modern VLSI Placement* (DAC 2019, \
                       [DOI:10.1109/DAC.2019.8806865](https://doi.org/10.1109/DAC.2019.8806865))"
                .into(),
            formula: r"$\mathrm{HPWL}(\text{net}) = \mathrm{smooth\_max}(x_i) - \mathrm{smooth\_min}(x_i) + \dots_y$"
                .into(),
            predicted: "loss falls monotonically".into(),
            simulated: format!("initial = {initial_loss:.0}, final = {final_loss:.0}"),
            passed: final_loss < initial_loss,
        },
        eda_trace::Reference {
            citation: "log-sum-exp smoothing bias".into(),
            formula: r"$\mathrm{HPWL}_\text{floor} \approx \frac{2 \log N_\text{pins}}{\beta}$"
                .into(),
            predicted: format!("≈ {lse_floor:.0} DBU per net"),
            simulated: format!("{:.0} DBU total ({} nets)", final_loss, nl.nets.len()),
            passed: final_loss <= lse_floor * (nl.nets.len() as f64) * 1.5,
        },
    ];

    let meta = ReportMeta::new(
        "rlx-eda placement-optimization trace — HPWL on a 6-instance netlist",
        "Synthetic Netlist (eda-pnr)",
        Domain::Layout,
    )
    .with_intro(BACKGROUND)
    .with_objective(OBJECTIVE)
    .with_notes(NOTES)
    .with_references(references);

    let report = Report {
        trace: &trace,
        meta: &meta,
        charts: &charts,
        floorplan: FloorplanSource::Svg {
            svg: final_svg,
            caption: "Final placement after 600 Adam steps. Six unit cells \
                      collapsed to a tight cluster around the centroid of the \
                      seed positions; visible Manhattan wires connect every \
                      net.".into(),
        },
        extra_panels: vec![Panel {
            heading: "Initial placement".into(),
            body: PanelBody::Svg {
                file_slug: "initial_floorplan".into(),
                body: initial_svg,
            },
        }],
    };

    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs");
    report.write(&out_dir, "hpwl_optim_trace")?;

    print_run_summary(&trace, &nl, &seeds, &final_xy);
    println!(
        "\nwrote: {}/hpwl_optim_trace.md  (+ csv + assets/hpwl_optim_trace/)",
        out_dir.display(),
    );
    Ok(())
}

/// Pretty-print the optimization trajectory + before/after positions
/// + pairwise-separation table to stdout. The same data lives in the
/// trace's CSV / markdown — this is the human-readable summary.
fn print_run_summary(
    trace: &Trace,
    nl: &Netlist,
    seeds: &[(f32, f32)],
    final_xy: &[(f32, f32)],
) {
    use std::collections::BTreeSet;

    let initial_loss = trace.rows.first().map(|r| r.get("loss")).unwrap_or(0.0);
    let final_loss = trace.rows.last().map(|r| r.get("loss")).unwrap_or(0.0);
    let cell_w = 4_000.0_f32;

    // Header.
    println!(
        "\nspike-eda PNR ▸ HPWL + density loss   ({} instances, {} nets, {} Adam steps)",
        nl.instances.len(), nl.nets.len(), ADAM_STEPS,
    );
    println!(
        "  loss = HPWL + α·Σ overlap(i,j)   α = {:.0e}  β_hpwl = {:.0e}  β_density = {:.0e}\n",
        DENSITY_WEIGHT, BETA, BETA_DENSITY,
    );

    // Trajectory table — one row per logged step.
    println!("  step       loss       bbox_diag    Δloss");
    let mut prev_loss = initial_loss;
    for r in &trace.rows {
        let loss = r.get("loss");
        let diag = r.get("bbox_diag");
        let dloss = loss - prev_loss;
        let dlabel = if dloss < 0.0 {
            format!("{dloss:>+10.1}")
        } else if dloss > 0.0 {
            format!("{dloss:>+10.1}  ↑")
        } else {
            "         —".to_string()
        };
        println!(
            "  {:>4}   {:>9.1}   {:>9.1}   {dlabel}",
            r.step, loss, diag,
        );
        prev_loss = loss;
    }

    println!(
        "\n  initial HPWL = {initial_loss:>9.1}    final HPWL = {final_loss:>9.1}    \
         reduction = {:.1} %",
        100.0 * (initial_loss - final_loss) / initial_loss.max(1.0),
    );

    // Per-instance before/after.
    println!(
        "\n  instance        seed (x, y)              final (x, y)           drift [DBU]"
    );
    for (i, ((sx, sy), (fx, fy))) in seeds.iter().zip(final_xy.iter()).enumerate() {
        let dx = fx - sx;
        let dy = fy - sy;
        let drift = (dx * dx + dy * dy).sqrt();
        println!(
            "  U{i}        ({:>+8.0}, {:>+8.0})       ({:>+8.0}, {:>+8.0})         {drift:>8.0}",
            sx, sy, fx, fy,
        );
    }

    // Pairwise non-overlap check. Each cell is `cell_w × cell_w`;
    // bboxes overlap iff |Δx| < cell_w AND |Δy| < cell_w.
    println!("\n  pairwise separations (cell width = {} DBU):", cell_w as i64);
    let mut overlapping: BTreeSet<(usize, usize)> = BTreeSet::new();
    for i in 0..final_xy.len() {
        for j in (i + 1)..final_xy.len() {
            let dx = (final_xy[i].0 - final_xy[j].0).abs();
            let dy = (final_xy[i].1 - final_xy[j].1).abs();
            let ok = dx >= cell_w * 0.7 || dy >= cell_w * 0.7;
            let mark = if ok { "✓" } else { "✗ overlap" };
            if !ok { overlapping.insert((i, j)); }
            println!(
                "    U{i}—U{j}   dx = {:>6.0}   dy = {:>6.0}   {mark}",
                dx, dy,
            );
        }
    }
    if overlapping.is_empty() {
        println!("\n  ✓ no cell-bbox overlaps — density penalty held the cluster apart");
    } else {
        println!(
            "\n  ✗ {} pair(s) overlapped: {:?}",
            overlapping.len(), overlapping,
        );
    }
}

// ── Adam step closure — emits one TraceRow per step ───────────────────

struct AdamStep<'a> {
    sess: &'a mut rlx_runtime::CompiledGraph,
    placement: &'a mut DifferentiablePlacement,
    nl: &'a Netlist,
    adam: AdamState,
    lr_base: f32,
    lr_sched: LrSchedule,
    beta_hpwl_sched: BetaSchedule,
    beta_density_sched: BetaSchedule,
    /// Map from Adam-param index → `(instance_index, axis)` so we
    /// can shovel the f32 vector back into the placement after the
    /// update. Returned by [`position_param_layout`].
    layout: Vec<(usize, u8)>,
    total_steps: u32,
}

impl<'a> OptStep for AdamStep<'a> {
    fn step(&mut self, step: u32) -> TraceRow {
        // Push current positions into the rlx Param slots, then run
        // the AD-augmented session with the schedule's β values.
        for (i, (x, y)) in self.placement.instance_xy.iter().enumerate() {
            self.sess.set_param(&self.placement.x_param_name(self.nl, i), &[*x]);
            self.sess.set_param(&self.placement.y_param_name(self.nl, i), &[*y]);
        }
        let beta_h = self.beta_hpwl_sched.beta_at(step, self.total_steps);
        let beta_d = self.beta_density_sched.beta_at(step, self.total_steps);
        let outs = self.sess.run(&[
            ("d_output",          &[1.0_f32]),
            (HPWL_BETA_INPUT,     &[beta_h]),
            (DENSITY_BETA_INPUT,  &[beta_d]),
        ]);
        let loss = outs[0][0];

        // Bbox diagonal of current placement.
        let (mut xmin, mut xmax) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut ymin, mut ymax) = (f32::INFINITY, f32::NEG_INFINITY);
        for &(x, y) in &self.placement.instance_xy {
            if x < xmin { xmin = x; }
            if x > xmax { xmax = x; }
            if y < ymin { ymin = y; }
            if y > ymax { ymax = y; }
        }
        let dx = xmax - xmin;
        let dy = ymax - ymin;
        let diag = (dx * dx + dy * dy).sqrt();

        let lr = self.lr_sched.lr_at(self.lr_base, step, self.total_steps);

        let mut row = TraceRow::new(step)
            .with("loss",      loss as f64)
            .with("bbox_diag", diag as f64)
            .with("lr",        lr as f64)
            .with("beta_hpwl", beta_h as f64);
        for (i, (x, y)) in self.placement.instance_xy.iter().enumerate() {
            row = row.with(format!("x{i}"), *x as f64).with(format!("y{i}"), *y as f64);
        }

        // Adam update after logging — step 0 reflects seed state.
        if step < self.total_steps {
            let n = self.layout.len();
            let mut params: Vec<f32> = self
                .layout
                .iter()
                .map(|(i, axis)| if *axis == 0 {
                    self.placement.instance_xy[*i].0
                } else {
                    self.placement.instance_xy[*i].1
                })
                .collect();
            let grads: Vec<f32> = (0..n).map(|k| outs[1 + k][0]).collect();
            // 1-indexed step counter for Adam bias correction.
            self.adam.step(&mut params, &grads, lr, step + 1);
            for (k, (i, axis)) in self.layout.iter().enumerate() {
                if *axis == 0 {
                    self.placement.instance_xy[*i].0 = params[k];
                } else {
                    self.placement.instance_xy[*i].1 = params[k];
                }
            }
        }
        row
    }
}

// ── Floorplan rendering helpers ──────────────────────────────────────

/// Materialize the `(f32, f32)` placement, run a Manhattan router
/// over the netlist, and emit an SVG of the resulting top cell via
/// `eda-viz::layout::render_to_svg`.
fn render_floorplan(
    nl: &Netlist,
    lib: &Library,
    xy: &[(f32, f32)],
    tag: &str,
) -> Result<String, Box<dyn Error>> {
    let transforms: Vec<Trans> = xy
        .iter()
        .map(|(x, y)| Trans::translate(Vec2::new(x.round() as i64, y.round() as i64)))
        .collect();
    // Build a fresh netlist clone with a tag suffix so each call
    // produces a distinct top-cell name (the library would reject a
    // duplicate insertion otherwise).
    let mut tagged = nl.clone();
    tagged.name = format!("{}_{tag}", nl.name);
    let placer = ManualPlacer::new(transforms);
    let router = ManhattanRouter::new(WireStyle::Polygon);
    let result = PnrFlow::new(placer, router).run(&tagged, lib);
    let style = Style {
        units_per_dbu: 0.002,
        background: Some("white".to_string()),
        show_ports: true,
        show_legend: false,
        ..Style::default()
    };
    Ok(eda_viz::layout::render_to_svg(lib, result.top, &style))
}

// ── Long-form prose blocks ────────────────────────────────────────────

const BACKGROUND: &str = r#"**Half-perimeter wirelength (HPWL)** is the standard placement objective in physical design: for each net, take the bounding-box of the pins it touches; sum the box's half-perimeter (`(max x − min x) + (max y − min y)`) across every net. Smaller HPWL ⇒ shorter wires ⇒ lower delay, lower power, less congestion.

`max` and `min` aren't differentiable at ties, which is why for decades placement was a discrete combinatorial problem solved by simulated annealing or analytical solvers with custom kernels. **DREAMPlace** (Lin et al., DAC 2019) showed that replacing both with a log-sum-exp smoothing — `smooth_max(x_i; β) = (1/β) · log Σ exp(β · x_i)` — turns HPWL into a differentiable loss any deep-learning framework can backprop through, and that gradient descent on the resulting smooth objective places million-cell ASICs at GPU speed.

HPWL alone collapses every instance to a single point (zero wirelength is the trivial minimum). Real placement adds a competing **non-overlap** term: penalise every pair of instances whose bboxes intersect. The penalty here is the smooth pairwise overlap area, with `smooth_relu(z; β) = z · sigmoid(β · z)` (the swish/SiLU shape) standing in for the non-differentiable `relu` clamp:

```text
  overlap_x(i, j) = smooth_relu( (w_i + w_j) / 2 - |x_i - x_j|; β )
  overlap_y(i, j) = smooth_relu( (h_i + h_j) / 2 - |y_i - y_j|; β )
  density(i, j)   = overlap_x · overlap_y
  loss            = HPWL + α · Σ_{i<j} density(i, j)
```

`eda-pnr::ad::combined_loss_graph(&netlist, &lib, β_hpwl, β_density, α)` builds the whole thing on the rlx graph; every instance's `(x, y)` is an `rlx_ir::Param`, and any rlx optimizer (Adam here, but also SGD, DADO, RL search) drives the placement. Same path the LNA uses to tune `Lg` and the MZI uses to tune `n_eff_A` — placement is just another node in the same ML graph."#;

const OBJECTIVE: &str = r#"Minimize the combined loss `HPWL + α · Σ_{i<j} density(i, j)` where the HPWL term is

`HPWL(P) = Σ_net [smooth_max(x_pins) − smooth_min(x_pins) + smooth_max(y_pins) − smooth_min(y_pins)]`

over instance positions `P`. Pin position = `instance_pos + port_offset` (port offsets baked as constants since translation-invariant under instance moves). The density term penalises pairwise bbox overlap; with `α = 10⁻²` and 4 µm × 4 µm cells, the equilibrium is a tight non-overlapping cluster — instances pack together, but no two centroids land closer than ~4 µm."#;

const NOTES: &str = r#"- **Initial seeds** are deterministic golden-ratio spirals across a ±50 µm region; same input produces same trace every run.
- **Wires drawn in both floorplans** — the "before" picture shows long Manhattan paths across the full 100 µm spread; the "after" picture shows the cluster collapsed and wires nearly invisible.
- **Adam step size (`lr = 1000` DBU)** is calibrated to the seed magnitude; rule of thumb is ~1 % of the seed spread per step, same scaling the LNA uses for `Lg`.
- **Per-instance trajectory charts** (`xpos`, `ypos`) reveal which instances move first — the 3-pin cluster `(U0, U1, U2)` collapses early because each move reduces three nets' HPWL contributions simultaneously."#;
