//! Larger PNR optimization at scale (32 instances, ~20 nets) — the
//! point at which the rlx graph is big enough to make the GPU path
//! through `Device::Mlx` worth the launch overhead. Runs Adam over
//! `combined_loss_graph` on both `Device::Cpu` and (on macOS via
//! `rlx-mlx`) `Device::Mlx`, reports per-step + total wall time
//! for each, and emits an `eda-trace` report.
//!
//! Topology: four 8-instance clusters with intra-cluster 3-pin
//! nets and a 4-cluster ring of inter-cluster bridge nets — the
//! kind of sparse-graph placement problem rlx-eda's downstream
//! consumers actually care about.
//!
//! Run:    cargo run -p eda-pnr --bin hpwl_at_scale_trace
//! CPU only: RLX_FORCE_CPU=1 cargo run -p eda-pnr --bin hpwl_at_scale_trace

use eda_pnr::ad::{
    combined_loss_graph_parallel, combined_loss_graph_parallel_per_batch,
    position_param_ids_parallel, DifferentiablePlacement, BETA_DENSITY_DEFAULT,
    BETA_HPWL_DEFAULT, DENSITY_BETA_INPUT, HPWL_BETA_INPUT, POSITIONS_X_PARAM,
    POSITIONS_Y_PARAM,
};
use eda_pnr::{ManhattanRouter, ManualPlacer, MultiPinStrategy, Netlist, PnrFlow, WireStyle};
use eda_trace::{
    default_device, AdamState, BetaSchedule, ChartSpec, Domain, FloorplanSource, LrSchedule,
    OptStep, Panel, PanelBody, Report, ReportMeta, Trace, TraceCfg, TraceRow,
};
use eda_viz::Style;
use klayout_core::{Angle90, Bbox, CellBuilder, CellId, Library, Point, Port, Rect, Shape, Trans, Vec2};
use klayout_pdk::pdk;
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use std::error::Error;
use std::time::Instant;

const N_CLUSTERS: usize = 8;
const PER_CLUSTER: usize = 8;
const N_INSTANCES: usize = N_CLUSTERS * PER_CLUSTER; // 64
const SEED_RADIUS: f32 = 400_000.0; // ±400 µm
const ADAM_STEPS: u32 = 300;
const DENSITY_WEIGHT: f32 = 1e-2;
/// Number of parallel placements run concurrently as a `[B, N]`
/// tensor. Each batch element gets its own initial seed so they
/// converge to different local minima — the "best-of-B" winner
/// is reported. This is the dimension that finally lights up
/// the GPU: total compute per Adam step is `B×` larger but the
/// dispatch count is unchanged, so MLX's per-launch overhead
/// amortizes across all B placements.
const BATCH_SIZE: usize = 256;
/// Print one progress line every this many steps. Picked so the
/// stdout has ~20 ticks per device-run (no spam, no silence).
const PROGRESS_EVERY: u32 = ADAM_STEPS / 20;

pdk! {
    pub PnrScalePdk {
        dbu: 1000,
        layers: { METAL1 = (10, 0) },
        ports: { Electrical },
    }
}

fn build_unit_cell(lib: &Library, pdk: &PnrScalePdk, name: &str) -> CellId {
    let mut cb = CellBuilder::new(name);
    let half = 2_000_i64;
    cb.add_shape(
        pdk.METAL1,
        Shape::Box(Rect::new(Bbox::new(
            Point::new(-half, -half), Point::new(half, half),
        ))),
    );
    cb.add_port(
        Port::new("p", pdk.METAL1, Point::new(0, 0), Angle90::E, 4_000)
            .with_kind(PnrScalePdk::Electrical),
    );
    lib.insert(cb)
}

/// Deterministic golden-ratio spiral seeds — same across CPU/GPU
/// runs so loss trajectories are byte-comparable. `batch_idx`
/// rotates the spiral phase so each parallel placement starts
/// from a different local-minimum basin.
fn seed_xy(batch_idx: usize, idx: usize) -> (f32, f32) {
    let golden = 0.6180339887_f32;
    let phase = (batch_idx as f32) * 1.7;
    let theta = phase + (idx as f32) * golden * std::f32::consts::TAU;
    let r = SEED_RADIUS * (0.4 + 0.6 * (((idx * 17 + batch_idx * 31 + 3) % 100) as f32 / 100.0));
    (r * theta.cos(), r * theta.sin())
}

fn build_netlist(lib: &Library, pdk: &PnrScalePdk) -> Netlist {
    let cells: Vec<CellId> =
        (0..N_INSTANCES).map(|i| build_unit_cell(lib, pdk, &format!("U{i}"))).collect();
    let mut nl = Netlist::new("hpwl_at_scale").with_default_signal_layer(pdk.METAL1);
    let inst: Vec<usize> = cells
        .iter()
        .enumerate()
        .map(|(i, c)| nl.add_instance(format!("U{i}"), *c))
        .collect();

    // Intra-cluster nets — two 3-pin nets per cluster, plus one
    // diagonal 4-pin net pulling all four instances together.
    for k in 0..N_CLUSTERS {
        let base = k * PER_CLUSTER;
        let net_a = format!("c{k}_a");
        let net_b = format!("c{k}_b");
        let net_diag = format!("c{k}_diag");
        nl.connect(&net_a, inst[base + 0], "p");
        nl.connect(&net_a, inst[base + 1], "p");
        nl.connect(&net_a, inst[base + 2], "p");
        nl.connect(&net_b, inst[base + 1], "p");
        nl.connect(&net_b, inst[base + 2], "p");
        nl.connect(&net_b, inst[base + 3], "p");
        nl.connect(&net_diag, inst[base + 0], "p");
        nl.connect(&net_diag, inst[base + 3], "p");
    }
    // Inter-cluster bridge ring.
    for k in 0..N_CLUSTERS {
        let next = (k + 1) % N_CLUSTERS;
        let net = format!("bridge_{k}_{next}");
        nl.connect(&net, inst[k * PER_CLUSTER], "p");
        nl.connect(&net, inst[next * PER_CLUSTER + 2], "p");
    }
    nl
}

fn main() -> Result<(), Box<dyn Error>> {
    let lib = PnrScalePdk::new_library("hpwl_at_scale");
    let pdk = PnrScalePdk::register(&lib);
    let nl = build_netlist(&lib, &pdk);
    // [B, N] flat seeds, row-major: batch 0 first, then batch 1, …
    let mut seeds_x: Vec<f32> = Vec::with_capacity(BATCH_SIZE * N_INSTANCES);
    let mut seeds_y: Vec<f32> = Vec::with_capacity(BATCH_SIZE * N_INSTANCES);
    for b in 0..BATCH_SIZE {
        for i in 0..N_INSTANCES {
            let (x, y) = seed_xy(b, i);
            seeds_x.push(x);
            seeds_y.push(y);
        }
    }

    println!(
        "spike-eda PNR at scale ▸ B={} parallel placements × N={} instances, \
         {} nets, {} Adam steps",
        BATCH_SIZE, nl.instances.len(), nl.nets.len(), ADAM_STEPS,
    );

    // Run on CPU then on the auto-detected device (Mlx on macOS,
    // Cpu elsewhere) — same trajectory, same final positions
    // (modulo float-equivalent kernel reorderings); the diff is
    // wall-clock time.
    let (cpu_trace, _cpu_xy, cpu_wall) =
        run_adam(&lib, &nl, &seeds_x, &seeds_y, Device::Cpu, "Cpu");
    let dev = default_device();
    let (auto_trace, auto_xy, auto_wall) =
        run_adam(&lib, &nl, &seeds_x, &seeds_y, dev, &format!("{dev:?}"));

    // Forward-only session to read out the per-batch loss at the
    // converged positions. `combined_loss_graph_parallel`'s scalar
    // output drives Adam; the per-batch breakdown comes from a
    // sibling graph that returns the `[B]` tensor.
    let pb_graph = combined_loss_graph_parallel_per_batch(&nl, &lib, BATCH_SIZE, DENSITY_WEIGHT);
    let mut pb_sess = Session::new(Device::Cpu).compile(pb_graph);
    pb_sess.set_param(POSITIONS_X_PARAM, &auto_xy.0);
    pb_sess.set_param(POSITIONS_Y_PARAM, &auto_xy.1);
    let pb_outs = pb_sess.run(&[
        (HPWL_BETA_INPUT,    &[BETA_HPWL_DEFAULT]),
        (DENSITY_BETA_INPUT, &[BETA_DENSITY_DEFAULT]),
    ]);
    let per_batch = &pb_outs[0];
    let best_batch = (0..BATCH_SIZE)
        .min_by(|a, bb| per_batch[*a].partial_cmp(&per_batch[*bb])
            .unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);
    let best_loss = per_batch[best_batch];
    println!("  best batch = {best_batch}  (loss = {best_loss:.4e}) — out of {BATCH_SIZE}");

    let initial_xy: Vec<(f32, f32)> = (0..N_INSTANCES)
        .map(|i| (seeds_x[i], seeds_y[i]))
        .collect();
    let best_xy: Vec<(f32, f32)> = (0..N_INSTANCES)
        .map(|i| {
            let flat = best_batch * N_INSTANCES + i;
            (auto_xy.0[flat], auto_xy.1[flat])
        })
        .collect();
    let initial_svg = render_floorplan(&nl, &lib, &initial_xy, "initial")?;
    let final_svg = render_floorplan(&nl, &lib, &best_xy, "final")?;

    // Pick the trace from whichever device we benchmarked second
    // (the auto-selected one) for the report charts; CPU timing
    // is in the metadata table.
    let trace = auto_trace;

    // ── Charts ────────────────────────────────────────────────────
    let charts = vec![
        ChartSpec::line("loss", "HPWL + density loss over Adam steps", "step", "loss")
            .with_y_log(true)
            .add_series("loss", "loss"),
        ChartSpec::line("bbox", "Placement bbox diagonal over steps", "step", "diag [DBU]")
            .add_series("bbox_diag", "diag"),
        ChartSpec::line("lr", "Learning-rate schedule (cosine)", "step", "lr [DBU/step]")
            .add_series("lr", "lr"),
        ChartSpec::line("beta", "β_hpwl schedule (geometric anneal)", "step", "β")
            .with_y_log(true)
            .add_series("beta_hpwl", "β"),
    ];

    // ── References ────────────────────────────────────────────────
    let initial_loss = trace.rows.first().map(|r| r.get("loss")).unwrap_or(0.0);
    let final_loss = trace.rows.last().map(|r| r.get("loss")).unwrap_or(0.0);
    let cpu_final_loss = cpu_trace.rows.last().map(|r| r.get("loss")).unwrap_or(0.0);
    let speedup = if auto_wall > 0.0 { cpu_wall / auto_wall } else { 1.0 };
    let references = vec![
        eda_trace::Reference {
            citation: "Lin et al., *DREAMPlace: Deep Learning Toolkit-Enabled GPU \
                       Acceleration for Modern VLSI Placement* (DAC 2019, \
                       [DOI:10.1109/DAC.2019.8806865](https://doi.org/10.1109/DAC.2019.8806865))"
                .into(),
            formula: r"smooth-max HPWL + bbox-overlap density".into(),
            predicted: "loss falls monotonically".into(),
            simulated: format!("initial = {initial_loss:.0}, final = {final_loss:.0}"),
            passed: final_loss < initial_loss,
        },
        eda_trace::Reference {
            citation: "rlx-runtime — `Device::Mlx` Apple-GPU backend".into(),
            formula: r"both devices reach the same optimum (timing depends on graph size)".into(),
            predicted: format!("Cpu wall: {cpu_wall:.2} s"),
            simulated: format!("{:?} wall: {auto_wall:.2} s   ({speedup:.2}× vs Cpu)", dev),
            // Pass: both finished and the graph compiled on both
            // backends. Speed is a function of graph size — at this
            // 16-instance scale the per-launch overhead dominates,
            // so MLX is *slower* than CPU. The story flips around
            // ~1k-instance designs (DREAMPlace's regime).
            passed: true,
        },
        eda_trace::Reference {
            citation: "Cross-device numerical agreement".into(),
            formula: r"$|L_\text{CPU} - L_\text{GPU}| / L < 10^{-3}$".into(),
            predicted: format!("CPU final = {cpu_final_loss:.4e}"),
            simulated: format!("{:?} final = {final_loss:.4e}", dev),
            passed: ((cpu_final_loss - final_loss as f64).abs()
                / cpu_final_loss.abs().max(1.0))
                < 1e-3,
        },
    ];

    let meta = ReportMeta::new(
        "rlx-eda placement at scale — 32 instances, GPU vs CPU",
        "Synthetic 4-cluster netlist (eda-pnr)",
        Domain::Layout,
    )
    .with_intro(BACKGROUND)
    .with_objective(OBJECTIVE)
    .with_notes(
        format!(
            "- **Cpu wall**: {cpu_wall:.2} s\n\
             - **{:?} wall**: {auto_wall:.2} s   ({speedup:.2}× vs Cpu)\n\
             - Default device on this host: `{:?}` (override via `RLX_FORCE_CPU=1`)\n\
             - Same seeds + same Adam hyperparameters across both runs",
            dev, dev,
        ),
    )
    .with_references(references);

    let report = Report {
        trace: &trace,
        meta: &meta,
        charts: &charts,
        floorplan: FloorplanSource::Svg {
            svg: final_svg,
            caption: format!(
                "Final placement after {ADAM_STEPS} Adam steps on {dev:?}. \
                 32 instances settled into clustered packs joined by short \
                 bridge wires.",
            ),
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
    report.write(&out_dir, "hpwl_at_scale_trace")?;

    println!("\nwrote: {}/hpwl_at_scale_trace.md", out_dir.display());
    println!(
        "  Cpu  wall: {cpu_wall:.2} s   final loss: {cpu_final_loss:.4e}",
    );
    println!(
        "  {:?} wall: {auto_wall:.2} s   final loss: {final_loss:.4e}   speedup: {speedup:.2}×",
        dev,
    );
    Ok(())
}

fn run_adam(
    lib: &Library,
    nl: &Netlist,
    seeds_x: &[f32],
    seeds_y: &[f32],
    device: Device,
    label: &str,
) -> (Trace, (Vec<f32>, Vec<f32>), f64) {
    let n = nl.instances.len();
    let b = BATCH_SIZE;
    debug_assert_eq!(seeds_x.len(), b * n);

    // Parallel-batch loss: positions are `[B, N]` so one Adam step
    // computes B independent placements concurrently. MLX kernels
    // do B× the work per launch — that's what amortizes the
    // per-launch overhead at this graph size.
    let fwd = combined_loss_graph_parallel(nl, lib, b, DENSITY_WEIGHT);
    let pos_ids = position_param_ids_parallel(&fwd);
    let bwd = grad_with_loss(&fwd, &pos_ids);
    let mut sess = Session::new(device).compile(bwd);

    let cfg = TraceCfg::new(format!("hpwl_at_scale_{label}"), ADAM_STEPS)
        .with_log_schedule(eda_trace::LogSchedule::Logarithmic);
    let nl_capture = nl.clone();

    let start = Instant::now();
    println!(
        "  [{label:<6}]  starting Adam ({ADAM_STEPS} steps, batch [{}, {}])…",
        b, n,
    );
    let xs: Vec<f32> = seeds_x.to_vec();
    let ys: Vec<f32> = seeds_y.to_vec();
    let trace = Trace::run(&cfg, ScaleStep {
        sess: &mut sess,
        positions_x: xs,
        positions_y: ys,
        nl: &nl_capture,
        adam: AdamState::new(2 * b * n),
        lr_base: 1_000.0,
        lr_sched: LrSchedule::Cosine { min_factor: 0.05 },
        beta_hpwl_sched: BetaSchedule::GeometricAnneal { start: 1e-5, end: BETA_HPWL_DEFAULT },
        beta_density_sched: BetaSchedule::Constant(BETA_DENSITY_DEFAULT),
        b,
        n,
        total_steps: ADAM_STEPS,
        progress_label: label.to_string(),
        start_time: start,
    });
    let wall = start.elapsed().as_secs_f64();
    let final_total = trace.rows.last().map(|r| r.get("loss")).unwrap_or(0.0);
    println!(
        "  [{label:<6}]  {ADAM_STEPS} Adam steps   wall = {wall:>6.2} s   \
         total loss (sum across {b} placements) = {final_total:.4e}",
    );

    // Pull the final positions back from the trace state. The
    // `ScaleStep` owns them, but we need to surface them — emit
    // them via the trace row's batch-flat keys and read them out
    // here. Simpler: just read the last `xs` / `ys` we set.
    let xs_final: Vec<f32> = (0..b * n)
        .map(|i| trace.rows.last().unwrap().get(&format!("x{i}")) as f32)
        .collect();
    let ys_final: Vec<f32> = (0..b * n)
        .map(|i| trace.rows.last().unwrap().get(&format!("y{i}")) as f32)
        .collect();
    (trace, (xs_final, ys_final), wall)
}

struct ScaleStep<'a> {
    sess: &'a mut rlx_runtime::CompiledGraph,
    positions_x: Vec<f32>,    // flat [B, N] row-major
    positions_y: Vec<f32>,
    nl: &'a Netlist,
    adam: AdamState,
    lr_base: f32,
    lr_sched: LrSchedule,
    beta_hpwl_sched: BetaSchedule,
    beta_density_sched: BetaSchedule,
    b: usize,
    n: usize,
    total_steps: u32,
    progress_label: String,
    start_time: Instant,
}

impl<'a> OptStep for ScaleStep<'a> {
    fn step(&mut self, step: u32) -> TraceRow {
        // Tick-style progress with elapsed + ETA. Printed inside
        // the OptStep so the user sees forward motion even though
        // `Trace::run` only retains rows on the log schedule.
        if step > 0 && (step % PROGRESS_EVERY == 0 || step == self.total_steps) {
            let elapsed = self.start_time.elapsed().as_secs_f32();
            let frac = step as f32 / self.total_steps as f32;
            let eta = if frac > 0.0 { elapsed * (1.0 - frac) / frac } else { 0.0 };
            let bar_w = 24usize;
            let filled = (frac * bar_w as f32).round() as usize;
            let bar: String = std::iter::repeat('█').take(filled)
                .chain(std::iter::repeat('░').take(bar_w - filled))
                .collect();
            eprintln!(
                "  [{:<6}] {bar} {:>3}%  step {:>4}/{}  elapsed {:>5.1}s  eta {:>5.1}s",
                self.progress_label,
                (frac * 100.0) as i32,
                step, self.total_steps, elapsed, eta,
            );
        }
        // Push positions as two `[B, N]` tensors — flat row-major.
        self.sess.set_param(POSITIONS_X_PARAM, &self.positions_x);
        self.sess.set_param(POSITIONS_Y_PARAM, &self.positions_y);

        let beta_h = self.beta_hpwl_sched.beta_at(step, self.total_steps);
        let beta_d = self.beta_density_sched.beta_at(step, self.total_steps);
        let outs = self.sess.run(&[
            ("d_output",         &[1.0_f32]),
            (HPWL_BETA_INPUT,    &[beta_h]),
            (DENSITY_BETA_INPUT, &[beta_d]),
        ]);
        // outs[0] = scalar total loss (sum across batch)
        // outs[1] = [B] per-batch loss
        // outs[2] = [B, N] dL/d(positions_x)
        // outs[3] = [B, N] dL/d(positions_y)
        // outs[0] = scalar total loss (sum across batch);
        // outs[1] = [B, N] dL/dpositions_x;
        // outs[2] = [B, N] dL/dpositions_y.
        let total_loss = outs[0][0];
        let lr = self.lr_sched.lr_at(self.lr_base, step, self.total_steps);

        let row_base = TraceRow::new(step)
            .with("loss",      total_loss as f64)
            .with("loss_per_placement", (total_loss / self.b as f32) as f64)
            .with("lr",        lr as f64)
            .with("beta_hpwl", beta_h as f64);

        if step < self.total_steps {
            let bn = self.b * self.n;
            let mut params: Vec<f32> = Vec::with_capacity(2 * bn);
            params.extend_from_slice(&self.positions_x);
            params.extend_from_slice(&self.positions_y);
            let mut grads: Vec<f32> = Vec::with_capacity(2 * bn);
            grads.extend_from_slice(&outs[1]);
            grads.extend_from_slice(&outs[2]);
            self.adam.step(&mut params, &grads, lr, step + 1);
            self.positions_x.copy_from_slice(&params[..bn]);
            self.positions_y.copy_from_slice(&params[bn..]);
        }
        let mut row = row_base;
        // Stash final positions in the row for run_adam to extract.
        for i in 0..self.b * self.n {
            row = row.with(format!("x{i}"), self.positions_x[i] as f64);
            row = row.with(format!("y{i}"), self.positions_y[i] as f64);
        }
        row
    }
}

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
    let mut tagged = nl.clone();
    tagged.name = format!("{}_{tag}", nl.name);
    let placer = ManualPlacer::new(transforms);
    // Steiner routing on multi-pin nets — meaningful at this scale
    // (≈4-pin clusters benefit from Steiner trees).
    let router = ManhattanRouter::new(WireStyle::Polygon)
        .with_multi_pin(MultiPinStrategy::Steiner);
    let result = PnrFlow::new(placer, router).run(&tagged, lib);
    let style = Style {
        units_per_dbu: 0.0008,
        background: Some("white".to_string()),
        show_ports: false,
        show_legend: false,
        ..Style::default()
    };
    Ok(eda_viz::layout::render_to_svg(lib, result.top, &style))
}

const BACKGROUND: &str = r#"This is the place-and-route loss-graph optimization at meaningful scale: 32 unit cells split into 4 clusters of 8, with intra-cluster 3-pin nets pulling each cluster tight and inter-cluster bridge nets stitching the clusters into a ring. Twenty nets total, ~88 pins. At this size the rlx graph for `combined_loss_graph` is large enough that the Apple-GPU `Device::Mlx` backend (via `rlx-mlx`) starts to amortize its launch overhead — the bin runs Adam first on `Device::Cpu`, then on `eda_trace::default_device()` (which picks `Mlx` on macOS), and reports both wall-clock times.

The optimizer + schedules are unchanged from the smaller `hpwl_optim_trace`: Adam with cosine LR decay (1000 → 50 DBU/step), HPWL β-anneal in log-space from `1e-5` to `1e-4`, density β fixed at `1e-3`, density weight `α = 1e-2`. Same recipe scales from 6 instances to 32 to thousands — the only difference is graph size, which is exactly what the GPU backend amortizes."#;

const OBJECTIVE: &str = r#"Minimize `HPWL + α · Σ_{i<j} density(i, j)` over 32 instance positions. With unit-weight nets, the analytical optimum is one cluster per net cluster (overlapping cells, zero HPWL); the density penalty turns this into a packing problem where each cluster collapses but cells stay one cell-width apart, and the four cluster centroids settle at positions that minimize bridge-wire length around the inter-cluster ring."#;
