//! 2.4 GHz LNA input-match trace — RF-domain consumer of the
//! [`eda-trace`] harness.
//!
//! Runs Adam on the gate inductor `Lg` of the canonical
//! inductively-degenerated cascode (`spike-lna::Lna`), with the trace
//! harness recording per-step loss / params / gradients, rendering
//! charts (loss, Lg trajectory, gradient signal, S₁₁ before/after
//! sweep), embedding the floorplan via `eda-viz::layout::render_to_svg`,
//! and templating a markdown report at
//! `crates/spike-lna/docs/lna_match_trace.md`.
//!
//! Every artifact is regenerated each run — tweak `Lna::lna_24ghz`
//! sizing or the literature references and rerun
//! `cargo run -p spike-lna --bin lna_match_trace` to see the report
//! update.

use eda_hir::Layout;
use eda_trace::{
    AdamState, ChartSpec, Domain, FloorplanSource, LrSchedule, OptStep, Reference, Report,
    ReportMeta, Trace, TraceCfg, TraceRow,
};
use eda_viz::Style;
use rlx_ir::{NodeId, Op};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use spike_lna::{Lna, RfDemo};
use std::error::Error;

const TAU: f32 = std::f32::consts::TAU;
const F0_HZ: f32 = 2.4e9;
const ADAM_STEPS: u32 = 4_000;

// Razavi §5.3.3 canonical sizing for Z₀ = 50 Ω at 2.4 GHz:
//   gm·Ls/Cgs = Z₀         → Ls = Z₀·Cgs / gm = 250 pH
//   ω₀²(Lg+Ls)·Cgs = 1     → Lg ≈ 17.34 nH at the canonical Cgs/Ls
const GM: f32 = 50e-3;     // 50 mS
const CGS: f32 = 250e-15;  // 250 fF
const LS: f32 = 250e-12;   // 250 pH
const LD: f32 = 10e-9;     // 10 nH
const RL: f32 = 500.0;     // 500 Ω drain load

fn lg_star() -> f32 {
    let omega0 = TAU * F0_HZ;
    1.0 / (omega0 * omega0 * CGS) - LS
}

fn main() -> Result<(), Box<dyn Error>> {
    let lna = Lna::lna_24ghz("trace");
    let lg_target = lg_star();
    let lg_init = lg_target * 0.5; // start half-detuned

    println!(
        "spike-lna :: lna_match_trace — Razavi inductively-degenerated cascode\n\
         f₀ = {:.2} GHz, Lg* = {:.3} nH, Lg_init = {:.3} nH\n",
        F0_HZ / 1e9,
        lg_target * 1e9,
        lg_init * 1e9,
    );

    // Build the loss graph + AD session once. The optimizer closure
    // captures both via &mut and emits a per-step row.
    let fwd = lna.build_match_loss_graph();
    let lg_id = find_param(&fwd, &lna.lg_param_name());
    let bwd = grad_with_loss(&fwd, &[lg_id]);
    let mut sess = Session::new(Device::Cpu).compile(bwd);
    sess.set_param(&lna.gm_param_name(),  &[GM]);
    sess.set_param(&lna.cgs_param_name(), &[CGS]);
    sess.set_param(&lna.ls_param_name(),  &[LS]);
    sess.set_param(&lna.ld_param_name(),  &[LD]);
    sess.set_param(&lna.rl_param_name(),  &[RL]);

    // Adam (eda-trace::AdamState) — lr base scaled to Lg's nH
    // magnitude so step sizes track the parameter scale; cosine
    // decay to 10 % to kill the late-iteration overshoot.
    let lr_base = lg_target * 1e-2;
    let lr_sched = LrSchedule::Cosine { min_factor: 0.1 };
    let mut lg = lg_init;

    let cfg = TraceCfg::new("lna_match_trace", ADAM_STEPS)
        .with_log_schedule(eda_trace::LogSchedule::Logarithmic);
    let lna_capture = lna.clone();
    let trace = Trace::run(&cfg, MakeStep {
        sess: &mut sess,
        lg: &mut lg,
        adam: AdamState::with_betas(1, 0.9, 0.999, 1e-12),
        lr_base,
        lr_sched,
        total_steps: ADAM_STEPS,
        lna: &lna_capture,
    });

    // Final Lg ends up in the last logged row.
    let lg_final = trace.rows.last().map(|r| r.get("lg")).unwrap_or(lg_init as f64) as f32;

    // ── Forward sweeps for the before/after S₁₁ chart ─────────────
    let before_sweep = sweep_s11(&lna, lg_init);
    let after_sweep = sweep_s11(&lna, lg_final);

    // ── Floorplan SVG via eda-viz::layout ─────────────────────────
    let lib = RfDemo::new_library("spike-lna-floorplan");
    let pdk = RfDemo::register(&lib);
    let top = lna.layout(&lib, &pdk);
    let style = Style {
        units_per_dbu: 0.0008,
        background: Some("white".to_string()),
        show_ports: true,
        show_legend: true,
        ..Style::default()
    };
    let floorplan_svg = eda_viz::layout::render_to_svg(&lib, top, &style);

    // ── Build report ──────────────────────────────────────────────
    let charts = vec![
        ChartSpec::line(
            "loss",
            "|S₁₁(f₀)|² over Adam steps",
            "step",
            "|S₁₁|²",
        )
        .with_y_log(true)
        .add_series("loss", "|S₁₁|²"),

        ChartSpec::line(
            "lg",
            "Gate-inductor trajectory",
            "step",
            "Lg [H]",
        )
        .add_series("lg", "Lg")
        .add_colored_series("lg_target", "Lg* (Razavi optimum)", "#777777"),

        ChartSpec::line(
            "grad",
            "∂|S₁₁|² / ∂Lg over steps",
            "step",
            "gradient",
        )
        .add_series("grad", "dL/dLg"),

        ChartSpec::line(
            "return_loss",
            "Return loss at f₀ (dB)",
            "step",
            "return loss [dB]",
        )
        .add_series("return_loss_db", "−10·log₁₀|S₁₁|²"),

        ChartSpec::before_after(
            "s11_sweep",
            "|S₁₁(f)|² — before vs after Adam",
            "frequency [GHz]",
            "|S₁₁|²",
            "before (Lg = {:.2} nH)".replace("{:.2}", &format!("{:.2}", lg_init * 1e9)),
            before_sweep,
            "after (Lg = {:.2} nH)".replace("{:.2}", &format!("{:.2}", lg_final * 1e9)),
            after_sweep,
        ),
    ];

    let meta = ReportMeta::new(
        "rlx-eda single-circuit ML optimization trace — Inductively-degenerated cascode LNA",
        "Lna (spike-lna)",
        Domain::Rf,
    )
    .with_intro(LNA_BACKGROUND.to_string())
    .with_objective(LNA_OBJECTIVE.to_string())
    .with_notes(LNA_NOTES.to_string())
    .with_references(razavi_references(lg_final, &lna, &mut Session::new(Device::Cpu).compile(lna.build_forward_graph())));

    let report = Report {
        trace: &trace,
        meta: &meta,
        charts: &charts,
        floorplan: FloorplanSource::Svg {
            svg: floorplan_svg,
            caption:
                "RfDemo PDK floorplan: M1+M2 cascode at the centre, Lg above, Ls below, \
                 Ld to the right; M1 contact pads at rf_in / rf_out / vdd / gnd / vbias. \
                 Spiral inductors render on a dedicated METAL_TOP layer."
                    .to_string(),
        },
        extra_panels: vec![],
    };

    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs");
    report.write(&out_dir, "lna_match_trace")?;

    println!(
        "wrote: {}/lna_match_trace.md  (+ csv + assets/lna_match_trace/)",
        out_dir.display(),
    );
    println!(
        "  Lg final  = {:.4} nH    Lg*       = {:.4} nH    rel error = {:.2e}",
        lg_final * 1e9,
        lg_target * 1e9,
        (lg_final - lg_target).abs() / lg_target,
    );
    let final_loss = trace.rows.last().map(|r| r.get("loss")).unwrap_or(0.0);
    println!(
        "  final |S₁₁|² = {:.3e}  ({:.1} dB return loss)",
        final_loss,
        if final_loss > 0.0 { -10.0 * final_loss.log10() } else { f64::INFINITY },
    );

    Ok(())
}

// ── Optimizer step (one Adam tick) ────────────────────────────────────

struct MakeStep<'a> {
    sess: &'a mut rlx_runtime::CompiledGraph,
    lg: &'a mut f32,
    adam: AdamState,
    lr_base: f32,
    lr_sched: LrSchedule,
    total_steps: u32,
    lna: &'a Lna,
}

impl<'a> OptStep for MakeStep<'a> {
    fn step(&mut self, step: u32) -> TraceRow {
        self.sess.set_param(&self.lna.lg_param_name(), &[*self.lg]);
        let outs = self
            .sess
            .run(&[("freq_hz", &[F0_HZ]), ("d_output", &[1.0_f32])]);
        let loss = outs[0][0];
        let grad = outs[1][0];
        let lr = self.lr_sched.lr_at(self.lr_base, step, self.total_steps);

        let row = TraceRow::new(step)
            .with("loss", loss as f64)
            .with("lg", *self.lg as f64)
            .with("lg_target", lg_star() as f64)
            .with("grad", grad as f64)
            .with("lr", lr as f64)
            .with(
                "return_loss_db",
                if loss > 0.0 { (-10.0 * (loss as f64).log10()).max(0.0) } else { 120.0 },
            );

        // Adam update *after* logging, so step 0 reflects the
        // initial state and step N the post-update state.
        if step < self.total_steps {
            let mut params = [*self.lg];
            self.adam.step(&mut params, &[grad], lr, step + 1);
            *self.lg = params[0];
        }
        row
    }
}

// ── S₁₁ frequency sweep (used for the before/after chart) ─────────────

fn sweep_s11(lna: &Lna, lg_h: f32) -> Vec<(f64, f64)> {
    let g = lna.build_forward_graph();
    let mut sess = Session::new(Device::Cpu).compile(g);
    sess.set_param(&lna.gm_param_name(),  &[GM]);
    sess.set_param(&lna.cgs_param_name(), &[CGS]);
    sess.set_param(&lna.lg_param_name(),  &[lg_h]);
    sess.set_param(&lna.ls_param_name(),  &[LS]);
    sess.set_param(&lna.ld_param_name(),  &[LD]);
    sess.set_param(&lna.rl_param_name(),  &[RL]);

    let mut out = Vec::new();
    for k in 0..=80 {
        let f_hz = 1.0e9 + (k as f32) * 0.05e9; // 1.0..5.0 GHz, 50 MHz step
        let o = sess.run(&[("freq_hz", &[f_hz])]);
        let mag2 = (o[0][0].powi(2) + o[1][0].powi(2)) as f64;
        out.push((f_hz as f64 / 1e9, mag2));
    }
    out
}

fn find_param(g: &rlx_ir::Graph, name: &str) -> NodeId {
    g.nodes()
        .iter()
        .enumerate()
        .find_map(|(i, n)| match &n.op {
            Op::Param { name: pn, .. } if pn == name => Some(NodeId(i as u32)),
            _ => None,
        })
        .expect("param missing")
}

// ── Razavi / Lee literature row builder ───────────────────────────────

fn razavi_references(
    lg_final: f32,
    lna: &Lna,
    sess: &mut rlx_runtime::CompiledGraph,
) -> Vec<Reference> {
    sess.set_param(&lna.gm_param_name(),  &[GM]);
    sess.set_param(&lna.cgs_param_name(), &[CGS]);
    sess.set_param(&lna.lg_param_name(),  &[lg_final]);
    sess.set_param(&lna.ls_param_name(),  &[LS]);
    sess.set_param(&lna.ld_param_name(),  &[LD]);
    sess.set_param(&lna.rl_param_name(),  &[RL]);

    let outs = sess.run(&[("freq_hz", &[F0_HZ])]);
    let s11_re = outs[0][0];
    let s11_im = outs[1][0];
    let s21_mag = outs[2][0];
    let s11_mag2 = s11_re * s11_re + s11_im * s11_im;

    let z0 = 50.0_f32;
    let omega0 = TAU * F0_HZ;
    let pred_match = z0;                       // Razavi: gm·Ls/Cgs = Z₀
    let sim_match = GM * LS / CGS;
    let pred_resonance = 1.0 / (omega0 * omega0 * CGS); // Razavi: Lg + Ls = 1/(ω₀²Cgs)
    let sim_resonance = lg_final + LS;
    let pred_gain = GM * RL / (2.0 * omega0 * CGS * z0); // Razavi 5.79

    vec![
        Reference {
            citation: "Razavi, *RF Microelectronics* (2nd ed., Pearson, 2011, ISBN 978-0-13-713473-1; \
                       [DOI:10.5555/2207144](https://doi.org/10.5555/2207144))".to_string(),
            formula: r"$g_m \cdot L_s / C_{gs} = Z_0$ (input-match real part)".to_string(),
            predicted: format!("{pred_match:.2} Ω"),
            simulated: format!("{sim_match:.2} Ω"),
            passed: (sim_match - pred_match).abs() / pred_match < 1e-3,
        },
        Reference {
            citation: "Razavi, *RF Microelectronics* (2nd ed.)".to_string(),
            formula: r"$\omega_0^2 (L_g + L_s) C_{gs} = 1$ (resonance)".to_string(),
            predicted: format!("{:.4e} H", pred_resonance),
            simulated: format!("{:.4e} H", sim_resonance),
            passed: (sim_resonance - pred_resonance).abs() / pred_resonance < 1e-2,
        },
        Reference {
            citation: "Lee, *The Design of CMOS Radio-Frequency Integrated Circuits* \
                       (2nd ed., Cambridge UP, 2003, ISBN 978-0-521-83539-8; \
                       [DOI:10.1017/CBO9780511817281](https://doi.org/10.1017/CBO9780511817281))"
                .to_string(),
            formula: r"$|S_{21}| = g_m R_L / (2 \omega_0 C_{gs} Z_0)$ (matched gain)".to_string(),
            predicted: format!("{pred_gain:.4}"),
            simulated: format!("{s21_mag:.4}"),
            passed: (s21_mag - pred_gain).abs() / pred_gain < 1e-3,
        },
        Reference {
            citation: "Pozar, *Microwave Engineering* (4th ed., Wiley, 2011, \
                       ISBN 978-0-470-63155-3; \
                       [DOI:10.1002/0471221015](https://doi.org/10.1002/0471221015))"
                .to_string(),
            formula: r"$S_{11} = (Z_\text{in} - Z_0)/(Z_\text{in} + Z_0)$, \
                       $|S_{11}|^2 \to 0$ at match".to_string(),
            predicted: "≤ 1e-4".to_string(),
            simulated: format!("{s11_mag2:.4e}"),
            passed: s11_mag2 < 1e-4,
        },
    ]
}

// ── Long-form prose blocks ────────────────────────────────────────────

const LNA_BACKGROUND: &str = r#"A **low-noise amplifier (LNA)** is the first active stage in nearly every RF receiver — a cellular front-end, a Wi-Fi radio, a GPS module, a radiotelescope IF chain, the readout of a Josephson-junction qubit. It sits between the antenna (or sensor) and the rest of the chain, and its job is to add **as much gain as possible while contributing as little noise as possible**, because every stage that follows scales the LNA's noise figure (`F`) into the system NF via Friis' equation.

The **inductively-degenerated cascode** is the canonical RF-CMOS LNA topology: a common-source NMOS (`M1`) with a small spiral inductor `Ls` in its source, a cascode NMOS (`M2`) on top for output isolation, a gate-side input inductor `Lg` for the input match, and a parallel-LC tank (`Ld`, `R_L`) at the drain. The reason this exact topology shows up in Razavi ch. 5, Lee ch. 11, and every 2.4 GHz / 5.8 GHz front-end paper since the late 1990s is that source-degeneration with a *purely inductive* element creates a real input impedance **without resistive loss** — the inductor `Ls` synthesises a `gm·Ls/Cgs`-valued resistor at the gate node, which can be matched to the antenna's `Z₀ = 50 Ω` while the inductor itself dissipates nothing. The LC at the drain tunes the output to the same operating frequency, parking the gain peak on the band of interest.

This makes the matching problem a clean exercise in differentiable RF design: the input impedance has a closed-form expression in `(gm, Cgs, Lg, Ls)`, the loss is `|S₁₁(ω₀)|²`, and gradient descent on `Lg` lands the optimum that Razavi derives algebraically — `ω₀² (Lg + Ls) Cgs = 1` — to within float precision. That's what the rest of this document shows."#;

const LNA_OBJECTIVE: &str = r#"**Loss:** `|S₁₁(f₀)|²` at the design frequency `f₀ = 2.4 GHz`.

**Closed-form input impedance** (Razavi §5.3.3):

$$Z_\mathrm{in}(\omega) = j\omega(L_g + L_s) + \frac{1}{j\omega C_{gs}} + \frac{g_m \, L_s}{C_{gs}}$$

**Match conditions** (real part = `Z₀`, imaginary part = 0 at `ω₀`):

$$\frac{g_m L_s}{C_{gs}} = Z_0, \qquad \omega_0^2 (L_g + L_s) C_{gs} = 1$$

The first condition is satisfied by sizing — `gm = 50 mS`, `Cgs = 250 fF`, `Ls = 250 pH` give `gm·Ls/Cgs = 50 Ω` exactly. That leaves the second as a one-parameter inverse-design problem: tune `Lg` until `Im Z_in(ω₀) = 0`, i.e. `Lg = 1/(ω₀² C_{gs}) − Ls ≈ 17.34 nH`. Adam on `Lg` reproduces this closed-form optimum below."#;

const LNA_NOTES: &str = r#"- **Reference impedance:** `Z₀ = 50 Ω` throughout.
- **Param scale:** Adam's learning rate is set to `Lg* · 0.01` so step sizes track the parameter's nano-henry magnitude — the same trick `mzi_ml_trace` uses to keep step sizes commensurate with `n_eff` units.
- **Behavioral model is single-frequency at resonance for `|S₂₁|`** (Razavi eq. 5.79). A full LC-tank `S₂₁(f)` model lands when the drain-tank `Ld + R_L` story justifies it.
- **Layout side is independent of behavioral side.** The Mosfet's geometric W/L doesn't drive the small-signal `gm` / `Cgs` — those are independent rlx params, matching how a designer hands PEX-extracted values to a layout-extraction step. Tying them together is a follow-up when MOSFET-model integration justifies it."#;
