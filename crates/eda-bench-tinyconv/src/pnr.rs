//! Optional PnR loop — Adam-on-HPWL placement of the controller
//! FSM cells, separate from the inner tile-sizing loop.
//!
//! ## Why optional
//!
//! The MAC tile array's uniform `tile_grid` placement is provably
//! optimal for a regular weight-stationary topology — any
//! non-grid placement strictly increases inter-tile wirelength.
//! AD-driven placement there is wasted complexity. The CONTROLLER
//! FSM is the irregular logic where placement matters; that's
//! where this module lives.
//!
//! ## Runtime toggle
//!
//! `PnrMode::Disabled` (default for backward compat) → `run` is a
//! no-op returning `None`. `PnrMode::AdamHpwl(cfg)` → builds a
//! synthetic controller netlist + runs Adam on HPWL via
//! `eda_pnr::ad::hpwl_loss_graph`. Either way, the inner tile
//! loop runs identically; PnR is a sibling, not a wrapper.
//!
//! Lift the pattern verbatim from `eda-pnr/tests/ad_hpwl.rs` —
//! that's the smoke test for the same machinery.

use eda_pnr::{
    ad::{hpwl_loss_graph, position_param_ids, DifferentiablePlacement},
    Netlist,
};
use klayout_core::{
    Angle90, Bbox, CellBuilder, CellId, Library, Point, Port, Rect, Shape,
};
use klayout_pdk::pdk;
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use spike_divider_block::{Adam, Optimizer};

use super::optimization::OptError;

/// Toggle for the PnR sub-step. Caller decides per-run; passes
/// to [`run`] which returns `None` when disabled.
///
/// Serializes via the adjacent-tagged shape so a `bench.toml`
/// section is human-friendly:
///
/// ```toml
/// [pnr]
/// enabled = true
/// [pnr.adam]
/// max_steps = 200
/// learning_rate = 5000.0
/// beta = 0.0001
/// ```
///
/// See `BenchConfig` (`crate::config`).
#[derive(Debug, Clone, Copy)]
pub enum PnrMode {
    /// Skip the PnR loop entirely. v1 default — keeps the bench
    /// pipeline lean for runs that only care about tile sizing.
    Disabled,
    /// Run Adam on HPWL of a synthetic controller netlist using
    /// the supplied schedule.
    AdamHpwl(PnrAdamConfig),
}

impl Default for PnrMode {
    fn default() -> Self {
        PnrMode::Disabled
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PnrAdamConfig {
    pub max_steps: usize,
    pub learning_rate: f32,
    /// Smooth-max sharpness for HPWL log-sum-exp. Smaller β =
    /// smoother gradients but looser HPWL approximation; larger
    /// β = exact bbox but ill-conditioned. `1e-4` works for
    /// initial seeds spread across ~50 µm.
    pub beta: f32,
}

impl Default for PnrAdamConfig {
    fn default() -> Self {
        Self {
            max_steps: 200,
            learning_rate: 5_000.0, // positions in DBU; large lr OK
            beta: 1e-4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PnrSummary {
    pub initial_hpwl: f32,
    pub final_hpwl: f32,
    pub n_steps: usize,
    pub final_positions: Vec<(f32, f32)>,
}

// Minimal PDK so the PnR loop doesn't depend on a foundry GDS or
// a populated mock library.
pdk! {
    pub PnrTestPdk {
        dbu: 1000,
        layers: { METAL1 = (10, 0) },
        ports: { Electrical },
    }
}

fn dummy_cell(lib: &Library, pdk: &PnrTestPdk, name: &str) -> CellId {
    let mut cb = CellBuilder::new(name);
    cb.add_shape(
        pdk.METAL1,
        Shape::Box(Rect::new(Bbox::new(
            Point::new(0, 0),
            Point::new(1_000, 1_000),
        ))),
    );
    cb.add_port(
        Port::new("p", pdk.METAL1, Point::new(0, 0), Angle90::E, 1_000)
            .with_kind(PnrTestPdk::Electrical),
    );
    lib.insert(cb)
}

/// Build the synthetic controller netlist: 4 cells touching one
/// shared net. v1.5+ replaces this with the real controller's
/// `Block` lowering once the FSM placement is wired through
/// `eda-stdcells`.
fn build_synth_netlist(lib: &Library, pdk: &PnrTestPdk) -> Netlist {
    let c = dummy_cell(lib, pdk, "ctrl_cell");
    let mut nl = Netlist::new("controller").with_default_signal_layer(pdk.METAL1);
    let i0 = nl.add_instance("ctrl_inv1", c);
    let i1 = nl.add_instance("ctrl_nand1", c);
    let i2 = nl.add_instance("ctrl_inv2", c);
    let i3 = nl.add_instance("ctrl_dff1", c);
    nl.connect("ctrl_chain", i0, "p");
    nl.connect("ctrl_chain", i1, "p");
    nl.connect("ctrl_chain", i2, "p");
    nl.connect("ctrl_chain", i3, "p");
    nl
}

/// Run the configured PnR mode. Returns `None` for `Disabled`,
/// `Some(summary)` for `AdamHpwl`. Surfacing the mode in the
/// signature makes the runtime toggle explicit at every call site.
pub fn run(mode: PnrMode) -> Result<Option<PnrSummary>, OptError> {
    match mode {
        PnrMode::Disabled => Ok(None),
        PnrMode::AdamHpwl(cfg) => run_adam_hpwl(cfg).map(Some),
    }
}

/// Pure-function variant of the Adam-on-HPWL pass. `run(PnrMode::AdamHpwl(cfg))`
/// dispatches here; tests can call this directly to bypass the toggle.
pub fn run_adam_hpwl(cfg: PnrAdamConfig) -> Result<PnrSummary, OptError> {
    let lib = PnrTestPdk::new_library("pnr_controller");
    let pdk = PnrTestPdk::register(&lib);
    let nl = build_synth_netlist(&lib, &pdk);

    // Seed: 4 cells spread across a 50 µm × 50 µm region. HPWL
    // optimum is everyone collapsed to a single point (modulo
    // smooth-max softness).
    let seed_xy = [
        (0.0_f32, 0.0_f32),
        (50_000.0, 0.0),
        (50_000.0, 50_000.0),
        (0.0, 50_000.0),
    ];
    let placement = DifferentiablePlacement {
        instance_xy: seed_xy.to_vec(),
        beta: cfg.beta,
    };

    let fwd = hpwl_loss_graph(&nl, &lib, placement.beta);
    let pos_ids = position_param_ids(&fwd, &nl);
    let bwd = grad_with_loss(&fwd, &pos_ids);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    // Push initial params.
    let push_params = |sess: &mut rlx_runtime::CompiledGraph, p: &DifferentiablePlacement| {
        for (idx, (x, y)) in p.instance_xy.iter().enumerate() {
            sess.set_param(&p.x_param_name(&nl, idx), &[*x]);
            sess.set_param(&p.y_param_name(&nl, idx), &[*y]);
        }
    };
    push_params(&mut sess, &placement);

    // Initial HPWL.
    let initial_hpwl = sess.run(&[("d_output", &[1.0_f32][..])])[0][0];

    // 2 params per cell × 4 cells = 8 Adam-targeted params.
    let mut params: Vec<f32> = placement
        .instance_xy
        .iter()
        .flat_map(|(x, y)| [*x, *y])
        .collect();
    let mut adam = Adam::new(cfg.learning_rate, params.len());

    let mut last_loss = initial_hpwl;
    for _ in 0..cfg.max_steps {
        // Update Param values from `params` vec.
        for (idx, chunk) in params.chunks(2).enumerate() {
            sess.set_param(&placement.x_param_name(&nl, idx), &chunk[0..1]);
            sess.set_param(&placement.y_param_name(&nl, idx), &chunk[1..2]);
        }
        let outs = sess.run(&[("d_output", &[1.0_f32][..])]);
        last_loss = outs[0][0];
        // Gradients: outs[1..] in order of pos_ids registration —
        // alternating x, y per instance.
        let grads: Vec<f32> = (1..outs.len()).map(|i| outs[i][0]).collect();
        if !last_loss.is_finite() || grads.iter().any(|g| !g.is_finite()) {
            return Err(OptError::InnerDiverged { steps: 0 });
        }
        adam.step(&mut params, &grads);
    }

    let final_positions: Vec<(f32, f32)> =
        params.chunks(2).map(|c| (c[0], c[1])).collect();

    Ok(PnrSummary {
        initial_hpwl,
        final_hpwl: last_loss,
        n_steps: cfg.max_steps,
        final_positions,
    })
}
