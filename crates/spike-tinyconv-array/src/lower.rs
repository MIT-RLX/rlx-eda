//! Lower a `rlx_fpga::model::Model` to an `ArrayBlock`.
//!
//! This is the *second* lowering of the FPGA model — the first is
//! `rlx-fpga::codegen` → SystemVerilog. Both consume the same source
//! of truth (`rlx_fpga::model::Model`), so any divergence between
//! FPGA bitstream and silicon GDS at the architectural level is a
//! bug in one of these two lowerings, not a difference in the model.
//!
//! v1 lowering: walk `model.layers`, count weight-stationary tiles
//! needed per layer (`Conv2d` = `c_in · c_out · kh · kw`,
//! `Dense` = `in_features · out_features`, `Relu`/`MaxPool` = 0),
//! and verify each layer fits the supplied `ArrayConfig`'s
//! `nx · ny` budget. Layers are packed onto the same tile grid
//! time-multiplexed by the controller (controller body deferred —
//! the lowering's job is to validate fit, not schedule).
//!
//! When real DADO lands (PLAN.md step 6+), it'll wrap this:
//! search over `ArrayConfig` candidates, lower each, score by
//! `inner::run` final loss + bench `measure_physical`.

use rlx_fpga::model::{Layer, Model};

use crate::array::{ArrayBlock, ArrayConfig};

#[derive(Debug, thiserror::Error)]
pub enum LowerError {
    #[error("model has no layers")]
    EmptyModel,
    #[error("layer {layer} requires {needed} MAC tiles, exceeds budget {budget}")]
    OverBudget {
        layer: usize,
        needed: usize,
        budget: usize,
    },
}

/// Lower `model` to an `ArrayBlock` under `config`. Validates that
/// every Conv2d / Dense layer fits the `nx · ny` tile budget;
/// returns `OverBudget` on the first overflow.
///
/// `Relu` and `MaxPool2d` layers don't consume MAC tiles —
/// they're handled by the controller's pass-through path.
pub fn lower(model: &Model, config: ArrayConfig) -> Result<ArrayBlock, LowerError> {
    if model.layers.is_empty() {
        return Err(LowerError::EmptyModel);
    }
    let (nx, ny) = config.grid;
    let budget = nx * ny;
    for (idx, layer) in model.layers.iter().enumerate() {
        let needed = weight_count(layer);
        if needed > budget {
            return Err(LowerError::OverBudget {
                layer: idx,
                needed,
                budget,
            });
        }
    }
    Ok(ArrayBlock::new(format!("lowered_{}", model.name), config))
}

/// Number of MAC tiles a layer needs in a weight-stationary array.
/// Public so the bench harness can pre-size an `ArrayConfig` from a
/// `Model` without calling `lower`.
pub fn weight_count(layer: &Layer) -> usize {
    match layer {
        Layer::Conv2d {
            c_in,
            c_out,
            kh,
            kw,
            ..
        } => c_in * c_out * kh * kw,
        Layer::Dense {
            in_features,
            out_features,
            ..
        } => in_features * out_features,
        // Relu / MaxPool2d / Argmax / etc. don't consume MAC tiles —
        // the controller passes activations through directly.
        _ => 0,
    }
}

/// Helper: max `weight_count` across all layers — the smallest
/// `nx · ny` that lets `lower` succeed without time-multiplexing.
pub fn min_required_tiles(model: &Model) -> usize {
    model.layers.iter().map(weight_count).max().unwrap_or(0)
}

/// Cycles a layer takes on a `budget`-tile weight-stationary array
/// **assuming time-multiplexing**: load the array with the next
/// `budget` weights, run one activation pass, repeat until every
/// weight has been used. `ceil(weight_count / budget)` per
/// activation pass; v1 assumes one pass per inference.
///
/// For layers that fit in one shot (`weight_count ≤ budget`) this
/// returns 1; layers that exceed the budget cycle the array
/// proportionally — and the time-multiplexing config swaps the
/// hard `OverBudget` error for a soft "this will be slow" signal.
pub fn cycles_per_layer(weight_count: usize, budget: usize) -> usize {
    if budget == 0 {
        return usize::MAX; // degenerate; caller should reject
    }
    weight_count.div_ceil(budget).max(1)
}

/// Total cycles to evaluate the whole model on a time-multiplexed
/// array of `budget = nx · ny` tiles. Sum of `cycles_per_layer`
/// over every layer (`Relu` / `MaxPool` / `Argmax` consume zero
/// MAC cycles → 0).
///
/// This is the natural denominator for "throughput vs grid size"
/// curves the bench reporter draws when DADO compares ArrayConfig
/// candidates: smaller grid → more cycles → larger end-to-end
/// latency.
pub fn total_cycles(model: &Model, budget: usize) -> usize {
    model
        .layers
        .iter()
        .map(|l| {
            let w = weight_count(l);
            if w == 0 {
                0
            } else {
                cycles_per_layer(w, budget)
            }
        })
        .sum()
}

/// Time-multiplexed alternative to `lower`: instead of erroring on
/// `OverBudget`, returns the resulting array along with the cycle
/// count so DADO can score grid-vs-throughput trade-offs. The
/// returned `ArrayBlock` is the *physical* grid (size = `config.grid`),
/// not a notional array sized for the largest layer.
pub fn lower_time_multiplexed(
    model: &Model,
    config: ArrayConfig,
) -> Result<(ArrayBlock, usize), LowerError> {
    if model.layers.is_empty() {
        return Err(LowerError::EmptyModel);
    }
    let (nx, ny) = config.grid;
    let budget = nx * ny;
    let cycles = total_cycles(model, budget);
    let array = ArrayBlock::new(format!("lowered_tm_{}", model.name), config);
    Ok((array, cycles))
}
