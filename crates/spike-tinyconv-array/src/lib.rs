//! `spike-tinyconv-array` — full TinyConv silicon, lowering
//! `rlx_fpga::model::Model` to `Block` composition.
//!
//! See `eda-bench-tinyconv/PLAN.md` build-order step 5.

pub mod array;
pub mod codegen;
pub mod controller;
pub mod lower;

pub use array::{ArrayBlock, ArrayConfig};
pub use controller::ControllerFsm;
pub use lower::{
    cycles_per_layer, lower, lower_time_multiplexed, min_required_tiles, total_cycles,
    weight_count, LowerError,
};
