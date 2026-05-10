//! SystemVerilog codegen for the rlx-eda silicon flow.
//!
//! Sibling of `rlx_fpga::codegen` — same Q0.31 arithmetic, same
//! semantics — but emits parallel weight-baked architectures
//! targeting ASIC instead of sequential BRAM-based controllers
//! targeting FPGA.
//!
//! Selected by [`super::array::WeightStrategy`]:
//!
//! - `Bram` — defer to `rlx-fpga::codegen` (no rlx-eda emit needed).
//! - `MaskRom` — stub, pending.
//! - `BakedConstants` — [`unrolled`] writes a fully-unrolled
//!   dedicated-multiplier-per-weight Verilog file with weights as
//!   `localparam` constants. No BRAM, no controller cycles —
//!   compute completes in pipeline-depth cycles, not work-count.

pub mod unrolled;

pub use unrolled::{
    emit_unrolled_dense_per_channel, emit_unrolled_dense_tb, emit_unrolled_dense_top, EmitError,
};
