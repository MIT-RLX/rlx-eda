//! FPGA backend: `rlx-fpga` emit → nextpnr-ecp5 → parsed reports
//! for physical metrics; `rlx_fpga::reference` for functional
//! measurement (bit-exact pure Rust — no toolchain required).
//!
//! Functional ground truth at scale. The only backend fast enough to
//! run the full 10k MNIST test set under any candidate config in
//! seconds. Therefore this is what L5 yield-gate evaluation calls,
//! and what the inner Adam loop's accuracy term consumes (with the
//! tile noise model injected) so SPICE never enters the inner loop.
//!
//! Serves L1, L2, L5. Cannot serve L3 or L4 (no SDF / no analog
//! parasitic sim).
//!
//! Physical measurements are gated by `bench-fpga` (nextpnr-ecp5 +
//! yosys); functional L1 is unconditional because `rlx_fpga::reference`
//! is pure Rust.

use rlx_fpga::model::Model;
use rlx_fpga::reference;

use super::{Backend, BackendError};
use crate::metrics::{functional::Level, Functional, Physical};

pub struct FpgaBackend {
    /// Output of `cargo run -p rlx-fpga --bin rlx-fpga-emit`. Used
    /// by `measure_physical` for the nextpnr-ecp5 report path.
    pub hw_dir: std::path::PathBuf,
    /// The TinyConv model the backend evaluates. Same value type
    /// `spike_tinyconv_array::lower` consumes — guarantees bench
    /// and lowering see the same source of truth.
    pub model: Model,
    /// Test set the functional arm runs against. Each entry is one
    /// `(image, label)` pair; the bench reports top-1 accuracy
    /// over the supplied set. Empty set → no functional eval.
    pub test_set: Vec<(Vec<i8>, u8)>,
}

impl FpgaBackend {
    pub fn new(hw_dir: std::path::PathBuf, model: Model) -> Self {
        Self {
            hw_dir,
            model,
            test_set: Vec::new(),
        }
    }

    /// Attach a test set. Caller supplies `(image, label)` pairs;
    /// each image must be `model.input_len` long.
    pub fn with_test_set(mut self, test_set: Vec<(Vec<i8>, u8)>) -> Self {
        self.test_set = test_set;
        self
    }
}

impl Backend for FpgaBackend {
    fn name(&self) -> &'static str {
        "fpga"
    }

    fn measure_physical(&self) -> Result<Physical, BackendError> {
        #[cfg(not(feature = "bench-fpga"))]
        {
            return Err(BackendError::NotEnabled("fpga", "bench-fpga"));
        }
        #[cfg(feature = "bench-fpga")]
        {
            unimplemented!("nextpnr-ecp5 --report → LUT/BRAM/Fmax")
        }
    }

    /// L1 reference (Rust pure, bit-exact) is always supported. L2
    /// (RTL sim) and L5 (PVT × MC at scale) require external
    /// toolchains and stay gated by `bench-fpga`.
    ///
    /// `images` is interpreted as **indices into `self.test_set`** —
    /// caller picks which subset to evaluate. Empty `images` → run
    /// the entire test set.
    fn measure_functional(
        &self,
        level: Level,
        images: &[u32],
    ) -> Result<Functional, BackendError> {
        match level {
            Level::L1Reference => self.measure_l1_reference(images),
            _ => {
                #[cfg(not(feature = "bench-fpga"))]
                {
                    return Err(BackendError::NotEnabled("fpga", "bench-fpga"));
                }
                #[cfg(feature = "bench-fpga")]
                {
                    unimplemented!("L2/L3/L5 sim path — needs external toolchain")
                }
            }
        }
    }
}

impl FpgaBackend {
    fn measure_l1_reference(&self, images: &[u32]) -> Result<Functional, BackendError> {
        if self.test_set.is_empty() {
            return Err(BackendError::Toolchain(
                "fpga: measure_functional(L1) called with empty test set; \
                 attach via `FpgaBackend::with_test_set`"
                    .into(),
            ));
        }

        // `images` empty → eval everything; otherwise index into the
        // test set and surface OOB indices as Toolchain errors.
        let n = self.test_set.len();
        let indices: Vec<usize> = if images.is_empty() {
            (0..n).collect()
        } else {
            for &i in images {
                if (i as usize) >= n {
                    return Err(BackendError::Toolchain(format!(
                        "fpga: image index {i} out of bounds (test set has {n} entries)"
                    )));
                }
            }
            images.iter().map(|&i| i as usize).collect()
        };

        let mut correct = 0usize;
        let mut per_class_correct = [0u32; 10];
        let mut per_class_total = [0u32; 10];

        for idx in &indices {
            let (image, label) = &self.test_set[*idx];
            let (pred, _intermediates) = reference::run(&self.model, image);
            per_class_total[*label as usize] += 1;
            if pred == *label as usize {
                correct += 1;
                per_class_correct[*label as usize] += 1;
            }
        }

        let n_eval = indices.len();
        let top1_acc = correct as f64 / n_eval as f64;
        let per_class_acc = std::array::from_fn(|i| {
            let total = per_class_total[i];
            if total == 0 {
                0.0
            } else {
                per_class_correct[i] as f64 / total as f64
            }
        });

        Ok(Functional {
            level: Level::L1Reference,
            top1_acc,
            per_class_acc,
            divergence_first_layer: None,
            n_images: n_eval,
        })
    }
}
