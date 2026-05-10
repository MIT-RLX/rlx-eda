//! RTL simulation backend — drives the rlx-fpga-emitted SystemVerilog
//! through Verilator (in docker) and produces real cycle counts.
//!
//! ## What this is
//!
//! A `Backend`-trait impl whose `measure_inference` method runs the
//! actual emitted RTL: writes the input image as `tb_image.mem`,
//! generates a testbench with a cycle counter, invokes Verilator
//! inside a pinned `verilator/verilator` container, parses
//! `RESULT pred=N cycles=N` from the simulator's stdout, returns
//! the prediction + cycle count.
//!
//! ## Why "the silicon time, not the wall-clock"
//!
//! Verilator on the host takes seconds to compile + run; that's
//! overhead. The number this backend reports is the **simulated
//! cycle count** — the latency the actual silicon would exhibit at
//! the configured `period_ns`. `InferenceMetrics::with_simulated`
//! threads it through.
//!
//! ## Image is the input, prediction is the output
//!
//! Bytes of the test image become Verilog signals on the
//! BRAM-write port (`in_addr`, `in_we`, `in_din`); after `start`
//! pulses, signals propagate through every emitted Conv2d / ReLU /
//! MaxPool / Dense / Argmax module; `pred` carries the
//! classification. **Inference happens via the simulated
//! datapath**, not via a Rust shortcut.
//!
//! Gated by `bench-rtl-sim` feature. Tests are `#[ignore]` by
//! default — Verilator compile + simulate takes ~30–60 s per run
//! and needs docker on PATH.

#![cfg(feature = "bench-rtl-sim")]

use std::path::PathBuf;

use eda_container::DockerRun;

use crate::inference::SimulatedLatency;

use super::{BackendError};

/// Default verilator image, digest-pinned. Must match
/// `crates/eda-bench-tinyconv/docker/verilator-digest.txt`.
pub const DEFAULT_VERILATOR_IMAGE: &str =
    "verilator/verilator@sha256:342bf0e4468892f1abec354433c8332f177bc589f3553dbbd07b43fcec13eeb2";

/// RTL sim configuration. Construct via [`RtlSimBackend::new`].
pub struct RtlSimBackend {
    /// Directory containing the rlx-fpga emit output —
    /// `top.sv` + `weights/*.mem` + `layers/*.sv` +
    /// `primitives/*.sv`. Typically
    /// `<rlx-workspace>/rlx-fpga/hw/tinyconv_mnist/`.
    pub hw_dir: PathBuf,
    /// Pinned Verilator container image. Defaults to
    /// [`DEFAULT_VERILATOR_IMAGE`] (verilator 5.048 arm64 native).
    pub docker_image: String,
    /// Target clock period in ns. Drives the simulated-silicon
    /// time conversion; doesn't affect the RTL run itself
    /// (Verilator simulates cycles abstractly). Real timing
    /// closure happens at L3 (gate-level + SDF).
    pub period_ns: f64,
    /// Activation BRAM input port width. Default 784 = 28×28 image
    /// for TinyConv-MNIST. Other models would override.
    pub input_len: usize,
}

impl RtlSimBackend {
    pub fn new(hw_dir: PathBuf) -> Self {
        Self {
            hw_dir,
            docker_image: DEFAULT_VERILATOR_IMAGE.to_string(),
            period_ns: 10.0, // 100 MHz target
            input_len: 784,
        }
    }

    /// Run one inference end-to-end: image bytes → simulated RTL
    /// → predicted class + cycle count. The whole computation
    /// happens via Verilator's simulation of the emitted SV.
    pub fn measure_inference_one(&self, image: &[i8]) -> Result<RtlSimResult, BackendError> {
        if image.len() != self.input_len {
            return Err(BackendError::Toolchain(format!(
                "rtl_sim: image length {} != expected {}",
                image.len(),
                self.input_len
            )));
        }

        // 1. Write the input image as `tb_image.mem` (one byte per
        //    line, hex; matches `$readmemh` semantics in the tb).
        let mem_path = self.hw_dir.join("tb_image.mem");
        let mem: String = image
            .iter()
            .map(|b| format!("{:02x}", *b as u8))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&mem_path, mem)
            .map_err(|e| BackendError::Toolchain(format!("write tb_image.mem: {e}")))?;

        // 2. Generate the cycle-counting testbench. The shipping
        //    tb.sv only prints `pred`; we need both pred + cycles
        //    in a parseable line.
        let tb = self.generate_testbench();
        let tb_path = self.hw_dir.join("tb_bench.sv");
        std::fs::write(&tb_path, tb)
            .map_err(|e| BackendError::Toolchain(format!("write tb_bench.sv: {e}")))?;

        // 3. Run Verilator inside docker, build + execute the sim.
        let workdir = "/work";
        let bash_cmd = format!(
            // List explicit SV files so verilator gets a stable
            // build set. `--binary` builds + invokes the sim in
            // one shot.
            "set -e; cd {workdir}; \
             SV_FILES=$(find . -name '*.sv' -not -name 'tb.sv' | sort); \
             verilator --binary --build -j 0 --top-module tb_bench \
               -Wno-fatal -Wno-WIDTHEXPAND -Wno-UNUSEDSIGNAL \
               $SV_FILES -o sim_bench 1>&2; \
             ./obj_dir/sim_bench"
        );
        let stdout = DockerRun::new(&self.docker_image)
            .entrypoint("bash")
            .workdir(workdir)
            .mount(self.hw_dir.clone(), PathBuf::from(workdir))
            .arg("-c")
            .arg(bash_cmd)
            .run_with_stdin(b"")
            .map_err(|e| BackendError::Toolchain(format!("verilator docker: {e}")))?;

        // 4. Parse `RESULT pred=N cycles=N`.
        parse_result(&stdout).map_err(BackendError::Toolchain)
    }

    fn generate_testbench(&self) -> String {
        let n = self.input_len;
        // Half-period in ns = 5 → 10 ns clock period @ 100 MHz.
        // (Verilator cycle counts are abstract; the period only
        // matters for the simulated-time conversion at the bench
        // reporter, not here.)
        format!(
            r#"// Auto-generated by RtlSimBackend. Drives one inference and
// prints `RESULT pred=N cycles=N` for the bench harness to parse.
`timescale 1ns/1ps
module tb_bench;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst = 1;
    logic start = 0;
    logic done;
    logic [9:0] in_addr = '0;
    logic in_we = 0;
    logic signed [7:0] in_din = '0;
    logic signed [7:0] pred;

    top u_top (
        .clk(clk), .rst(rst), .start(start), .done(done),
        .in_addr(in_addr), .in_we(in_we), .in_din(in_din),
        .pred(pred)
    );

    logic signed [7:0] image_mem [0:{n_minus_1}];

    // Cycle counter: counts rising edges between start=1 and done=1.
    longint cycles_counter = 0;
    logic counting = 0;
    always @(posedge clk) begin
        if (counting && !done) cycles_counter <= cycles_counter + 1;
        if (start) counting <= 1;
        if (done) counting <= 0;
    end

    initial begin
        $readmemh("tb_image.mem", image_mem);
        rst = 1; #20; rst = 0;
        for (int i = 0; i < {n}; i++) begin
            @(posedge clk);
            in_addr <= i[9:0];
            in_we <= 1'b1;
            in_din <= image_mem[i];
        end
        @(posedge clk); in_we <= 1'b0;
        @(posedge clk); start <= 1'b1;
        wait (done);
        @(posedge clk); start <= 1'b0;
        $display("RESULT pred=%0d cycles=%0d", $signed(pred), cycles_counter);
        $finish;
    end
endmodule
"#,
            n = n,
            n_minus_1 = n - 1
        )
    }
}

/// Result of one RTL-simulated inference.
#[derive(Debug, Clone, Copy)]
pub struct RtlSimResult {
    pub prediction: i8,
    pub cycles: u64,
}

impl RtlSimResult {
    /// Project to a [`SimulatedLatency`] for the bench harness.
    pub fn to_simulated(&self, period_ns: f64) -> SimulatedLatency {
        SimulatedLatency::from_cycles(self.cycles, period_ns)
    }
}

fn parse_result(stdout: &str) -> Result<RtlSimResult, String> {
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("RESULT ") {
            let mut prediction: Option<i8> = None;
            let mut cycles: Option<u64> = None;
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("pred=") {
                    prediction = v.parse().ok();
                } else if let Some(v) = tok.strip_prefix("cycles=") {
                    cycles = v.parse().ok();
                }
            }
            if let (Some(p), Some(c)) = (prediction, cycles) {
                return Ok(RtlSimResult { prediction: p, cycles: c });
            }
        }
    }
    Err(format!(
        "rtl_sim: missing RESULT line in verilator stdout. Output:\n{stdout}"
    ))
}
