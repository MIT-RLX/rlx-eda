//! End-to-end VCD reader test on a realistic dump: nested scopes,
//! `$date`/`$version`/`$comment` preamble, mixed widths, and a bus that
//! transitions through unknown bits before settling.

use eda_waveform::vcd;

const SAMPLE: &str = r#"$date
   Mon May  9 12:00:00 2026
$end
$version
   cocotb 1.9.0
$end
$comment
   3-bit SAR ADC bench: clk + sample + 3-bit code bus.
$end
$timescale 100 ps $end
$scope module tb $end
$scope module dut $end
$var wire 1 ! clk $end
$var wire 1 " sample $end
$var wire 3 # code [2:0] $end
$upscope $end
$var wire 1 $ done $end
$upscope $end
$enddefinitions $end
$dumpvars
0!
0"
bxxx #
0$
$end
#10
1!
b000 #
#20
0!
1"
b101 #
#30
1!
0"
b111 #
#40
0!
1$
"#;

#[test]
fn realistic_vcd_with_nested_scopes() {
    let w = vcd::read(SAMPLE.as_bytes()).unwrap();

    // 100ps timescale × ticks {0, 10, 20, 30, 40} = 0..4 ns.
    assert_eq!(w.axis(), &[0.0, 1e-9, 2e-9, 3e-9, 4e-9]);

    // Nested scopes flatten with dots: tb -> dut -> clk, plus tb -> done.
    let names = w.signal_names();
    assert!(names.contains(&"tb.dut.clk"));
    assert!(names.contains(&"tb.dut.sample"));
    assert!(names.contains(&"tb.dut.code[2:0]"));
    assert!(names.contains(&"tb.done"));

    let clk = w.real("tb.dut.clk").unwrap();
    assert_eq!(clk, &[0.0, 1.0, 0.0, 1.0, 0.0]);

    let sample = w.real("tb.dut.sample").unwrap();
    assert_eq!(sample, &[0.0, 0.0, 1.0, 0.0, 0.0]);

    // Bus: starts unknown, then 0b000=0, 0b101=5, 0b111=7, then carries.
    let code = w.real("tb.dut.code[2:0]").unwrap();
    assert!(code[0].is_nan());
    assert_eq!(code[1], 0.0);
    assert_eq!(code[2], 5.0);
    assert_eq!(code[3], 7.0);
    assert_eq!(code[4], 7.0); // last value carries forward when no change

    // `done` is asserted late; carries 0 forward up to its first change.
    let done = w.real("tb.done").unwrap();
    assert_eq!(done, &[0.0, 0.0, 0.0, 0.0, 1.0]);
}
