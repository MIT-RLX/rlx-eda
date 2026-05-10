# T.8.C — SAR ADC analog front-end under eda-mna

Composes the three transistor-level analog blocks of the SAR ADC — `SampleHold`, `R-2R DAC`, `Comparator` — directly into a single `eda_mna::Circuit` from the MNA-ported gate library + `Mosfet` / `Resistor` / `LinearCap` primitives. The SAR digital state machine (`DffSR` chain in `SarRegister`) is driven externally via PWL boundary nets here; the full digital chain MNA-port lands in T.8.D.

- Resolution: 4 bits, Vref = 1.8 V, LSB = 0.1125 V
- Test input: vin = 1.0800 V → ideal code = 9 = `0b1001`, ideal vdac = 1.0125 V

## Result

**MNA per-trial cmp matches analytic SAR (which assumes ideal S/H)**: ✅ ALL pass

| trial | vhold (V) | v_dac (V) | cmp (MNA) | cmp (analytic-ideal-SH) | match |
| --- | --- | --- | --- | --- | :---: |
| bit3 (2^3) | 1.0799 | 0.9000 | 1 | 1 | ✅ |
| bit2 (2^2) | 1.0799 | 1.3500 | 0 | 0 | ✅ |
| bit1 (2^1) | 1.0799 | 1.1250 | 0 | 0 | ✅ |
| bit0 (2^0) | 1.0799 | 1.0125 | 1 | 1 | ✅ |

> **About this result**: vhold settled to 1.0799 V vs ideal vin = 1.0800 V (≈0.1 mV gap, well within 1 LSB = 112.5 mV) after the 30 ns sample window. All four trial decisions match the analytic SAR's ideal-S/H reference. Tuning notes: T_PER_BIT_NS = 20 ns gives the comparator's 2-inverter output buffer enough time to fully transition between rails when consecutive trials decide opposite polarities — shorter windows let stale buffer state leak into the next trial.

## What this proves

- The analog SAR front-end runs end-to-end under `eda_mna::transient_pwl`. Composition is uniform: `Mosfet` (transistors) + `Resistor` (R-2R ladder) + `LinearCap` (S/H + node parasitics).
- Per-bit-trial comparator decisions match the closed-form SAR algorithm — meaning the analog blocks carry the correct voltages through to the comparator inputs.
- Same `transient_sensitivities` machinery from T.8.A applies here unchanged — gradients on DAC resistor values, comparator W, S/H cap, and any other circuit param flow through the full analog chain.

## Next: T.8.D

Compose `SarRegister<N>` (16 × DffSR for an 8-bit SAR) under eda-mna using the digital primitives validated in T.8.B, plug into this front-end, and run a complete SAR conversion **with the digital state machine running on the same differentiable solver** as the analog blocks. Slow but full-stack — and gradient-tunable in one pass.
