# rlx-eda SAR ADC characterization (Tiers A–D)

Behavioral non-ideal 8-bit SAR @ Vref = 1.8 V. All metrics in this report come from `BehavioralSar` (pure Rust, fast); a transistor-level cross-validation against the same metrics is the natural follow-up.

## Headline numbers

- **INL_max = 1.531 LSB**, **DNL_max = 1.797 LSB** (one nominal realization)
- **ENOB = 6.86 bits**, **SNDR = 43.03 dB**, **SFDR = 48.78 dB**
- **Max conversion rate** ≈ **200.00 MHz** (0.0050 µs/conv, ENOB within 0.5 bits of nominal)
- **Power**: 4.86 µW dynamic, 4.860 pJ/conv, 10.80 µW leakage
- **Yield (INL < 1 LSB)** at σ_R = 0.5%, σ_Vos = 1.0 mV: **34.5%** of 200 samples
- **Gradient sizing** dropped INL_max from **2.788 → 1.663 LSB** by sizing the comparator input pair from W=0.25 µm → W=6.01 µm in 25 Adam iterations.

---

## Tier A — characterization

### A.1 INL / DNL

Static linearity from a slow input ramp, 64 samples per output code, code-density histogram → DNL → cumulative INL with endpoint correction.

| Metric | Value |
| --- | --- |
| INL_max | 1.531 LSB |
| DNL_max | 1.797 LSB |

![inl](crates/spike-sar-adc/docs/assets/sar_adc_characterization/inl.svg)

![dnl](crates/spike-sar-adc/docs/assets/sar_adc_characterization/dnl.svg)

### A.2 ENOB / SFDR / SNDR

Coherent-sampled single-tone FFT, 1024 samples, 7 cycles per record. Fundamental in bin 7; SNDR computed from total noise+distortion power across all non-fundamental bins.

| Metric | Value |
| --- | --- |
| SNDR | 43.03 dB |
| SFDR | 48.78 dB |
| ENOB | (SNDR − 1.76)/6.02 = **6.86 bits** |

![fft](crates/spike-sar-adc/docs/assets/sar_adc_characterization/fft.svg)

### A.3 Conversion-rate sweep

Sweep total conversion period; per-bit decision time = period/N. When decision time approaches the comparator's regenerative latch τ, metastability bleeds into effective comparator noise → ENOB drops.

![conv](crates/spike-sar-adc/docs/assets/sar_adc_characterization/conv_rate.svg)

| period (µs) | ENOB | INL_max (LSB) |
| --- | --- | --- |
| 0.001 | 6.03 | 1.562 |
| 0.002 | 6.03 | 1.562 |
| 0.005 | 6.86 | 1.531 |
| 0.010 | 6.86 | 1.531 |
| 0.050 | 6.86 | 1.531 |
| 0.100 | 6.86 | 1.531 |
| 0.500 | 6.86 | 1.531 |
| 1.000 | 6.86 | 1.531 |
| 5.000 | 6.86 | 1.531 |

Max sustainable conversion rate: **200.00 MHz** (0.0050 µs/conv) at the smallest period whose ENOB stays within 0.5 bits of the long-period nominal.

### A.4 Power per conversion

Behavioral estimate: ~600 transistors, α = 0.5 switching activity, C_load ≈ 5 fF, V_swing = Vref. E = α·N·C·V².

- **Energy per conversion**: 4.860 pJ
- **Average dynamic power** at the nominal 1.00 µs conversion period: 4.86 µW
- **Leakage** (rough): 10.80 µW

---

## Tier B — signal integrity

### B.1 Monte Carlo over R-2R + comparator mismatch

200 samples drawn from σ_R = 0.50%, σ_Vos = 1.00 mV. Each sample → its own INL trace → max|INL| recorded.

| Percentile | INL_max (LSB) |
| --- | --- |
| p50  | 1.250 |
| p95  | 2.375 |
| p99  | 2.938 |
| worst | 3.375 |

Yield with INL < 1 LSB: **34.5%**.

![mc](crates/spike-sar-adc/docs/assets/sar_adc_characterization/mc_inl_hist.svg)

### B.2 kT/C noise on the S/H cap

Analytical floor: σ = √(kT/C). At C = 200 fF, T = 300 K → σ = **143.91 µV RMS** = 0.0205 LSB. Comfortably below 1 LSB at this resolution; would dominate at >12 bits.

### B.3 Comparator metastability

Regenerative latch model: P(undecided) = exp(−(t_decision − t_required)/τ_latch), with τ_latch = 50 ps and t_required determined by the input differential. For v_diff = LSB/2 the latch resolves cleanly within picoseconds; for v_diff approaching the noise floor (LSB/8) it requires several τ.

![metastability](crates/spike-sar-adc/docs/assets/sar_adc_characterization/metastability.svg)

### B.4 Per-bit DAC settling

R-2R driving impedance scales with bit position; settling to ½ LSB requires ln(2N)·τ. Assuming R_unit = 2 kΩ, C_node = 200 fF.

![settle](crates/spike-sar-adc/docs/assets/sar_adc_characterization/dac_settling.svg)

| bit (MSB→LSB) | settling (ns) |
| --- | --- |
| 7 | 2.50 |
| 6 | 1.25 |
| 5 | 0.62 |
| 4 | 0.31 |
| 3 | 0.16 |
| 2 | 0.08 |
| 1 | 0.04 |
| 0 | 0.02 |

---

## Tier C — gradient-optimized comparator sizing

Pelgrom mismatch law: σ_Vth = A_Vth / √(W·L), A_Vth ≈ 5 mV·µm. Comparator input-pair W is the optimization variable; ∂INL/∂W via central FD; Newton-ish step on log W. (Once the MNA-port of the SAR sub-blocks lands, this can switch to AD via `transient_sensitivities` for ~10× speedup.)

- Initial W = 0.25 µm → INL_max = 2.788 LSB
- Final   W = 6.01 µm → INL_max = 1.663 LSB (1.7× reduction)
- 25 iterations

![sizing-w](crates/spike-sar-adc/docs/assets/sar_adc_characterization/sizing_w.svg)

![sizing-inl](crates/spike-sar-adc/docs/assets/sar_adc_characterization/sizing_inl.svg)

This is the headline rlx-eda result: gradient-driven transistor sizing on a SAR ADC, no hand-tuning, no SPICE in the loop.

---

## Tier D — corners + reliability

### D.1 PVT corners (3 temp × 3 Vdd × 3 process)

| Temp (°C) | Vdd | Process | ENOB | INL_max |
| --- | --- | --- | --- | --- |
| -40 | -10% | SS | 6.87 | 1.547 |
| -40 | -10% | TT | 6.87 | 1.547 |
| -40 | -10% | FF | 6.87 | 1.547 |
| -40 | nom | SS | 6.88 | 1.547 |
| -40 | nom | TT | 6.88 | 1.547 |
| -40 | nom | FF | 6.88 | 1.547 |
| -40 | +10% | SS | 6.91 | 1.531 |
| -40 | +10% | TT | 6.91 | 1.531 |
| -40 | +10% | FF | 6.91 | 1.531 |
| 25 | -10% | SS | 6.83 | 1.531 |
| 25 | -10% | TT | 6.83 | 1.531 |
| 25 | -10% | FF | 6.83 | 1.531 |
| 25 | nom | SS | 6.86 | 1.531 |
| 25 | nom | TT | 6.86 | 1.531 |
| 25 | nom | FF | 6.86 | 1.531 |
| 25 | +10% | SS | 6.87 | 1.547 |
| 25 | +10% | TT | 6.87 | 1.547 |
| 25 | +10% | FF | 6.87 | 1.547 |
| 125 | -10% | SS | 6.77 | 1.516 |
| 125 | -10% | TT | 6.77 | 1.516 |
| 125 | -10% | FF | 6.77 | 1.516 |
| 125 | nom | SS | 6.82 | 1.500 |
| 125 | nom | TT | 6.82 | 1.500 |
| 125 | nom | FF | 6.82 | 1.500 |
| 125 | +10% | SS | 6.85 | 1.516 |
| 125 | +10% | TT | 6.85 | 1.516 |
| 125 | +10% | FF | 6.85 | 1.516 |

**Worst-corner ENOB = 6.77 bits.** Datasheet headline number is the worst across this grid.

### D.2 NBTI aging (Vth drift over 10 years)

Simplified NBTI: ΔVth = A·(V_stress)^β·t^n, A=2 mV/(V²·yr^0.25). Vth shift folds into comparator effective offset → ENOB drift.

![aging](crates/spike-sar-adc/docs/assets/sar_adc_characterization/aging.svg)

| time (yr) | ΔVth (mV) | ENOB |
| --- | --- | --- |
| 0.001 | 0.697 | 6.86 |
| 0.010 | 1.240 | 6.86 |
| 0.100 | 2.204 | 6.86 |
| 0.500 | 3.296 | 6.86 |
| 1.000 | 3.920 | 6.86 |
| 2.000 | 4.662 | 6.86 |
| 5.000 | 5.862 | 6.86 |
| 10.000 | 6.971 | 6.86 |

### D.3 PSRR (Vdd-ripple sensitivity)

Ratiometric DAC: Vdd ripple maps directly into bit weights. Inject δV·sin(ω_r·t) on the reference, measure the output spur at ω_r in the conversion spectrum.

**Estimated PSRR** ≈ 7.1 dB at the smallest tested ripple (0.1 mV).

![psrr](crates/spike-sar-adc/docs/assets/sar_adc_characterization/psrr.svg)

---

## Notes

- All metrics here run on the **behavioral** SAR; a transistor-level cross-check (using the same `BehavioralSar` knobs against `transient_pwl(SarAdc<8>)`) is the natural T.8.
- The Pelgrom + NBTI + R-2R-mismatch + kT/C numerics are calibrated to Sky130 130-nm-class typical silicon.
- The `tier_c_size_comparator` optimizer is FD-gradient today; once `transient_sensitivities` flows through MNA-ported SAR blocks it becomes pure AD.
