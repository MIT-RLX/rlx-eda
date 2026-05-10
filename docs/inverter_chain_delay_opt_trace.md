# rlx-eda inverter-chain delay optimization (Adam, transient gradients)

Circuit: 3-stage CMOS inverter chain (NMOS + PMOS per stage), with a 200 fF ground-tied load cap on each internal node setting the RC propagation delay. `vin` steps from 0 → Vdd at t = 5 ns; the output `vout` falls 3 inversions later, on a delay set by the per-stage NMOS Vth and the load caps.

Stimulus: `Vdd = 1 V`, `t_target = 15 ns`, `vout_target = 0 V`.

Loss:

$$L = (V_{out}(t_{\text{target}}) - V_{out}^*)^2$$

Per-parameter gradient via reverse-mode AD on the BE-step residual at each timestep, propagated forward through the cap history coupling. See `eda_mna::transient_sensitivities` for the IFT recurrence.

## Optimization outcome

- initial: `Vth_n = (0.450, 0.450, 0.450) V`, `vout(15 ns) = 0.9036 V`, `loss = 8.165e-1`
- final:   `Vth_n = (0.050, 0.570, 0.050) V`, `vout(15 ns) = 0.0400 V`, `loss = 1.598e-3`, `steps = 60`

All gradients computed by reverse-mode AD on the BE residual graph + per-step IFT recurrence — no SPICE oracle, no finite differences.

## Schematic

Three CMOS inverter stages in series, with a 200 fF ground-tied load cap on each internal node setting the per-stage RC delay. The three NMOS Vth values (`Vth_n1`, `Vth_n2`, `Vth_n3`) are the optimization parameters; PMOS Vth's and W/L are held fixed.

![schematic](crates/spike-divider-block/docs/assets/inverter_chain_delay_opt/schematic.svg)

## Floorplan (Sky130)

Real PDK-driven layout — 6 `Mosfet` cells (3 NMOS bottom, 3 PMOS top) placed into a Sky130 `Library`, with full electrical routing on metal1 + poly. Layers + colors come from the Sky130 `.lyp`; rendered via `eda_viz::layout::render_to_svg`.

**Routing scheme** (every net is electrically distinct — no stray shorts):

- **Power**: wide horizontal M1 Vdd / GND rails top + bottom, with short vertical source straps from each transistor's source port.
- **Body bias**: NMOS body-tap drops directly to GND (clear column south of the device). PMOS body-tap routes UP, jogs east through the narrow M1 channel between the PMOS drain pad and gate pad (y ≈ 17.2 µm), then up to Vdd at x = +4.5 µm — clear of both the drain bus and the gate bus.
- **Gate bus**: each stage's PMOS gate ↔ NMOS gate is on POLY (RES layer), filling the gap between the existing per-cell poly sticks. POLY-only gate routing keeps M1 clear of the body-tap pad column entirely (the original M1-only attempt had the gate net merging with the PMOS body pad — fatal short).
- **Drain bus**: vertical M1 per stage shorting PMOS drain to NMOS drain — this *is* the stage output net.
- **Inter-stage**: horizontal M1 in the routing channel (y ≈ 7 µm) carries each stage's output to the next stage's gate, where a VIA1 (M1↔poly contact) drops the signal onto the poly gate bus.
- **External**: vin pad on the left feeds stage-0 gate via the same M1+VIA1 path; vout pad on the right is driven by stage-2's drain bus.

> **Project rule**: every floorplan in this repo must be rendered against a real PDK (Sky130, Gf180mcu, …) — no stylized hand-drawn floorplans. Layer geometry has to match a real foundry stack so reviewers can spot routing/DRC issues by eye.

![floorplan](crates/spike-divider-block/docs/assets/inverter_chain_delay_opt/floorplan.png)

[Open as SVG](crates/spike-divider-block/docs/assets/inverter_chain_delay_opt/floorplan.svg) (vector, zoomable).

## DRC (Sky130)

✅ **DRC clean** — 0 violations across the rule set below.

Selected min-rule subset of the published Sky130A deck (real foundry decks have hundreds of rules; this is the load-bearing slice for hand-routed analog floorplans):

| Rule | Min (DBU) | Violations |
| --- | ---: | ---: |
| `M1 min width   (≥ 140 nm)` | 140 | 0 |
| `M1 min space   (≥ 140 nm)` | 140 | 0 |
| `POLY min width (≥ 150 nm)` | 150 | 0 |
| `POLY min space (≥ 210 nm)` | 210 | 0 |
| `DIFF min width (≥ 150 nm)` | 150 | 0 |
| `DIFF min space (≥ 270 nm)` | 270 | 0 |
| `NWELL min width (≥ 840 nm)` | 840 | 0 |
| `NWELL min space (≥ 1270 nm)` | 1270 | 0 |
| `VIA1 min width  (≥ 170 nm)` | 170 | 0 |
| `VIA1 min space  (≥ 170 nm)` | 170 | 0 |
| `NWELL enc PMOS-DIFF (≥ 180 nm)` | 180 | 0 |
| `M1 enc VIA1 (≥ 30 nm)` | 30 | 0 |

Run via `klayout_drc::{width, space, enclosing}` over a `klayout_geom::Region` extracted from the floorplan's top cell, per layer. Each rule returns a region of failing geometry; the violation count is the polygon count of that region.

## LVS (layout-vs-schematic)

✅ **LVS pass** — extracted net count matches the schematic, every expected probe lands in its own distinct net.

Extracted **6** nets (M1 + POLY merged via VIA1); schematic expects **6**.

Probe-to-net match (each row asserts that a known coordinate inside the named net's wire actually lands inside an extracted net):

| Net | Probe (DBU) | Matched extracted net | Result |
| --- | --- | ---: | :---: |
| **vin** | (-1500, 7000) | net_3 | ✅ |
| **n1** | (5000, 8000) | net_4 | ✅ |
| **n2** | (15000, 8000) | net_5 | ✅ |
| **vout** | (27250, 7000) | net_2 | ✅ |
| **Vdd** | (12000, 22000) | net_1 | ✅ |
| **GND** | (12000, -5000) | net_0 | ✅ |

Distinct extracted nets matched: **6** of 6 expected.

Run via `klayout_connect::extract_hierarchical` with M1 + POLY conductors and a VIA1 join rule. The extractor walks every shape on the conductor layers, merges polygons that touch (per layer), then joins layers across via cuts.

## PEX (parasitic-aware re-optimization)

Per-net wire R + per-node parasitic C estimated from the floorplan's actual routing geometry, using published Sky130A sheet values:

- M1 sheet R: 125 mΩ/sq, M1 area C: 38 aF/µm²
- POLY sheet R: 48 Ω/sq, POLY area C: 110 aF/µm²
- inter-stage M1 horizontal: 8.75 µm × 0.6 µm
- drain bus M1 vertical: 15 µm × 0.6 µm
- POLY gate U-bridge: 12 µm × 1 µm

→ **R_gate_path = 577.8 Ω** (in series at each stage's gate input — drain-bus M1 + inter-stage M1 + poly gate bridge)
→ **C_per_node = 1.862 fF** (parasitic cap from M1 + poly area to substrate, attached on each gate-input net AND each drain net in addition to the 200 fF Cload)

These values are wired into the simulated circuit as new `Resistor` + `LinearCap` devices. The differentiable solver doesn't know which nets came from "the layout" vs "the schematic" — `transient_sensitivities` propagates ∂L/∂Vth through the *augmented* circuit just as it did through the ideal one. **No code path in eda-mna changes** — extracted parasitics are first-class circuit elements.

Comparison of converged Adam state:

| Variant | Vth_n1 | Vth_n2 | Vth_n3 | vout(t*) | loss | steps |
| --- | --- | --- | --- | --- | --- | ---: |
| initial   | 0.450 | 0.450 | 0.450 | 0.9036 | 8.165e-1 | — |
| ideal     | 0.050 | 0.570 | 0.050 | 0.0400 | 1.598e-3 | 60 |
| pex-aware | 0.050 | 0.603 | 0.050 | 0.0447 | 1.994e-3 | 60 |

Vth shift attributable to parasitics: ΔVth = (+0.0000, +0.0330, +0.0000) V. The chain has to compensate for the added gate-side RC delay; whichever Vth's the optimizer pulls further down is the gradient telling us where the parasitics bite hardest.

## Rendered charts

| Loss over steps | Per-stage Vth trajectories |
| --- | --- |
| ![loss](crates/spike-divider-block/docs/assets/inverter_chain_delay_opt/loss.svg) | ![params](crates/spike-divider-block/docs/assets/inverter_chain_delay_opt/params.svg) |

| vout(t_target) tracking | Per-parameter gradient |
| --- | --- |
| ![output](crates/spike-divider-block/docs/assets/inverter_chain_delay_opt/output.svg) | ![grads](crates/spike-divider-block/docs/assets/inverter_chain_delay_opt/grads.svg) |

## Step-by-step trace

| step | Vth_n1 | Vth_n2 | Vth_n3 | vout(t*) | loss | g1 | g2 | g3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.4500 | 0.4500 | 0.4500 | 0.9036 | 8.165e-1 | 5.762e0 | -1.459e-4 | 1.140e0 |
| 1 | 0.4300 | 0.4700 | 0.4300 | 0.8526 | 7.269e-1 | 6.259e0 | -0.000e0 | 1.447e0 |
| 2 | 0.4100 | 0.4834 | 0.4100 | 0.7939 | 6.303e-1 | 6.482e0 | -0.000e0 | 1.704e0 |
| 3 | 0.3899 | 0.4938 | 0.3900 | 0.7293 | 5.319e-1 | 6.301e0 | -0.000e0 | 1.894e0 |
| 4 | 0.3699 | 0.5022 | 0.3699 | 0.6580 | 4.329e-1 | 5.937e0 | -0.000e0 | 2.006e0 |
| 5 | 0.3499 | 0.5094 | 0.3497 | 0.5827 | 3.396e-1 | 5.403e0 | -0.000e0 | 2.019e0 |
| 6 | 0.3300 | 0.5156 | 0.3295 | 0.5061 | 2.561e-1 | 4.599e0 | -0.000e0 | 1.824e0 |
| 7 | 0.3104 | 0.5210 | 0.3092 | 0.4335 | 1.879e-1 | 3.582e0 | -0.000e0 | 1.522e0 |
| 8 | 0.2914 | 0.5257 | 0.2891 | 0.3684 | 1.357e-1 | 2.703e0 | -0.000e0 | 1.204e0 |
| 9 | 0.2731 | 0.5300 | 0.2695 | 0.3119 | 9.725e-2 | 2.011e0 | -0.000e0 | 9.258e-1 |
| 10 | 0.2556 | 0.5337 | 0.2505 | 0.2643 | 6.985e-2 | 1.496e0 | -0.000e0 | 7.070e-1 |
| 11 | 0.2392 | 0.5371 | 0.2323 | 0.2251 | 5.066e-2 | 1.079e0 | -0.000e0 | 5.295e-1 |
| 12 | 0.2238 | 0.5402 | 0.2150 | 0.1930 | 3.725e-2 | 8.158e-1 | -0.000e0 | 4.046e-1 |
| 13 | 0.2095 | 0.5429 | 0.1987 | 0.1667 | 2.779e-2 | 6.063e-1 | -0.000e0 | 3.072e-1 |
| 14 | 0.1962 | 0.5454 | 0.1835 | 0.1454 | 2.113e-2 | 4.634e-1 | -0.000e0 | 2.385e-1 |
| 15 | 0.1839 | 0.5476 | 0.1692 | 0.1278 | 1.633e-2 | 3.591e-1 | -0.000e0 | 1.881e-1 |
| 16 | 0.1725 | 0.5496 | 0.1560 | 0.1135 | 1.287e-2 | 2.780e-1 | -0.000e0 | 1.491e-1 |
| 17 | 0.1620 | 0.5515 | 0.1437 | 0.1017 | 1.034e-2 | 2.211e-1 | -0.000e0 | 1.210e-1 |
| 18 | 0.1524 | 0.5531 | 0.1323 | 0.0918 | 8.435e-3 | 1.811e-1 | -0.000e0 | 1.001e-1 |
| 19 | 0.1435 | 0.5547 | 0.1217 | 0.0836 | 6.988e-3 | 1.506e-1 | -0.000e0 | 8.345e-2 |
| 20 | 0.1353 | 0.5560 | 0.1120 | 0.0766 | 5.874e-3 | 1.266e-1 | -0.000e0 | 7.029e-2 |
| 21 | 0.1278 | 0.5573 | 0.1030 | 0.0708 | 5.007e-3 | 1.076e-1 | -0.000e0 | 6.016e-2 |
| 22 | 0.1209 | 0.5585 | 0.0946 | 0.0657 | 4.321e-3 | 9.303e-2 | -0.000e0 | 5.222e-2 |
| 23 | 0.1146 | 0.5595 | 0.0870 | 0.0614 | 3.771e-3 | 8.166e-2 | -0.000e0 | 4.592e-2 |
| 24 | 0.1087 | 0.5604 | 0.0799 | 0.0575 | 3.305e-3 | 7.205e-2 | -0.000e0 | 4.067e-2 |
| 25 | 0.1034 | 0.5613 | 0.0733 | 0.0542 | 2.943e-3 | 6.382e-2 | -0.000e0 | 3.633e-2 |
| 26 | 0.0984 | 0.5621 | 0.0673 | 0.0514 | 2.645e-3 | 5.680e-2 | -0.000e0 | 3.262e-2 |
| 27 | 0.0939 | 0.5628 | 0.0617 | 0.0490 | 2.400e-3 | 5.113e-2 | -0.000e0 | 2.958e-2 |
| 28 | 0.0898 | 0.5635 | 0.0566 | 0.0468 | 2.194e-3 | 4.648e-2 | -0.000e0 | 2.706e-2 |
| 29 | 0.0859 | 0.5641 | 0.0518 | 0.0450 | 2.021e-3 | 4.264e-2 | -0.000e0 | 2.494e-2 |
| 30 | 0.0824 | 0.5646 | 0.0500 | 0.0440 | 1.933e-3 | 4.059e-2 | -0.000e0 | 2.390e-2 |
| 31 | 0.0792 | 0.5651 | 0.0500 | 0.0435 | 1.895e-3 | 3.957e-2 | -0.000e0 | 2.346e-2 |
| 32 | 0.0762 | 0.5655 | 0.0500 | 0.0431 | 1.861e-3 | 3.866e-2 | -0.000e0 | 2.307e-2 |
| 33 | 0.0735 | 0.5660 | 0.0500 | 0.0428 | 1.831e-3 | 3.786e-2 | -0.000e0 | 2.272e-2 |
| 34 | 0.0710 | 0.5663 | 0.0500 | 0.0425 | 1.804e-3 | 3.714e-2 | -0.000e0 | 2.240e-2 |
| 35 | 0.0686 | 0.5667 | 0.0500 | 0.0422 | 1.779e-3 | 3.650e-2 | -0.000e0 | 2.212e-2 |
| 36 | 0.0665 | 0.5670 | 0.0500 | 0.0419 | 1.757e-3 | 3.593e-2 | -0.000e0 | 2.187e-2 |
| 37 | 0.0645 | 0.5673 | 0.0500 | 0.0417 | 1.736e-3 | 3.541e-2 | -0.000e0 | 2.163e-2 |
| 38 | 0.0627 | 0.5675 | 0.0500 | 0.0414 | 1.718e-3 | 3.495e-2 | -0.000e0 | 2.142e-2 |
| 39 | 0.0610 | 0.5678 | 0.0500 | 0.0412 | 1.701e-3 | 3.452e-2 | -0.000e0 | 2.123e-2 |
| 40 | 0.0594 | 0.5680 | 0.0500 | 0.0411 | 1.686e-3 | 3.414e-2 | -0.000e0 | 2.106e-2 |
| 41 | 0.0580 | 0.5682 | 0.0500 | 0.0409 | 1.672e-3 | 3.379e-2 | -0.000e0 | 2.090e-2 |
| 42 | 0.0566 | 0.5683 | 0.0500 | 0.0407 | 1.659e-3 | 3.347e-2 | -0.000e0 | 2.075e-2 |
| 43 | 0.0554 | 0.5685 | 0.0500 | 0.0406 | 1.647e-3 | 3.318e-2 | -0.000e0 | 2.062e-2 |
| 44 | 0.0542 | 0.5686 | 0.0500 | 0.0404 | 1.636e-3 | 3.291e-2 | -0.000e0 | 2.049e-2 |
| 45 | 0.0531 | 0.5688 | 0.0500 | 0.0403 | 1.626e-3 | 3.267e-2 | -0.000e0 | 2.038e-2 |
| 46 | 0.0521 | 0.5689 | 0.0500 | 0.0402 | 1.617e-3 | 3.246e-2 | -0.000e0 | 2.027e-2 |
| 47 | 0.0511 | 0.5690 | 0.0500 | 0.0401 | 1.608e-3 | 3.226e-2 | -0.000e0 | 2.018e-2 |
| 48 | 0.0503 | 0.5691 | 0.0500 | 0.0400 | 1.600e-3 | 3.208e-2 | -0.000e0 | 2.008e-2 |
| 49 | 0.0500 | 0.5692 | 0.0500 | 0.0400 | 1.598e-3 | 3.203e-2 | -0.000e0 | 2.006e-2 |
| 50 | 0.0500 | 0.5693 | 0.0500 | 0.0400 | 1.598e-3 | 3.203e-2 | -0.000e0 | 2.006e-2 |
| 51 | 0.0500 | 0.5694 | 0.0500 | 0.0400 | 1.598e-3 | 3.203e-2 | -0.000e0 | 2.006e-2 |
| 52 | 0.0500 | 0.5694 | 0.0500 | 0.0400 | 1.598e-3 | 3.203e-2 | -0.000e0 | 2.006e-2 |
| 53 | 0.0500 | 0.5695 | 0.0500 | 0.0400 | 1.598e-3 | 3.203e-2 | -0.000e0 | 2.006e-2 |
| 54 | 0.0500 | 0.5695 | 0.0500 | 0.0400 | 1.598e-3 | 3.203e-2 | -0.000e0 | 2.006e-2 |
| 55 | 0.0500 | 0.5696 | 0.0500 | 0.0400 | 1.598e-3 | 3.203e-2 | -0.000e0 | 2.006e-2 |
| 56 | 0.0500 | 0.5696 | 0.0500 | 0.0400 | 1.598e-3 | 3.203e-2 | -0.000e0 | 2.006e-2 |
| 57 | 0.0500 | 0.5697 | 0.0500 | 0.0400 | 1.598e-3 | 3.203e-2 | -0.000e0 | 2.006e-2 |
| 58 | 0.0500 | 0.5697 | 0.0500 | 0.0400 | 1.598e-3 | 3.203e-2 | -0.000e0 | 2.006e-2 |
| 59 | 0.0500 | 0.5698 | 0.0500 | 0.0400 | 1.598e-3 | 3.203e-2 | -0.000e0 | 2.006e-2 |
| 60 | 0.0500 | 0.5698 | 0.0500 | 0.0400 | 1.598e-3 | 3.203e-2 | -0.000e0 | 2.006e-2 |
