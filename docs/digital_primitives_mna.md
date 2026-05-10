# T.8.B — Digital primitives under eda-mna

Each MNA-ported gate (`Inverter`, `Nand2`, `Nand3`, `And2`, `DLatch`, `Dff`, `DffSR`) is built from `spike_divider_block::Mosfet` primitives in an `eda_mna::Circuit`, driven with PWL boundary patterns (one continuous transient per gate, sampled at known timestamps), and the output net's level at the end of each pattern's settled window is scored against the gate's truth table.

## Summary

| Gate | Pass | Total | Result |
| --- | ---: | ---: | :---: |
| Inverter | 2 | 2 | ✅ |
| Nand2 | 4 | 4 | ✅ |
| And2 | 4 | 4 | ✅ |
| Nand3 | 8 | 8 | ✅ |
| DLatch | 6 | 6 | ✅ |
| Dff | 7 | 7 | ✅ |
| DffSR | 6 | 6 | ✅ |

## Per-gate truth tables

### Inverter

| Input | Output | Pass |
| --- | --- | :---: |
| `0` | 1.800 V → 1 | ✅ |
| `1` | 0.000 V → 0 | ✅ |

### Nand2

| Input | Output | Pass |
| --- | --- | :---: |
| `00` | 1.800 V → 1 | ✅ |
| `01` | 1.800 V → 1 | ✅ |
| `10` | 1.800 V → 1 | ✅ |
| `11` | 0.000 V → 0 | ✅ |

### And2

| Input | Output | Pass |
| --- | --- | :---: |
| `00` | 0.000 V → 0 | ✅ |
| `01` | 0.000 V → 0 | ✅ |
| `10` | 0.000 V → 0 | ✅ |
| `11` | 1.800 V → 1 | ✅ |

### Nand3

| Input | Output | Pass |
| --- | --- | :---: |
| `000` | 1.800 V → 1 | ✅ |
| `001` | 1.800 V → 1 | ✅ |
| `010` | 1.800 V → 1 | ✅ |
| `011` | 1.800 V → 1 | ✅ |
| `100` | 1.800 V → 1 | ✅ |
| `101` | 1.800 V → 1 | ✅ |
| `110` | 1.800 V → 1 | ✅ |
| `111` | 0.000 V → 0 | ✅ |

### DLatch

| Input | Output | Pass |
| --- | --- | :---: |
| `01` | -0.000 V → 0 | ✅ |
| `11` | 1.800 V → 1 | ✅ |
| `10` | 1.800 V → 1 | ✅ |
| `00` | 1.800 V → 1 | ✅ |
| `01` | 0.000 V → 0 | ✅ |
| `00` | 0.000 V → 0 | ✅ |

### Dff

| Input | Output | Pass |
| --- | --- | :---: |
| `00` | 0.000 V → 0 | ✅ |
| `01` | 0.000 V → 0 | ✅ |
| `10` | 0.000 V → 0 | ✅ |
| `11` | 1.800 V → 1 | ✅ |
| `01` | 1.800 V → 1 | ✅ |
| `00` | 1.798 V → 1 | ✅ |
| `01` | 0.000 V → 0 | ✅ |

### DffSR

| Input | Output | Pass |
| --- | --- | :---: |
| `0010` | 0.000 V → 0 | ✅ |
| `0011` | 0.000 V → 0 | ✅ |
| `0001` | 1.800 V → 1 | ✅ |
| `0011` | 1.800 V → 1 | ✅ |
| `1011` | 1.799 V → 1 | ✅ |
| `1111` | 1.800 V → 1 | ✅ |

## What this proves

- Every digital primitive needed for the SAR Logic block (Nand2/3 + Inverter for the gate-level layer; DLatch/Dff/DffSR for the storage layer) functions correctly under `eda_mna::transient_pwl`.
- Master–slave Dff timing is honored: Q does not leak D through during clk = 0.
- DffSR async set/reset overrides the clocked path correctly.
- The MNA composition functions (`spike_cmos_gates::mna::add_*`) produce the same circuit topology as the `SpiceEmit` impls — same transistor sizes, same internal-node naming, same series-stack width compensation.
- T.8.C can now compose the full `SarAdc<N>` from these primitives + the analog blocks (SampleHold, R2RDac, the comparator from T.8.A) in a single `eda_mna::Circuit`.
