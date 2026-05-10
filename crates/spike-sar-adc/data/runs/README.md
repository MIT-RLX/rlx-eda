# Cached run logs

Raw stderr captures from the T.11.D / T.11.E / T.11.G runs that
back the charts in [`../../docs/sar_adc_mc_sweep.md`](../../docs/sar_adc_mc_sweep.md)
and [`../../../spike-divider-block/docs/comparator_sizing_opt_ad.md`](../../../spike-divider-block/docs/comparator_sizing_opt_ad.md).

Cached here so the chart bin (`sar_charts`) renders identical
figures without requiring a re-run of the (slow) batched solver.

| file | source command | what it backs |
| --- | --- | --- |
| `sar_v0.log` | `RLX_BATCHED_PROGRESS=1 RLX_BATCHED_PER_CHIP_ALPHA=0 RLX_SAR_PHASE_FRAC=0.50 ./target/release/sar_adc_mc_sweep` | `convergence.svg` v0 series + `version_compare.svg` v0 bars |
| `sar_v1.log` | `… RLX_BATCHED_PER_CHIP_ALPHA=1 RLX_SAR_PHASE_FRAC=0.50 …` | v1 series / bars |
| `sar_v2.log` | `… RLX_BATCHED_PER_CHIP_ALPHA=1 RLX_SAR_PHASE_FRAC=0.70 …` | v2 series / bars |
| `sar_v3.log` | `… RLX_SAR_PHASE_FRAC=0.70 RLX_BATCHED_ADAPTIVE_DT=1 …` | v3 series / bars |
| `mlx_4096.log` | `RLX_BATCHED_DEVICE=mlx RLX_MLX_MODE=compiled RLX_N_VIN=16 RLX_N_DRAWS=256 ./target/release/comparator_vin_sweep_mc` | `mlx_scaling.svg` MLX-Compiled @ B=4096 datapoint |
| `dado_cascade.log` | `./target/release/comparator_sizing_opt_ad` | per-stage trace for `comparator_sizing_opt_ad.md` |

## Lookup order in `sar_charts`

For each version, the chart bin checks (in order):
1. `crates/spike-sar-adc/data/runs/{name}.log` — this directory
2. `/tmp/{name}.log` — a just-finished interactive run
3. synthetic representative curve baked into the source (fallback for clean checkouts that haven't run the sweep)

## Refreshing

```sh
cd /path/to/rlx-eda
RLX_BATCHED_PROGRESS=1 RLX_BATCHED_PER_CHIP_ALPHA=0 RLX_SAR_PHASE_FRAC=0.50 \
  ./target/release/sar_adc_mc_sweep > crates/spike-sar-adc/data/runs/sar_v0.log 2>&1
RLX_BATCHED_PROGRESS=1 RLX_BATCHED_PER_CHIP_ALPHA=1 RLX_SAR_PHASE_FRAC=0.50 \
  ./target/release/sar_adc_mc_sweep > crates/spike-sar-adc/data/runs/sar_v1.log 2>&1
RLX_BATCHED_PROGRESS=1 RLX_BATCHED_PER_CHIP_ALPHA=1 RLX_SAR_PHASE_FRAC=0.70 \
  ./target/release/sar_adc_mc_sweep > crates/spike-sar-adc/data/runs/sar_v2.log 2>&1
RLX_BATCHED_PROGRESS=1 RLX_BATCHED_PER_CHIP_ALPHA=1 RLX_SAR_PHASE_FRAC=0.70 \
  RLX_BATCHED_ADAPTIVE_DT=1 \
  ./target/release/sar_adc_mc_sweep > crates/spike-sar-adc/data/runs/sar_v3.log 2>&1

RLX_BATCHED_DEVICE=mlx RLX_MLX_MODE=compiled \
  RLX_N_VIN=16 RLX_N_DRAWS=256 \
  ./target/release/comparator_vin_sweep_mc > crates/spike-sar-adc/data/runs/mlx_4096.log 2>&1

./target/release/comparator_sizing_opt_ad > crates/spike-sar-adc/data/runs/dado_cascade.log 2>&1

# Re-render the SVGs from the refreshed logs:
./target/release/sar_charts
```
