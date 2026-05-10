# DADO vs naive EDA — R-2R DAC artifacts

Run: K=100, n_iters=80, seeds=12, snapshots@[0, 9, 24, 49, 79]

## Final-iteration averages

- **Synthetic decomposable**: DADO mean final best = `0.00000`, EDA mean final best = `-3.75000`
- **R-2R max-INL (V)**: DADO mean final best = `-0.00092`, EDA mean final best = `-0.00087`

## Top-level artifacts

- `00_trajectory_synth.png` — DADO/EDA best+mean trajectories on the synthetic objective.
- `00_trajectory_inl.png` — DADO/EDA best+mean trajectories on the R-2R INL objective.
- `00_param_evolution_dado.png` / `00_param_evolution_eda.png` — per-resistor expected deviation index over snapshot iterations (seed 0).
- `00_final_staircase.png` — DADO best vs EDA best vs ideal staircase, all 256 codes.
- `00_final_inl.png` — INL curves for DADO best and EDA best.
- `00_final_schematic_*.svg` — annotated R-2R schematic for each algorithm's final-best design.
- `00_final_layout_*.gds` — real GDS file for each algorithm's final-best design (open in KLayout).
- `00_final_ngspice_*.txt` — ngspice cross-validation report (256 codes; max |ngspice - analytical|).

## Per-snapshot packages (R-2R objective, seed 0)

Each iteration in [0, 9, 24, 49, 79] produces, for both `dado` and `eda`:
- `iter_NN_<alg>_marginals.png` — 16 per-resistor probability tracks across the deviation alphabet.
- `iter_NN_<alg>_schematic.svg` — schematic of the best-so-far design with resistor values annotated.
- `iter_NN_<alg>_layout.gds` — GDS of the best-so-far design (resistor body lengths encode actual ohms).
- `iter_NN_<alg>_staircase.png` — analytical DAC staircase.
- `iter_NN_<alg>_inl.png` — analytical INL curve.

