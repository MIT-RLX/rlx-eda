# Glossary and references

Every abbreviation that appears in the rlx-eda README or this `docs/`
tree, with a one-line gloss and (where one exists) a citable
reference. DOI links resolve through [doi.org](https://doi.org);
arXiv links go to the abstract page. All references below were
verified via Crossref / arXiv at the time of writing.

## Circuits and EDA

| abbr. | expansion | reference |
| --- | --- | --- |
| **EDA** | Electronic Design Automation — the field; also the workspace name. (Distinct from "EDA" in the optimization-algorithm sense below.) | general industry term |
| **ADC** | Analog-to-Digital Converter | terminology + test methods: IEEE Std 1241-2010, [DOI 10.1109/IEEESTD.2011.5692956](https://doi.org/10.1109/IEEESTD.2011.5692956) |
| **DAC** | Digital-to-Analog Converter | terminology + test methods: IEEE Std 1658-2011, [DOI 10.1109/IEEESTD.2012.6152113](https://doi.org/10.1109/IEEESTD.2012.6152113) |
| **SAR** | Successive Approximation Register — the binary-search ADC architecture used by `spike-sar-adc` | classic textbook, no single canonical paper |
| **INL** | Integral Non-Linearity — `vout(code) − ideal_vout(code)`, V | IEEE Std 1241-2010 §4.4.2 |
| **DNL** | Differential Non-Linearity — adjacent-code step error, `vout(k+1) − vout(k) − LSB`, V | IEEE Std 1241-2010 §4.4.1 |
| **ENOB** | Effective Number of Bits — `(SNDR_dB − 1.76) / 6.02` | IEEE Std 1241-2010 §4.5.5 |
| **SNDR** | Signal-to-Noise-and-Distortion Ratio | IEEE Std 1241-2010 §4.5.4 |
| **LSB** | Least Significant Bit — also the voltage step `vref / 2ⁿ` | general |
| **MOSFET** / **NMOS** / **PMOS** | Metal-Oxide-Semiconductor Field-Effect Transistor (n-channel / p-channel) | general |
| **CMOS** | Complementary MOS — both NMOS and PMOS on the same die | general |
| **AC** / **DC** | Alternating / Direct Current — also SPICE analysis modes | general |
| **PWL** | Piece-Wise Linear — SPICE source waveform spec | SPICE3 manual |
| **AND** / **NAND** | Boolean conjunction (and its negation); CMOS gate types | general |
| **MNA** | Modified Nodal Analysis — the matrix-stamp approach our `eda-mna` and `solve_r2r` solver use | Ho, Ruehli, Brennan (1975), [DOI 10.1109/TCS.1975.1084079](https://doi.org/10.1109/TCS.1975.1084079) |
| **SPICE** | Simulation Program with Integrated Circuit Emphasis — the simulator family | original: Nagel & Pederson, ERL-M382 (UC Berkeley, 1973), no DOI; modern impl: [`ngspice.sourceforge.io`](https://ngspice.sourceforge.io) |
| **PDK** | Process Design Kit — foundry-supplied layer / device definitions | general |
| **GDS** / **GDSII** | Graphic Database System — the layout file format `klayout-io::write_gds_*` produces | Calma Co. (1971), now a SEMI standard ([SEMI P39](https://www.semi.org)) |
| **HIR** / **IR** | High-level / Intermediate Representation — workspace-internal layering: HIR is the typed-block trait layer in `eda-hir`; the residual / Jacobian / batched analysis IR is `rlx_ir::Graph` | workspace convention |
| **DRC** / **LVS** | Design Rule Check / Layout vs Schematic — physical-verification stages | general industry terms |
| **DSE** | Design Space Exploration — the early-stage optimization loop the hybrid pipeline targets | general |
| **DBU** | Database Unit — KLayout's coordinate quantum (1 nm at `dbu = 1000`) | KLayout manual |
| **MZI** | Mach-Zehnder Interferometer — the photonic two-arm interferometer used in `spike-waveguide-block::mzi_match_trace` | general photonics |
| **LNA** | Low-Noise Amplifier — the RF block in `spike-lna` | Razavi, *RF Microelectronics* §5.3 |

## Optimization and machine learning

| abbr. | expansion | reference |
| --- | --- | --- |
| **DADO** | Decomposition-Aware Distributional Optimization | Bowden, Levine, Listgarten, ICLR 2026, [arXiv:2511.03032](https://arxiv.org/abs/2511.03032) |
| **EDA** *(when in the DADO context)* | Estimation of Distribution Algorithm — the family DADO generalizes | survey: Larrañaga & Lozano (2002), [DOI 10.1007/978-1-4615-1539-5](https://doi.org/10.1007/978-1-4615-1539-5) |
| **JT** | Junction Tree — the data structure DADO does message passing on | Lauritzen & Spiegelhalter (1988), [DOI 10.1111/j.2517-6161.1988.tb01721.x](https://doi.org/10.1111/j.2517-6161.1988.tb01721.x) |
| **KL** | Kullback–Leibler divergence — distance between distributions | Kullback & Leibler (1951), [DOI 10.1214/aoms/1177729694](https://doi.org/10.1214/aoms/1177729694) |
| **MLE** | Maximum Likelihood Estimation — what the weighted-MLE update step does | Fisher (1922), [DOI 10.1098/rsta.1922.0009](https://doi.org/10.1098/rsta.1922.0009) |
| **MSE** | Mean Squared Error — the SPICE B objective | general statistics term |
| **VAE** | Variational AutoEncoder — the search-distribution parameterization the DADO paper actually uses (we use tabular categoricals instead) | Kingma & Welling (2014), [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) |
| **MLP** | Multi-Layer Perceptron — what `spike-surrogate` trains | general |
| **NN** | Neural Network | general |
| **PINN** | Physics-Informed Neural Network | Raissi, Perdikaris, Karniadakis (2019), [DOI 10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045) |
| **AD** | Automatic Differentiation — the rlx-backed gradient flow | survey: Baydin et al., JMLR 18(153):1–43 (2018), [arXiv:1502.05767](https://arxiv.org/abs/1502.05767) |
| **ML** | Machine Learning | general |
| **SA** | Simulated Annealing — discrete-optimization baseline mentioned in the ablation table | Kirkpatrick, Gelatt, Vecchi (1983), [DOI 10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671) |
| **CMA-ES** | Covariance Matrix Adaptation Evolution Strategy — continuous-optimization baseline | Hansen & Ostermeier (2001), [DOI 10.1162/106365601750190398](https://doi.org/10.1162/106365601750190398) |
| **BO** | Bayesian Optimization — surrogate-based optimization (related to but distinct from our hybrid pipeline) | Mockus, *On the Bayes Methods for Seeking the Extremal Point*, IFAC Proc. Vol. (1975), [DOI 10.1016/S1474-6670(17)67769-3](https://doi.org/10.1016/S1474-6670(17)67769-3) |
| **DbAS** | Conditioning by adaptive sampling — related distributional optimizer cited in the DADO paper | Brookes, Park, Listgarten (2019), [arXiv:1901.10060](https://arxiv.org/abs/1901.10060) |
| **DP** | Dynamic Programming — the chain-JT suffix-sum DADO uses for `Q_c` | Bellman (1957), no DOI; modern reference: Cormen et al., *Introduction to Algorithms* |
| **ROM** *(when surrogate context)* | Reduced-Order Model — alternative to a closed-form analytical surrogate | general |
| **HPWL** | Half-Perimeter Wirelength — the standard placement objective `eda-pnr` uses | Caldwell, Kahng, Markov, *VLSI Physical Design* §7 |

## Tooling, formats, common tech

| abbr. | expansion | reference |
| --- | --- | --- |
| **CLI** | Command Line Interface | general |
| **API** | Application Programming Interface | general |
| **OS** | Operating System | general |
| **ETA** | Estimated Time of Arrival — shown by `indicatif` progress bars | — |
| **DX** | Developer Experience | general |
| **TTY** | Teletype — for `indicatif`'s "is stdout a terminal?" check | POSIX |
| **LOC** | Lines of Code | general |
| **CSV** | Comma-Separated Values — the format `eda-waveform` writes for trace export | RFC 4180 |
| **SVG** | Scalable Vector Graphics — `eda-viz` schematic / `eda-waveform` plot output | [W3C SVG 1.1 (Second Edition), 2011](https://www.w3.org/TR/SVG11/) |
| **PNG** | Portable Network Graphics — `eda-waveform`'s default plot format | [ISO/IEC 15948:2004](https://www.iso.org/standard/29581.html) |
| **VCD** | Value Change Dump — digital-waveform format `eda-waveform` reads | IEEE 1364-2005 §18 |
| **FFT** | Fast Fourier Transform — used by `eda-waveform::spectrum::adc_metrics` for ENOB | Cooley & Tukey, Math. Comp. 19(90):297–301 (1965), [DOI 10.1090/S0025-5718-1965-0178586-1](https://doi.org/10.1090/S0025-5718-1965-0178586-1) |
| **arXiv** | Open-access preprint repository | [arxiv.org](https://arxiv.org) |
| **ICLR** | International Conference on Learning Representations | [iclr.cc](https://iclr.cc) |
| **DFF** | D-type Flip-Flop — `spike-cmos-gates::DffSR` | general digital-design term |
| **NN** *(distinct from ML usage)* | sometimes also "near-neighbor" — disambiguated by context | — |

## Workspace-internal symbols

| abbr. | expansion | meaning |
| --- | --- | --- |
| **rlx** | (no expansion — sibling project name) | numerical / autodiff runtime at `../rlx` |
| **rlx-eda** | (this workspace) | EDA crates that consume `rlx` |
| **MTL** *(in path)* | (project name) | Sibling workspace at `../mtl/klayout-rs/` |
| **HIR** | see Circuits table above | — |
| **R0..R270** | KLayout 0°/90°/180°/270° rotations (`klayout_core::Rot4`) | — |
| **A** / **B** / **H** *(in `spike-dado-sar`)* | Phase A (analytical) / B (SPICE) / H (Hybrid) of the experiment | — |
| **ORFS** | OpenROAD Flow Scripts — the full ASIC docker image used by `eda-bench-tinyconv::backends::orfs` | [github.com/The-OpenROAD-Project/OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) |
