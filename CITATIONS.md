# Citations

How to cite `rlx-eda`, and the load-bearing external work it builds on.

## Citing this workspace

A paper describing `rlx-eda` is in preparation; the arXiv link will
replace the placeholder in [`README.md`](README.md) when it is posted.
Until then, please cite the software directly:

```bibtex
@software{rlx_eda_2026,
  title   = {rlx-eda: a code-defined Rust EDA workspace with a differentiable MNA core},
  author  = {Hauptmann, Eugene and Kosmyna, Nataliya},
  year    = {2026},
  url     = {https://github.com/MIT-RLX/rlx-eda},
  note    = {Companion workspaces: \url{https://github.com/MIT-RLX/rlx} (autodiff / GPU runtime)
             and \url{https://github.com/MIT-RLX/klayout-rs} (layout). Paper forthcoming.}
}
```

If you cite a specific contribution, please also cite the corresponding
section of [`docs/contributions.md`](docs/contributions.md) so reviewers
can find the experimental setup and the witness scripts.

## External references this workspace depends on

The DOI / arXiv links below are also embedded in
[`docs/glossary.md`](docs/glossary.md); this file collects the
load-bearing entries (the ones a reader of the README needs to find a
paper for) into one BibTeX-shaped list.

### Solver and simulator foundations

- **MNA — Modified Nodal Analysis** (the matrix-stamp method `eda-mna`
  uses): Ho, Ruehli, Brennan, *The Modified Nodal Approach to Network
  Analysis*, IEEE Trans. Circuits and Systems, 1975.
  [DOI 10.1109/TCS.1975.1084079](https://doi.org/10.1109/TCS.1975.1084079)
- **SPICE** (the original simulator we cross-validate against):
  Nagel & Pederson, *SPICE — Simulation Program with Integrated
  Circuit Emphasis*, ERL Memo M382, UC Berkeley, 1973. No DOI; modern
  open implementation: [`ngspice.sourceforge.io`](https://ngspice.sourceforge.io).
- **AD — Automatic Differentiation** (the rlx-backed gradient flow we
  push through MNA): Baydin, Pearlmutter, Radul, Siskind, *Automatic
  Differentiation in Machine Learning: a Survey*, JMLR 18(153):1–43,
  2018. [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)
- **FFT — Fast Fourier Transform** (used for ENOB / SNDR in
  `eda-waveform::spectrum`): Cooley & Tukey, *An algorithm for the
  machine calculation of complex Fourier series*, Math. Comp. 19(90):
  297–301, 1965.
  [DOI 10.1090/S0025-5718-1965-0178586-1](https://doi.org/10.1090/S0025-5718-1965-0178586-1)

### Optimization stack

- **DADO — Decomposition-Aware Distributional Optimization** (the
  per-block decomposition baseline `spike-dado-*` evaluates;
  algorithm name from the paper, whose arXiv title is *Leveraging
  Discrete Function Decomposability for Scientific Design*):
  Bowden, Levine, Listgarten, ICLR 2026.
  [arXiv:2511.03032](https://arxiv.org/abs/2511.03032)
- **EDA — Estimation-of-Distribution Algorithms** (the family DADO
  generalizes): Larrañaga & Lozano (eds.), *Estimation of Distribution
  Algorithms*, Kluwer, 2002.
  [DOI 10.1007/978-1-4615-1539-5](https://doi.org/10.1007/978-1-4615-1539-5)
- **DbAS — Conditioning by adaptive sampling** (related distributional
  optimizer): Brookes, Park, Listgarten, 2019.
  [arXiv:1901.10060](https://arxiv.org/abs/1901.10060)
- **VAE — Variational Autoencoder** (DADO's search-distribution
  parameterization): Kingma & Welling, *Auto-Encoding Variational
  Bayes*, 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- **PINN — Physics-Informed Neural Networks** (the `spike-pinn-*`
  experiment series): Raissi, Perdikaris, Karniadakis, JCP 2019.
  [DOI 10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
- **CMA-ES** (continuous-optimization baseline): Hansen & Ostermeier,
  *Completely Derandomized Self-Adaptation in Evolution Strategies*,
  Evol. Comput. 2001.
  [DOI 10.1162/106365601750190398](https://doi.org/10.1162/106365601750190398)
- **Simulated Annealing** (discrete-optimization baseline):
  Kirkpatrick, Gelatt, Vecchi, *Optimization by Simulated Annealing*,
  Science 1983.
  [DOI 10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671)
- **Bayesian Optimization** (related surrogate-based optimizer):
  Mockus, *On the Bayes Methods for Seeking the Extremal Point*,
  IFAC Proceedings Volumes, 1975.
  [DOI 10.1016/S1474-6670(17)67769-3](https://doi.org/10.1016/S1474-6670(17)67769-3)
- **KL divergence**: Kullback & Leibler, 1951.
  [DOI 10.1214/aoms/1177729694](https://doi.org/10.1214/aoms/1177729694)
- **Junction Tree message passing**: Lauritzen & Spiegelhalter, JRSS-B 1988.
  [DOI 10.1111/j.2517-6161.1988.tb01721.x](https://doi.org/10.1111/j.2517-6161.1988.tb01721.x)

### Place-and-route and layout

- **HPWL — Half-Perimeter Wirelength** (the placement objective
  `eda-pnr` differentiates): Caldwell, Kahng, Markov, *VLSI Physical
  Design: From Graph Partitioning to Timing Closure*, Springer 2011.
- **GDSII format** (the layout file format `klayout-io::write_gds_*`
  emits): Calma Co., 1971; now a SEMI standard,
  [SEMI P39](https://www.semi.org).
- **OpenROAD Flow Scripts** (the full ASIC backend used by
  `eda-bench-tinyconv::backends::orfs`):
  [github.com/The-OpenROAD-Project/OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts).

### Standards and measurement

- **IEEE Std 1241-2010** — ADC terminology and test methods (INL, DNL,
  ENOB, SNDR definitions used throughout the SAR ADC docs).
  [DOI 10.1109/IEEESTD.2011.5692956](https://doi.org/10.1109/IEEESTD.2011.5692956)
- **IEEE Std 1658-2011** — DAC terminology and test methods.
  [DOI 10.1109/IEEESTD.2012.6152113](https://doi.org/10.1109/IEEESTD.2012.6152113)
- **IEEE 1364-2005 §18** — VCD format (`eda-waveform`'s digital input).
- **W3C SVG 1.1 (Second Edition), 2011** — the format `eda-viz` and
  `eda-waveform` emit. <https://www.w3.org/TR/SVG11/>
- **ISO/IEC 15948:2004** — PNG, the default `eda-waveform` raster
  output. <https://www.iso.org/standard/29581.html>

### Photonic and RF references

- **MZI silicon-photonics validation set** — the four canonical
  results `spike-waveguide-block` is cross-checked against are listed
  inline in
  [`crates/spike-waveguide-block/docs/mzi_ml_trace.md`](crates/spike-waveguide-block/docs/mzi_ml_trace.md)
  and exercised as Rust tests in
  `crates/spike-waveguide-block/tests/literature_validation.rs`.
- **LNA noise-figure / matching theory** (the RF block in `spike-lna`):
  B. Razavi, *RF Microelectronics*, 2nd ed., Prentice Hall, 2011, §5.3.

For glossary entries that point at primary sources but are not
load-bearing for the README's headline claims (e.g. MLE / MSE / MLP),
see [`docs/glossary.md`](docs/glossary.md).

## Verification

Every DOI and arXiv ID above was re-resolved against the Crossref API
and `arxiv.org` on 2026-05-10, with the resolved title / authors /
publication year compared back to the surface text in this file (and
the matching entries in [`docs/glossary.md`](docs/glossary.md)).
All 15 references match. Two notes on metadata edge cases:

- **Larrañaga & Lozano (EDA book, `10.1007/978-1-4615-1539-5`)** —
  Crossref returns an empty author list for this entry because the
  volume is edited rather than authored; the editors are correctly
  rendered above as "Larrañaga & Lozano (eds.)", and the Crossref
  title and 2002 publication year both match.
- **IEEE 1241-2010 / 1658-2011** — Crossref does not surface a
  publication year for IEEE-SA standards. The "-YYYY" suffix in each
  standard's name (1241-**2010**, 1658-**2011**) is the canonical
  approval year and is preserved; the IEEEXplore DOI ingest dates
  (2011 / 2012) are not used in the citation.

Three entries from an earlier draft of this file were corrected when
their DOIs failed to resolve: the Baydin AD survey (`10.5555/...`
404'd; the JMLR paper has no Crossref DOI, so the canonical pointer
is the arXiv preprint), Cooley-Tukey FFT (switched from a JSTOR DOI
that doesn't appear in Crossref to the AMS DOI which does), and the
Mockus Bayesian-optimization reference (the previous Springer-LNCS
DOI 404'd; replaced with the IFAC Proceedings DOI for the same
paper). [`docs/glossary.md`](docs/glossary.md) was updated to match.
