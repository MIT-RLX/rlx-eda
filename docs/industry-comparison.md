# How rlx-eda compares to industry tools

`rlx-eda` is a research-stage Rust workspace, not a tape-out flow.
The fair peer set isn't Cadence — it's projects like Berkeley
[BAG3](https://github.com/ucb-art/BAG_framework),
[OpenROAD](https://theopenroadproject.org/) +
[ngspice](https://ngspice.sourceforge.io/) +
[KLayout](https://www.klayout.de/) +
[Magic](http://opencircuitdesign.com/magic/),
[Xyce](https://xyce.sandia.gov/), and academic differentiable-SPICE
prototypes. The table below is honest about scope: most rows where
rlx-eda lands `none` reflect deliberate choice (out of scope for a
research repo), not aspiration.

Items marked `none` that we plan to address are tracked in
[`../PLAN.md`](../PLAN.md) under *Industry-tool parity*; the *PDK
breadth* section there expands the second row.

| Capability | Cadence / Synopsys / Siemens | Open-source stack (ngspice / KLayout / OpenROAD / Magic / Xyce) | rlx-eda |
| --- | --- | --- | --- |
| PDK breadth | every foundry, every node | sky130, gf180mcu, ihp-sg13g2 | typed-layer PDKs auto-generated from `.lyp` for sky130 + gf180mcu (RcDivider lays out, extracts, and round-trips through ngspice under both, plus three smaller in-tree PDKs — RcDemo / Sky130Lite / Gf180Lite); ciel-install also covers gf180mcuA-D and ihp-sg13g2 |
| Device models | BSIM4/6, BSIM-CMG, HiSIM, PSP, Verilog-A | BSIM via ngspice | Shichman-Hodges level-1 + KT1/UTE thermal corners; BSIM4-in-rlx planned |
| DRC / LVS / PEX | Calibre, PVS, Pegasus (sign-off grade) | Magic + netgen | klayout-drc DRC + klayout-connect connectivity-LVS wired on RcDivider; eda-extract round-trips layout → SPICE → ngspice and cross-checks against the block's behavioral model (V·R₂/(R₁+R₂) within 1 ppm); PEX planned |
| AMS / mixed-signal | Spectre AMS, Xcelium AMS | limited | AC + transient; no event-driven HDL yet |
| RF / EM | Spectre RF, EMX, Clarity | — | none (S-params / HB / openEMS adapter planned) |
| Layout editor / schematic capture | Virtuoso | KLayout, Xschem | SVG/PNG render only; GDSII/OASIS/Xschem export planned |
| P&R, STA, power | Innovus, Tempus, Voltus, Fusion Compiler | OpenROAD, OpenSTA | none (OpenROAD/OpenSTA adapters planned) |
| Reliability / aging / EM | Spectre RelXpert, PrimeSim Reliability | — | none (BSIM4 aging + EM current-density planned) |
| Foundry sign-off | yes | no | no (out of scope) |
| Differentiable inverse design | ML-assisted layers (Cerebrus, DSO.ai) treat solver as black box | — | gradient-based, *through* the MNA solver; one IR drives both AD loop and SPICE deck |
| Surrogate-then-verify hybrid | proprietary | — | open, ~10 LOC integration; **36× wall-clock speedup vs direct ngspice** at ~5% relative quality loss on a 4-bit SAR ADC |
| IR layering | OA → CDL → SPICE → Liberty → DEF (lossy hops) | format-bridge soup | one typed-block IR (Device → Cell → Macro → Tile) end-to-end |
| Honest negative results | rare in vendor whitepapers | varies | DADO-vs-naive across 4 objectives × 25 experiments documented as null on real circuit metrics |
| Scale | thousand-engineer ecosystems, decades | community | single-author spike crates |

Bottom line: rlx-eda is not a Cadence replacement. It's a place to
test whether *typed-IR + differentiable-by-default + transparent
surrogate-then-verify* is a credible direction the big tools
haven't taken — and the SAR ADC numbers are the most honest
evidence so far.
