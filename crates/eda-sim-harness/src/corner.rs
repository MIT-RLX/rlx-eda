//! Corners: typical / extreme / Monte Carlo.
//!
//! cicsim's `make typical etc mc` flow maps to a [`CornerSet`] with the
//! three [`CornerKind`] members. A [`Corner`] is one concrete run — a
//! kind, a label (e.g. `"typical"`, `"mc_007"`), and the `.lib` section
//! tag the deck should include (`tt`, `ff_n40c`, …). Monte Carlo corners
//! also carry a seed for downstream `.param` randomization.

/// Family of a corner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CornerKind {
    /// Nominal process, nominal V/T.
    Typical,
    /// Extreme test condition: ff/ss/fs/sf at hot/cold extremes.
    Etc,
    /// Monte Carlo random draw.
    Mc,
}

/// Netlist view: schematic-ideal vs. layout-extracted (with parasitics).
/// Mirrors cicsim's `Sch` / `Lay` distinction. Pre-tape-out flow runs
/// every corner against both views and diffs the deltas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum View {
    /// Schematic netlist with ideal models. Default — fast, used for
    /// design exploration and architecture-level decisions.
    #[default]
    Schematic,
    /// Layout-extracted netlist (`.lpe.spi`) with parasitic R/C from
    /// magic / klayout extraction. Slower, used for tape-out gating.
    Layout,
}

impl View {
    pub fn as_str(self) -> &'static str {
        match self { View::Schematic => "Sch", View::Layout => "Lay" }
    }
}

impl CornerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CornerKind::Typical => "typical",
            CornerKind::Etc => "etc",
            CornerKind::Mc => "mc",
        }
    }
}

/// One concrete corner run.
#[derive(Debug, Clone)]
pub struct Corner {
    pub kind: CornerKind,
    /// Human label, used in reports and as the file suffix.
    pub label: String,
    /// `.lib` section name (e.g. `"tt"`, `"ff"`, `"ss"`). Some PDKs use
    /// compound names like `"ff_n40c"`; pass the exact string.
    pub lib_section: String,
    /// Supply voltage for this corner. The testbench reads this when
    /// emitting Vdd and any references to it.
    pub vdd: f64,
    /// Temperature in Celsius. Emitted as `.options temp=<t>`.
    pub temp_c: f64,
    /// Monte Carlo seed (`Mc` corners only).
    pub seed: Option<u64>,
    /// Schematic vs. layout-extracted netlist. Default `Schematic`.
    pub view: View,
}

impl Corner {
    pub fn typical(lib_section: impl Into<String>, vdd: f64) -> Self {
        Self {
            kind: CornerKind::Typical,
            label: "typical".into(),
            lib_section: lib_section.into(),
            vdd,
            temp_c: 27.0,
            seed: None,
            view: View::default(),
        }
    }

    pub fn etc(label: impl Into<String>, lib_section: impl Into<String>, vdd: f64, temp_c: f64) -> Self {
        Self {
            kind: CornerKind::Etc,
            label: label.into(),
            lib_section: lib_section.into(),
            vdd,
            temp_c,
            seed: None,
            view: View::default(),
        }
    }

    pub fn mc(label: impl Into<String>, lib_section: impl Into<String>, vdd: f64, temp_c: f64, seed: u64) -> Self {
        Self {
            kind: CornerKind::Mc,
            label: label.into(),
            lib_section: lib_section.into(),
            vdd,
            temp_c,
            seed: Some(seed),
            view: View::default(),
        }
    }

    /// Builder: switch this corner to the layout-extracted view.
    /// Used by `CornerSet::expand_views` to fan out into Sch + Lay pairs.
    pub fn with_view(mut self, view: View) -> Self {
        self.view = view;
        self
    }
}

/// A planned set of corners. Drives the harness's outer loop.
#[derive(Debug, Default, Clone)]
pub struct CornerSet {
    pub corners: Vec<Corner>,
}

impl CornerSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(mut self, c: Corner) -> Self {
        self.corners.push(c);
        self
    }

    /// `make typical` — single nominal run.
    pub fn typical_only(lib_section: impl Into<String>, vdd: f64) -> Self {
        Self::new().push(Corner::typical(lib_section, vdd))
    }

    /// `make typical etc` — nominal + four PVT extremes (ff_hot, ss_cold,
    /// fs_hot, sf_cold) at ±10 % vdd. Caller supplies the lib-section
    /// names since they vary per PDK (`ff` vs `ff_n40c` vs `ff_85`…).
    pub fn typical_etc(
        tt: &str,
        ff_hot: &str,
        ss_cold: &str,
        fs_hot: &str,
        sf_cold: &str,
        vdd_nom: f64,
    ) -> Self {
        Self::new()
            .push(Corner::typical(tt, vdd_nom))
            .push(Corner::etc("ff_hot", ff_hot, vdd_nom * 1.10, 85.0))
            .push(Corner::etc("ss_cold", ss_cold, vdd_nom * 0.90, -40.0))
            .push(Corner::etc("fs_hot", fs_hot, vdd_nom * 1.10, 85.0))
            .push(Corner::etc("sf_cold", sf_cold, vdd_nom * 0.90, -40.0))
    }

    /// `make mc` — append `n_runs` Monte Carlo corners against
    /// `lib_section` (e.g. sky130's `mc`, gf180mcu's `stat_mc`). Seeds
    /// start at `seed_base` and increment so every MC run is unique
    /// and reproducible across reruns.
    pub fn add_mc(mut self, lib_section: &str, vdd: f64, n_runs: usize, seed_base: u64) -> Self {
        for i in 0..n_runs {
            self.corners.push(Corner::mc(
                format!("mc_{:03}", i),
                lib_section,
                vdd,
                27.0,
                seed_base.wrapping_add(i as u64),
            ));
        }
        self
    }

    /// `make typical etc mc` — typical + four PVT extremes + `n_mc`
    /// Monte Carlo runs in one go. Mirrors cicsim's all-in-one batch.
    /// `mc_lib_section` is the PDK-specific MC section (sky130: `"mc"`,
    /// gf180mcu: `"stat_mc"`, IHP: pass the user-registered section
    /// name).
    pub fn typical_etc_mc(
        tt: &str,
        ff_hot: &str,
        ss_cold: &str,
        fs_hot: &str,
        sf_cold: &str,
        mc_lib_section: &str,
        vdd_nom: f64,
        n_mc: usize,
        seed_base: u64,
    ) -> Self {
        Self::typical_etc(tt, ff_hot, ss_cold, fs_hot, sf_cold, vdd_nom)
            .add_mc(mc_lib_section, vdd_nom, n_mc, seed_base)
    }

    /// Duplicate every corner into Schematic + Layout pairs. Doubles
    /// the run count but gives you `Sch_<label>` + `Lay_<label>` for
    /// every base corner — the cicsim `make all` regression flow.
    /// Layout corners get the same lib_section; the testbench is
    /// responsible for swapping the netlist `.include` to the extracted
    /// `.lpe.spi` when `corner.view == View::Layout`.
    pub fn expand_views(self) -> Self {
        let mut out = Vec::with_capacity(self.corners.len() * 2);
        for c in self.corners {
            out.push(c.clone().with_view(View::Schematic));
            out.push(c.with_view(View::Layout));
        }
        Self { corners: out }
    }
}
