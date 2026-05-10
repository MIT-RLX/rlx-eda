//! Known-PDK metadata: which ciel family each variant belongs to + the
//! relative path inside ciel's PDK-root layout where the SPICE lib lives
//! + the nominal supply voltage.

#[derive(Debug, Clone, Copy)]
pub struct KnownPdk {
    pub variant: &'static str,
    pub ciel_family: &'static str,
    /// Relative path from `<pdk_root>/<variant>/` to the top-level `.lib`
    /// SPICE file. Same shape across PDK families today.
    pub lib_subpath: &'static str,
    pub vdd_nom: f64,
}

pub const ALL: &[KnownPdk] = &[
    KnownPdk {
        variant: "sky130A",
        ciel_family: "sky130",
        lib_subpath: "libs.tech/ngspice/sky130.lib.spice",
        vdd_nom: 1.8,
    },
    KnownPdk {
        variant: "sky130B",
        ciel_family: "sky130",
        lib_subpath: "libs.tech/ngspice/sky130.lib.spice",
        vdd_nom: 1.8,
    },
    KnownPdk {
        variant: "gf180mcuA",
        ciel_family: "gf180mcu",
        // gf180mcu's open_pdks-built ngspice lib lives here in the ciel
        // tree; if a future ciel layout shifts this we'll auto-glob in
        // discover.rs.
        lib_subpath: "libs.tech/ngspice/sm141064.ngspice",
        vdd_nom: 5.0,
    },
    KnownPdk {
        variant: "gf180mcuB",
        ciel_family: "gf180mcu",
        lib_subpath: "libs.tech/ngspice/sm141064.ngspice",
        vdd_nom: 5.0,
    },
    KnownPdk {
        variant: "gf180mcuC",
        ciel_family: "gf180mcu",
        lib_subpath: "libs.tech/ngspice/sm141064.ngspice",
        vdd_nom: 5.0,
    },
    KnownPdk {
        variant: "gf180mcuD",
        ciel_family: "gf180mcu",
        lib_subpath: "libs.tech/ngspice/sm141064.ngspice",
        vdd_nom: 5.0,
    },
    KnownPdk {
        variant: "ihp-sg13g2",
        ciel_family: "ihp-sg13g2",
        // IHP's ngspice tree has no top-level sectioned `.lib` file —
        // corner selection is done by `.include`-ing one of the model
        // files directly. We point at the high-voltage MOS module as a
        // sensible default; sections will auto-detect as empty and the
        // user can `pdk register --sections` to add their own labels.
        lib_subpath: "libs.tech/ngspice/models/sg13g2_moshv_mod.lib",
        vdd_nom: 1.2,
    },
];

pub fn lookup(name: &str) -> Option<&'static KnownPdk> {
    ALL.iter().find(|p| p.variant.eq_ignore_ascii_case(name))
}

pub fn all_names() -> Vec<String> {
    ALL.iter().map(|p| p.variant.to_string()).collect()
}
