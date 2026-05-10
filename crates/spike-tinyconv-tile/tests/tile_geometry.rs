//! `Tile<P>` geometry tests for the Digital MAC tile. No foundry
//! GDS required — exercises only the const-shaped `pitch` / `rails`
//! / `edge_ports` bodies against the `MosfetPdk` trait.

use eda_tile::{current_density_check, Side, Tile};
use klayout_core::Library;
use spike_divider_block::MosfetPdk;
use spike_tinyconv_tile::{Mac8x8Tile, MacTopology, TileParams};

/// Sky130 PDK registered against a fresh test library — same
/// pattern as `eda-stdcells/tests/stdcell_layout.rs`.
fn fake_pdk(lib: &Library) -> eda_pdks::Sky130 {
    eda_pdks::Sky130::register(lib)
}

fn digital_tile() -> Mac8x8Tile {
    Mac8x8Tile::with_topology(
        "u0",
        TileParams::default(),
        MacTopology::Digital,
    )
}

#[test]
fn pitch_matches_floorplan_constants() {
    let tile = digital_tile();
    let p = <Mac8x8Tile as Tile<eda_pdks::Sky130>>::pitch(&tile);
    // 220 µm × 10.88 µm in DBU; sky130 dbu_per_um = 1000.
    // (Original 24 µm guess corrected as cell placement landed —
    //  row 1 — multiplier upper half + accumulator low — needs
    //  ~196 µm; 220 µm gives ~12 % routing margin.)
    assert_eq!(p.x, 220_000);
    assert_eq!(p.y, 10_880);
}

#[test]
fn pitch_y_equals_four_sc_hd_rows() {
    // sc_hd row height = 2.72 µm = 2720 DBU. Tile is 4 rows tall.
    let tile = digital_tile();
    let p = <Mac8x8Tile as Tile<eda_pdks::Sky130>>::pitch(&tile);
    assert_eq!(p.y, 4 * 2_720);
}

#[test]
fn rails_have_alternating_vdd_gnd_at_sc_hd_pitch() {
    let lib = Library::new("rails-test", 1000);
    let pdk = fake_pdk(&lib);
    let tile = digital_tile();

    let r = tile.rails(&pdk);
    // 4 rows + 1 = 5 rail tracks: row=0..4, even=VDD odd=GND.
    assert_eq!(r.vdd_tracks, vec![0, 5_440, 10_880]);
    assert_eq!(r.gnd_tracks, vec![2_720, 8_160]);
    // Both rails on metal1 in v1; sc_hd convention.
    assert_eq!(r.vdd_layer, pdk.metal1());
    assert_eq!(r.gnd_layer, pdk.metal1());
    assert_eq!(r.width_dbu, 480);
    assert_eq!(r.dbu_per_um, 1_000);
}

#[test]
fn rails_pass_compose_time_pdn_check_for_4x4_grid() {
    let lib = Library::new("pdn-test", 1000);
    let pdk = fake_pdk(&lib);
    let r = digital_tile().rails(&pdk);

    // 4×4 grid (16 tiles per strap), per-tile peak ~0.1 mA, sky130
    // metal1 Jmax ~1 mA/µm → 0.48 mA budget on a 480 nm strap.
    // 16 × 0.1 = 1.6 mA, OVER budget — confirm the check catches it.
    assert!(current_density_check(&r, 0.1, 16, 1.0).is_err());

    // Same strap with looser per-tile load: 0.025 mA → 0.4 mA total
    // → under the 0.48 mA budget → pass.
    assert!(current_density_check(&r, 0.025, 16, 1.0).is_ok());
}

#[test]
fn each_side_returns_eight_bit_ports() {
    let lib = Library::new("ports-test", 1000);
    let pdk = fake_pdk(&lib);
    let tile = digital_tile();

    for side in [Side::West, Side::East, Side::North, Side::South] {
        assert_eq!(tile.edge_ports(side, &pdk).len(), 8, "{side:?}");
    }
}

#[test]
fn west_and_east_offsets_match_for_abutment() {
    let lib = Library::new("abut-test", 1000);
    let pdk = fake_pdk(&lib);
    let tile = digital_tile();

    let west = tile.edge_ports(Side::West, &pdk);
    let east = tile.edge_ports(Side::East, &pdk);

    // Abutment contract: act_in[bit] on west must align with
    // act_pass[bit] on east at the same y-offset (the activation
    // bus runs straight across the tile row).
    assert_eq!(west.len(), east.len());
    for (w, e) in west.iter().zip(east.iter()) {
        assert_eq!(w.offset_dbu, e.offset_dbu, "bit offset mismatch");
        assert_eq!(w.layer, e.layer);
    }
}

#[test]
fn north_and_south_offsets_match_for_abutment() {
    let lib = Library::new("abut-test", 1000);
    let pdk = fake_pdk(&lib);
    let tile = digital_tile();

    let north = tile.edge_ports(Side::North, &pdk);
    let south = tile.edge_ports(Side::South, &pdk);

    assert_eq!(north.len(), south.len());
    for (n, s) in north.iter().zip(south.iter()) {
        assert_eq!(n.offset_dbu, s.offset_dbu, "bit offset mismatch");
        assert_eq!(n.layer, s.layer);
    }
}

#[test]
fn port_offsets_lie_inside_tile_pitch() {
    let lib = Library::new("pitch-test", 1000);
    let pdk = fake_pdk(&lib);
    let tile = digital_tile();
    let pitch = <Mac8x8Tile as Tile<eda_pdks::Sky130>>::pitch(&tile);

    for side in [Side::West, Side::East, Side::North, Side::South] {
        let ports = tile.edge_ports(side, &pdk);
        let bound = match side {
            Side::West | Side::East => pitch.y,
            Side::North | Side::South => pitch.x,
        };
        for p in &ports {
            assert!(
                p.offset_dbu >= 0 && p.offset_dbu <= bound,
                "{} on {:?} at offset {} outside [0, {}]",
                p.name, p.side, p.offset_dbu, bound,
            );
        }
    }
}

#[test]
fn port_names_follow_floorplan_convention() {
    let lib = Library::new("names-test", 1000);
    let pdk = fake_pdk(&lib);
    let tile = digital_tile();

    let names_for = |s: Side| -> Vec<String> {
        tile.edge_ports(s, &pdk).into_iter().map(|p| p.name).collect()
    };
    let west = names_for(Side::West);
    let east = names_for(Side::East);
    let north = names_for(Side::North);
    let south = names_for(Side::South);

    assert_eq!(west[0], "act_in[0]");
    assert_eq!(west[7], "act_in[7]");
    assert_eq!(east[0], "act_pass[0]");
    assert_eq!(north[0], "weight_in[0]");
    assert_eq!(south[0], "weight_pass[0]");
}