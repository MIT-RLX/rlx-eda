//! Verifies `Mac8x8Tile::cell_inventory` matches what `place_row`
//! actually places, and that the inventory is `ScHdLibrary`-keyable.
//!
//! Two sources of truth for the cell counts (the row-placement
//! constants in `layout.rs` and the inventory function) — drift
//! between them is the kind of bug `bisect.rs` would have to chase
//! later, so catch it at compile time via tests instead.

use spike_tinyconv_tile::{Mac8x8Tile, MacTopology, TileParams};

fn digital_tile() -> Mac8x8Tile {
    Mac8x8Tile::with_topology("u_inv", TileParams::default(), MacTopology::Digital)
}

#[test]
fn inventory_total_count_matches_floorplan() {
    let inv = digital_tile().cell_inventory();
    let total: usize = inv.iter().map(|(_, n)| n).sum();
    // PLAN.md floorplan: 8 + 32 + 24 + 32 + 32 + 16 + 16 + 16 + 16 + 10 = 202.
    assert_eq!(total, 202);
}

#[test]
fn inventory_includes_every_cell_type_used_by_place_row() {
    let inv: Vec<&str> = digital_tile().cell_inventory().into_iter().map(|(n, _)| n).collect();
    for name in [
        "sky130_fd_sc_hd__dfxtp_1",
        "sky130_fd_sc_hd__and2_1",
        "sky130_fd_sc_hd__fa_1",
        "sky130_fd_sc_hd__inv_1",
    ] {
        assert!(
            inv.iter().any(|n| *n == name),
            "{name} should appear in inventory"
        );
    }
}

#[test]
fn dff_count_matches_weight_register_plus_accumulator() {
    // Weight register = 8 DFFs, accumulator = 32 DFFs (low + high
    // halves of 16 each). Total = 40.
    let inv = digital_tile().cell_inventory();
    let dffs = inv
        .iter()
        .find(|(n, _)| *n == "sky130_fd_sc_hd__dfxtp_1")
        .map(|(_, c)| *c)
        .expect("dfxtp_1 in inventory");
    assert_eq!(dffs, 40);
}

#[test]
fn and2_count_matches_full_multiplier_pp_array() {
    // 8×8 multiplier has 8 partial-product rows × 8 AND2 = 64.
    let inv = digital_tile().cell_inventory();
    let and2 = inv
        .iter()
        .find(|(n, _)| *n == "sky130_fd_sc_hd__and2_1")
        .map(|(_, c)| *c)
        .expect("and2_1 in inventory");
    assert_eq!(and2, 64);
}

#[test]
fn fa_count_matches_multiplier_summing_plus_final_adder() {
    // Multiplier: 7 sum rows × 8 = 56 (split 24+32 across rows 0+1)
    // Final adder: 32 (split 16+16 across rows 2+3)
    // Total: 88.
    let inv = digital_tile().cell_inventory();
    let fa = inv
        .iter()
        .find(|(n, _)| *n == "sky130_fd_sc_hd__fa_1")
        .map(|(_, c)| *c)
        .expect("fa_1 in inventory");
    assert_eq!(fa, 88);
}
