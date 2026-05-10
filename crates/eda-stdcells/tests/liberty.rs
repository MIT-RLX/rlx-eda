//! Liberty parser tests — synthetic snippets shaped like
//! sky130_fd_sc_hd's exported `.lib`.

use eda_stdcells::liberty::{parse_lib, PinDirection};

const TWO_CELL_LIB: &str = r#"
library (sky130_fd_sc_hd_tt_025C_1v80) {
    /* drift comment */
    delay_model : table_lookup;
    cell ("sky130_fd_sc_hd__inv_1") {
        area : 5.0048;
        // line comment between attrs
        pin ("A") {
            direction : input;
            capacitance : 0.0023;
        }
        pin ("Y") {
            direction : output;
            function : "(!A)";
        }
        pin ("VPWR") {
            direction : inout;
        }
    }
    cell ("sky130_fd_sc_hd__nand2_1") {
        area : 6.2560;
        pin ("A") { direction : input; }
        pin ("B") { direction : input; }
        pin ("Y") {
            direction : output;
            function : "(!(A&B))";
        }
        leakage_power () {
            when : "A&B";
            value : 0.0017;
        }
    }
}
"#;

#[test]
fn parses_two_cells_with_correct_area() {
    let cells = parse_lib(TWO_CELL_LIB).expect("parse");
    assert_eq!(cells.len(), 2);
    assert_eq!(cells[0].cell_name, "sky130_fd_sc_hd__inv_1");
    assert_eq!(cells[0].area_um2_x1000, 5005); // 5.0048 → ×1000 round
    assert_eq!(cells[1].cell_name, "sky130_fd_sc_hd__nand2_1");
    assert_eq!(cells[1].area_um2_x1000, 6256);
}

#[test]
fn captures_pin_directions_and_functions() {
    let cells = parse_lib(TWO_CELL_LIB).expect("parse");
    let inv = &cells[0];
    assert_eq!(inv.pins.len(), 3);
    assert_eq!(inv.pins[0].name, "A");
    assert_eq!(inv.pins[0].direction, PinDirection::Input);
    assert_eq!(inv.pins[0].function, None);
    assert_eq!(inv.pins[1].name, "Y");
    assert_eq!(inv.pins[1].direction, PinDirection::Output);
    assert_eq!(inv.pins[1].function.as_deref(), Some("(!A)"));
    assert_eq!(inv.pins[2].direction, PinDirection::Inout);

    let nand = &cells[1];
    assert_eq!(nand.pins.len(), 3);
    assert_eq!(nand.pins[2].function.as_deref(), Some("(!(A&B))"));
}

#[test]
fn skips_unknown_groups_and_attributes() {
    // `leakage_power` and `delay_model` should not derail parsing.
    let cells = parse_lib(TWO_CELL_LIB).expect("parse");
    assert_eq!(cells.len(), 2, "leakage_power group should not produce a phantom cell");
}

#[test]
fn parses_empty_library() {
    let txt = r#"library (foo) { }"#;
    let cells = parse_lib(txt).expect("parse");
    assert!(cells.is_empty());
}

#[test]
fn parses_cell_with_no_pins() {
    let txt = r#"
        library (foo) {
            cell ("naked") {
                area : 1.0;
            }
        }"#;
    let cells = parse_lib(txt).expect("parse");
    assert_eq!(cells.len(), 1);
    assert_eq!(cells[0].cell_name, "naked");
    assert_eq!(cells[0].area_um2_x1000, 1000);
    assert!(cells[0].pins.is_empty());
}

#[test]
fn rejects_unterminated_string() {
    let txt = r#"library (foo) { cell ("bar"#;
    assert!(parse_lib(txt).is_err());
}

#[test]
fn handles_block_and_line_comments_together() {
    let txt = r#"
        /* multiple
           line block */
        library (foo) {
            // a comment
            cell ("c1") {
                /* inline */ area : 2.5;
                pin ("P") { direction : input; } // trailing
            }
        }"#;
    let cells = parse_lib(txt).expect("parse");
    assert_eq!(cells.len(), 1);
    assert_eq!(cells[0].area_um2_x1000, 2500);
    assert_eq!(cells[0].pins.len(), 1);
}
