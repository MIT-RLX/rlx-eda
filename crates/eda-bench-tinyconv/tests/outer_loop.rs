//! End-to-end outer loop test. Walks a small grid of `ArrayConfig`
//! candidates, runs the inner Adam loop on each, picks the winner.
//!
//! v1 outer = brute-force grid search; full DADO is a strategy
//! swap-in later.

use eda_bench_tinyconv::optimization::{
    inner::InnerConfig,
    outer::{run as run_outer, OuterResult},
    OptError,
};
use spike_tinyconv_array::array::ArrayConfig;
use spike_tinyconv_tile::{MacTopology, TileParams};

fn cfg(weight_bits: u8, w_l: f64, vdd: f64) -> ArrayConfig {
    ArrayConfig {
        grid: (4, 4),
        pipeline_depth: 1,
        topology: MacTopology::Digital,
        tile_params: TileParams {
            w_l_n: w_l,
            w_l_p: w_l,
            vdd,
            bias_v: 0.0,
            weight_bits,
        },
    }
}

fn inner_short() -> InnerConfig {
    // Keep tests fast — 20 Adam steps is plenty to demonstrate
    // selection. Real benches run 200+.
    InnerConfig {
        max_steps: 20,
        ..InnerConfig::default()
    }
}

#[test]
fn outer_run_empty_candidates_returns_outer_budget() {
    match run_outer(&[], &inner_short()) {
        Err(OptError::OuterBudget) => {}
        other => panic!("expected OuterBudget, got {other:?}"),
    }
}

#[test]
fn outer_run_picks_lowest_final_loss_candidate() {
    // Three candidates with different starting params. The winner
    // should be whichever candidate has the smallest final_loss
    // after Adam runs — independent of starting position. With
    // the multi-term loss (energy + delay + area + accuracy gate),
    // the optimal point is interior, so neither extreme starting
    // point is guaranteed to win.
    let candidates = vec![
        cfg(8, 4.0, 1.8), // high
        cfg(8, 1.0, 1.2), // mid
        cfg(8, 0.2, 0.7), // low
    ];
    let result: OuterResult = run_outer(&candidates, &inner_short()).expect("outer runs");

    assert_eq!(result.all_results.len(), 3);
    assert!(result.best_final_loss.is_finite());

    // The winner must have the strictly smallest final_loss across
    // all non-diverged candidates.
    for (i, c) in result.all_results.iter().enumerate() {
        if let Some(loss) = c.final_loss {
            if i != result.best_index {
                assert!(
                    loss >= result.best_final_loss,
                    "candidate {} has loss {} < best {}",
                    i,
                    loss,
                    result.best_final_loss,
                );
            }
        }
    }
}

#[test]
fn outer_run_skips_diverged_candidates_but_keeps_them_in_results() {
    // Two candidates: one Digital (works), one CR (no body → inner
    // diverges immediately). Outer should pick the Digital one and
    // record the CR as `final_loss = None`.
    let cr = ArrayConfig {
        topology: MacTopology::ChargeRedistribution,
        ..cfg(8, 1.0, 1.5)
    };
    let candidates = vec![cfg(8, 1.0, 1.5), cr];

    let result = run_outer(&candidates, &inner_short()).expect("outer runs");
    assert_eq!(result.best_index, 0, "Digital candidate must win");

    let cr_result = &result.all_results[1];
    assert!(
        cr_result.final_loss.is_none(),
        "CR candidate should be marked diverged"
    );
    assert_eq!(cr_result.n_steps, 0);
}

#[test]
fn outer_run_returns_outer_budget_when_all_candidates_diverge() {
    let cr1 = ArrayConfig {
        topology: MacTopology::ChargeRedistribution,
        ..cfg(8, 1.0, 1.5)
    };
    let cr2 = ArrayConfig {
        topology: MacTopology::CurrentMode,
        ..cfg(8, 1.0, 1.5)
    };
    match run_outer(&[cr1, cr2], &inner_short()) {
        Err(OptError::OuterBudget) => {}
        other => panic!("expected OuterBudget, got {other:?}"),
    }
}

#[test]
fn outer_run_preserves_candidate_order_in_all_results() {
    let candidates = vec![cfg(2, 1.0, 1.5), cfg(4, 1.0, 1.5), cfg(8, 1.0, 1.5)];
    let result = run_outer(&candidates, &inner_short()).expect("outer runs");
    let weight_bits: Vec<u8> = result
        .all_results
        .iter()
        .map(|r| r.config.tile_params.weight_bits)
        .collect();
    assert_eq!(weight_bits, vec![2, 4, 8]);
}

#[test]
fn outer_run_best_trace_matches_best_index() {
    // best_trace is the inner-loop trace from the winning candidate.
    // Sanity: it's non-empty and the last step's p_total matches
    // best_final_loss exactly.
    let candidates = vec![cfg(8, 0.5, 1.0), cfg(8, 2.0, 1.6)];
    let result = run_outer(&candidates, &inner_short()).expect("outer runs");
    let last = result.best_trace.last().expect("non-empty trace");
    assert_eq!(last.p_total, result.best_final_loss);
}
