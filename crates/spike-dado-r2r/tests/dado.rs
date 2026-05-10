//! Convergence comparison: on a perfectly-decomposable synthetic
//! objective, DADO should beat naive EDA averaged over a handful of
//! seeds. This is the chain-JT analogue of the paper's Fig 1c.

use spike_dado_r2r::{random_design, run, score_synth, Design, Rng};

#[test]
fn dado_beats_eda_on_synthetic_chain_average() {
    let n_iters    = 60;
    let k_samples  = 100;
    let tau        = 1.0;
    let alpha      = 0.1;
    let n_seeds    = 6;

    // Fix a random target_design so the optimum is well-defined.
    let mut target_rng = Rng::new(13);
    let target: Design = random_design(&mut target_rng);
    let score = move |x: &Design| score_synth(x, &target);

    let mut dado_finals = Vec::with_capacity(n_seeds);
    let mut eda_finals  = Vec::with_capacity(n_seeds);
    for s in 0..n_seeds {
        let seed = (s as u32) * 2 + 101;
        let dado = run(&score, n_iters, k_samples, tau, alpha, true,  seed, &[]);
        let eda  = run(&score, n_iters, k_samples, tau, alpha, false, seed, &[]);
        dado_finals.push(*dado.best.last().unwrap());
        eda_finals .push(*eda .best.last().unwrap());
    }

    let dado_mean: f64 = dado_finals.iter().sum::<f64>() / n_seeds as f64;
    let eda_mean : f64 = eda_finals .iter().sum::<f64>() / n_seeds as f64;

    // Both are non-positive (score = -hamming distance, max is 0).
    // DADO should average noticeably closer to zero than EDA.
    assert!(dado_mean > eda_mean,
            "DADO ({dado_mean}) did not beat EDA ({eda_mean}) on average");
    // And DADO should be near the optimum (sweep shows it usually hits 0).
    assert!(dado_mean >= -1.0,
            "DADO underperformed: final mean best = {dado_mean}");
}
