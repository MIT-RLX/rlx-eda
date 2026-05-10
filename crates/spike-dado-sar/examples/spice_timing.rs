use spike_dado_sar::{catalog, score_spice};
use std::time::Instant;
fn main() {
    let invoker = score_spice::invoker_from_env().expect("invoker");
    let nominal = catalog::NOMINAL;
    // Warm-up.
    let _ = score_spice::score_spice(invoker.as_ref(), &nominal, 1);
    // Time 3 evals at n_vins=4.
    for n_vins in [1usize, 4] {
        let t0 = Instant::now();
        let n_runs = 3;
        let mut total = 0.0_f64;
        for _ in 0..n_runs {
            let (s, _) = score_spice::score_spice(invoker.as_ref(), &nominal, n_vins);
            total += s;
        }
        let dt = t0.elapsed();
        println!("n_vins={} : {:.2}s for {} runs ({:.2}s/run, score≈{:.4})",
            n_vins, dt.as_secs_f64(), n_runs, dt.as_secs_f64() / n_runs as f64, total / n_runs as f64);
    }
}
