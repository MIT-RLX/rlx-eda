//! Architectural test: training reduces loss substantially. We don't
//! pin a specific final loss (depends on init / batches) — just assert
//! the trajectory's last quartile is much smaller than the first.

use spike_surrogate::*;

#[test]
fn training_reduces_loss_by_at_least_10x() {
    let res = train(/* n_steps */ 3_000, /* lr */ 5e-3, /* seed */ 1);
    let n = res.losses.len();

    let first_quart_avg: f32 =
        res.losses[..n/4].iter().sum::<f32>() / (n/4) as f32;
    let last_quart_avg: f32 =
        res.losses[3*n/4..].iter().sum::<f32>() / (n - 3*n/4) as f32;

    println!("first_quart_avg = {first_quart_avg:.4e}");
    println!("last_quart_avg  = {last_quart_avg:.4e}");
    println!("ratio           = {:.2}", first_quart_avg / last_quart_avg);

    assert!(last_quart_avg < first_quart_avg / 10.0,
        "training failed to reduce loss 10×: first={first_quart_avg:.3e}, last={last_quart_avg:.3e}");
    // Loss should be small in absolute terms too — divider Vout is in [0,1].
    // MSE 5e-3 ⇒ RMS ~0.07, the kind of fit a 16-hidden-unit MLP gets after
    // 3000 steps without learning-rate scheduling. Real production surrogate
    // would tune lr / use larger MLP / batchnorm to get to ~1e-4.
    assert!(last_quart_avg < 1e-2,
        "final loss {last_quart_avg:.3e} too high — surrogate didn't fit");
}

#[test]
fn training_is_deterministic_under_fixed_seed() {
    // Same seed → same trajectory. Required for reproducible tests.
    let a = train(50, 5e-3, 42);
    let b = train(50, 5e-3, 42);
    for (la, lb) in a.losses.iter().zip(b.losses.iter()) {
        assert!((la - lb).abs() < 1e-6,
            "non-deterministic step: {la} vs {lb}");
    }
}
