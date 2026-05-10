//! Train the surrogate MLP and print the loss trajectory at a few
//! checkpoints so you can eyeball convergence.

use spike_surrogate::*;

fn main() {
    let res = train(/* n_steps */ 1_000, /* lr */ 5e-3, /* seed */ 0xC0FFEE);
    let n = res.losses.len();
    let pct = |p: usize| res.losses[(n - 1).min(p * n / 100)];
    println!("MLP surrogate training (BATCH={BATCH}, HIDDEN={HIDDEN}, lr=5e-3)");
    println!("  step    {:>4}    loss = {:+.6e}", 0,         pct(0));
    println!("  step    {:>4}    loss = {:+.6e}", n / 10,    pct(10));
    println!("  step    {:>4}    loss = {:+.6e}", n / 4,     pct(25));
    println!("  step    {:>4}    loss = {:+.6e}", n / 2,     pct(50));
    println!("  step    {:>4}    loss = {:+.6e}", 3 * n / 4, pct(75));
    println!("  step    {:>4}    loss = {:+.6e}", n - 1,     pct(100 - 1));
    println!();
    println!("  parameters: {} total", res.final_weights.len());
}
