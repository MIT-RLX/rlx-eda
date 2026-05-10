//! Uniform 1-D sampling on `[0, 1)` with frozen RNG seeds.

use eda_nn::Rng;

pub fn uniform_samples(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new((seed as u32) ^ ((seed >> 32) as u32));
    (0..n).map(|_| rng.next_unit()).collect()
}
