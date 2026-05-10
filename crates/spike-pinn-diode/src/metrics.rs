//! Accuracy + latency metrics.

use crate::config::V_REF;

#[derive(Clone, Copy, Debug)]
pub struct Accuracy {
    pub max_abs:    f32,
    pub rms:        f32,
    pub max_abs_fs: f32,
}

pub fn accuracy(pred: &[f32], truth: &[f32]) -> Accuracy {
    debug_assert_eq!(pred.len(), truth.len());
    let n = pred.len() as f32;
    let mut max_abs = 0.0_f32;
    let mut sse = 0.0_f32;
    for (p, t) in pred.iter().zip(truth) {
        let d = (p - t).abs();
        if d > max_abs { max_abs = d; }
        sse += d * d;
    }
    let rms = (sse / n).sqrt();
    Accuracy {
        max_abs,
        rms,
        max_abs_fs: max_abs / V_REF,
    }
}
