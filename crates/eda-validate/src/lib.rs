//! Numerical validation primitives for the rlx-eda stack.
//!
//! Roles:
//! - `assert_close` / `is_close`: relative+absolute tolerance comparison.
//! - `central_difference`: independent FD reference for any scalar f32 → f32 fn.
//! - `gradcheck_scalar`: triangulates AD-grad against FD-grad for a multi-input
//!   scalar-output function.
//! - `assert_traces_close` (f64): compares two time-domain (or
//!   frequency-domain) traces on potentially different sample grids by
//!   linearly interpolating onto the coarser grid before tolerance-checking.
//!
//! Convention: every component model and analysis lands with a `gradcheck`
//! test alongside its implementation.

/// Element-wise tolerance check used everywhere in the stack.
///
/// Returns true iff `|a - b| <= atol + rtol * |b|`.
#[inline]
pub fn is_close(a: f32, b: f32, rtol: f32, atol: f32) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

/// Panics with a useful message if `a` and `b` differ by more than the
/// rtol/atol envelope.
#[track_caller]
pub fn assert_close(a: f32, b: f32, rtol: f32, atol: f32, label: &str) {
    if !is_close(a, b, rtol, atol) {
        panic!(
            "[{label}] not close:\n  a    = {a:.9e}\n  b    = {b:.9e}\n  |a-b|= {diff:.3e}  (envelope = {env:.3e})",
            diff = (a - b).abs(),
            env  = atol + rtol * b.abs(),
        );
    }
}

/// Central-difference derivative: `(f(x + eps) - f(x - eps)) / (2 * eps)`.
///
/// Two evaluations per call; truncation error is `O(eps^2)` for smooth `f`,
/// roundoff floor `~ eps_machine / eps`. For f32, `eps = 1e-3` is a good
/// default — the truncation error and roundoff cross around there.
pub fn central_difference<F: FnMut(f32) -> f32>(mut f: F, x: f32, eps: f32) -> f32 {
    let fp = f(x + eps);
    let fm = f(x - eps);
    (fp - fm) / (2.0 * eps)
}

/// Compare an AD-computed gradient vector against finite differences of the
/// same forward function.
///
/// `forward(params) -> scalar` is evaluated `2 * params.len()` times. Useful
/// as a baseline when no analytic reference exists. `eps` is per-parameter;
/// pass `1e-3` for f32 unless you have a reason to scale.
///
/// Returns `Ok(())` if every entry is close; otherwise an error describing
/// the first mismatch.
pub fn gradcheck_scalar<F>(
    forward: &mut F,
    params: &[f32],
    ad_grad: &[f32],
    eps: f32,
    rtol: f32,
    atol: f32,
) -> Result<(), String>
where
    F: FnMut(&[f32]) -> f32,
{
    assert_eq!(params.len(), ad_grad.len(), "params and grad length mismatch");
    for i in 0..params.len() {
        let mut probe = params.to_vec();
        let xi = params[i];
        probe[i] = xi + eps;
        let fp = forward(&probe);
        probe[i] = xi - eps;
        let fm = forward(&probe);
        let fd = (fp - fm) / (2.0 * eps);
        if !is_close(ad_grad[i], fd, rtol, atol) {
            return Err(format!(
                "gradcheck mismatch at param[{i}]:\n  AD = {ad:.9e}\n  FD = {fd:.9e}\n  diff = {diff:.3e}",
                ad = ad_grad[i],
                diff = (ad_grad[i] - fd).abs(),
            ));
        }
    }
    Ok(())
}

// ── Trace comparison (f64) ─────────────────────────────────────────────
//
// Time-domain comparisons need to handle the case where the two simulators
// produce different sample grids: rlx with our outer-loop BE uses uniform
// timesteps, ngspice's `.tran` uses LTE-controlled adaptive stepping. The
// trace shape is the same; only the sample locations differ.

/// `|a - b| <= atol + rtol * |b|` for f64.
#[inline]
pub fn is_close_f64(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

/// Linear interpolation of a (x, y) trace at `xq`. `xs` must be sorted
/// ascending. Out-of-range queries clamp to the nearest endpoint — for
/// transient/AC tests we always evaluate *inside* the simulator's range.
pub fn lerp(xs: &[f64], ys: &[f64], xq: f64) -> f64 {
    debug_assert_eq!(xs.len(), ys.len());
    debug_assert!(!xs.is_empty(), "empty trace");
    if xq <= xs[0] { return ys[0]; }
    if xq >= xs[xs.len() - 1] { return ys[ys.len() - 1]; }
    // Binary search for the interval [xs[i], xs[i+1]] containing xq.
    let i = match xs.binary_search_by(|x| x.partial_cmp(&xq).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(j) => return ys[j],
        Err(j) => j - 1,
    };
    let (x0, x1) = (xs[i], xs[i + 1]);
    let (y0, y1) = (ys[i], ys[i + 1]);
    let t = (xq - x0) / (x1 - x0);
    y0 + t * (y1 - y0)
}

/// Compare two traces sampled on (possibly different) grids.
///
/// `(t_a, y_a)` and `(t_b, y_b)` must each be ascending in their `t`. We
/// pick whichever grid is **shorter** as the comparison grid (it has the
/// fewest evaluation points and tends to be the lower-resolution one), and
/// interpolate the other onto it. Each sample is checked with `is_close_f64`.
///
/// Returns the index of the first mismatch as `Err`, or `Ok(())` on full
/// agreement. Panics with a useful message when used as `assert_*`.
pub fn traces_close(
    t_a: &[f64], y_a: &[f64],
    t_b: &[f64], y_b: &[f64],
    rtol: f64, atol: f64,
) -> std::result::Result<(), (usize, f64, f64, f64)> {
    assert_eq!(t_a.len(), y_a.len());
    assert_eq!(t_b.len(), y_b.len());
    assert!(!t_a.is_empty() && !t_b.is_empty());

    // Use the shorter (likely coarser) grid as the comparison axis.
    let (t_grid, y_grid, t_other, y_other) = if t_a.len() <= t_b.len() {
        (t_a, y_a, t_b, y_b)
    } else {
        (t_b, y_b, t_a, y_a)
    };

    // Restrict to the overlap of the two ranges; outside the overlap the
    // result is dominated by clamping artifacts, not numerical agreement.
    let t_lo = t_a[0].max(t_b[0]);
    let t_hi = t_a[t_a.len() - 1].min(t_b[t_b.len() - 1]);

    for (i, (&t, &y)) in t_grid.iter().zip(y_grid.iter()).enumerate() {
        if t < t_lo || t > t_hi { continue; }
        let y_other_at_t = lerp(t_other, y_other, t);
        if !is_close_f64(y, y_other_at_t, rtol, atol) {
            return Err((i, t, y, y_other_at_t));
        }
    }
    Ok(())
}

#[track_caller]
pub fn assert_traces_close(
    t_a: &[f64], y_a: &[f64],
    t_b: &[f64], y_b: &[f64],
    rtol: f64, atol: f64, label: &str,
) {
    if let Err((i, t, ya, yb)) = traces_close(t_a, y_a, t_b, y_b, rtol, atol) {
        panic!(
            "[{label}] traces diverge at idx {i}, t={t:+.6e}:\n  a   = {ya:+.9e}\n  b@t = {yb:+.9e}\n  |a-b| = {diff:.3e}  (envelope = {env:.3e})",
            diff = (ya - yb).abs(),
            env  = atol + rtol * yb.abs(),
        );
    }
}

// ── Cross-simulator triangulation ──────────────────────────────────────
//
// The Invoker results from `eda-extern-ngspice` and `eda-extern-ltspice`
// share a `HashMap<String, f64>` (DC) and `(Vec<f64>, HashMap<String,
// Vec<f64>>)` (transient) shape. These helpers compare those shapes
// without depending on either crate, so callers wire them together with
// 3 lines of glue code in a test.
//
// The reports include a `worst` mismatch + per-node breakdown so a
// failing CI gate points at exactly which node disagrees and by how
// much — not just "ngspice and LTspice disagree somewhere".

use std::collections::HashMap;

/// Per-node difference between two DC operating-point results.
#[derive(Debug, Clone)]
pub struct DcDiffReport {
    /// `node → (a_value, b_value, |a - b|)`.
    pub per_node: HashMap<String, (f64, f64, f64)>,
    /// `(node, |diff|)` for the worst node, if any.
    pub worst: Option<(String, f64)>,
    /// RMS of the per-node `|a - b|` differences.
    pub rms: f64,
}

impl DcDiffReport {
    /// Panic with a useful summary if any node exceeds the rtol/atol envelope.
    #[track_caller]
    pub fn assert_within(&self, rtol: f64, atol: f64, label: &str) {
        for (node, (a, b, _)) in &self.per_node {
            if !is_close_f64(*a, *b, rtol, atol) {
                panic!(
                    "[{label}] DC mismatch at node {node:?}:\n  a = {a:+.9e}\n  b = {b:+.9e}\n  |a-b| = {diff:.3e}  (envelope = {env:.3e})\n  rms across nodes = {rms:.3e}",
                    diff = (a - b).abs(),
                    env  = atol + rtol * b.abs(),
                    rms  = self.rms,
                );
            }
        }
    }
}

/// Compare two DC operating-point result maps. Both maps must contain
/// the same set of nodes — missing keys are panic-worthy bugs in the
/// caller, not numerical disagreements.
pub fn compare_dc_voltages(
    a: &HashMap<String, f64>,
    b: &HashMap<String, f64>,
) -> DcDiffReport {
    assert!(!a.is_empty(), "left DC map is empty");
    assert_eq!(
        a.len(), b.len(),
        "DC maps differ in length: {} vs {}", a.len(), b.len()
    );

    let mut per_node = HashMap::new();
    let mut sum_sq = 0.0_f64;
    let mut worst: Option<(String, f64)> = None;
    for (node, &av) in a {
        let bv = *b.get(node).unwrap_or_else(|| {
            panic!("right DC map missing node {node:?}; left has it = {av}")
        });
        let d = (av - bv).abs();
        sum_sq += d * d;
        if worst.as_ref().is_none_or(|(_, w)| d > *w) {
            worst = Some((node.clone(), d));
        }
        per_node.insert(node.clone(), (av, bv, d));
    }
    let rms = (sum_sq / per_node.len() as f64).sqrt();
    DcDiffReport { per_node, worst, rms }
}

/// Per-node transient diff: max-abs and RMS over the overlap interval,
/// computed by interpolating `b` onto `a`'s time grid.
#[derive(Debug, Clone)]
pub struct TransientDiffReport {
    /// `node → (max_abs_diff, rms_diff)`.
    pub per_node: HashMap<String, (f64, f64)>,
    /// `(node, max_abs_diff)` for the worst node.
    pub worst: Option<(String, f64)>,
}

impl TransientDiffReport {
    /// Panic if any node's max-abs diff exceeds the envelope. Uses
    /// scalar tolerance (the trace's peak `|y|` is the natural reference
    /// scale), via `max_abs <= atol + rtol * peak`.
    #[track_caller]
    pub fn assert_within(&self, rtol: f64, atol: f64, peak_per_node: f64, label: &str) {
        for (node, &(max_abs, rms)) in &self.per_node {
            let env = atol + rtol * peak_per_node;
            if max_abs > env {
                panic!(
                    "[{label}] transient mismatch at node {node:?}:\n  max|a-b| = {max_abs:.3e}\n  rms      = {rms:.3e}\n  envelope = {env:.3e}",
                );
            }
        }
    }
}

/// Compare two transient traces node-by-node. Both must have the same
/// node set; time grids may differ — `b` is interpolated onto `a`'s grid.
pub fn compare_transient_traces(
    t_a: &[f64], a: &HashMap<String, Vec<f64>>,
    t_b: &[f64], b: &HashMap<String, Vec<f64>>,
) -> TransientDiffReport {
    assert!(!a.is_empty());
    assert_eq!(a.len(), b.len(), "trace maps differ in length");

    let t_lo = t_a[0].max(t_b[0]);
    let t_hi = t_a[t_a.len() - 1].min(t_b[t_b.len() - 1]);

    let mut per_node = HashMap::new();
    let mut worst: Option<(String, f64)> = None;
    for (node, ya) in a {
        let yb = b.get(node).unwrap_or_else(|| {
            panic!("right trace map missing node {node:?}")
        });
        assert_eq!(ya.len(), t_a.len(), "node {node:?}: |y_a| != |t_a|");
        assert_eq!(yb.len(), t_b.len(), "node {node:?}: |y_b| != |t_b|");

        let mut max_abs = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut count = 0usize;
        for (t, y) in t_a.iter().copied().zip(ya.iter().copied()) {
            if t < t_lo || t > t_hi { continue; }
            let yb_at_t = lerp(t_b, yb, t);
            let d = (y - yb_at_t).abs();
            if d > max_abs { max_abs = d; }
            sum_sq += d * d;
            count += 1;
        }
        let rms = if count > 0 { (sum_sq / count as f64).sqrt() } else { 0.0 };
        if worst.as_ref().is_none_or(|(_, w)| max_abs > *w) {
            worst = Some((node.clone(), max_abs));
        }
        per_node.insert(node.clone(), (max_abs, rms));
    }
    TransientDiffReport { per_node, worst }
}

// ── Layout-vs-behavioral witness ───────────────────────────────────────
//
// The third leg of the validation pyramid. The first two compare two
// *simulators* run on the same circuit description (rlx vs ngspice,
// ngspice vs LTspice). This one compares two *circuit descriptions* of
// the same block — the behavioral / analytic model the block carries
// (DcBehavioral, S-params, MNA stamps) against the SPICE deck emitted
// from extracting its laid-out geometry. Disagreement flags one of:
//
//   - the layout dropped a wire (LNA/MZI symptom — pads exist, routes
//     don't, extracted nets short or split)
//   - the device-recognizer maps a child cell to the wrong R / C value
//   - the behavioral model and the layout-PDK rules drifted apart
//     (e.g. sheet-ρ change in length_to_resistance)
//
// Wraps `compare_dc_voltages` rather than re-implementing it; the value
// here is the failure phrasing — "extracted-from-layout deck disagrees
// with behavioral model" is a different debugging story from
// "two DC simulators disagree".

/// DC-level layout-vs-behavioral comparison. Build via [`Self::dc`] and
/// drive the assertion with [`Self::assert_within`].
#[derive(Debug, Clone)]
pub struct LayoutVsBehavioralReport {
    pub inner: DcDiffReport,
}

impl LayoutVsBehavioralReport {
    /// Compare an extracted-deck DC operating point against the block's
    /// behavioral / analytic prediction. Both maps must share node names.
    pub fn dc(
        extracted: &HashMap<String, f64>,
        behavioral: &HashMap<String, f64>,
    ) -> Self {
        Self { inner: compare_dc_voltages(extracted, behavioral) }
    }

    /// Worst-disagreeing node, if any: `(node, |extracted - behavioral|)`.
    pub fn worst(&self) -> Option<(&str, f64)> {
        self.inner.worst.as_ref().map(|(n, d)| (n.as_str(), *d))
    }

    /// Panic if any node breaches the rtol/atol envelope. The failure
    /// message names the offender so a CI gate points at the right
    /// likely culprit (missing wire vs. recognizer mis-value vs. PDK
    /// drift) instead of just "DC mismatch".
    #[track_caller]
    pub fn assert_within(&self, rtol: f64, atol: f64, label: &str) {
        for (node, (ex, be, _)) in &self.inner.per_node {
            if !is_close_f64(*ex, *be, rtol, atol) {
                panic!(
                    "[{label}] layout-vs-behavioral mismatch at node {node:?}:\n\
                     \n  extracted-deck: {ex:+.9e}\n  behavioral:     {be:+.9e}\n  |Δ|           = {diff:.3e}  (envelope = {env:.3e})\n  rms across nodes = {rms:.3e}\n\n\
                     Likely causes (in order):\n  1. missing wire in Layout::layout — extraction split a net\n  2. DeviceRecognizer returned the wrong value for this block\n  3. behavioral model and layout PDK rules drifted (e.g. sheet ρ)",
                    diff = (ex - be).abs(),
                    env  = atol + rtol * be.abs(),
                    rms  = self.inner.rms,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_within_tolerance() {
        assert!(is_close(1.0, 1.0 + 1e-6, 1e-5, 0.0));
        assert!(!is_close(1.0, 1.1, 1e-5, 0.0));
    }

    #[test]
    fn fd_quadratic() {
        // f(x) = x^2, f'(2) = 4.
        let g = central_difference(|x| x * x, 2.0, 1e-3);
        assert_close(g, 4.0, 1e-4, 0.0, "central_difference quadratic");
    }

    #[test]
    fn gradcheck_linear_ok() {
        // f(a, b) = 3a + 5b, true grad = [3, 5].
        let mut fwd = |p: &[f32]| 3.0 * p[0] + 5.0 * p[1];
        gradcheck_scalar(&mut fwd, &[0.7, -0.2], &[3.0, 5.0], 1e-3, 1e-4, 1e-6).unwrap();
    }

    #[test]
    fn gradcheck_linear_detects_wrong_grad() {
        let mut fwd = |p: &[f32]| 3.0 * p[0] + 5.0 * p[1];
        let bad = gradcheck_scalar(&mut fwd, &[0.7, -0.2], &[3.0, 4.0], 1e-3, 1e-4, 1e-6);
        assert!(bad.is_err());
    }

    #[test]
    fn lerp_endpoints_and_midpoint() {
        let xs = [0.0_f64, 1.0, 2.0];
        let ys = [10.0_f64, 20.0, 25.0];
        assert_eq!(lerp(&xs, &ys, -1.0), 10.0);   // clamp low
        assert_eq!(lerp(&xs, &ys, 3.0), 25.0);    // clamp high
        assert_eq!(lerp(&xs, &ys, 0.5), 15.0);    // interior
        assert_eq!(lerp(&xs, &ys, 1.5), 22.5);    // interior
        assert_eq!(lerp(&xs, &ys, 1.0), 20.0);    // exact knot
    }

    #[test]
    fn traces_match_on_shifted_grid() {
        // Same underlying signal y = 2t, sampled on different grids.
        let t_a: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let y_a: Vec<f64> = t_a.iter().map(|t| 2.0 * t).collect();
        let t_b: Vec<f64> = (0..=20).map(|i| i as f64 * 0.05 + 1e-9).collect();
        let y_b: Vec<f64> = t_b.iter().map(|t| 2.0 * t).collect();
        traces_close(&t_a, &y_a, &t_b, &y_b, 1e-9, 1e-9).unwrap();
    }

    #[test]
    fn traces_detect_mismatch() {
        let t_a = vec![0.0_f64, 1.0, 2.0];
        let y_a = vec![0.0_f64, 1.0, 2.0];
        let t_b = vec![0.0_f64, 1.0, 2.0];
        let y_b = vec![0.0_f64, 1.0, 3.0]; // last point off
        let res = traces_close(&t_a, &y_a, &t_b, &y_b, 1e-6, 1e-6);
        assert!(res.is_err());
    }

    #[test]
    fn dc_diff_report_finds_worst() {
        let mut a = HashMap::new();
        a.insert("vout".into(), 0.5);
        a.insert("mid".into(), 0.25);
        let mut b = HashMap::new();
        b.insert("vout".into(), 0.500_001);
        b.insert("mid".into(), 0.255);
        let r = compare_dc_voltages(&a, &b);
        assert_eq!(r.per_node.len(), 2);
        assert_eq!(r.worst.as_ref().unwrap().0, "mid");
        assert!((r.worst.as_ref().unwrap().1 - 0.005).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "DC mismatch")]
    fn dc_diff_panics_on_envelope_breach() {
        let mut a = HashMap::new();
        a.insert("v".into(), 1.0);
        let mut b = HashMap::new();
        b.insert("v".into(), 1.5);
        let r = compare_dc_voltages(&a, &b);
        r.assert_within(1e-6, 1e-3, "test");
    }

    #[test]
    fn dc_diff_passes_within_envelope() {
        let mut a = HashMap::new();
        a.insert("v".into(), 1.0);
        let mut b = HashMap::new();
        b.insert("v".into(), 1.000_001);
        let r = compare_dc_voltages(&a, &b);
        r.assert_within(1e-3, 1e-5, "test");
    }

    #[test]
    fn layout_vs_behavioral_passes_within_envelope() {
        // Extracted-deck Vout matches analytic Vout = V·R2/(R1+R2) for
        // R1=1k, R2=3k, V=1.0 → 0.75. The ngspice deck has finite-tolerance
        // floor; rtol=1e-6 mirrors what a real ngspice run delivers.
        let mut ex = HashMap::new();
        ex.insert("vout".into(), 0.749_999_5);
        let mut be = HashMap::new();
        be.insert("vout".into(), 0.750_000_0);
        LayoutVsBehavioralReport::dc(&ex, &be)
            .assert_within(1e-5, 1e-6, "divider");
    }

    #[test]
    #[should_panic(expected = "layout-vs-behavioral mismatch")]
    fn layout_vs_behavioral_panics_on_envelope_breach() {
        // Simulates the LNA-style failure: extracted deck reports 0V at
        // vout because the routing wire is missing — net got split, so
        // the divider degenerates.
        let mut ex = HashMap::new();
        ex.insert("vout".into(), 0.0);
        let mut be = HashMap::new();
        be.insert("vout".into(), 0.75);
        LayoutVsBehavioralReport::dc(&ex, &be)
            .assert_within(1e-3, 1e-3, "divider-with-missing-wire");
    }

    #[test]
    fn transient_diff_handles_different_grids() {
        // Same y = 2t signal sampled finely vs coarsely.
        let t_a: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let y_a: Vec<f64> = t_a.iter().map(|t| 2.0 * t).collect();
        let t_b: Vec<f64> = (0..=20).map(|i| i as f64 * 0.05).collect();
        let y_b: Vec<f64> = t_b.iter().map(|t| 2.0 * t).collect();
        let mut am = HashMap::new();
        am.insert("y".to_string(), y_a);
        let mut bm = HashMap::new();
        bm.insert("y".to_string(), y_b);
        let r = compare_transient_traces(&t_a, &am, &t_b, &bm);
        let (max_abs, _rms) = r.per_node["y"];
        assert!(max_abs < 1e-12, "got max_abs = {max_abs:.3e}");
    }
}
