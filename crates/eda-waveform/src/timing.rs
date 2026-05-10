//! Timing extraction for analog/mixed-signal waveforms.
//!
//! Verification engineers spend most of their time on three numbers:
//! when does a signal cross a level, how fast does it transition, and is
//! the data stable around the clock edge. This module gives those three
//! primitives a Waveform-shaped home.
//!
//! All helpers operate on raw `(t, y)` slices and use linear
//! interpolation between samples to land sub-sample-accurate edge times
//! — important when the simulator's sample grid is coarser than the
//! transition you're measuring (which it usually is, since LTE-controlled
//! adaptive solvers don't spend extra steps inside a fast edge).
//!
//! ## Conventions
//!
//! - **Crossings** are returned as the *interpolated* time at which the
//!   signal equals `level`, not the bracket sample.
//! - **Rise/fall time** uses absolute levels (e.g. 0.1V and 0.9V), not
//!   percentages. The caller computes those from peak/valley if it wants
//!   "10%–90% of swing" — the primitive doesn't guess at the steady
//!   states.
//! - **Setup/hold** is positive when the data was stable on the
//!   correct side of the edge. A negative setup means data changed
//!   *after* the previous-cycle timing window started — i.e. a violation
//!   from the perspective of the receiving flop.
//!
//! ## Limitations
//!
//! No glitch filtering / hysteresis. If the input has noise that crosses
//! the level multiple times near an edge, you'll see multiple crossings.
//! Filter beforehand (or pick a level that's clear of the noise band).

/// Which polarity of crossing to report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    /// `y` was below the level and is now at-or-above.
    Rise,
    /// `y` was at-or-above the level and is now below.
    Fall,
    /// Either polarity.
    Either,
}

/// All times at which `y` crosses `level`, with linear interpolation
/// between bracketing samples. `t` must be ascending.
///
/// A sample exactly equal to `level` counts as a crossing if the
/// neighboring samples are on opposite sides; consecutive equal-to-level
/// samples don't multi-count.
pub fn crossings(t: &[f64], y: &[f64], level: f64, edge: Edge) -> Vec<f64> {
    debug_assert_eq!(t.len(), y.len());
    if t.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 0..t.len() - 1 {
        let y0 = y[i];
        let y1 = y[i + 1];
        // Skip samples where either side is NaN.
        if !y0.is_finite() || !y1.is_finite() {
            continue;
        }
        let d0 = y0 - level;
        let d1 = y1 - level;
        // Need a sign change. Treat exact-equal at i+1 as a crossing
        // (so a step from below→exact lands here) but suppress the
        // mirror case at i to avoid double counting.
        let crossed = (d0 < 0.0 && d1 >= 0.0) || (d0 > 0.0 && d1 <= 0.0);
        if !crossed {
            continue;
        }
        let polarity = if d0 < 0.0 { Edge::Rise } else { Edge::Fall };
        if !matches!(edge, Edge::Either) && edge != polarity {
            continue;
        }
        // Linear interp: t0 + (level - y0) / (y1 - y0) * (t1 - t0)
        let frac = (level - y0) / (y1 - y0);
        out.push(t[i] + frac * (t[i + 1] - t[i]));
    }
    out
}

/// Result of a transition-time measurement.
#[derive(Debug, Clone, Copy)]
pub struct EdgeMetrics {
    /// Interpolated time at which `y` first crosses `low_level` (rising)
    /// or `high_level` (falling).
    pub start_t: f64,
    /// Interpolated time at which `y` crosses the other level.
    pub end_t: f64,
    /// `end_t - start_t`. Always positive; `None` if the transition
    /// wasn't found.
    pub duration: f64,
}

/// Time for the first low→high transition: from `low_level` to `high_level`.
///
/// Returns `None` if there's no rising crossing of `low_level` followed
/// by a rising crossing of `high_level`. `low_level` must be `<` `high_level`.
pub fn rise_time(t: &[f64], y: &[f64], low_level: f64, high_level: f64) -> Option<EdgeMetrics> {
    assert!(low_level < high_level, "low_level must be < high_level");
    let lo = crossings(t, y, low_level, Edge::Rise);
    let hi = crossings(t, y, high_level, Edge::Rise);
    let &lo_t = lo.first()?;
    // Find the first high crossing that happens *after* the low crossing.
    let &hi_t = hi.iter().find(|&&h| h >= lo_t)?;
    Some(EdgeMetrics {
        start_t: lo_t,
        end_t: hi_t,
        duration: hi_t - lo_t,
    })
}

/// Time for the first high→low transition: from `high_level` to `low_level`.
pub fn fall_time(t: &[f64], y: &[f64], high_level: f64, low_level: f64) -> Option<EdgeMetrics> {
    assert!(low_level < high_level, "low_level must be < high_level");
    let hi = crossings(t, y, high_level, Edge::Fall);
    let lo = crossings(t, y, low_level, Edge::Fall);
    let &hi_t = hi.first()?;
    let &lo_t = lo.iter().find(|&&l| l >= hi_t)?;
    Some(EdgeMetrics {
        start_t: hi_t,
        end_t: lo_t,
        duration: lo_t - hi_t,
    })
}

/// Setup/hold pair around one clock edge.
#[derive(Debug, Clone, Copy)]
pub struct SetupHold {
    /// Interpolated clock-edge time.
    pub clk_t: f64,
    /// Time from the most recent data transition *before* `clk_t`
    /// to `clk_t`. `+inf` if data never changed before the edge.
    pub setup: f64,
    /// Time from `clk_t` to the next data transition *after* it. `+inf`
    /// if data never changes after the edge.
    pub hold: f64,
}

/// For each clock edge of polarity `clk_edge`, find the surrounding
/// data-transition times (Either-polarity) and return setup/hold.
///
/// Both clock and data are thresholded against their own levels — pass
/// `clk_threshold = vdd/2` and `data_threshold = vdd/2` for typical
/// rail-to-rail digital. The clock can have a different sample grid
/// from the data; both axes must be ascending.
pub fn setup_hold(
    t_clk: &[f64],
    y_clk: &[f64],
    clk_threshold: f64,
    clk_edge: Edge,
    t_data: &[f64],
    y_data: &[f64],
    data_threshold: f64,
) -> Vec<SetupHold> {
    let clk_edges = crossings(t_clk, y_clk, clk_threshold, clk_edge);
    let data_edges = crossings(t_data, y_data, data_threshold, Edge::Either);
    let mut out = Vec::with_capacity(clk_edges.len());
    for ce in clk_edges {
        // Most recent data edge ≤ ce.
        let setup = match data_edges.iter().rev().find(|&&d| d <= ce) {
            Some(&d) => ce - d,
            None => f64::INFINITY,
        };
        // Next data edge > ce.
        let hold = match data_edges.iter().find(|&&d| d > ce) {
            Some(&d) => d - ce,
            None => f64::INFINITY,
        };
        out.push(SetupHold {
            clk_t: ce,
            setup,
            hold,
        });
    }
    out
}

/// One pulse: a `[start, end]` interval where the signal was above
/// `threshold`. Times are interpolated to sub-sample accuracy.
#[derive(Debug, Clone, Copy)]
pub struct Pulse {
    pub start: f64,
    pub end: f64,
    pub width: f64,
}

/// All complete above-threshold pulses in `(t, y)`.
///
/// A pulse is a Rise → next-Fall pair through `threshold`. Edge cases:
/// - Signal starts already above threshold → the first segment has no
///   Rise to anchor it, so it's omitted (we don't know when it began).
/// - Signal ends still above threshold → the final segment has no
///   Fall to close it, so it's omitted (we don't know when it ends).
///
/// In other words: only pulses that fully fit inside `[t[0], t[last]]`
/// are returned.
pub fn pulse_widths(t: &[f64], y: &[f64], threshold: f64) -> Vec<Pulse> {
    let rises = crossings(t, y, threshold, Edge::Rise);
    let falls = crossings(t, y, threshold, Edge::Fall);
    let mut out = Vec::new();
    let mut ri = 0;
    let mut fi = 0;
    while ri < rises.len() && fi < falls.len() {
        if falls[fi] <= rises[ri] {
            // Fall before any rise — signal started high; skip.
            fi += 1;
            continue;
        }
        let start = rises[ri];
        let end = falls[fi];
        out.push(Pulse {
            start,
            end,
            width: end - start,
        });
        ri += 1;
        fi += 1;
    }
    out
}

/// Filter pulses whose width is *less than* `min_width` — typically
/// used to flag glitches.
pub fn glitches_below(pulses: &[Pulse], min_width: f64) -> Vec<Pulse> {
    pulses.iter().filter(|p| p.width < min_width).copied().collect()
}

/// Period / jitter summary derived from a sorted list of crossing times.
#[derive(Debug, Clone, Copy)]
pub struct PeriodStats {
    /// Number of *periods* observed (= `crossings.len() - 1`).
    pub n_periods: usize,
    /// Mean period (seconds).
    pub mean: f64,
    /// RMS of `period[i] - mean` over all periods (seconds).
    pub rms_jitter: f64,
    /// `max(period) - min(period)` (seconds).
    pub p2p_jitter: f64,
    /// RMS of `period[i+1] - period[i]` over consecutive periods —
    /// the "cycle-to-cycle" jitter that close-in noise dominates.
    pub cycle_to_cycle: f64,
}

/// Period / jitter statistics from a sequence of (sorted) crossing
/// times — typically the output of [`crossings`] applied to a clock
/// signal at its midpoint with `Edge::Rise`.
///
/// Returns `None` if fewer than 2 crossings (no period to measure)
/// or 3 crossings (no cycle-to-cycle differences). For exactly 2
/// crossings, jitter terms are reported as `0.0`.
pub fn period_stats(crossings: &[f64]) -> Option<PeriodStats> {
    if crossings.len() < 2 {
        return None;
    }
    let n_periods = crossings.len() - 1;
    let periods: Vec<f64> = crossings.windows(2).map(|w| w[1] - w[0]).collect();
    let mean = periods.iter().sum::<f64>() / n_periods as f64;
    let rms_jitter = (periods
        .iter()
        .map(|p| {
            let d = p - mean;
            d * d
        })
        .sum::<f64>()
        / n_periods as f64)
        .sqrt();
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &p in &periods {
        if p < min {
            min = p;
        }
        if p > max {
            max = p;
        }
    }
    let p2p_jitter = max - min;
    let cycle_to_cycle = if periods.len() < 2 {
        0.0
    } else {
        let diffs: Vec<f64> = periods.windows(2).map(|w| w[1] - w[0]).collect();
        (diffs.iter().map(|d| d * d).sum::<f64>() / diffs.len() as f64).sqrt()
    };
    Some(PeriodStats {
        n_periods,
        mean,
        rms_jitter,
        p2p_jitter,
        cycle_to_cycle,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn crossings_finds_interpolated_rise() {
        // Linear ramp from 0→1 over t ∈ [0, 1]. Crossing 0.5 at t=0.5.
        let t = [0.0, 1.0];
        let y = [0.0, 1.0];
        let xs = crossings(&t, &y, 0.5, Edge::Rise);
        assert_eq!(xs.len(), 1);
        assert!(approx(xs[0], 0.5));
    }

    #[test]
    fn crossings_filters_by_polarity() {
        // Triangle wave: 0 → 1 → 0. Two crossings of 0.5: one rise, one fall.
        let t = [0.0, 1.0, 2.0];
        let y = [0.0, 1.0, 0.0];
        let rises = crossings(&t, &y, 0.5, Edge::Rise);
        let falls = crossings(&t, &y, 0.5, Edge::Fall);
        let either = crossings(&t, &y, 0.5, Edge::Either);
        assert_eq!(rises.len(), 1);
        assert_eq!(falls.len(), 1);
        assert_eq!(either.len(), 2);
        assert!(approx(rises[0], 0.5));
        assert!(approx(falls[0], 1.5));
    }

    #[test]
    fn crossings_handles_nan_samples() {
        // NaN sample shouldn't produce a phantom crossing.
        let t = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, f64::NAN, 1.0, 0.0];
        let xs = crossings(&t, &y, 0.5, Edge::Either);
        // Only the fall through 0.5 between t=2 and t=3 (interp at 2.5).
        assert_eq!(xs.len(), 1);
        assert!(approx(xs[0], 2.5));
    }

    #[test]
    fn rise_time_picks_first_lo_then_hi() {
        // 0 at t=0, ramp to 1 between t=10 and t=11.
        let t = [0.0, 10.0, 10.5, 11.0, 12.0];
        let y = [0.0, 0.0, 0.5, 1.0, 1.0];
        // 10%–90% of unit swing: 0.1 → 0.9.
        let m = rise_time(&t, &y, 0.1, 0.9).unwrap();
        // start crosses 0.1 between 10 and 10.5 → t=10.1.
        assert!(approx(m.start_t, 10.1));
        // end crosses 0.9 between 10.5 and 11 → t=10.9.
        assert!(approx(m.end_t, 10.9));
        assert!(approx(m.duration, 0.8));
    }

    #[test]
    fn fall_time_mirrors_rise() {
        let t = [0.0, 10.0, 10.5, 11.0, 12.0];
        let y = [1.0, 1.0, 0.5, 0.0, 0.0];
        let m = fall_time(&t, &y, 0.9, 0.1).unwrap();
        assert!(approx(m.start_t, 10.1));
        assert!(approx(m.end_t, 10.9));
        assert!(approx(m.duration, 0.8));
    }

    #[test]
    fn rise_time_returns_none_when_no_transition() {
        let t = [0.0, 1.0, 2.0];
        let y = [0.0, 0.0, 0.0];
        assert!(rise_time(&t, &y, 0.1, 0.9).is_none());
    }

    #[test]
    fn setup_hold_around_clock_rise() {
        // Clock rises at t=10. Data changes at t=8 (setup=2) and t=12 (hold=2).
        let t_clk = [0.0, 10.0, 10.0001, 20.0];
        let y_clk = [0.0, 0.0, 1.0, 1.0];
        let t_data = [0.0, 8.0, 8.0001, 12.0, 12.0001, 20.0];
        let y_data = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let sh = setup_hold(&t_clk, &y_clk, 0.5, Edge::Rise, &t_data, &y_data, 0.5);
        assert_eq!(sh.len(), 1);
        // Clock crossing of 0.5 lands ≈ midway between 10 and 10.0001 → ~10.00005.
        assert!((sh[0].clk_t - 10.00005).abs() < 1e-3);
        // Data rise lands ≈ 8.00005 → setup ≈ 2.0.
        assert!((sh[0].setup - 2.0).abs() < 1e-3);
        // Data fall lands ≈ 12.00005 → hold ≈ 2.0.
        assert!((sh[0].hold - 2.0).abs() < 1e-3);
    }

    #[test]
    fn setup_hold_infinite_when_no_data_changes() {
        // Clock has 1 rising edge; data never changes.
        let t_clk = [0.0, 10.0, 10.0001, 20.0];
        let y_clk = [0.0, 0.0, 1.0, 1.0];
        let t_data = [0.0, 20.0];
        let y_data = [1.0, 1.0];
        let sh = setup_hold(&t_clk, &y_clk, 0.5, Edge::Rise, &t_data, &y_data, 0.5);
        assert_eq!(sh.len(), 1);
        assert!(sh[0].setup.is_infinite());
        assert!(sh[0].hold.is_infinite());
    }

    #[test]
    fn crossings_empty_or_single_sample() {
        assert!(crossings(&[], &[], 0.0, Edge::Either).is_empty());
        assert!(crossings(&[1.0], &[0.0], 0.0, Edge::Either).is_empty());
    }

    #[test]
    fn period_stats_perfect_clock_has_zero_jitter() {
        // 10 evenly-spaced edges at 1 GHz (1 ns period).
        let xs: Vec<f64> = (0..10).map(|i| i as f64 * 1e-9).collect();
        let s = period_stats(&xs).unwrap();
        assert_eq!(s.n_periods, 9);
        assert!(approx(s.mean, 1e-9));
        assert!(s.rms_jitter < 1e-18);
        assert!(s.p2p_jitter < 1e-18);
        assert!(s.cycle_to_cycle < 1e-18);
    }

    #[test]
    fn period_stats_alternating_long_short_jitter() {
        // Edges at t = 0, 1, 1.5, 2.5, 3, 4 ns → periods = [1, 0.5, 1, 0.5, 1].
        // Mean = 0.8; deviations = [0.2, -0.3, 0.2, -0.3, 0.2].
        // sum(d²) = 0.04 + 0.09 + 0.04 + 0.09 + 0.04 = 0.30.
        // RMS jitter = sqrt(0.30 / 5) = sqrt(0.06) ≈ 0.2449.
        // p2p = 1 - 0.5 = 0.5.
        // cycle-to-cycle: diffs = [-0.5, 0.5, -0.5, 0.5] → RMS = 0.5.
        let xs = vec![0.0, 1.0, 1.5, 2.5, 3.0, 4.0];
        let s = period_stats(&xs).unwrap();
        assert!(approx(s.mean, 0.8));
        assert!(approx(s.rms_jitter, (0.06_f64).sqrt()));
        assert!(approx(s.p2p_jitter, 0.5));
        assert!(approx(s.cycle_to_cycle, 0.5));
    }

    #[test]
    fn period_stats_too_few_edges() {
        assert!(period_stats(&[]).is_none());
        assert!(period_stats(&[1.0]).is_none());
    }

    #[test]
    fn pulse_widths_recovers_square_wave_widths() {
        // Three above-threshold pulses, each 0.5 s wide.
        let t: Vec<f64> = (0..=12).map(|i| i as f64 * 0.25).collect(); // 0..3 in 0.25 steps
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| {
                // High during [0.5, 1.0], [1.5, 2.0], [2.5, 3.0).
                let in_pulse = (ti >= 0.5 && ti < 1.0)
                    || (ti >= 1.5 && ti < 2.0)
                    || (ti >= 2.5 && ti < 3.0);
                if in_pulse { 1.0 } else { 0.0 }
            })
            .collect();
        let pulses = pulse_widths(&t, &y, 0.5);
        // First pulse: rise interpolates within (0.25, 0.5); fall within (1.0, 1.25). Etc.
        // We expect 2 complete pulses (the third extends past the trace end? Actually
        // it falls at t=1.0 so it IS complete; let's count). With y high in
        // [0.5..1.0), [1.5..2.0), [2.5..3.0):
        //   rise at ~0.375, fall at ~1.125 → pulse1
        //   rise at ~1.375, fall at ~2.125 → pulse2
        //   rise at ~2.375, fall at ~3.0 → pulse3 (falls right at last sample)
        assert!(
            pulses.len() >= 2,
            "expected at least 2 complete pulses, got {}",
            pulses.len()
        );
        // Each width should be ~0.75 (sample-grid artifact: rise/fall
        // straddle 0.5, so the interpolated boundary is ~0.5 wide of
        // sample-pair midpoints, giving ~0.75 not 0.5). Verify they're
        // consistent.
        let w0 = pulses[0].width;
        for p in &pulses[1..] {
            assert!(
                (p.width - w0).abs() < 0.05,
                "widths inconsistent: {} vs {}",
                p.width,
                w0
            );
        }
    }

    #[test]
    fn pulse_widths_omits_pulse_that_starts_high() {
        // Signal starts above threshold and falls at t=1, then rises again at t=2.5,
        // falls at t=3.5. Only the second pulse should be reported.
        let t = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let y = vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let pulses = pulse_widths(&t, &y, 0.5);
        assert_eq!(pulses.len(), 1);
        // Rise interp between t=2.0 (y=0) and t=2.5 (y=1) at level 0.5 → t=2.25.
        // Fall interp between t=3.0 (y=1) and t=3.5 (y=0) at level 0.5 → t=3.25.
        assert!(approx(pulses[0].width, 1.0));
    }

    #[test]
    fn glitches_below_filters_correctly() {
        let pulses = vec![
            Pulse { start: 0.0, end: 1.0, width: 1.0 },
            Pulse { start: 2.0, end: 2.05, width: 0.05 },
            Pulse { start: 3.0, end: 4.0, width: 1.0 },
        ];
        let glitches = glitches_below(&pulses, 0.1);
        assert_eq!(glitches.len(), 1);
        assert!(approx(glitches[0].width, 0.05));
    }

    #[test]
    fn pulse_widths_handles_no_pulses() {
        let t = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 0.0, 0.0];
        assert!(pulse_widths(&t, &y, 0.5).is_empty());
    }

    #[test]
    fn period_stats_two_edges_zero_jitter() {
        let s = period_stats(&[0.0, 1.0]).unwrap();
        assert_eq!(s.n_periods, 1);
        assert!(approx(s.mean, 1.0));
        assert!(approx(s.rms_jitter, 0.0));
        assert!(approx(s.cycle_to_cycle, 0.0));
    }
}
