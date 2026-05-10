//! Analytic witnesses for vector fitting — first rung of the
//! validation pyramid. Each test constructs an exact rational
//! response, samples it on a frequency grid, runs the fitter, and
//! checks that the recovered (poles, residues) match within
//! numerical tolerance.
//!
//! No FD or ngspice rung yet — vector fitting is a pure-numerical
//! algorithm with no derivative-of-loss surface to FD-check, and
//! ngspice doesn't speak rational-fit ASTs. The cross-check we'll
//! add later is a scikit-rf parity witness once we wire the Python
//! plumbing.

use eda_vfit::*;

fn log_freqs(omega_min: f64, omega_max: f64, n: usize) -> Vec<f64> {
    let lmin = omega_min.ln();
    let lmax = omega_max.ln();
    (0..n).map(|i| {
        let t = i as f64 / (n as f64 - 1.0);
        (lmin + (lmax - lmin) * t).exp()
    }).collect()
}

fn sample_response(
    freqs: &[f64], poles: &[C64], residues: &[C64], d: f64, e: f64,
) -> Vec<C64> {
    freqs.iter().map(|&w| eval_model(w, poles, residues, d, e)).collect()
}

#[test]
fn fit_residues_recovers_known_residues_at_known_poles() {
    // H(s) = 0.5/(s+10) + 1.5/(s+1000) + 0.2
    let true_poles    = vec![C64::new(-10.0, 0.0), C64::new(-1000.0, 0.0)];
    let true_residues = vec![C64::new(0.5, 0.0),   C64::new(1.5, 0.0)];
    let true_d        = 0.2;

    let freqs = log_freqs(0.1, 1e5, 200);
    let h     = sample_response(&freqs, &true_poles, &true_residues,
                                true_d, 0.0);

    let (residues, d, e, rms) = fit_residues(
        &freqs, &h, &true_poles,
        /* d */ true, /* e */ false,
    ).unwrap();

    assert!(rms < 1e-9, "rms {rms} too large");
    assert!((residues[0].re - 0.5).abs() < 1e-6, "r0 {:?}", residues[0]);
    assert!((residues[1].re - 1.5).abs() < 1e-6, "r1 {:?}", residues[1]);
    assert!(residues[0].im.abs() < 1e-6);
    assert!(residues[1].im.abs() < 1e-6);
    assert!((d - 0.2).abs() < 1e-6);
    assert!(e.abs() < 1e-9);
}

#[test]
fn vector_fit_relocates_single_real_pole() {
    // H(s) = 1/(s + 10).  Start with a wrong pole, expect convergence.
    let true_poles    = vec![C64::new(-10.0, 0.0)];
    let true_residues = vec![C64::new(1.0, 0.0)];
    let freqs = log_freqs(0.1, 1e3, 200);
    let h     = sample_response(&freqs, &true_poles, &true_residues, 0.0, 0.0);

    let initial = vec![C64::new(-1.0, 0.0)];     // an order of magnitude off
    let opt = VfitOptions {
        n_iters: 8,
        asymptotic_d: false,
        asymptotic_e: false,
        ..Default::default()
    };
    let res = vector_fit(&freqs, &h, &initial, opt).unwrap();

    assert!(res.rms_error < 1e-8, "rms {}", res.rms_error);
    let p = res.poles[0];
    let r = res.residues[0];
    assert!((p.re + 10.0).abs() / 10.0 < 1e-3,
            "pole {p:?} should be near (-10, 0)");
    assert!(p.im.abs() < 1e-6);
    assert!((r.re - 1.0).abs() < 1e-3, "residue {r:?} should be near (1, 0)");
}

#[test]
fn vector_fit_relocates_two_real_poles() {
    // H(s) = 0.5/(s+5) + 2.0/(s+500).   Two-decade pole separation.
    let true_poles    = vec![C64::new(-5.0,   0.0),
                             C64::new(-500.0, 0.0)];
    let true_residues = vec![C64::new(0.5, 0.0),
                             C64::new(2.0, 0.0)];
    let freqs = log_freqs(0.05, 5e3, 300);
    let h     = sample_response(&freqs, &true_poles, &true_residues, 0.0, 0.0);

    // Start poles log-spaced across the band — Gustavsen's standard init.
    let initial = log_spaced_real_poles(0.05, 5e3, 2);
    let opt = VfitOptions {
        n_iters: 10,
        asymptotic_d: false,
        asymptotic_e: false,
        ..Default::default()
    };
    let res = vector_fit(&freqs, &h, &initial, opt).unwrap();

    assert!(res.rms_error < 1e-7, "rms {}", res.rms_error);

    // Recovered poles unordered → match each true pole to its closest fit.
    let mut found: Vec<(f64, f64)> = res.poles.iter()
        .zip(res.residues.iter())
        .map(|(p, r)| (p.re, r.re))
        .collect();
    for (true_p, true_r) in true_poles.iter().zip(&true_residues) {
        let (idx, _) = found.iter().enumerate()
            .min_by(|(_, a), (_, b)| {
                ((a.0 - true_p.re).abs())
                    .partial_cmp(&((b.0 - true_p.re).abs())).unwrap()
            }).unwrap();
        let (p_re, r_re) = found.remove(idx);
        assert!((p_re - true_p.re).abs() / true_p.re.abs() < 5e-3,
                "pole {p_re} should be near {}", true_p.re);
        assert!((r_re - true_r.re).abs() / true_r.re.abs() < 5e-3,
                "residue {r_re} should be near {}", true_r.re);
    }
}

#[test]
fn rejects_dimension_mismatch() {
    let freqs = vec![1.0, 2.0];
    let h = vec![C64::ONE];
    let poles = vec![C64::new(-1.0, 0.0)];
    let err = fit_residues(&freqs, &h, &poles, true, false).unwrap_err();
    matches!(err, VfitError::DimensionMismatch { .. });
}
