//! `eda-vfit` — Vector fitting for frequency-domain rational models.
//!
//! Given frequency samples `(ω_n, H_n)` — typically S-parameters of a
//! photonic / RF block — fits a sum-of-poles rational model
//!
//! ```text
//!   H(s) ≈ Σ_k r_k / (s - p_k) + d + s·e
//! ```
//!
//! that can be realized as a small linear ODE system and stamped into
//! the MNA solver via `spike-waveguide-block` or similar. Designed to
//! pair with delay differential equations for the bulk group-delay
//! component (circulax issues #2 / #3).
//!
//! ## What's in scope (MVP)
//!
//! - [`fit_residues`] — given a fixed pole set, solve the complex
//!   linear least-squares for `(residues, d, e)`. Useful when the
//!   user already knows where the poles are (from a scikit-rf fit,
//!   or from a physical model) and just wants the residue projection.
//! - [`vector_fit`] — Gustavsen-style iterative pole relocation,
//!   restricted to **real poles**. Each iteration solves an LSQ for
//!   `(residues, σ-residues, d, e)`, then relocates poles to the real
//!   zeros of `σ(s)` recovered as eigenvalues of the (small) companion
//!   matrix `A_σ = diag(p_k) − 1·σ̃ᵀ` via Durand–Kerner on the explicit
//!   characteristic polynomial.
//!
//! ## What's *not* in scope (yet)
//!
//! - Complex-conjugate pole pairs (broadband resonance fits). The
//!   secular-equation root finder used here returns whichever roots
//!   Durand–Kerner lands on; complex roots are projected onto their
//!   real part with a stability flip, which is fine for over-damped
//!   responses but loses information for under-damped ones. Tracked
//!   as a follow-up — extend to the standard real-form companion
//!   matrix `[diag(p_k) ± off-diag]` for conjugate pairs.
//! - Passivity / causality enforcement (PRVF, MFVF variants). MVP
//!   returns whatever fit minimizes RMS frequency error; downstream
//!   consumers should check passivity before stamping into MNA.
//! - State-space realization (`A`, `B`, `C`, `D` matrices). Once you
//!   have poles + residues, the realization is mechanical; lives in
//!   `spike-waveguide-block` or a future `eda-vfit-realize`.
//!
//! ## Algorithm reference
//!
//! Gustavsen & Semlyen, "Rational approximation of frequency-domain
//! responses by vector fitting", IEEE Trans. Power Delivery **14**(3),
//! 1052–1061 (1999).

use std::ops::{Add, Div, Mul, Neg, Sub};

// ── Minimal complex-f64 type ──────────────────────────────────────────

/// Inline complex-f64. Kept in-crate to avoid pulling `num-complex`
/// into the workspace just for this module — the surface is small
/// enough that re-implementing it is cheaper than the dep churn.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct C64 { pub re: f64, pub im: f64 }

impl C64 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE:  Self = Self { re: 1.0, im: 0.0 };
    pub const I:    Self = Self { re: 0.0, im: 1.0 };
    pub const fn new(re: f64, im: f64) -> Self { Self { re, im } }
    pub fn norm_sqr(self) -> f64 { self.re * self.re + self.im * self.im }
    pub fn norm(self) -> f64 { self.norm_sqr().sqrt() }
    pub fn conj(self) -> Self { Self::new(self.re, -self.im) }
    pub fn recip(self) -> Self {
        let n = self.norm_sqr();
        Self::new(self.re / n, -self.im / n)
    }
}
impl Add for C64 { type Output = Self;
    fn add(self, o: Self) -> Self { Self::new(self.re + o.re, self.im + o.im) } }
impl Sub for C64 { type Output = Self;
    fn sub(self, o: Self) -> Self { Self::new(self.re - o.re, self.im - o.im) } }
impl Mul for C64 { type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(self.re * o.re - self.im * o.im,
                  self.re * o.im + self.im * o.re)
    } }
impl Mul<f64> for C64 { type Output = Self;
    fn mul(self, s: f64) -> Self { Self::new(self.re * s, self.im * s) } }
impl Div for C64 { type Output = Self;
    fn div(self, o: Self) -> Self { self * o.recip() } }
impl Neg for C64 { type Output = Self;
    fn neg(self) -> Self { Self::new(-self.re, -self.im) } }

// ── Fit options / result ──────────────────────────────────────────────

#[derive(Copy, Clone, Debug)]
pub struct VfitOptions {
    /// Outer iterations for pole relocation. Gustavsen reports
    /// 3–5 typical for smooth responses; pathological cases can need
    /// more. Ignored by [`fit_residues`].
    pub n_iters: usize,
    /// Include constant term `d` in the model.
    pub asymptotic_d: bool,
    /// Include `s·e` term (high-frequency direct coupling).
    pub asymptotic_e: bool,
    /// After pole relocation, force `Re(p_k) < 0` by reflection
    /// (`p_k ← -|Re(p_k)| + j·Im(p_k)`). Mandatory for stable
    /// time-domain realization.
    pub enforce_stability: bool,
    /// Durand–Kerner iteration cap inside the σ-zero finder.
    pub root_finder_iters: usize,
    /// Tiny floor below which `|Re(p_k)|` after stability flip is
    /// raised to this value, so a near-jω pole doesn't yield a
    /// numerically stiff state-space realization.
    pub min_decay_rate: f64,
}
impl Default for VfitOptions {
    fn default() -> Self {
        Self {
            n_iters:           5,
            asymptotic_d:      true,
            asymptotic_e:      false,
            enforce_stability: true,
            root_finder_iters: 200,
            min_decay_rate:    1e-6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VfitResult {
    pub poles:    Vec<C64>,
    pub residues: Vec<C64>,
    pub d:        f64,
    pub e:        f64,
    /// Root-mean-square fit error across the input frequency grid:
    /// `sqrt( mean_n |H_n − H_fit(jω_n)|² )`.
    pub rms_error: f64,
}

#[derive(Clone, Debug)]
pub enum VfitError {
    DimensionMismatch { freqs: usize, response: usize },
    NoPoles,
    SingularLstsq,
    /// Durand–Kerner failed to converge within `root_finder_iters`.
    /// Returns the maximum residual update at the last iteration —
    /// useful for tuning the cap.
    RootFinderDiverged { last_max_dx: f64 },
}
impl std::fmt::Display for VfitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { freqs, response } =>
                write!(f, "freqs ({freqs}) != response ({response})"),
            Self::NoPoles => write!(f, "initial pole list is empty"),
            Self::SingularLstsq =>
                write!(f, "least-squares normal equations singular"),
            Self::RootFinderDiverged { last_max_dx } =>
                write!(f, "Durand–Kerner did not converge \
                          (last max |Δroot| = {last_max_dx:e})"),
        }
    }
}
impl std::error::Error for VfitError {}

// ── Public API ────────────────────────────────────────────────────────

/// Evaluate the rational model
/// `H(jω) = Σ r_k / (jω − p_k) + d + jω·e`
/// at a single angular frequency. Useful for plotting fit-vs-data
/// and for the [`VfitResult::rms_error`] computation.
pub fn eval_model(
    omega: f64,
    poles: &[C64],
    residues: &[C64],
    d: f64,
    e: f64,
) -> C64 {
    let s = C64::new(0.0, omega);
    let mut acc = C64::new(d, 0.0) + s * e;
    for (p, r) in poles.iter().zip(residues) {
        acc = acc + (*r) / (s - *p);
    }
    acc
}

/// Direct residue identification at a fixed pole set.
///
/// Solves the complex linear least-squares
///
/// ```text
///   minimize_{r,d,e} Σ_n |Σ_k r_k/(jω_n - p_k) + d + jω_n·e − H_n|²
/// ```
///
/// Useful as a building block (called once per outer iteration of
/// [`vector_fit`]) and as a standalone API when the user already
/// knows where the poles are.
///
/// `freqs_rad` is the angular-frequency grid (rad/s); `response` is
/// the measured `H(jω_n)`. Returned `residues` align 1:1 with
/// `poles`.
pub fn fit_residues(
    freqs_rad: &[f64],
    response: &[C64],
    poles: &[C64],
    asymptotic_d: bool,
    asymptotic_e: bool,
) -> Result<(Vec<C64>, f64, f64, f64), VfitError> {
    if freqs_rad.len() != response.len() {
        return Err(VfitError::DimensionMismatch {
            freqs: freqs_rad.len(), response: response.len(),
        });
    }
    if poles.is_empty() && !asymptotic_d && !asymptotic_e {
        return Err(VfitError::NoPoles);
    }
    let k = poles.len();
    let n_extra = (asymptotic_d as usize) + (asymptotic_e as usize);
    let n_unknowns_c = k + n_extra;        // complex unknowns
    let n_rows_c = freqs_rad.len();        // complex equations

    // Build A (n_rows_c × n_unknowns_c, complex) and b (n_rows_c,
    // complex). Each row n has columns [1/(jω_n − p_k) for k] then
    // (optionally) [1] then (optionally) [jω_n], and rhs H_n.
    let mut a_cmplx = vec![C64::ZERO; n_rows_c * n_unknowns_c];
    let mut b_cmplx = vec![C64::ZERO; n_rows_c];
    for (n, (&omega, &h)) in freqs_rad.iter().zip(response).enumerate() {
        let s = C64::new(0.0, omega);
        for (i, &p) in poles.iter().enumerate() {
            a_cmplx[n * n_unknowns_c + i] = (s - p).recip();
        }
        let mut col = k;
        if asymptotic_d {
            a_cmplx[n * n_unknowns_c + col] = C64::ONE;
            col += 1;
        }
        if asymptotic_e {
            a_cmplx[n * n_unknowns_c + col] = s;
            // col += 1;
        }
        b_cmplx[n] = h;
    }

    let theta = solve_complex_lstsq(&a_cmplx, &b_cmplx,
                                    n_rows_c, n_unknowns_c)?;
    let residues: Vec<C64> = theta[..k].to_vec();
    let mut col = k;
    let d = if asymptotic_d {
        let v = theta[col].re; col += 1; v
    } else { 0.0 };
    let e = if asymptotic_e { theta[col].re } else { 0.0 };

    let rms = rms_error(freqs_rad, response, poles, &residues, d, e);
    Ok((residues, d, e, rms))
}

/// Iterative vector fitting with pole relocation.
///
/// Implements one Gustavsen iteration per pass:
///
/// 1. Solve the *augmented* LSQ for `(r_k, d, e, σ_k)` simultaneously,
///    with `σ(s) = 1 + Σ σ_k/(s − p_k)` and `N(s) = Σ r_k/(s − p_k) +
///    d + s·e`, enforcing `σ(s) · H(s) = N(s)` at every sample.
/// 2. Relocate poles to the **zeros of σ(s)** — found via
///    Durand–Kerner on the explicit characteristic polynomial.
/// 3. Optionally enforce stability (flip `Re(p) > 0` to its negative).
///
/// After `n_iters` poles, runs one final [`fit_residues`] on the
/// converged pole set so the returned residues are consistent with
/// `σ(s) = 1` (Gustavsen's "post-processing" step).
///
/// **Real-pole MVP**: `initial_poles` should all have `im = 0`.
/// Complex starting poles are accepted but the relocation step
/// projects roots onto the real axis (their `re` part), which biases
/// fits with under-damped resonances. Track via `rms_error` and
/// extend to conjugate pairs when needed.
pub fn vector_fit(
    freqs_rad: &[f64],
    response: &[C64],
    initial_poles: &[C64],
    opt: VfitOptions,
) -> Result<VfitResult, VfitError> {
    if freqs_rad.len() != response.len() {
        return Err(VfitError::DimensionMismatch {
            freqs: freqs_rad.len(), response: response.len(),
        });
    }
    if initial_poles.is_empty() {
        return Err(VfitError::NoPoles);
    }
    let k = initial_poles.len();
    let mut poles: Vec<C64> = initial_poles.to_vec();

    for _ in 0..opt.n_iters {
        // ── Augmented LSQ ──
        // Unknowns: [r_1..r_K, d?, e?, σ_1..σ_K]. Each row equation:
        //   Σ r_k · x_k(jω) + d + jω·e − Σ σ_k · x_k(jω) · H_n = H_n
        // where x_k(jω) = 1/(jω − p_k).
        let n_extra = (opt.asymptotic_d as usize)
                    + (opt.asymptotic_e as usize);
        let cols = 2 * k + n_extra;
        let n_rows_c = freqs_rad.len();
        let mut a = vec![C64::ZERO; n_rows_c * cols];
        let mut b = vec![C64::ZERO; n_rows_c];
        for (n, (&omega, &h)) in freqs_rad.iter().zip(response).enumerate() {
            let s = C64::new(0.0, omega);
            // Cache x_k = 1/(s - p_k).
            let xs: Vec<C64> = poles.iter()
                .map(|&p| (s - p).recip()).collect();
            for (i, &x) in xs.iter().enumerate() {
                a[n * cols + i] = x;
            }
            let mut col = k;
            if opt.asymptotic_d {
                a[n * cols + col] = C64::ONE;
                col += 1;
            }
            if opt.asymptotic_e {
                a[n * cols + col] = s;
                col += 1;
            }
            for (i, &x) in xs.iter().enumerate() {
                a[n * cols + col + i] = -(x * h);
            }
            b[n] = h;
        }

        let theta = solve_complex_lstsq(&a, &b, n_rows_c, cols)?;
        // σ-residues are the last K entries.
        let sigma_offset = cols - k;
        let sigma_residues: Vec<f64> = theta[sigma_offset..]
            .iter()
            .map(|c| c.re)        // real-pole MVP → take real part
            .collect();

        // ── Pole relocation: zeros of σ(s) = 1 + Σ σ_k/(s - p_k) ──
        let pole_re: Vec<f64> = poles.iter().map(|p| p.re).collect();
        let coeffs = sigma_zero_polynomial(&pole_re, &sigma_residues);
        let roots = durand_kerner(&coeffs, opt.root_finder_iters)?;

        // Project to real (real-pole MVP), enforce stability.
        poles = roots.into_iter().map(|root| {
            let mut re = root.re;
            if opt.enforce_stability && re > 0.0 {
                re = -re;
            }
            if opt.enforce_stability
               && re.abs() < opt.min_decay_rate
            {
                re = -opt.min_decay_rate;
            }
            C64::new(re, 0.0)
        }).collect();
    }

    // Final residue solve on the converged pole set (σ ≡ 1).
    let (residues, d, e, rms) = fit_residues(
        freqs_rad, response, &poles,
        opt.asymptotic_d, opt.asymptotic_e,
    )?;
    Ok(VfitResult { poles, residues, d, e, rms_error: rms })
}

// ── Internals ─────────────────────────────────────────────────────────

/// Build the polynomial whose roots are the zeros of
/// `σ(s) = 1 + Σ_k σ_k / (s − p_k)`. Multiplying through by
/// `∏(s − p_k)` gives
///
/// ```text
///   σ_poly(s) = ∏_k(s − p_k)  +  Σ_k σ_k · ∏_{j≠k}(s − p_j)
/// ```
///
/// which is monic of degree `k`. Returned in **descending** order
/// of `s`: `coeffs[0]` is the leading coefficient (≡ 1), `coeffs[k]`
/// is the constant term.
fn sigma_zero_polynomial(poles: &[f64], sigma_res: &[f64]) -> Vec<f64> {
    debug_assert_eq!(poles.len(), sigma_res.len());
    let k = poles.len();
    if k == 0 { return vec![1.0]; }

    // Product term: ∏(s − p_k) — degree k, length k+1.
    let mut prod = vec![0.0_f64; k + 1];
    prod[0] = 1.0;
    for (idx, &p) in poles.iter().enumerate() {
        // multiply current poly (degree idx) by (s − p)
        let cur_len = idx + 1;
        let mut new = vec![0.0_f64; cur_len + 1];
        for (i, &c) in prod[..cur_len].iter().enumerate() {
            new[i]     += c;        // s · old
            new[i + 1] -= p * c;    // −p · old
        }
        prod[..cur_len + 1].copy_from_slice(&new);
    }

    // Σ_k σ_k · ∏_{j≠k}(s − p_j). Each term has degree k−1 (length k).
    let mut sum_lo = vec![0.0_f64; k];
    for j in 0..k {
        let mut prod_j = vec![0.0_f64; k];
        prod_j[0] = 1.0;
        let mut idx = 0;
        for (i, &p_i) in poles.iter().enumerate() {
            if i == j { continue; }
            let cur_len = idx + 1;
            let mut new = vec![0.0_f64; cur_len + 1];
            for (m, &c) in prod_j[..cur_len].iter().enumerate() {
                new[m]     += c;
                new[m + 1] -= p_i * c;
            }
            prod_j[..cur_len + 1].copy_from_slice(&new);
            idx += 1;
        }
        for (i, &c) in prod_j[..k].iter().enumerate() {
            sum_lo[i] += sigma_res[j] * c;
        }
    }

    // Sum: prod (length k+1) leading at s^k; sum_lo (length k) leading
    // at s^{k−1} → align to prod[1..].
    let mut out = prod;
    for (i, &c) in sum_lo.iter().enumerate() {
        out[i + 1] += c;
    }
    out
}

/// Durand–Kerner root finder for a real-coefficient polynomial.
/// Coefficients in **descending** order; `coeffs[0]` is leading.
/// Returns one complex root per polynomial degree.
fn durand_kerner(coeffs: &[f64], max_iter: usize)
    -> Result<Vec<C64>, VfitError>
{
    let n = coeffs.len() - 1;
    if n == 0 { return Ok(Vec::new()); }
    let lead = coeffs[0];
    let a: Vec<C64> = coeffs.iter().map(|c| C64::new(c / lead, 0.0)).collect();

    // Initial guess: scaled rotated complex points (Aberth-style spread).
    let mut roots: Vec<C64> = (0..n).map(|i| {
        let theta = 2.0 * std::f64::consts::PI * (i as f64 + 0.4)
                    / n as f64;
        C64::new(0.4 + 0.9 * theta.cos(), 0.9 * theta.sin())
    }).collect();

    let mut last_max_dx = f64::INFINITY;
    for _ in 0..max_iter {
        let mut max_dx = 0.0_f64;
        for i in 0..n {
            // Horner eval of monic polynomial at roots[i].
            let mut p_val = C64::ZERO;
            for &c in &a {
                p_val = p_val * roots[i] + c;
            }
            // Denominator: ∏_{j≠i}(roots[i] − roots[j]).
            let mut denom = C64::ONE;
            for j in 0..n {
                if j == i { continue; }
                denom = denom * (roots[i] - roots[j]);
            }
            if denom.norm() < 1e-300 { continue; }
            let dx = p_val / denom;
            roots[i] = roots[i] - dx;
            max_dx = max_dx.max(dx.norm());
        }
        last_max_dx = max_dx;
        if max_dx < 1e-12 { return Ok(roots); }
    }
    Err(VfitError::RootFinderDiverged { last_max_dx })
}

/// Solve `A · x = b` in the least-squares sense for complex `A` (size
/// `n_rows × n_cols`, row-major) and complex `b`. Forms the normal
/// equations `(Aᴴ A) x = Aᴴ b` and runs Gauss-Jordan with partial
/// pivoting on the n_cols × n_cols Hermitian-PSD system.
///
/// Adequate for the small fit sizes (K poles ≤ ~20) we hit. For
/// ill-conditioned bases (closely-spaced poles relative to the
/// frequency span), upgrade to a complex QR via Householder.
fn solve_complex_lstsq(
    a: &[C64], b: &[C64],
    n_rows: usize, n_cols: usize,
) -> Result<Vec<C64>, VfitError> {
    debug_assert_eq!(a.len(), n_rows * n_cols);
    debug_assert_eq!(b.len(), n_rows);

    // Normal-equations matrix: H = Aᴴ · A   (n_cols × n_cols)
    let mut h = vec![C64::ZERO; n_cols * n_cols];
    for i in 0..n_cols {
        for j in 0..n_cols {
            let mut acc = C64::ZERO;
            for r in 0..n_rows {
                acc = acc + a[r * n_cols + i].conj() * a[r * n_cols + j];
            }
            h[i * n_cols + j] = acc;
        }
    }
    // Rhs:  g = Aᴴ · b  (length n_cols)
    let mut g = vec![C64::ZERO; n_cols];
    for i in 0..n_cols {
        let mut acc = C64::ZERO;
        for r in 0..n_rows {
            acc = acc + a[r * n_cols + i].conj() * b[r];
        }
        g[i] = acc;
    }

    // Gauss-Jordan with partial pivoting on (h, g).
    let n = n_cols;
    for k in 0..n {
        let mut piv = k;
        for r in (k + 1)..n {
            if h[r * n + k].norm() > h[piv * n + k].norm() { piv = r; }
        }
        if h[piv * n + k].norm() < 1e-30 {
            return Err(VfitError::SingularLstsq);
        }
        if piv != k {
            for c in 0..n { h.swap(k * n + c, piv * n + c); }
            g.swap(k, piv);
        }
        let akk = h[k * n + k];
        for r in 0..n {
            if r == k { continue; }
            let f = h[r * n + k] / akk;
            if f.norm() == 0.0 { continue; }
            for c in k..n {
                h[r * n + c] = h[r * n + c] - f * h[k * n + c];
            }
            g[r] = g[r] - f * g[k];
        }
    }
    let mut x = vec![C64::ZERO; n];
    for i in 0..n { x[i] = g[i] / h[i * n + i]; }
    Ok(x)
}

fn rms_error(
    freqs: &[f64], response: &[C64],
    poles: &[C64], residues: &[C64], d: f64, e: f64,
) -> f64 {
    let mut acc = 0.0_f64;
    for (&omega, &h) in freqs.iter().zip(response) {
        let h_fit = eval_model(omega, poles, residues, d, e);
        acc += (h - h_fit).norm_sqr();
    }
    (acc / freqs.len() as f64).sqrt()
}

/// Convenience: log-spaced negative real initial poles spanning
/// `[ω_min, ω_max]`. Standard Gustavsen starting set for the
/// real-pole MVP.
pub fn log_spaced_real_poles(omega_min: f64, omega_max: f64, k: usize)
    -> Vec<C64>
{
    if k == 0 { return Vec::new(); }
    if k == 1 { return vec![C64::new(-(omega_min * omega_max).sqrt(), 0.0)]; }
    let lmin = omega_min.ln();
    let lmax = omega_max.ln();
    (0..k).map(|i| {
        let t = i as f64 / (k as f64 - 1.0);
        let omega = (lmin + (lmax - lmin) * t).exp();
        C64::new(-omega, 0.0)
    }).collect()
}
