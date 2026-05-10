//! DADO (Decomposition-Aware Distributional Optimization) on the 8-bit
//! R-2R DAC, mirroring the ICLR 2026 paper *Leveraging Discrete Function
//! Decomposability for Scientific Design* (Bowden, Levine, Listgarten —
//! arXiv 2511.03032v2).
//!
//! ## What this spike demonstrates
//!
//! The paper introduces a distributional optimizer for discrete design
//! that exploits a junction-tree decomposition of the objective f(x) =
//! Σᵢ Cᵢ(x̂ᵢ). The paper names circuit design as a target application
//! but only evaluates on synthetic problems and proteins. This spike is
//! the smallest meaningful end-to-end EDA experiment: discrete sizing of
//! 16 resistors in the R-2R DAC, optimized by DADO and by a
//! decomposition-unaware EDA baseline, scored on the same objective.
//!
//! ## Two objectives
//!
//! 1. `score_synth` — a perfectly decomposable target-matching objective
//!    on the same chain JT. The optimum is known analytically (score 0
//!    at `target_design`). Used as a sanity check that the algorithm is
//!    implemented correctly: DADO should converge much faster than EDA.
//! 2. `score_inl`   — negative max-INL of the R-2R DAC under the
//!    perturbed resistor design, computed by solving the linear network
//!    on every code 0..256. The decomposition (per-bit error) is an
//!    approximation; the paper notes that DADO is robust to imperfect
//!    decompositions.
//!
//! ## Simplifications vs the paper
//!
//! The paper uses VAE-style neural-net search distributions. We use a
//! tabular categorical factorized along the chain JT. Both fit the same
//! family `p_θ(x) = ∏ p_θ(C_i | S_{i-1})`; only the parameterization
//! differs. With 16 vars over a 5-element alphabet on this JT, tabular
//! params total < 1k logits — no NN needed.

#![allow(clippy::needless_range_loop)]

pub mod charts;
pub mod layout;
pub mod schem;
pub mod sim;

use spike_dac_r2r::ideal_vout;

// ---------------------------------------------------------------------
// Problem constants
// ---------------------------------------------------------------------

/// Alphabet size: number of discrete resistor-deviation values.
pub const D: usize = 5;

/// Discrete deviation alphabet — fractional deviations from nominal.
pub const DEVIATIONS: [f64; D] = [-0.05, -0.025, 0.0, 0.025, 0.05];

/// Number of design variables (resistors). 1 termination + 8 input
/// feeders + 7 spine = 16, matching `spike-dac-r2r`'s 8-bit ladder.
pub const L: usize = 16;

/// DAC resolution.
pub const N_BITS: usize = 8;
/// Code count = 2^N_BITS.
pub const N_CODES: usize = 1 << N_BITS;
/// Number of internal R-2R nodes (n_0…n_{N-1}, with n_{N-1} == vout).
pub const N_NODES: usize = N_BITS;

/// Nominal "R" value. The 2R legs are 2× this.
pub const R_NOMINAL: f64 = 10_000.0;

// ---------------------------------------------------------------------
// Resistor indexing
//
//  idx 0          : r_term  (LSB termination, 2R, between vlow and n_0)
//  idx 1..=8      : r_in[bit] for bit ∈ 0..8 (2R, between in_bit and n_bit)
//  idx 9..=15     : r_sp[s]  for s   ∈ 0..7 (R,  between n_s   and n_{s+1})
//
// Mirrors the netlist order in spike-dac-r2r/src/lib.rs:82-91.
// ---------------------------------------------------------------------

#[inline] pub const fn r_term_idx() -> usize { 0 }
#[inline] pub const fn r_in_idx(bit: usize) -> usize { 1 + bit }
#[inline] pub const fn r_sp_idx(s: usize)   -> usize { 9 + s }

/// One categorical assignment per resistor — value in `0..D`.
pub type Design = [u8; L];

/// Nominal value (R or 2R) of resistor `idx` before deviation.
fn nominal_ohms(idx: usize) -> f64 {
    // r_term (idx 0) and r_in[*] (idx 1..=8) are 2R; spine (idx 9..=15) is R.
    if idx <= 8 { 2.0 * R_NOMINAL } else { R_NOMINAL }
}

/// Resolve `design[idx]` to an actual resistance in ohms.
pub fn r_value(design: &Design, idx: usize) -> f64 {
    nominal_ohms(idx) * (1.0 + DEVIATIONS[design[idx] as usize])
}

// ── Thermal-corner extension ───────────────────────────────────────────
//
// For a poly resistor on a CMOS process, R(T) = R₀·(1 + TC1·(T−Tnom)).
// With *uniform* TC1 across all 16 resistors, the R-2R divider's INL
// is invariant to T — every R scales by the same factor and the
// transfer ratio is unchanged. That's a no-op DADO can't exploit.
//
// To get a believable thermal-aware optimization problem, we couple
// TC1 to the discrete *deviation* choice: a wider/narrower poly
// resistor has a slightly different TC1 than nominal (substrate
// coupling + dimension-dependent bandgap-shift effects). Modeled here
// as `TC1_eff(idx) = TC1_NOM · (1 + κ · dev[idx])`. With κ = 2.0, a
// +5 % deviation also gets +10 % more thermal drift than nominal —
// large enough that the worst-corner INL diverges from nominal-T INL,
// small enough that it stays physically defensible.
//
// All thermal helpers are additive: existing `r_value` / `solve_r2r` /
// `score_inl` still evaluate at Tnom by construction, so the DADO
// binaries that already exist keep working unchanged.

/// Nominal temperature for resistor parameters, °C.
pub const T_NOMINAL_C: f64 = 27.0;

/// Process corners DADO scores against. Sky130 + GF180 datasheets cite
/// these as the standard digital-PVT temperature triple.
pub const T_CORNERS_C: [f64; 3] = [-40.0, 27.0, 125.0];

/// Nominal poly-resistor linear temperature coefficient, 1/°C.
/// −800 ppm/°C is the textbook value for the high-resistance poly
/// option in sky130 / GF180; high-density poly is closer to −1500
/// ppm/°C, but the demo's behavior is qualitatively the same.
pub const TC1_NOM: f64 = -8.0e-4;

/// Coupling between deviation and TC1: `TC1_eff = TC1_NOM·(1 + κ·dev)`.
/// κ = 2 makes the worst-corner INL ~50 % larger than nominal-T INL
/// for adversarial designs — a comfortable signal for DADO to chase.
pub const TC1_DEV_KAPPA: f64 = 2.0;

/// Effective TC1 of resistor `idx` given its discrete deviation choice.
fn tc1_eff(design: &Design, idx: usize) -> f64 {
    TC1_NOM * (1.0 + TC1_DEV_KAPPA * DEVIATIONS[design[idx] as usize])
}

/// Resistance at corner temperature `t_celsius`. At `T_NOMINAL_C`
/// reduces exactly to `r_value(design, idx)`.
pub fn r_value_at_temp(design: &Design, idx: usize, t_celsius: f64) -> f64 {
    let r0 = r_value(design, idx);
    let dt = t_celsius - T_NOMINAL_C;
    r0 * (1.0 + tc1_eff(design, idx) * dt)
}

// ---------------------------------------------------------------------
// Non-ideal R-2R evaluator
//
// Linear network → one nodal-analysis solve per code. With perturbed
// resistors the closed form `code/2^N · vref` no longer holds, so we
// build the conductance matrix G and solve G v = i.
//
// 8×8 system. Hand-rolled Gaussian elimination with partial pivoting.
// We deliberately avoid eda-mna here: it's built around behavioral
// devices + Newton solves over rlx-graph residuals, which is overkill
// for this purely-linear inner loop.
// ---------------------------------------------------------------------

/// Solve the R-2R network for one code, return vout (= node n_{N-1}).
///
/// Inputs `in_bit` are at `vref` if bit set, else at `vlow`. Bit 0 is LSB.
pub fn solve_r2r(design: &Design, code: u32, vref: f64, vlow: f64) -> f64 {
    let mut g = [[0.0_f64; N_NODES]; N_NODES];
    let mut i_vec = [0.0_f64; N_NODES];

    // r_term: vlow -- 2R -- n_0  (one-port to ground-equivalent)
    let g_term = 1.0 / r_value(design, r_term_idx());
    g[0][0] += g_term;
    i_vec[0] += g_term * vlow;

    // r_in[bit]: in_bit -- 2R -- n_bit  (driven port per bit)
    for bit in 0..N_BITS {
        let v_in = if (code >> bit) & 1 == 1 { vref } else { vlow };
        let g_in = 1.0 / r_value(design, r_in_idx(bit));
        g[bit][bit] += g_in;
        i_vec[bit] += g_in * v_in;
    }

    // r_sp[s]: n_s -- R -- n_{s+1}  (spine, two-port between nodes)
    for s in 0..(N_NODES - 1) {
        let g_sp = 1.0 / r_value(design, r_sp_idx(s));
        g[s][s]     += g_sp;
        g[s + 1][s + 1] += g_sp;
        g[s][s + 1] -= g_sp;
        g[s + 1][s] -= g_sp;
    }

    let v = gauss_solve(g, i_vec);
    v[N_NODES - 1]
}

/// Like `solve_r2r` but evaluates conductances at corner temperature
/// `t_celsius` via [`r_value_at_temp`]. At `T_NOMINAL_C` reduces
/// numerically to `solve_r2r`.
pub fn solve_r2r_at_temp(
    design: &Design, code: u32, vref: f64, vlow: f64, t_celsius: f64,
) -> f64 {
    let mut g = [[0.0_f64; N_NODES]; N_NODES];
    let mut i_vec = [0.0_f64; N_NODES];

    let g_term = 1.0 / r_value_at_temp(design, r_term_idx(), t_celsius);
    g[0][0] += g_term;
    i_vec[0] += g_term * vlow;

    for bit in 0..N_BITS {
        let v_in = if (code >> bit) & 1 == 1 { vref } else { vlow };
        let g_in = 1.0 / r_value_at_temp(design, r_in_idx(bit), t_celsius);
        g[bit][bit] += g_in;
        i_vec[bit] += g_in * v_in;
    }
    for s in 0..(N_NODES - 1) {
        let g_sp = 1.0 / r_value_at_temp(design, r_sp_idx(s), t_celsius);
        g[s][s]         += g_sp;
        g[s + 1][s + 1] += g_sp;
        g[s][s + 1]     -= g_sp;
        g[s + 1][s]     -= g_sp;
    }

    let v = gauss_solve(g, i_vec);
    v[N_NODES - 1]
}

/// In-place Gaussian elimination with partial pivoting on a fixed-size
/// system. Specialised to N_NODES; small enough that we don't bother
/// generalising. Returns the solution vector.
fn gauss_solve(mut a: [[f64; N_NODES]; N_NODES], mut b: [f64; N_NODES]) -> [f64; N_NODES] {
    for k in 0..N_NODES {
        // Partial pivot.
        let mut piv = k;
        let mut best = a[k][k].abs();
        for r in (k + 1)..N_NODES {
            if a[r][k].abs() > best { best = a[r][k].abs(); piv = r; }
        }
        if piv != k {
            a.swap(k, piv);
            b.swap(k, piv);
        }
        let akk = a[k][k];
        // Singular conductance matrix would mean an unconnected node;
        // our R-2R topology guarantees this can't happen for finite R.
        debug_assert!(akk.abs() > 0.0, "singular conductance matrix");

        for r in (k + 1)..N_NODES {
            let f = a[r][k] / akk;
            if f == 0.0 { continue; }
            for c in k..N_NODES { a[r][c] -= f * a[k][c]; }
            b[r] -= f * b[k];
        }
    }
    // Back-substitute.
    let mut x = [0.0_f64; N_NODES];
    for i in (0..N_NODES).rev() {
        let mut s = b[i];
        for j in (i + 1)..N_NODES { s -= a[i][j] * x[j]; }
        x[i] = s / a[i][i];
    }
    x
}

// ---------------------------------------------------------------------
// Junction tree
//
// Chain of N_BITS cliques. Clique `i` covers the variables that
// dominate bit i's contribution to vout: the bit's own input feeder
// plus the spine resistors immediately adjacent to its node n_i. The
// LSB termination joins clique 0. Adjacent cliques overlap on a single
// spine resistor (the size-1 separator).
//
//   C_0 = { r_term, r_in[0],            r_sp[0]            }
//   C_i = {         r_in[i], r_sp[i-1], r_sp[i]            }   1 ≤ i ≤ 6
//   C_7 = {         r_in[7], r_sp[6]                       }
//   S_i = C_i ∩ C_{i+1} = { r_sp[i] }                          0 ≤ i ≤ 6
// ---------------------------------------------------------------------

/// Variables (resistor indices) in clique `i`. Order matters — it's the
/// canonical layout used to encode/decode tabular table positions.
pub fn clique_vars(i: usize) -> Vec<usize> {
    match i {
        0 => vec![r_term_idx(), r_in_idx(0), r_sp_idx(0)],
        7 => vec![r_in_idx(7), r_sp_idx(6)],
        _ => vec![r_in_idx(i), r_sp_idx(i - 1), r_sp_idx(i)],
    }
}

/// Variables in the separator between clique `i` and clique `i+1`.
/// Always a single spine resistor for our chain.
pub fn separator_vars(i: usize) -> Vec<usize> {
    debug_assert!(i < N_BITS - 1);
    vec![r_sp_idx(i)]
}

/// Variables in `C_i` that are *not* in the parent separator `S_{i-1}`.
/// These are the "new" vars introduced by clique i, the ones the
/// conditional p(C_i | S_{i-1}) actually parameterises.
pub fn new_vars(i: usize) -> Vec<usize> {
    if i == 0 { return clique_vars(0); }
    let parent_sep = separator_vars(i - 1);
    clique_vars(i).into_iter().filter(|v| !parent_sep.contains(v)).collect()
}

// ---------------------------------------------------------------------
// Factorised search distribution
//
// p_θ(x) = p(C_0) · ∏_{i=1..K-1} p(C_i | S_{i-1}).
//
// Each conditional is a categorical over the assignments of `new_vars(i)`,
// indexed by the assignment of the parent separator. Stored as logits
// in a flat Vec, with index = sep_index * D^|new_vars| + new_index.
// ---------------------------------------------------------------------

#[inline] fn d_pow(k: usize) -> usize {
    let mut r = 1usize;
    for _ in 0..k { r *= D; }
    r
}

/// Encode a tuple of categorical values into a flat index (little-endian
/// in `D`).
fn encode(vars: &[u8]) -> usize {
    let mut idx = 0usize;
    let mut mult = 1usize;
    for &v in vars { idx += (v as usize) * mult; mult *= D; }
    idx
}

/// Inverse of `encode`.
fn decode(mut idx: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(k);
    for _ in 0..k { out.push((idx % D) as u8); idx /= D; }
    out
}

/// Tabular factorised categorical along the chain JT.
#[derive(Clone)]
pub struct ChainDist {
    /// `logits[i]` has length `D^|S_{i-1}|` × `D^|new_vars(i)|`,
    /// row-major: outer index is separator value, inner is new-vars value.
    logits: Vec<Vec<f64>>,
    /// Cached `D^|new_vars(i)|` for each clique.
    n_new: Vec<usize>,
    /// Cached `D^|S_{i-1}|` for each clique (1 for the root).
    n_sep: Vec<usize>,
}

impl ChainDist {
    /// Uniform-prior initialisation (all logits zero).
    pub fn uniform() -> Self {
        let mut logits = Vec::with_capacity(N_BITS);
        let mut n_new = Vec::with_capacity(N_BITS);
        let mut n_sep = Vec::with_capacity(N_BITS);
        for i in 0..N_BITS {
            let nn = d_pow(new_vars(i).len());
            let ns = if i == 0 { 1 } else { d_pow(separator_vars(i - 1).len()) };
            logits.push(vec![0.0; nn * ns]);
            n_new.push(nn);
            n_sep.push(ns);
        }
        Self { logits, n_new, n_sep }
    }

    /// Sample one design from the current parameters, given an RNG.
    pub fn sample(&self, rng: &mut Rng) -> Design {
        let mut x = [0u8; L];
        for i in 0..N_BITS {
            // Read separator value from already-sampled vars.
            let sep_idx = if i == 0 {
                0
            } else {
                let sep = separator_vars(i - 1);
                let sep_vals: Vec<u8> = sep.iter().map(|&v| x[v]).collect();
                encode(&sep_vals)
            };
            let row = &self.logits[i][sep_idx * self.n_new[i] .. (sep_idx + 1) * self.n_new[i]];
            let pick = sample_softmax(row, rng);
            // Decode `pick` back into the new-vars values, write into x.
            let nv = new_vars(i);
            let vals = decode(pick, nv.len());
            for (k, &var) in nv.iter().enumerate() { x[var] = vals[k]; }
        }
        x
    }

    /// Replace logits by the (smoothed) log of weighted counts. Counts
    /// are accumulated per-clique with sample weights `weights[k]` for
    /// sample `k`. Smoothing (`alpha`) keeps tails alive — without it
    /// any unseen separator slice would collapse to NaN.
    pub fn fit_weighted(&mut self, samples: &[Design], weights: &[Vec<f64>], alpha: f64) {
        // weights[i][k] = weight for sample k when fitting clique i's conditional.
        debug_assert_eq!(weights.len(), N_BITS);
        for i in 0..N_BITS {
            let w_i = &weights[i];
            debug_assert_eq!(w_i.len(), samples.len());
            let nn = self.n_new[i];
            let ns = self.n_sep[i];
            let mut counts = vec![alpha; nn * ns];
            let nv = new_vars(i);
            for (k, x) in samples.iter().enumerate() {
                let w = w_i[k];
                if w == 0.0 { continue; }
                let sep_idx = if i == 0 {
                    0
                } else {
                    let sep = separator_vars(i - 1);
                    let sep_vals: Vec<u8> = sep.iter().map(|&v| x[v]).collect();
                    encode(&sep_vals)
                };
                let new_vals: Vec<u8> = nv.iter().map(|&v| x[v]).collect();
                let new_idx = encode(&new_vals);
                counts[sep_idx * nn + new_idx] += w;
            }
            // Renormalise per separator slice and take logs.
            for s in 0..ns {
                let row = &mut counts[s * nn .. (s + 1) * nn];
                let z: f64 = row.iter().sum();
                if z > 0.0 {
                    for c in row.iter_mut() { *c = (*c / z).ln(); }
                }
            }
            self.logits[i] = counts;
        }
    }
}

/// Sample a categorical from raw logits via the softmax-Gumbel trick.
fn sample_softmax(logits: &[f64], rng: &mut Rng) -> usize {
    // Stable: subtract max, then sample by inverse-CDF over softmax.
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut probs = vec![0.0; logits.len()];
    let mut z = 0.0;
    for (i, &l) in logits.iter().enumerate() {
        let p = (l - m).exp();
        probs[i] = p;
        z += p;
    }
    let u = rng.next_unit() * z;
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u <= acc { return i; }
    }
    probs.len() - 1
}

// ---------------------------------------------------------------------
// Score decomposition
// ---------------------------------------------------------------------

/// Synthetic, perfectly-decomposable target-matching objective on the
/// chain JT. Each clique contributes negative Hamming distance from a
/// fixed `target_design`, restricted to the variables in that clique.
/// Optimum is `target_design` with score 0.
///
/// Returns `(total_score, per_clique_components)`.
pub fn score_synth(design: &Design, target: &Design) -> (f64, [f64; N_BITS]) {
    let mut comps = [0.0_f64; N_BITS];
    for i in 0..N_BITS {
        let mut c = 0.0;
        for &v in &clique_vars(i) {
            if design[v] != target[v] { c -= 1.0; }
        }
        comps[i] = c;
    }
    let total: f64 = comps.iter().sum();
    (total, comps)
}

/// Negative *sum* of squared INL over all codes, in V². Replaces the
/// `max_k |·|` reduction with a `Σ_k (·)²` reduction — same per-code
/// errors, but a sum-of-squares is naturally compatible with DADO's
/// suffix-sum value functions in a way `max` is not. Per-clique
/// attribution still uses the highest set bit of each code, but now the
/// component values *add* across codes within a clique instead of being
/// hidden under a `max`.
pub fn score_sse_inl(design: &Design) -> (f64, [f64; N_BITS]) {
    let mut total = 0.0_f64;
    let mut comps = [0.0_f64; N_BITS];
    for code in 0..N_CODES as u32 {
        let v = solve_r2r(design, code, 1.0, 0.0);
        let target = ideal_vout(code, N_BITS as u32, 1.0, 0.0);
        let err2 = (v - target).powi(2);
        total += err2;
        // Code 0 is exact at all-nominal, attribute it to bit 0.
        let hi = if code == 0 { 0 } else { (31 - code.leading_zeros()) as usize };
        comps[hi] -= err2;
    }
    (-total, comps)
}

/// Negative max-DNL across the 255 adjacent-code transitions, in volts.
///
/// DNL(k) = vout(k+1) − vout(k) − LSB, with LSB = vref / 2ᴺ. The
/// decomposition attributes each transition's |DNL| to the *highest
/// bit that flips* in that transition, which is determined by the carry
/// chain: for k → k+1, the highest flipped bit is `trailing_ones(k)`.
/// Most adjacent codes only flip bit 0 (so bit-0's clique gets most of
/// the credit); the rare full-carry transitions land on the high-bit
/// cliques. This is a much cleaner per-bit factorisation than max-INL
/// admits.
pub fn score_dnl(design: &Design) -> (f64, [f64; N_BITS]) {
    let lsb = 1.0 / (N_CODES as f64);
    let mut max_abs = 0.0_f64;
    let mut comps = [0.0_f64; N_BITS];
    let mut prev = solve_r2r(design, 0, 1.0, 0.0);
    for code in 1..N_CODES as u32 {
        let cur = solve_r2r(design, code, 1.0, 0.0);
        let dnl = (cur - prev) - lsb;
        let abs_dnl = dnl.abs();
        if abs_dnl > max_abs { max_abs = abs_dnl; }
        // Highest flipped bit for transition (code-1) → code is the
        // count of trailing 1s in (code-1). Clamp to N_BITS-1.
        let hi = ((code - 1).trailing_ones() as usize).min(N_BITS - 1);
        comps[hi] -= dnl * dnl;   // squared so signed errors don't cancel
        prev = cur;
    }
    (-max_abs, comps)
}

/// Negative max-INL across all 256 codes, in volts. The decomposition
/// attributes each code's error to its single highest set bit (the bit
/// that dominates the code's contribution). This is approximate — INL
/// genuinely couples all resistors — but the paper observes DADO is
/// robust to imperfect decompositions.
/// Negative max-INL evaluated at `t_celsius` (one corner).
/// Decomposition is the same highest-set-bit attribution as
/// [`score_inl`].
pub fn score_inl_at_temp(
    design: &Design, t_celsius: f64,
) -> (f64, [f64; N_BITS]) {
    let mut max_abs = 0.0_f64;
    let mut comps = [0.0_f64; N_BITS];
    let mut counts = [0.0_f64; N_BITS];
    for code in 0..N_CODES as u32 {
        let v = solve_r2r_at_temp(design, code, 1.0, 0.0, t_celsius);
        let target = ideal_vout(code, N_BITS as u32, 1.0, 0.0);
        let err = (v - target).abs();
        if err > max_abs { max_abs = err; }
        if code > 0 {
            let hi = (31 - code.leading_zeros()) as usize;
            comps[hi] -= err * err;
            counts[hi] += 1.0;
        }
    }
    for i in 0..N_BITS {
        if counts[i] > 0.0 { comps[i] /= counts[i]; }
    }
    (-max_abs, comps)
}

/// Worst-corner INL across [`T_CORNERS_C`]: `max_T |INL(design, T)|`,
/// returned as a negated score (higher = better) so it slots into
/// DADO's existing `ScoreFn` shape. Per-clique components come from
/// the corner that hit the worst max-INL — DADO then attributes blame
/// to whichever clique dominated at the actual failing corner.
pub fn score_inl_worst_corner(design: &Design) -> (f64, [f64; N_BITS]) {
    let mut worst_score = f64::INFINITY; // looking for the most-negative score
    let mut worst_comps = [0.0_f64; N_BITS];
    for &t in &T_CORNERS_C {
        let (score, comps) = score_inl_at_temp(design, t);
        if score < worst_score {
            worst_score = score;
            worst_comps = comps;
        }
    }
    (worst_score, worst_comps)
}

pub fn score_inl(design: &Design) -> (f64, [f64; N_BITS]) {
    let mut max_abs = 0.0_f64;
    let mut comps = [0.0_f64; N_BITS];
    let mut counts = [0.0_f64; N_BITS];
    for code in 0..N_CODES as u32 {
        let v = solve_r2r(design, code, 1.0, 0.0);
        let target = ideal_vout(code, N_BITS as u32, 1.0, 0.0);
        let err = (v - target).abs();
        if err > max_abs { max_abs = err; }
        // Attribute the squared error to the highest set bit of `code`
        // (codes 1..256 — code 0 is exact by construction). Summing
        // squared errors per-bit gives a smooth, additive surrogate
        // that's a usable per-clique signal even if it's not the same
        // as max-INL.
        if code > 0 {
            let hi = (31 - code.leading_zeros()) as usize;
            comps[hi] -= err * err;
            counts[hi] += 1.0;
        }
    }
    // Normalise per-bit contributions so they're scaled comparably.
    for i in 0..N_BITS {
        if counts[i] > 0.0 { comps[i] /= counts[i]; }
    }
    (-max_abs, comps)
}

// ---------------------------------------------------------------------
// DADO + naive EDA loops
//
// Both algorithms share the same factorised distribution, sampling, and
// weighted-MLE update. The only difference is what weights they use:
//
//   naive EDA: weights[i][k] = w(score(x^k))           (same for all i)
//   DADO:      weights[i][k] = w(Q_i(x̂_i^k))
//
// `Q_i(x̂_i^k)` is the value-function message for clique i evaluated at
// the sample's clique-i assignment. For our root-at-clique-0 chain it
// collapses to a suffix sum over component scores: Q_i^k = Σ_{j ≥ i} C_j^k.
//
// Weights are obtained by exp-normalising the (rescaled) Q values across
// the K samples, which is the standard EDA "prioritise high-scoring
// samples" trick — equivalent to a weighted MLE that chases the
// Boltzmann distribution at temperature `tau`.
// ---------------------------------------------------------------------

/// Score function alias: returns (total, per-clique components).
pub type ScoreFn<'a> = dyn Fn(&Design) -> (f64, [f64; N_BITS]) + 'a;

/// Boltzmann-style sample weights: w_k = exp((score_k - max_k) / tau).
/// Deliberately unnormalised — the best sample gets weight 1 and others
/// fall off from there. Total weight (effective sample size) lands in
/// [1, K], which is the right scale to compare against the per-category
/// smoothing prior `alpha` in `fit_weighted`. Normalising to sum=1 made
/// `alpha` swamp the data.
fn softmax_weights(scores: &[f64], tau: f64) -> Vec<f64> {
    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    scores.iter().map(|s| ((s - m) / tau).exp()).collect()
}

/// State captured at one snapshot iteration during optimization.
#[derive(Clone, Debug)]
pub struct DistSnapshot {
    pub iter: usize,
    /// Per-resistor marginal distribution: `marginals[var][cat]` is the
    /// probability of resistor `var` taking deviation index `cat`.
    pub marginals: [[f64; D]; L],
    /// Best concrete design seen up to and including this iteration.
    pub best_design: Design,
    /// Score of `best_design`.
    pub best_score: f64,
    /// Per-resistor expected deviation index (centroid of `marginals`).
    pub expected_dev_idx: [f64; L],
}

/// Outcome of one optimization run.
#[derive(Clone, Debug)]
pub struct RunTrace {
    /// Best-so-far total score after each iteration.
    pub best: Vec<f64>,
    /// Mean total score per iteration.
    pub mean: Vec<f64>,
    /// Best concrete design ever seen across all iterations.
    pub best_design: Design,
    /// Score of `best_design`.
    pub best_score: f64,
    /// Distribution + best-design snapshots at user-requested iters.
    pub snapshots: Vec<DistSnapshot>,
}

/// Estimate the per-resistor marginal of `dist` by sampling. With
/// `n_samples = 4000` standard-error per cell is ~0.008 — fine for
/// visualisation. Cheap (≤ 1 ms) at our problem size.
pub fn estimate_marginals(dist: &ChainDist, rng: &mut Rng) -> [[f64; D]; L] {
    const N: usize = 4000;
    let mut counts = [[0.0_f64; D]; L];
    for _ in 0..N {
        let x = dist.sample(rng);
        for v in 0..L { counts[v][x[v] as usize] += 1.0; }
    }
    for v in 0..L {
        for c in 0..D { counts[v][c] /= N as f64; }
    }
    counts
}

fn expected_idx(marginals: &[[f64; D]; L]) -> [f64; L] {
    let mut out = [0.0_f64; L];
    for v in 0..L {
        let mut s = 0.0;
        for c in 0..D { s += (c as f64) * marginals[v][c]; }
        out[v] = s;
    }
    out
}

/// Run DADO (`use_decomposition = true`) or naive EDA (`= false`).
/// `snapshot_iters` lists 1-based iteration indices to snapshot at; the
/// snapshot is taken *after* that iteration's update (so iter 0 means
/// after the first update). Pass `&[]` to skip snapshots entirely.
pub fn run(
    score: &ScoreFn,
    n_iters: usize,
    k_samples: usize,
    tau: f64,
    alpha: f64,
    use_decomposition: bool,
    seed: u32,
    snapshot_iters: &[usize],
) -> RunTrace {
    let mut rng = Rng::new(seed);
    let mut dist = ChainDist::uniform();
    let mut best_so_far = f64::NEG_INFINITY;
    let mut best_design_so_far: Design = [(D / 2) as u8; L]; // sensible default
    let mut best = Vec::with_capacity(n_iters);
    let mut mean = Vec::with_capacity(n_iters);
    let mut snapshots = Vec::with_capacity(snapshot_iters.len());
    for it in 0..n_iters {
        let (b, bd, m) = eda_step_with_best(
            &mut dist, score, k_samples, tau, alpha, use_decomposition, &mut rng,
        );
        if b > best_so_far { best_so_far = b; best_design_so_far = bd; }
        best.push(best_so_far);
        mean.push(m);
        if snapshot_iters.contains(&it) {
            let marginals = estimate_marginals(&dist, &mut rng);
            let expected_dev_idx = expected_idx(&marginals);
            snapshots.push(DistSnapshot {
                iter: it,
                marginals,
                best_design: best_design_so_far,
                best_score: best_so_far,
                expected_dev_idx,
            });
        }
    }
    RunTrace {
        best,
        mean,
        best_design: best_design_so_far,
        best_score: best_so_far,
        snapshots,
    }
}

/// Same as `eda_step` but additionally returns the best concrete design
/// from the batch (so `run` can keep a running best-design pointer).
fn eda_step_with_best(
    dist: &mut ChainDist,
    score: &ScoreFn,
    k_samples: usize,
    tau: f64,
    alpha: f64,
    use_decomposition: bool,
    rng: &mut Rng,
) -> (f64, Design, f64) {
    let mut samples = Vec::with_capacity(k_samples);
    let mut totals  = Vec::with_capacity(k_samples);
    let mut comps   = Vec::with_capacity(k_samples);
    for _ in 0..k_samples {
        let x = dist.sample(rng);
        let (t, c) = score(&x);
        samples.push(x);
        totals.push(t);
        comps.push(c);
    }
    let mut weights: Vec<Vec<f64>> = Vec::with_capacity(N_BITS);
    for i in 0..N_BITS {
        let raw: Vec<f64> = if use_decomposition {
            (0..k_samples).map(|k| {
                let mut s = 0.0;
                for j in i..N_BITS { s += comps[k][j]; }
                s
            }).collect()
        } else {
            totals.clone()
        };
        weights.push(softmax_weights(&raw, tau));
    }
    dist.fit_weighted(&samples, &weights, alpha);
    let (best_idx, &best) = totals
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let mean = totals.iter().sum::<f64>() / k_samples as f64;
    (best, samples[best_idx], mean)
}

// ---------------------------------------------------------------------
// PRNG — xorshift32, copied from spike-surrogate so we don't drag `rand`.
// ---------------------------------------------------------------------

#[derive(Clone)]
pub struct Rng(u32);
impl Rng {
    pub fn new(seed: u32) -> Self { Self(seed.max(1)) }
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.0 = x;
        x
    }
    pub fn next_unit(&mut self) -> f64 { self.next_u32() as f64 / u32::MAX as f64 }
}

/// Convenience: draw a uniform-random design (used for tests + targets).
pub fn random_design(rng: &mut Rng) -> Design {
    let mut x = [0u8; L];
    for i in 0..L { x[i] = (rng.next_u32() as usize % D) as u8; }
    x
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// At all-nominal deviations the closed-form `ideal_vout` must agree
    /// with our MNA solver across every code 0..256.
    #[test]
    fn evaluator_matches_ideal_at_nominal() {
        let nominal = [2u8; L]; // index 2 → 0% deviation
        for code in 0..N_CODES as u32 {
            let v = solve_r2r(&nominal, code, 1.0, 0.0);
            let target = ideal_vout(code, N_BITS as u32, 1.0, 0.0);
            assert!((v - target).abs() < 1e-12,
                    "code {code}: solver gave {v}, ideal {target}");
        }
    }

    /// Junction tree shape sanity.
    #[test]
    fn jt_shape() {
        // 16 distinct vars covered when we union all cliques.
        use std::collections::BTreeSet;
        let mut covered = BTreeSet::new();
        for i in 0..N_BITS { for &v in &clique_vars(i) { covered.insert(v); } }
        assert_eq!(covered.len(), L);
        // Each separator is a single spine resistor.
        for i in 0..(N_BITS - 1) {
            let s = separator_vars(i);
            assert_eq!(s.len(), 1);
            assert_eq!(s[0], r_sp_idx(i));
        }
    }

    /// Round-trip encode/decode of categorical tuples.
    #[test]
    fn encode_decode_roundtrip() {
        let cases: &[&[u8]] = &[&[0,0,0], &[4,4,4], &[1,2,3], &[3,0,4,2]];
        for case in cases {
            let idx = encode(case);
            let back = decode(idx, case.len());
            assert_eq!(&back[..], *case);
        }
    }

    /// Smoke test: DADO on the synthetic objective converges to a high
    /// score within a small budget. Uses the tuned defaults from
    /// `examples/sweep.rs`: K=100, alpha=0.1, tau=1.0.
    #[test]
    fn dado_synth_converges() {
        let mut rng = Rng::new(7);
        let target = random_design(&mut rng);
        let score = move |x: &Design| score_synth(x, &target);
        let trace = run(&score, 60, 100, 1.0, 0.1, true, 11, &[]);
        let final_best = *trace.best.last().unwrap();
        // 16 vars × 8 cliques (size 2-3) max score = 0; worst random ≈ -19.
        // DADO with these settings reliably reaches ≥ -1 (the sweep shows
        // it usually hits 0).
        assert!(final_best >= -1.0,
                "DADO did not converge on synthetic: best={final_best}");
    }
}
