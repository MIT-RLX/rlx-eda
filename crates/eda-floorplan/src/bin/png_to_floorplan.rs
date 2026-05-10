//! Greedy primitive-fit floorplan from a raster image — RGB,
//! double-resolution variant.
//!
//! Implements the classic "primitive painting" recipe (Fogleman's
//! [`primitive`](https://github.com/fogleman/primitive), 2016 — also
//! the precursor to differentiable-primitive papers like DiffVG): add
//! one rectangle at a time, pick each new rect by sampling K random
//! candidates biased toward the largest current residual, hill-climb
//! each candidate's `(cx, cy, hw, hh, α)` on an L1 image loss, **pick
//! that rect's RGB color analytically** via the closed-form L2-best
//! color over its soft mask, and composite the result onto a running
//! RGB canvas via Porter-Duff `over`. Repeat for `N` rects.
//!
//! Why best-color is closed-form: with the rect's geometry + α fixed,
//! `canvas_new[c] = canvas[c]·(1-w) + color[c]·w` is linear in
//! `color[c]`; squared error has a quadratic minimum
//! `color[c] = Σ_ij (w·target + (w² − w)·canvas) / Σ w²` per channel.
//! No need to spend a hill-climb dimension on it — every candidate's
//! optimal color is one matrix dot product.
//!
//! Why greedy beats global AD on this problem:
//!   * Each rect's gradient signal is isolated to its own footprint —
//!     no permutation-invariance saddles, no mutual cancellation.
//!   * Loss landscape is locally smooth + monotone under residual
//!     compositing: every accepted rect strictly reduces L1.
//!   * Inner refine is on a 5-DoF problem; AD or hill-climb both
//!     finish in microseconds. The `[N, H, W, 3]` broadcast tensor
//!     never gets allocated → no rlx CPU SIGSEGV territory.
//!
//! AD is still demonstrated end-to-end via [`rlx_refine_one_rect`]:
//! the *first* rect's refine runs through `rlx-ir` +
//! `rlx_opt::autodiff::grad_with_loss` so the "rlx-eda hits silicon
//! via reverse-mode AD" story has a live witness. Subsequent rects
//! use the much faster Rust hill-climb.
//!
//! Outputs land in `target/floorplans/beaver_optim/`:
//!   floorplan.svg / floorplan.gds  — fitted rects on metal1
//!                                    (chip layout is single-layer;
//!                                    color is for the convergence
//!                                    raster only)
//!   convergence.png                — target | rasterized | layout
//!                                    side-by-side, RGB
//!   loss.csv                       — per-rect (idx, Δloss, L1, geom, color)
//!   summary.txt                    — bbox, kept count, final loss

use std::fs;
use std::path::{Path, PathBuf};

use eda_viz::Style;
use image::{ImageBuffer, Rgb, RgbImage};
use klayout_core::{Bbox, CellBuilder, CellId, Library, Point, Rect as KRect, Shape};
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape as TShape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use spike_divider_block::pdks::Sky130Lite;
use spike_divider_block::{Adam, Optimizer, RcLikePdk};

const RASTER_W: usize = 256;
const RASTER_H: usize = 384;
const N_RECTS: usize = 600;
/// Random candidates per rect — Fogleman's default is ~50–500. More
/// candidates = better fit, slower. K=50 is a good speed/quality knob
/// at this resolution.
const K_CANDIDATES: usize = 50;
/// Hill-climb steps per candidate. Each step perturbs one parameter
/// by ±δ; δ decays over the inner loop (simulated-annealing style).
const HILL_STEPS: usize = 100;
/// Polish passes — disabled (set to 0). Refitting each rect against
/// "all others as baseline" sounded principled but Porter-Duff `over`
/// isn't commutative, so the per-rect optima don't compose into a
/// global optimum: the refits made things slightly worse on every
/// experiment. Kept as code (`polish_pass`) for posterity and in
/// case a commutative-composite mode is added later.
const N_POLISH_PASSES: usize = 0;
/// Max half-extent in pixels. Bumped with the resolution so individual
/// rects can still cover meaningful features (eyes, ears, hands).
const HW_MAX_PX: f32 = 40.0;
const HH_MAX_PX: f32 = 40.0;
const HW_MIN_PX: f32 = 0.8;
const HH_MIN_PX: f32 = 0.8;
/// Sigmoid edge sharpness for soft rect masks. Smaller = crisper
/// edges = closer to a hard rect; larger = smoother gradients during
/// refinement.
const TAU: f32 = 0.7;
/// 1 raster pixel = 1.5 µm = 1500 DBU. Final cell at 256 × 384 px
/// = 384 × 576 µm — same physical chip size as the lower-res
/// variants; we just keep stuffing more pixels in.
const DBU_PER_PIXEL: i64 = 1_500;

/// Number of color channels. RGB = 3; the canvas/target buffers are
/// laid out as `[r, g, b, r, g, b, ...]` so a stride-3 indexer hits
/// the right channel without a helper struct.
const C: usize = 3;

#[inline(always)]
fn idx_pixel(i: usize, j: usize, w: usize) -> usize {
    (i * w + j) * C
}

fn load_target(path: &Path, w: usize, h: usize) -> Vec<f32> {
    // RGBA → composite onto white → resize. Preserves color.
    let rgba = image::open(path)
        .unwrap_or_else(|e| panic!("read {path:?}: {e}"))
        .to_rgba8();
    let mut rgb =
        ImageBuffer::<Rgb<u8>, Vec<u8>>::new(rgba.width(), rgba.height());
    for (px, gp) in rgba.pixels().zip(rgb.pixels_mut()) {
        let [r, g, b, a] = px.0;
        let alpha = a as f32 / 255.0;
        let blend = |c: u8| (c as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
        gp.0 = [blend(r), blend(g), blend(b)];
    }
    let resized = image::imageops::resize(
        &rgb,
        w as u32,
        h as u32,
        image::imageops::FilterType::Triangle,
    );
    // Linear [0,1] floats, channel-interleaved.
    resized.iter().map(|p| *p as f32 / 255.0).collect()
}

fn save_rgb(path: &Path, w: usize, h: usize, data: &[f32]) {
    let buf: Vec<u8> = data
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    let img: RgbImage =
        ImageBuffer::from_raw(w as u32, h as u32, buf).expect("rgb buffer size");
    img.save(path).expect("write png");
}

fn save_triptych(
    path: &Path,
    w: usize,
    h: usize,
    target: &[f32],
    raster: &[f32],
    layout: &[f32],
) {
    let total_w = 3 * w + 2;
    let mut out = vec![255u8; total_w * h * C]; // white dividers
    for (panel_idx, panel) in [target, raster, layout].iter().enumerate() {
        let x0 = panel_idx * (w + 1);
        for i in 0..h {
            for j in 0..w {
                let src = (i * w + j) * C;
                let dst = (i * total_w + (x0 + j)) * C;
                for c in 0..C {
                    out[dst + c] = (panel[src + c].clamp(0.0, 1.0) * 255.0) as u8;
                }
            }
        }
        if panel_idx < 2 {
            for i in 0..h {
                let dst = (i * total_w + (x0 + w)) * C;
                out[dst] = 128;
                out[dst + 1] = 128;
                out[dst + 2] = 128;
            }
        }
    }
    let img: RgbImage =
        ImageBuffer::from_raw(total_w as u32, h as u32, out).expect("triptych");
    img.save(path).expect("write triptych");
}

#[derive(Clone, Copy, Debug)]
struct Rect {
    cx: f32,
    cy: f32,
    hw: f32,
    hh: f32,
    /// Opacity in [0,1].
    alpha: f32,
    /// RGB color in [0,1]³.
    color: [f32; C],
}

impl Rect {
    fn x0(&self) -> f32 { self.cx - self.hw }
    fn x1(&self) -> f32 { self.cx + self.hw }
    fn y0(&self) -> f32 { self.cy - self.hh }
    fn y1(&self) -> f32 { self.cy + self.hh }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn soft_mask_at(jx: f32, iy: f32, x0: f32, x1: f32, y0: f32, y1: f32) -> f32 {
    let l = sigmoid((jx - x0) / TAU);
    let r = sigmoid((x1 - jx) / TAU);
    let t = sigmoid((iy - y0) / TAU);
    let b = sigmoid((y1 - iy) / TAU);
    l * r * t * b
}

fn affected_bbox(r: &Rect, w: usize, h: usize) -> (usize, usize, usize, usize) {
    let pad = 4.0 * TAU + 1.0;
    let j0 = (r.x0() - pad).floor().max(0.0) as usize;
    let j1 = ((r.x1() + pad).ceil() as i64).max(0).min(w as i64) as usize;
    let i0 = (r.y0() - pad).floor().max(0.0) as usize;
    let i1 = ((r.y1() + pad).ceil() as i64).max(0).min(h as i64) as usize;
    (j0.min(w), i0.min(h), j1, i1)
}

/// Closed-form best RGB color for a rect of fixed geometry/α, by
/// minimizing per-pixel L2 of the post-composite canvas vs target.
/// Derivation:
///   `canvas_new[c] = canvas[c]·(1−w) + color[c]·w`,  `w = α·mask`
///   `∂/∂color[c] Σ (canvas_new[c] − target[c])² = 0`
///   `color[c] = (Σ w·target[c] + Σ (w² − w)·canvas[c]) / Σ w²`
/// Returns `None` if `Σw² ≈ 0` (rect doesn't actually cover anything).
fn best_color(canvas: &[f32], target: &[f32], w: usize, h: usize, r: &Rect)
    -> Option<[f32; C]>
{
    let (j0, i0, j1, i1) = affected_bbox(r, w, h);
    let (x0, x1, y0, y1) = (r.x0(), r.x1(), r.y0(), r.y1());
    let mut sum_ww = 0.0_f32;
    let mut num = [0.0_f32; C];
    for i in i0..i1 {
        let iy = i as f32 + 0.5;
        for j in j0..j1 {
            let jx = j as f32 + 0.5;
            let m = soft_mask_at(jx, iy, x0, x1, y0, y1);
            let ww = r.alpha * m;
            sum_ww += ww * ww;
            let pidx = idx_pixel(i, j, w);
            for c in 0..C {
                num[c] += ww * target[pidx + c]
                    + (ww * ww - ww) * canvas[pidx + c];
            }
        }
    }
    if sum_ww < 1e-6 { return None; }
    let mut col = [0.0_f32; C];
    for c in 0..C {
        col[c] = (num[c] / sum_ww).clamp(0.0, 1.0);
    }
    Some(col)
}

/// Δ in L1 image loss if `r` (with its current `color`) is composited
/// onto `canvas`. Negative = strict improvement. Touches only the
/// affected bbox.
fn delta_l1_loss(canvas: &[f32], target: &[f32], w: usize, h: usize, r: &Rect)
    -> f32
{
    let (j0, i0, j1, i1) = affected_bbox(r, w, h);
    let (x0, x1, y0, y1) = (r.x0(), r.x1(), r.y0(), r.y1());
    let mut delta = 0.0_f32;
    for i in i0..i1 {
        let iy = i as f32 + 0.5;
        for j in j0..j1 {
            let jx = j as f32 + 0.5;
            let m = soft_mask_at(jx, iy, x0, x1, y0, y1);
            let ww = r.alpha * m;
            let pidx = idx_pixel(i, j, w);
            for c in 0..C {
                let cv = canvas[pidx + c];
                let tv = target[pidx + c];
                let cn = cv + ww * (r.color[c] - cv);
                delta += (cn - tv).abs() - (cv - tv).abs();
            }
        }
    }
    delta
}

fn composite_over(canvas: &mut [f32], w: usize, h: usize, r: &Rect) {
    let (j0, i0, j1, i1) = affected_bbox(r, w, h);
    let (x0, x1, y0, y1) = (r.x0(), r.x1(), r.y0(), r.y1());
    for i in i0..i1 {
        let iy = i as f32 + 0.5;
        for j in j0..j1 {
            let jx = j as f32 + 0.5;
            let m = soft_mask_at(jx, iy, x0, x1, y0, y1);
            let ww = r.alpha * m;
            let pidx = idx_pixel(i, j, w);
            for c in 0..C {
                let cv = canvas[pidx + c];
                canvas[pidx + c] = cv + ww * (r.color[c] - cv);
            }
        }
    }
}

/// xorshift64 RNG.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed.max(1)) }
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 as u32
    }
    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
    fn signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

/// Sample a pixel index proportional to ‖canvas − target‖₁ (per-pixel,
/// summed over channels). Picks where the canvas needs help most.
fn sample_residual_weighted(
    canvas: &[f32],
    target: &[f32],
    w: usize,
    h: usize,
    rng: &mut Rng,
) -> usize {
    let mut total = 0.0_f32;
    let mut accept = 0usize;
    for p in 0..(w * h) {
        let mut r = 1e-3_f32;
        for c in 0..C {
            r += (canvas[p * C + c] - target[p * C + c]).abs();
        }
        total += r;
        if rng.next_f32() * total <= r {
            accept = p;
        }
    }
    let _ = total;
    accept
}

/// Random rect proposal — position biased toward residual; size + α
/// uniform. Color is *not* a free parameter; it gets set by
/// `best_color` after each geometry move.
fn random_proposal(
    canvas: &[f32],
    target: &[f32],
    w: usize,
    h: usize,
    rng: &mut Rng,
) -> Rect {
    let p = sample_residual_weighted(canvas, target, w, h, rng);
    let cy = (p / w) as f32 + rng.signed() * 1.5;
    let cx = (p % w) as f32 + rng.signed() * 1.5;
    let hw = HW_MIN_PX + rng.next_f32() * (HW_MAX_PX * 0.4 - HW_MIN_PX);
    let hh = HH_MIN_PX + rng.next_f32() * (HH_MAX_PX * 0.4 - HH_MIN_PX);
    let alpha = 0.4 + rng.next_f32() * 0.5;
    let mut r = Rect {
        cx: cx.clamp(0.0, w as f32),
        cy: cy.clamp(0.0, h as f32),
        hw, hh, alpha,
        color: [0.5, 0.5, 0.5],
    };
    if let Some(c) = best_color(canvas, target, w, h, &r) {
        r.color = c;
    }
    r
}

fn clamp_rect(r: &mut Rect, w: usize, h: usize) {
    r.cx = r.cx.clamp(0.0, w as f32);
    r.cy = r.cy.clamp(0.0, h as f32);
    r.hw = r.hw.clamp(HW_MIN_PX, HW_MAX_PX);
    r.hh = r.hh.clamp(HH_MIN_PX, HH_MAX_PX);
    r.alpha = r.alpha.clamp(0.05, 1.0);
}

/// Hill-climb: perturb one geometry/α param by ±δ (annealed), recompute
/// the closed-form best color, accept if Δloss improves.
fn hill_climb(
    canvas: &[f32],
    target: &[f32],
    w: usize,
    h: usize,
    init: Rect,
    rng: &mut Rng,
) -> (Rect, f32) {
    let mut best = init;
    clamp_rect(&mut best, w, h);
    if let Some(c) = best_color(canvas, target, w, h, &best) {
        best.color = c;
    }
    let mut best_delta = delta_l1_loss(canvas, target, w, h, &best);

    for step in 0..HILL_STEPS {
        let progress = step as f32 / HILL_STEPS as f32;
        let pos_amp = 4.0 * (1.0 - progress) + 0.4 * progress;
        let size_amp = 3.0 * (1.0 - progress) + 0.3 * progress;
        let alpha_amp = 0.2 * (1.0 - progress) + 0.02 * progress;

        // Perturb one of the 5 geometry/α params (color is closed-form).
        let which = rng.next_u32() as usize % 5;
        let mut trial = best;
        match which {
            0 => trial.cx += rng.signed() * pos_amp,
            1 => trial.cy += rng.signed() * pos_amp,
            2 => trial.hw += rng.signed() * size_amp,
            3 => trial.hh += rng.signed() * size_amp,
            _ => trial.alpha += rng.signed() * alpha_amp,
        }
        clamp_rect(&mut trial, w, h);
        if let Some(c) = best_color(canvas, target, w, h, &trial) {
            trial.color = c;
        }
        let trial_delta = delta_l1_loss(canvas, target, w, h, &trial);
        if trial_delta < best_delta {
            best = trial;
            best_delta = trial_delta;
        }
    }
    (best, best_delta)
}

fn fit_one_rect(
    canvas: &[f32],
    target: &[f32],
    w: usize,
    h: usize,
    rng: &mut Rng,
) -> (Rect, f32) {
    let mut best: Option<(Rect, f32)> = None;
    for _ in 0..K_CANDIDATES {
        let prop = random_proposal(canvas, target, w, h, rng);
        let (refined, delta) = hill_climb(canvas, target, w, h, prop, rng);
        match best {
            None => best = Some((refined, delta)),
            Some((_, d_best)) if delta < d_best => best = Some((refined, delta)),
            _ => {}
        }
    }
    best.expect("K_CANDIDATES > 0")
}

/// Render the canvas from scratch by compositing every rect except
/// optionally one (`skip`). Used by the polish pass to get a
/// "without rect k" baseline cheaply — total cost is the sum of all
/// rect bboxes, not O(N·W·H).
fn replay_canvas(
    rects: &[Rect],
    skip: Option<usize>,
    w: usize,
    h: usize,
    base_color: [f32; C],
) -> Vec<f32> {
    let mut canvas = vec![0.0f32; w * h * C];
    for p in canvas.chunks_exact_mut(C) {
        p.copy_from_slice(&base_color);
    }
    for (k, r) in rects.iter().enumerate() {
        if Some(k) == skip { continue; }
        composite_over(&mut canvas, w, h, r);
    }
    canvas
}

/// One polish sweep — coordinate descent over rects.
///
/// Subtle point: Porter-Duff `over` doesn't commute, so a rect's
/// effective contribution depends on what's *above* it (subsequent
/// rects can paint over its ink) as well as below. To keep the
/// refit math consistent (`hill_climb` measures Δloss assuming the
/// rect is composited *last*, i.e. on top of the baseline), we
/// process rects through a queue: pop the front, refit against the
/// rest of the queue as baseline, push the refitted rect to the
/// back. After one pass every rect has been refitted "as last"
/// against a baseline that contained every other rect — and the
/// final canvas, replayed in queue order, *is* the loss the
/// hill-climbs were optimizing.
fn polish_pass(
    rects: &mut Vec<Rect>,
    target: &[f32],
    w: usize,
    h: usize,
    base_color: [f32; C],
    rng: &mut Rng,
) -> f32 {
    use std::collections::VecDeque;
    let mut queue: VecDeque<Rect> = rects.iter().copied().collect();
    let n = queue.len();
    let mut total_delta = 0.0f32;
    for _ in 0..n {
        let r = queue.pop_front().unwrap();
        let remaining: Vec<Rect> = queue.iter().copied().collect();
        let baseline = replay_canvas(&remaining, None, w, h, base_color);
        let init_delta = delta_l1_loss(&baseline, target, w, h, &r);
        let (refined, refined_delta) = hill_climb(&baseline, target, w, h, r, rng);
        // hill_climb only accepts trials with delta < best_delta, where
        // best_delta starts at init_delta. So refined_delta ≤ init_delta;
        // strict inequality only when something improved. Either way
        // the queue retains the rect — `refined == r` when the rect was
        // already locally optimal.
        queue.push_back(refined);
        total_delta += refined_delta - init_delta;
    }
    rects.clear();
    rects.extend(queue);
    total_delta
}

// ── rlx-AD demo: refine a single (luma) rect via reverse-mode AD ────
//
// AD demo runs against the *luminance* projection of the RGB canvas
// — the rlx graph is built once for a single channel so the first-rect
// witness stays a small graph with no [N,H,W,3] tensor. Color for
// this rect is then re-fit by the closed-form `best_color` after
// the geometry settles.

fn rlx_refine_one_rect(
    canvas_luma: &[f32],
    target_luma: &[f32],
    w: usize,
    h: usize,
    init: Rect,
) -> (Rect, f32, usize) {
    let img_2d = TShape::new(&[h, w], DType::F32);
    let scalar = TShape::new(&[1], DType::F32);

    let mut g = Graph::new("rlx_one_rect_luma");

    let xx_data: Vec<f32> = (0..h)
        .flat_map(|_| (0..w).map(|j| j as f32 + 0.5))
        .collect();
    let yy_data: Vec<f32> = (0..h)
        .flat_map(|i| (0..w).map(move |_| i as f32 + 0.5))
        .collect();
    let xx = g.add_node(
        Op::Constant { data: f32_bytes(&xx_data) },
        vec![],
        img_2d.clone(),
    );
    let yy = g.add_node(
        Op::Constant { data: f32_bytes(&yy_data) },
        vec![],
        img_2d.clone(),
    );
    let canvas_in = g.input("canvas", img_2d.clone());
    let target_in = g.input("target", img_2d.clone());

    let cx_p = g.param("cx", scalar.clone());
    let cy_p = g.param("cy", scalar.clone());
    let hw_p = g.param("hw", scalar.clone());
    let hh_p = g.param("hh", scalar.clone());
    let al_p = g.param("alpha", scalar.clone());

    let hw_abs = g.activation(Activation::Abs, hw_p, scalar.clone());
    let hh_abs = g.activation(Activation::Abs, hh_p, scalar.clone());
    let al_sig = g.activation(Activation::Sigmoid, al_p, scalar.clone());

    let x0 = g.binary(BinaryOp::Sub, cx_p, hw_abs, scalar.clone());
    let x1 = g.binary(BinaryOp::Add, cx_p, hw_abs, scalar.clone());
    let y0 = g.binary(BinaryOp::Sub, cy_p, hh_abs, scalar.clone());
    let y1 = g.binary(BinaryOp::Add, cy_p, hh_abs, scalar.clone());

    let inv_tau = g.add_node(
        Op::Constant { data: f32_bytes(&[1.0 / TAU]) },
        vec![],
        scalar.clone(),
    );
    let edge =
        |g: &mut Graph, coord: NodeId, bound: NodeId, sign: bool| -> NodeId {
            let diff = if sign {
                g.binary(BinaryOp::Sub, coord, bound, img_2d.clone())
            } else {
                g.binary(BinaryOp::Sub, bound, coord, img_2d.clone())
            };
            let scaled = g.binary(BinaryOp::Mul, diff, inv_tau, img_2d.clone());
            g.activation(Activation::Sigmoid, scaled, img_2d.clone())
        };
    let l = edge(&mut g, xx, x0, true);
    let r = edge(&mut g, xx, x1, false);
    let t = edge(&mut g, yy, y0, true);
    let b = edge(&mut g, yy, y1, false);
    let lr = g.binary(BinaryOp::Mul, l, r, img_2d.clone());
    let tb = g.binary(BinaryOp::Mul, t, b, img_2d.clone());
    let mask = g.binary(BinaryOp::Mul, lr, tb, img_2d.clone());

    // Rect's luma color held as a Constant — best-color closed form
    // gives it analytically each iteration; in this AD demo we just
    // use 1.0 (full ink) since we're proving differentiability of
    // *geometry*, not color.
    // canvas_new = canvas + α·mask·(ink − canvas).  ink = 1.
    let one_img = g.add_node(
        Op::Constant { data: f32_bytes(&vec![1.0; w * h]) },
        vec![],
        img_2d.clone(),
    );
    let one_minus_c = g.binary(BinaryOp::Sub, one_img, canvas_in, img_2d.clone());
    let m_times = g.binary(BinaryOp::Mul, mask, one_minus_c, img_2d.clone());
    let weighted = g.binary(BinaryOp::Mul, m_times, al_sig, img_2d.clone());
    let canvas_new = g.binary(BinaryOp::Add, canvas_in, weighted, img_2d.clone());

    let diff = g.binary(BinaryOp::Sub, canvas_new, target_in, img_2d.clone());
    let sq = g.binary(BinaryOp::Mul, diff, diff, img_2d.clone());
    let loss = g.reduce(sq, ReduceOp::Sum, vec![0, 1], false, scalar.clone());
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[cx_p, cy_p, hw_p, hh_p, al_p]);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    let mut params = [
        init.cx, init.cy, init.hw, init.hh,
        (init.alpha.clamp(1e-3, 0.999) / (1.0 - init.alpha.clamp(1e-3, 0.999))).ln(),
    ];
    let mut adam = Adam::new(0.05, 5);

    const N_ITER: usize = 200;
    let mut retries = 0u32;
    let mut step = 0usize;
    while step < N_ITER {
        sess.set_param("cx", &params[0..1]);
        sess.set_param("cy", &params[1..2]);
        sess.set_param("hw", &params[2..3]);
        sess.set_param("hh", &params[3..4]);
        sess.set_param("alpha", &params[4..5]);
        let outs = sess.run(&[
            ("d_output", &[1.0_f32][..]),
            ("canvas", canvas_luma),
            ("target", target_luma),
        ]);
        let last_loss = outs[0][0];
        if !last_loss.is_finite() {
            if retries < 8 { retries += 1; continue; }
            break;
        }
        let grads: [f32; 5] = [
            outs[1][0], outs[2][0], outs[3][0], outs[4][0], outs[5][0],
        ];
        adam.step(&mut params, &grads);
        params[2] = params[2].clamp(-HW_MAX_PX, HW_MAX_PX);
        params[3] = params[3].clamp(-HH_MAX_PX, HH_MAX_PX);
        params[0] = params[0].clamp(0.0, w as f32);
        params[1] = params[1].clamp(0.0, h as f32);
        step += 1;
    }

    let r = Rect {
        cx: params[0],
        cy: params[1],
        hw: params[2].abs().clamp(HW_MIN_PX, HW_MAX_PX),
        hh: params[3].abs().clamp(HH_MIN_PX, HH_MAX_PX),
        alpha: sigmoid(params[4]),
        color: [0.5; C], // overridden below by best_color
    };
    (r, 0.0, step)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn rgb_to_luma(rgb: &[f32]) -> Vec<f32> {
    rgb.chunks_exact(C)
        .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
        .collect()
}

/// Box-downsample the RGB image to a smaller luma grid (for the
/// rlx-AD demo, which runs at lower resolution to stay clear of the
/// rlx CPU executor's shape-dependent SIGSEGV regime).
fn downsample_luma(
    rgb: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; dst_w * dst_h];
    let sx = src_w as f32 / dst_w as f32;
    let sy = src_h as f32 / dst_h as f32;
    for di in 0..dst_h {
        for dj in 0..dst_w {
            let i0 = (di as f32 * sy) as usize;
            let i1 = ((di + 1) as f32 * sy).ceil() as usize;
            let j0 = (dj as f32 * sx) as usize;
            let j1 = ((dj + 1) as f32 * sx).ceil() as usize;
            let mut acc = 0.0f32;
            let mut count = 0;
            for i in i0..i1.min(src_h) {
                for j in j0..j1.min(src_w) {
                    let p = (i * src_w + j) * C;
                    acc += 0.2126 * rgb[p] + 0.7152 * rgb[p + 1] + 0.0722 * rgb[p + 2];
                    count += 1;
                }
            }
            out[di * dst_w + dj] = acc / count.max(1) as f32;
        }
    }
    out
}

// ── Output ─────────────────────────────────────────────────────────

/// K-means cluster the rect colors into `K` clusters in RGB space,
/// returning each rect's cluster index. Used to map rects to PDK
/// layers so the chip layout shows up in multiple colors when loaded
/// into KLayout (rather than one big metal1 stack).
fn kmeans_assign<const K: usize>(rects: &[Rect], iters: usize, rng: &mut Rng) -> Vec<usize> {
    let n = rects.len();
    // Init centers from random rects.
    let mut centers: [[f32; C]; K] = [[0.0; C]; K];
    for k in 0..K {
        let pick = rng.next_u32() as usize % n;
        centers[k] = rects[pick].color;
    }
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        // Assignment step.
        for (i, r) in rects.iter().enumerate() {
            let mut best = 0;
            let mut best_d = f32::INFINITY;
            for (k, c) in centers.iter().enumerate() {
                let d = (r.color[0] - c[0]).powi(2)
                    + (r.color[1] - c[1]).powi(2)
                    + (r.color[2] - c[2]).powi(2);
                if d < best_d { best_d = d; best = k; }
            }
            assign[i] = best;
        }
        // Update step.
        let mut sums: [[f32; C]; K] = [[0.0; C]; K];
        let mut counts: [u32; K] = [0; K];
        for (i, r) in rects.iter().enumerate() {
            let k = assign[i];
            for c in 0..C { sums[k][c] += r.color[c]; }
            counts[k] += 1;
        }
        for k in 0..K {
            if counts[k] == 0 {
                // Re-seed empty clusters from a random rect.
                let pick = rng.next_u32() as usize % n;
                centers[k] = rects[pick].color;
            } else {
                for c in 0..C { centers[k][c] = sums[k][c] / counts[k] as f32; }
            }
        }
    }
    // Sort clusters by perceived luma so the layer assignment is
    // deterministic + interpretable (darkest cluster → layer 0,
    // lightest → layer K-1). Rebuild assignment with sorted indices.
    let mut order: Vec<usize> = (0..K).collect();
    order.sort_by(|&a, &b| {
        let la = 0.2126 * centers[a][0] + 0.7152 * centers[a][1] + 0.0722 * centers[a][2];
        let lb = 0.2126 * centers[b][0] + 0.7152 * centers[b][1] + 0.0722 * centers[b][2];
        la.partial_cmp(&lb).unwrap()
    });
    let mut remap = [0usize; K];
    for (new, &old) in order.iter().enumerate() { remap[old] = new; }
    for a in &mut assign { *a = remap[*a]; }
    assign
}

/// Number of sky130 layers we paint into. Sky130Lite exposes 7 layers
/// (POLY, MET1, LICON1, DIFF, NWELL, NSDM, PSDM); we use all of them
/// so KLayout's natural per-layer coloring renders the beaver in
/// multiple colors instead of one solid metal1 mass.
const N_LAYERS: usize = 7;

fn build_layout_cell(
    lib: &Library,
    pdk: &Sky130Lite,
    rects: &[Rect],
    raster_h: usize,
    assign: &[usize],
) -> CellId {
    let mut cb = CellBuilder::new("BeaverOptim".to_string());
    // Layer ordering matches `kmeans_assign`'s luma-sorted output:
    // index 0 = darkest cluster, index N_LAYERS-1 = lightest.
    let layers: [_; N_LAYERS] = [
        pdk.MET1,    // dark — beaver fur, MIT text
        pdk.POLY,    // mid-dark
        pdk.LICON1,  // mid
        pdk.DIFF,    //
        pdk.NWELL,   //
        pdk.PSDM,    //
        pdk.NSDM,    // light — shirt, highlights
    ];
    let h_dbu = (raster_h as i64) * DBU_PER_PIXEL;
    for (i, r) in rects.iter().enumerate() {
        let xa = (r.x0() * DBU_PER_PIXEL as f32) as i64;
        let xb = (r.x1() * DBU_PER_PIXEL as f32) as i64;
        let ya = h_dbu - (r.y1() * DBU_PER_PIXEL as f32) as i64;
        let yb = h_dbu - (r.y0() * DBU_PER_PIXEL as f32) as i64;
        if xb <= xa || yb <= ya { continue; }
        let lyr = layers[assign[i].min(N_LAYERS - 1)];
        cb.add_shape(
            lyr,
            Shape::Box(KRect::new(Bbox::new(
                Point::new(xa, ya),
                Point::new(xb, yb),
            ))),
        );
    }
    lib.insert(cb)
}

/// Write a colored SVG of the rect stack — each rect rendered at
/// its own RGB color. The standard `eda_viz::layout::write_svg`
/// path uses the metal1 layer color uniformly (correct for chip
/// rendering, but it makes the beaver invisible). This writer is
/// used alongside the layer-correct render so reviewers can see
/// what the optimizer actually picked.
fn write_colored_svg(
    path: &Path,
    rects: &[Rect],
    raster_w: usize,
    raster_h: usize,
) -> std::io::Result<()> {
    let pad = 4.0;
    let w = raster_w as f32;
    let h = raster_h as f32;
    let mut s = String::new();
    s.push_str(&format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <svg xmlns=\"http://www.w3.org/2000/svg\" \
         viewBox=\"{:.2} {:.2} {:.2} {:.2}\" width=\"{}\" height=\"{}\">\n",
        -pad,
        -pad,
        w + 2.0 * pad,
        h + 2.0 * pad,
        (w + 2.0 * pad) as i32 * 4,
        (h + 2.0 * pad) as i32 * 4,
    ));
    s.push_str(&format!(
        "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"white\" stroke=\"none\"/>\n",
        -pad, -pad, w + 2.0 * pad, h + 2.0 * pad,
    ));
    for r in rects {
        let x0 = r.x0().clamp(0.0, w);
        let x1 = r.x1().clamp(0.0, w);
        let y0 = r.y0().clamp(0.0, h);
        let y1 = r.y1().clamp(0.0, h);
        if x1 <= x0 || y1 <= y0 { continue; }
        let r8 = (r.color[0].clamp(0.0, 1.0) * 255.0) as u8;
        let g8 = (r.color[1].clamp(0.0, 1.0) * 255.0) as u8;
        let b8 = (r.color[2].clamp(0.0, 1.0) * 255.0) as u8;
        s.push_str(&format!(
            "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" \
             fill=\"#{r8:02x}{g8:02x}{b8:02x}\" fill-opacity=\"{:.3}\" stroke=\"none\"/>\n",
            x0, y0, x1 - x0, y1 - y0, r.alpha,
        ));
    }
    s.push_str("</svg>\n");
    fs::write(path, s)
}

/// Hard-edge composite for the layout panel — what the rectangles
/// actually look like once edges aren't smoothed.
fn rasterize_hard(rects: &[Rect], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![1.0_f32; w * h * C]; // white background
    for r in rects {
        let i0 = (r.y0().floor() as i64).max(0) as usize;
        let i1 = ((r.y1().ceil() as i64).max(0) as usize).min(h);
        let j0 = (r.x0().floor() as i64).max(0) as usize;
        let j1 = ((r.x1().ceil() as i64).max(0) as usize).min(w);
        for i in i0..i1 {
            for j in j0..j1 {
                let pidx = idx_pixel(i, j, w);
                for c in 0..C {
                    let cv = out[pidx + c];
                    out[pidx + c] = cv + r.alpha * (r.color[c] - cv);
                }
            }
        }
    }
    out
}

fn out_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .join("target/floorplans/beaver_optim")
}

fn main() {
    // Big stack — same belt-and-braces from the v1 binary; the rlx-ad
    // demo path stack-allocates intermediates that overflow the
    // default 8 MB at this raster size.
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run)
        .expect("spawn worker")
        .join()
        .expect("worker panic");
}

fn run() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf();
    let png_path = workspace.join("logos/beaver.png");
    println!(
        "Loading {png_path:?} → {RASTER_W}×{RASTER_H} RGB"
    );
    let target = load_target(&png_path, RASTER_W, RASTER_H);
    let target_mean: [f32; C] = {
        let mut m = [0.0_f32; C];
        for p in target.chunks_exact(C) {
            for c in 0..C { m[c] += p[c]; }
        }
        let n = (RASTER_W * RASTER_H) as f32;
        for c in 0..C { m[c] /= n; }
        m
    };
    println!(
        "  target mean RGB: ({:.3}, {:.3}, {:.3})",
        target_mean[0], target_mean[1], target_mean[2]
    );

    // Canvas init = target mean. Residuals start zero-mean — every
    // rect has to actively pull a region away from the average toward
    // its true color.
    let mut canvas = vec![0.0f32; RASTER_W * RASTER_H * C];
    for p in canvas.chunks_exact_mut(C) {
        p.copy_from_slice(&target_mean);
    }
    let initial_l1 = l1(&canvas, &target);
    println!(
        "Greedy fit: N={N_RECTS} rects, K={K_CANDIDATES} candidates, hill {HILL_STEPS} steps. Initial L1 = {initial_l1:.2}"
    );

    let mut rects: Vec<Rect> = Vec::with_capacity(N_RECTS);
    let mut rng = Rng::new(0xdead_beef_1234_5678);
    let mut csv: Vec<String> = vec![
        "idx,delta,running_l1,cx,cy,hw,hh,alpha,r,g,b".to_string()
    ];
    let t_start = std::time::Instant::now();

    // Demo: rlx-AD refine on a *downsampled luminance* projection for
    // the first rect — proves the AD path still differentiates a
    // real-image fitting problem end-to-end. Done at a small fixed
    // raster (48 × 72) so the rlx CPU executor's thunk allocator
    // doesn't trip the same out-of-bounds memmove that bit us at
    // [N, H, W] = [200, 96, 64].
    {
        const AD_W: usize = 48;
        const AD_H: usize = 72;
        let target_small = downsample_luma(&target, RASTER_W, RASTER_H, AD_W, AD_H);
        let canvas_small = vec![
            target_small.iter().sum::<f32>() / target_small.len() as f32;
            AD_W * AD_H
        ];
        let scale_x = AD_W as f32 / RASTER_W as f32;
        let scale_y = AD_H as f32 / RASTER_H as f32;
        let prop_small = Rect {
            cx: 0.5 * AD_W as f32,
            cy: 0.5 * AD_H as f32,
            hw: 0.25 * AD_W as f32,
            hh: 0.25 * AD_H as f32,
            alpha: 0.6,
            color: [0.5; C],
        };
        let t_ad = std::time::Instant::now();
        let (rect_small, _l, iters) =
            rlx_refine_one_rect(&canvas_small, &target_small, AD_W, AD_H, prop_small);
        let ad_ms = t_ad.elapsed().as_secs_f32() * 1000.0;
        // Lift back to full-res and refit color on RGB.
        let mut rect = Rect {
            cx: rect_small.cx / scale_x,
            cy: rect_small.cy / scale_y,
            hw: rect_small.hw / scale_x,
            hh: rect_small.hh / scale_y,
            alpha: rect_small.alpha,
            color: [0.5; C],
        };
        clamp_rect(&mut rect, RASTER_W, RASTER_H);
        if let Some(c) = best_color(&canvas, &target, RASTER_W, RASTER_H, &rect) {
            rect.color = c;
        }
        let delta = delta_l1_loss(&canvas, &target, RASTER_W, RASTER_H, &rect);
        println!(
            "  rect   0  [rlx-ad refine @ {AD_W}×{AD_H}, {iters:3} iters, {ad_ms:.1} ms]  Δ={delta:+.3}  rgb=({:.2},{:.2},{:.2})  α={:.2}",
            rect.color[0], rect.color[1], rect.color[2], rect.alpha
        );
        composite_over(&mut canvas, RASTER_W, RASTER_H, &rect);
        let running = l1(&canvas, &target);
        csv.push(format!(
            "0,{delta:.4},{running:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            rect.cx, rect.cy, rect.hw, rect.hh, rect.alpha,
            rect.color[0], rect.color[1], rect.color[2]
        ));
        rects.push(rect);
    }

    for n in 1..N_RECTS {
        let (rect, delta) = fit_one_rect(&canvas, &target, RASTER_W, RASTER_H, &mut rng);
        if delta >= 0.0 {
            println!(
                "  rect {n:3}  no improvement over {K_CANDIDATES} candidates — stopping early"
            );
            break;
        }
        composite_over(&mut canvas, RASTER_W, RASTER_H, &rect);
        rects.push(rect);
        let running = l1(&canvas, &target);
        csv.push(format!(
            "{n},{delta:.4},{running:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            rect.cx, rect.cy, rect.hw, rect.hh, rect.alpha,
            rect.color[0], rect.color[1], rect.color[2]
        ));
        if n % 25 == 0 || n == N_RECTS - 1 {
            println!(
                "  rect {n:3}  Δ={delta:+.3}  L1={running:8.2}  ({:.1}% reduction)",
                100.0 * (1.0 - running / initial_l1.max(1.0))
            );
        }
    }

    let after_greedy = t_start.elapsed();
    let l1_after_greedy = l1(&canvas, &target);
    println!(
        "Greedy phase done in {:.2}s. L1 {initial_l1:.1} → {l1_after_greedy:.1} ({:.1}% reduction).",
        after_greedy.as_secs_f32(),
        100.0 * (1.0 - l1_after_greedy / initial_l1.max(1.0))
    );

    // ── Polish pass (coordinate descent over rects) ─────────────────
    for pass in 0..N_POLISH_PASSES {
        let t_pass = std::time::Instant::now();
        let _delta = polish_pass(&mut rects, &target, RASTER_W, RASTER_H, target_mean, &mut rng);
        // Replay the canvas to reflect the refined rects.
        canvas = replay_canvas(&rects, None, RASTER_W, RASTER_H, target_mean);
        let l1_now = l1(&canvas, &target);
        println!(
            "Polish pass {} ({:.2}s):  L1 → {l1_now:.1}  ({:.1}% total reduction)",
            pass + 1,
            t_pass.elapsed().as_secs_f32(),
            100.0 * (1.0 - l1_now / initial_l1.max(1.0))
        );
    }
    let elapsed = t_start.elapsed();
    let final_l1 = l1(&canvas, &target);
    println!(
        "Done in {:.2}s total. {} rects placed. L1 {initial_l1:.1} → {final_l1:.1} ({:.1}% reduction).",
        elapsed.as_secs_f32(),
        rects.len(),
        100.0 * (1.0 - final_l1 / initial_l1.max(1.0))
    );

    // ── Outputs ───────────────────────────────────────────────────
    let out_dir = out_root();
    fs::create_dir_all(&out_dir).expect("create out dir");

    let hard_raster = rasterize_hard(&rects, RASTER_W, RASTER_H);
    save_triptych(
        &out_dir.join("convergence.png"),
        RASTER_W, RASTER_H,
        &target, &canvas, &hard_raster,
    );
    save_rgb(&out_dir.join("target.png"), RASTER_W, RASTER_H, &target);
    save_rgb(&out_dir.join("rasterized.png"), RASTER_W, RASTER_H, &canvas);

    fs::write(out_dir.join("loss.csv"), csv.join("\n") + "\n").expect("csv");

    // Quantize each rect's RGB color into one of N_LAYERS clusters
    // (k-means in RGB), then map clusters to sky130 layers in luma
    // order. The layout is multi-layer — each rect lives on the
    // sky130 layer corresponding to its color cluster — so KLayout
    // renders the chip in multiple colors and the GDS carries the
    // layered structure rather than collapsing to one big metal1
    // pour.
    let layer_assign = kmeans_assign::<N_LAYERS>(&rects, 20, &mut rng);
    let layer_counts: [usize; N_LAYERS] = {
        let mut c = [0usize; N_LAYERS];
        for &a in &layer_assign { c[a] += 1; }
        c
    };
    println!(
        "Layer assignment (luma-sorted, dark→light): {:?}",
        layer_counts
    );

    let lib = Sky130Lite::new_library("beaver_optim");
    let pdk = Sky130Lite::register(&lib);
    let top = build_layout_cell(&lib, &pdk, &rects, RASTER_H, &layer_assign);
    let mut style = Style::default();
    style.show_instance_labels = false;
    style.show_ports = false;
    style.show_legend = true;
    style.units_per_dbu = 0.01;
    // Per-sky130-layer rendering — each layer gets its own palette
    // color so the chip view IS multi-colored. Same rectangles as
    // the colored-by-rect SVG, just colored by which layer the
    // k-means assigned them to.
    eda_viz::layout::write_svg(
        &lib,
        top,
        &style,
        &out_dir.join("floorplan_layers.svg"),
    )
    .expect("svg");
    // Colored view — same rect geometry, painted with each rect's
    // RGB color so the layout reads as a *picture* of the beaver
    // (humans care about this; KLayout doesn't). This is the "see
    // what was fitted" file readers will look at first.
    write_colored_svg(
        &out_dir.join("floorplan.svg"),
        &rects,
        RASTER_W,
        RASTER_H,
    )
    .expect("colored svg");
    klayout_io::write_gds_path(
        &lib,
        out_dir.join("floorplan.gds").to_str().expect("utf-8"),
    )
    .map_err(|e| std::io::Error::other(format!("gds: {e}")))
    .expect("gds");

    let summary = format!(
        "raster        : {RASTER_W} × {RASTER_H} px (RGB)\n\
         rects placed  : {}  (of {N_RECTS} budget)\n\
         K candidates  : {K_CANDIDATES}\n\
         hill steps    : {HILL_STEPS}\n\
         τ (sigmoid)   : {TAU}\n\
         L1 initial    : {initial_l1:.3}\n\
         L1 final      : {final_l1:.3}\n\
         L1 reduction  : {:.1}%\n\
         wall time     : {:.2}s\n",
        rects.len(),
        100.0 * (1.0 - final_l1 / initial_l1.max(1.0)),
        elapsed.as_secs_f32(),
    );
    fs::write(out_dir.join("summary.txt"), &summary).expect("summary");
    println!("\n{summary}");
}

fn l1(canvas: &[f32], target: &[f32]) -> f32 {
    canvas
        .iter()
        .zip(target)
        .map(|(c, t)| (c - t).abs())
        .sum()
}
