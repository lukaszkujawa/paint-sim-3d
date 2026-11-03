use crate::brush::BrushShape;
use crate::color::{self, PigmentSet};
use crate::film::{Deposit, Method};
use crate::tools::ToolKind;
use crate::App;
use glam::{DVec2, DVec3};
use image::ImageDecoder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

const EVAL_WIDTH: u32 = 1536;
const EDGE_WEIGHT: f32 = 2.0;
const APPROACH_HW: f64 = 3.0;
const APPROACH_MIN: f64 = 0.004;
const APPROACH_MAX: f64 = 0.03;
const HOVER: f64 = 0.004;
const PRESS_PER_SIZE: f64 = 1.0;
const PRESS_MIN: f64 = 0.0005;
const APPROACH_STEPS: f64 = 4.0;
const TRAVEL_PER_SIZE: f64 = 1.0;
const REFINEMENTS: usize = 16;
const TILE: usize = 8;
const MIN_HW: f64 = 0.75;
const STOPS: [f32; 4] = [f32::INFINITY, 30.0, 18.0, 10.0];
const LOOKAHEAD: usize = 160;
const EXECUTE: usize = 48;
const PRUNE_VISIBLE: f32 = 0.25;
const MISS_LIMIT: usize = 24;
const TABU: f32 = 0.5;
const SHARPEN_MM: f32 = 1.5;
const REF_SIGMA: f32 = 0.5;
const REF_RANGE: f32 = 12.0;
const REF_RANGE_WIDE: f32 = 18.0;
const REF_RANGE_HW: f32 = 40.0;
const T_MIN: f32 = 5.0;
const STROKE_COST: f32 = 2.0;
const SEED_QUANTILE: f32 = 0.88;
const MIN_BLOB: usize = 3;
const BLOB_WINDOW_HW: f32 = 8.0;
const BLOB_WINDOW_MIN: usize = 24;
const BLOB_WINDOW_MAX: usize = 512;
const BLOBS_KEEP: usize = 4000;
const REBLOB: usize = LOOKAHEAD;
const T_FOCUS: f32 = 3.0;
const TILE_EFF_RATE: f32 = 0.3;
const VISIT_DECAY: f32 = 0.3;
const BLOB_TRIES: usize = 3;
const ISO_STOP_DE: f32 = 22.0;
const STOP_FINE: f32 = 0.5;
const ISO_MIN_STEPS: usize = 2;
const RIDGE_LAMBDA: f32 = 0.8;
const RIDGE_FAN: [f32; 5] = [-0.5, -0.25, 0.0, 0.25, 0.5];
const EMPTY_BATCHES: u32 = 3;
const RETRY_SECS: f32 = 3.0;
const PRESS_START: f64 = 0.35;
const PRESS_RAMP_SIZES: f64 = 2.0;
const PRESS_JITTER: f64 = 0.15;
const EFFICIENCY_RATE: f32 = 0.1;
const STRUCT_SIZE_LO: f64 = 0.7;
const QUANTILE_STRIDE: usize = 7;
const KNIFE_LOAD_MARGIN: f64 = 0.8;
const KNIFE_STREAK_LIGHT: f32 = 0.25;
const KNIFE_STREAK_DARK: f32 = 0.05;
const SMOOTH_RANGE: f32 = 0.1;
const BILATERAL_PASSES: usize = 2;
const SEGMENT_SMOOTH_PX: f32 = 3.0;
const SEGMENT_RANGE: f32 = 10.0;
const SEGMENT_STRIDE: usize = 13;
const REGION_COUNT: usize = 16;
const KMEANS_ROUNDS: usize = 12;
const MODE_PASSES: usize = 2;
const MODE_RADIUS: usize = 2;
const OUTSIDE_DISCOUNT: f32 = 0.85;
const KIN_DE: f32 = 15.0;
const TAPER_MIN: f32 = 0.45;
const TAPER_END: f32 = 0.2;
const PREP_SETTLE: f32 = 0.35;
const FLOW_JITTER: f32 = 0.8;
const MEDIAN_COVER: f32 = 0.5;
const GATE: f32 = 0.9;
const GATE_OPEN_DE: f32 = 6.0;
const GATE_FULL_DE: f32 = 10.0;
const DETAIL_RADIUS: usize = 6;
const ETF_RADIUS: usize = 10;
const ETF_ITERS: usize = 3;
const ETF_PASS_ITERS: usize = 2;
const ETF_ETA: f32 = 4.0;
const ETF_SEED_EDGE: f32 = 0.2;
const ETF_REACH: f32 = 10.0;
const LEN_WIDTHS: f32 = 20.0;
const RIDGE_MIN_HW: f32 = 8.0;
const BRISTLE_PER_SIZE: f64 = 5.0;
const BRISTLE_MIN: f64 = 0.006;
const SIZE_SPREAD: f64 = 0.75;
const SIZE_CHANGED: f64 = 0.01;
const MIN_GAIN: f32 = 2.0;
const CHANGED_DE: f32 = 4.0;
const LOWER_FRAMES: u32 = 1;
const WARMUP_FRAMES: u32 = 3;
const LIFT_FRAMES: u32 = 1;
const SETTLE_FRAMES: u32 = 2;
const THREAD_SETTLE_MAX: u32 = 180;
const POUR_FILM: f64 = 2.0e-4;
const PREVIEW_W: u32 = 288;
const EDGE_PX: f32 = 2.0;
const TRACE_TURN: f32 = 0.3;
const TRACE_COHERENT: f32 = 0.1;
const MAX_POINTS: f32 = 96.0;
const TENSOR_RADIUS: usize = 4;
const CHROMA_WEIGHT: f32 = 0.7;
const WHITE: usize = 0;
const BLACK: usize = 1;
const GAMUT_SLACK: f32 = 3.0;
const GAMUT_BIN: f32 = 1.5;
const TONE_LOW: f32 = 0.01;
const TONE_HIGH: f32 = 0.995;
const TONE_MIN_SPAN: f32 = 20.0;
const TONE_GAMMA: f32 = 1.3;
const LOAD_SIZE: f64 = 0.016;
const STRIDE_HW: f32 = 6.0;
const LOAD_MAX: f64 = 10.0;

pub struct Photo {
    pub name: String,
    pub image: image::RgbaImage,
    pub texture: Option<egui::TextureHandle>,
    shown: Option<Prep>,
}

#[derive(Clone, Copy, PartialEq)]
pub struct Prep {
    pub contrast: f32,
    pub blur_mm: f32,
    pub sharpen: f32,
    pub smooth_mm: f32,
    pub lift_ev: f32,
}

impl Prep {
    pub fn none() -> Self {
        Self { contrast: 1.0, blur_mm: 0.0, sharpen: 0.0, smooth_mm: 0.0, lift_ev: 0.0 }
    }

    fn is_none(&self) -> bool {
        *self == Self::none()
    }
}

fn bilateral(src: &[[f32; 3]], w: usize, h: usize, sigma_px: f32, range: f32) -> Vec<[f32; 3]> {
    if sigma_px < 0.5 {
        return src.to_vec();
    }
    let reach = (sigma_px * 2.0).ceil() as isize;
    let kernel: Vec<f32> = (-reach..=reach).map(|i| (-(i * i) as f32 / (2.0 * sigma_px * sigma_px)).exp()).collect();
    let inv = 1.0 / (2.0 * range * range);
    let mut cur = src.to_vec();
    for _ in 0..BILATERAL_PASSES {
        for horizontal in [true, false] {
            let prev = cur;
            cur = (0..w * h)
                .into_par_iter()
                .map(|k| {
                    let (x, y) = ((k % w) as isize, (k / w) as isize);
                    let c = prev[k];
                    let (mut sum, mut tot) = ([0.0f32; 3], 0.0f32);
                    for (o, kw) in kernel.iter().enumerate() {
                        let d = o as isize - reach;
                        let (sx, sy) = if horizontal { ((x + d).clamp(0, w as isize - 1), y) } else { (x, (y + d).clamp(0, h as isize - 1)) };
                        let v = prev[sy as usize * w + sx as usize];
                        let dist = (v[0] - c[0]).powi(2) + (v[1] - c[1]).powi(2) + (v[2] - c[2]).powi(2);
                        let wgt = kw * (-dist * inv).exp();
                        for i in 0..3 {
                            sum[i] += v[i] * wgt;
                        }
                        tot += wgt;
                    }
                    sum.map(|v| v / tot.max(1e-12))
                })
                .collect();
        }
    }
    cur
}

fn segment(lab: &[[f32; 3]], w: usize, h: usize, seed: u64) -> (Vec<u8>, Vec<[f32; 3]>) {
    let soft = bilateral(lab, w, h, SEGMENT_SMOOTH_PX, SEGMENT_RANGE);
    let mut rng = StdRng::seed_from_u64(seed);
    let sample: Vec<[f32; 3]> = soft.iter().step_by(SEGMENT_STRIDE).copied().collect();
    let mut centres = vec![sample[rng.random_range(0..sample.len())]];
    while centres.len() < REGION_COUNT {
        let d2: Vec<f64> = sample.iter().map(|p| centres.iter().map(|c| de(*p, *c)).fold(f32::MAX, f32::min).powi(2) as f64).collect();
        let total: f64 = d2.iter().sum();
        let mut pick = rng.random::<f64>() * total;
        let mut chosen = sample.len() - 1;
        for (i, d) in d2.iter().enumerate() {
            pick -= d;
            if pick <= 0.0 {
                chosen = i;
                break;
            }
        }
        centres.push(sample[chosen]);
    }
    let nearest = |p: &[f32; 3], centres: &[[f32; 3]]| centres.iter().enumerate().map(|(i, c)| (i, de(*p, *c))).min_by(|a, b| a.1.total_cmp(&b.1)).map(|(i, _)| i).unwrap_or(0);
    for _ in 0..KMEANS_ROUNDS {
        let mut acc = vec![[0.0f64; 4]; centres.len()];
        for p in &sample {
            let i = nearest(p, &centres);
            for c in 0..3 {
                acc[i][c] += p[c] as f64;
            }
            acc[i][3] += 1.0;
        }
        for (c, a) in centres.iter_mut().zip(&acc) {
            if a[3] > 0.0 {
                *c = [(a[0] / a[3]) as f32, (a[1] / a[3]) as f32, (a[2] / a[3]) as f32];
            }
        }
    }
    let mut label: Vec<u8> = soft.par_iter().map(|p| nearest(p, &centres) as u8).collect();
    for _ in 0..MODE_PASSES {
        let prev = label;
        label = (0..w * h)
            .into_par_iter()
            .map(|k| {
                let (x, y) = ((k % w) as isize, (k / w) as isize);
                let mut votes = [0u16; 256];
                let r = MODE_RADIUS as isize;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (sx, sy) = ((x + dx).clamp(0, w as isize - 1), (y + dy).clamp(0, h as isize - 1));
                        votes[prev[sy as usize * w + sx as usize] as usize] += 1;
                    }
                }
                votes.iter().enumerate().max_by_key(|(_, v)| **v).map(|(i, _)| i as u8).unwrap_or(prev[k])
            })
            .collect();
    }
    (label, centres)
}

fn taper(t: f32) -> f32 {
    let e = (t.min(1.0 - t) / TAPER_END).clamp(0.0, 1.0);
    TAPER_MIN + (1.0 - TAPER_MIN) * e * e * (3.0 - 2.0 * e)
}

fn gaussian(src: &[f32], w: usize, h: usize, sigma_px: f32) -> Vec<f32> {
    if sigma_px < 0.3 {
        return src.to_vec();
    }
    let r = ((((12.0 * sigma_px * sigma_px / 3.0) + 1.0).sqrt() - 1.0) * 0.5).round().max(1.0) as usize;
    box_blur(&box_blur(&box_blur(src, w, h, r), w, h, r), w, h, r)
}

pub fn prepare(img: &image::RgbaImage, prep: Prep, px_mm: f32) -> image::RgbaImage {
    if prep.is_none() {
        return img.clone();
    }
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut rgb: Vec<[f32; 3]> = img.pixels().map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]).collect();
    if prep.smooth_mm > 0.0 {
        rgb = bilateral(&rgb, w, h, prep.smooth_mm / px_mm.max(1e-6), SMOOTH_RANGE);
    }
    let mut chans: Vec<Vec<f32>> = (0..3).map(|c| rgb.iter().map(|p| p[c]).collect()).collect();
    for ch in chans.iter_mut() {
        let mut v = gaussian(ch, w, h, prep.blur_mm / px_mm.max(1e-6));
        if prep.sharpen > 0.0 {
            let soft = gaussian(&v, w, h, (SHARPEN_MM / px_mm.max(1e-6)).max(1.0));
            for (x, b) in v.iter_mut().zip(&soft) {
                *x += prep.sharpen * (*x - b);
            }
        }
        for x in v.iter_mut() {
            *x = (0.5 + (*x - 0.5) * prep.contrast).clamp(0.0, 1.0);
        }
        *ch = v;
    }
    let mut out = image::RgbaImage::new(w as u32, h as u32);
    for (k, p) in out.pixels_mut().enumerate() {
        *p = image::Rgba([(chans[0][k] * 255.0).round() as u8, (chans[1][k] * 255.0).round() as u8, (chans[2][k] * 255.0).round() as u8, 255]);
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FocusShape {
    Off,
    Circle,
    Rect,
}

#[derive(Clone, Copy, PartialEq)]
pub struct Focus {
    pub shape: FocusShape,
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub soft: f32,
    pub outside: f32,
}

impl Focus {
    pub fn rect(&self) -> [f32; 4] {
        [self.a[0].min(self.b[0]), self.a[1].min(self.b[1]), self.a[0].max(self.b[0]), self.a[1].max(self.b[1])]
    }

    fn weight(&self, x: f32, y: f32, aspect: f32) -> f32 {
        let (sd, scale) = match self.shape {
            FocusShape::Off => return 1.0,
            FocusShape::Circle => {
                let r = ((self.b[0] - self.a[0]).powi(2) + ((self.b[1] - self.a[1]) / aspect).powi(2)).sqrt().max(1e-4);
                ((((x - self.a[0]).powi(2) + ((y - self.a[1]) / aspect).powi(2)).sqrt() - r), r)
            }
            FocusShape::Rect => {
                let r = self.rect();
                let (hx, hy) = (0.5 * (r[2] - r[0]), 0.5 * (r[3] - r[1]) / aspect);
                let (dx, dy) = ((x - 0.5 * (r[0] + r[2])).abs() - hx, ((y - 0.5 * (r[1] + r[3])) / aspect).abs() - hy);
                let out = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
                (out + dx.max(dy).min(0.0), hx.min(hy).max(1e-4))
            }
        };
        let f = (self.soft * scale).max(1e-4);
        let t = ((sd + 0.5 * f) / f).clamp(0.0, 1.0);
        let s = t * t * (3.0 - 2.0 * t);
        1.0 - s * (1.0 - self.outside)
    }
}

pub struct EvolveState {
    pub photo: Option<Photo>,
    pub focus: Focus,
    pub run: Option<Evolver>,
    pub candidates: usize,
    pub follow: f64,
    pub intent: f64,
    pub prep: Prep,
    pub prep_touched: Option<std::time::Instant>,
    pub dry_passes: bool,
    pub longest: f64,
    pub active: bool,
    photo_gen: u64,
    saved_paint: Option<[f32; 16]>,
    saved_height: Option<f64>,
    saved_load: Option<f64>,
    saved_len: Option<f64>,
    user_sizes: Vec<(ToolKind, BrushShape, f64)>,
    stroke_press: f64,
}

impl EvolveState {
    pub fn new() -> Self {
        Self { photo: None, focus: Focus { shape: FocusShape::Off, a: [0.5, 0.5], b: [0.75, 0.5], soft: 0.5, outside: 0.0 }, run: None, candidates: 32, follow: 0.7, intent: 0.7, prep: Prep::none(), prep_touched: None, dry_passes: true, longest: 0.25, active: false, photo_gen: 0, saved_paint: None, saved_height: None, saved_load: None, saved_len: None, user_sizes: Vec::new(), stroke_press: 0.0 }
    }
}

#[derive(Clone, Copy)]
struct ToolRange {
    lo: f64,
    hi: f64,
    half_width: f64,
    len: (f32, f32),
    speed: f64,
    long: bool,
    reach: f64,
}

fn tool_range(kind: ToolKind, shape: BrushShape) -> Option<ToolRange> {
    let reach = if kind == ToolKind::Brush { 30.0 } else { f64::INFINITY };
    let r = |lo, hi, half_width, len, speed| Some(ToolRange { lo, hi, half_width, len, speed, long: !matches!(kind, ToolKind::Sponge | ToolKind::Knife), reach });
    match kind {
        ToolKind::Pour => r(0.0008, 0.02, 1.15, (2.0, 10.0), 0.5),
        ToolKind::Brush if shape == BrushShape::Flat => r(0.0001, 0.2, 0.5, (2.0, 8.0), 0.3),
        ToolKind::Brush => r(0.0001, 0.03, 0.55, (2.0, 8.0), 0.3),
        ToolKind::Roller => r(0.04, 0.25, 0.5, (1.0, 4.0), 0.35),
        ToolKind::Sponge => r(0.02, 0.15, 0.4, (0.0, 1.2), 0.12),
        ToolKind::Stick => r(0.003, 0.03, 0.2, (6.0, 30.0), 0.4),
        ToolKind::Bottle => r(0.0005, 0.006, 1.5, (6.0, 30.0), 0.2),
        ToolKind::Bucket => r(0.003, 0.03, 1.0, (6.0, 30.0), 0.4),
        ToolKind::Knife => r(0.005, 0.15, 0.5, (1.5, 7.0), 0.15),
        ToolKind::Rubber => None,
    }
}

fn is_thread(kind: ToolKind) -> bool {
    matches!(kind, ToolKind::Stick | ToolKind::Bottle | ToolKind::Bucket)
}

fn decode(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn lab(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = rgb.map(|c| c.max(0.0));
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    let f = |t: f32| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn de(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn tonemap_inverse(y: f32) -> f32 {
    let (a, b, c, d, e) = (2.51f32, 0.03f32, 2.43f32, 0.59f32, 0.14f32);
    let y = y.clamp(0.0, 0.995);
    let qa = a - y * c;
    let qb = b - y * d;
    (-qb + (qb * qb + 4.0 * qa * y * e).sqrt()) / (2.0 * qa)
}

fn tonemap(x: f32) -> f32 {
    ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
}

fn encode(c: f32) -> f32 {
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn view_lab(rgb: [f32; 3], gain: [f32; 3]) -> [f32; 3] {
    lab([tonemap(gain[0] * rgb[0]), tonemap(gain[1] * rgb[1]), tonemap(gain[2] * rgb[2])])
}

fn gamut_map(target: Vec<[f32; 3]>, palette: &[Swatch]) -> Vec<[f32; 3]> {
    let bin = |v: f32| (v / GAMUT_BIN).round() as i32;
    let keys: std::collections::HashSet<[i32; 3]> = target.iter().map(|t| t.map(bin)).collect();
    let keys: Vec<[i32; 3]> = keys.into_iter().collect();
    let found: std::collections::HashMap<[i32; 3], [f32; 3]> = keys
        .par_iter()
        .map(|k| {
            let want = k.map(|v| v as f32 * GAMUT_BIN);
            let near = palette.iter().map(|s| s.lab).min_by(|a, b| de(*a, want).total_cmp(&de(*b, want))).unwrap_or(want);
            (*k, near)
        })
        .collect();
    target
        .into_par_iter()
        .map(|t| {
            let near = found[&t.map(bin)];
            let d = de(t, near);
            if d <= GAMUT_SLACK {
                return t;
            }
            let f = GAMUT_SLACK / d;
            [near[0] + f * (t[0] - near[0]), near[1] + f * (t[1] - near[1]), near[2] + f * (t[2] - near[2])]
        })
        .collect()
}

fn gauss(rng: &mut StdRng) -> f32 {
    let u: f32 = rng.random::<f32>().max(1e-7);
    let v: f32 = rng.random();
    (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
}

fn cover_crop(img: &image::RgbaImage, aspect: f32) -> image::RgbaImage {
    let (w, h) = img.dimensions();
    let (cw, ch) = if w as f32 / h as f32 > aspect { (((h as f32) * aspect).round() as u32, h) } else { (w, ((w as f32) / aspect).round() as u32) };
    image::imageops::crop_imm(img, (w - cw.min(w)) / 2, (h - ch.min(h)) / 2, cw.min(w).max(1), ch.min(h).max(1)).to_image()
}

pub fn load_photo(path: &Path, aspect: f32) -> Result<Photo, String> {
    let reader = image::ImageReader::open(path).map_err(|e| e.to_string())?.with_guessed_format().map_err(|e| e.to_string())?;
    let mut dec = reader.into_decoder().map_err(|e| e.to_string())?;
    let orient = dec.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = image::DynamicImage::from_decoder(dec).map_err(|e| e.to_string())?;
    img.apply_orientation(orient);
    let crop = cover_crop(&img.to_rgba8(), aspect);
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    Ok(Photo { name, image: crop, texture: None, shown: None })
}

pub fn preview(photo: &mut Photo, ctx: &egui::Context, prep: Prep, canvas_mm: f32) -> egui::TextureHandle {
    if let Some(t) = photo.texture.as_ref().filter(|_| photo.shown == Some(prep)) {
        return t.clone();
    }
    let (w, h) = photo.image.dimensions();
    let ph = ((PREVIEW_W as f32) * h as f32 / w as f32).round().max(1.0) as u32;
    let small = prepare(&image::imageops::resize(&photo.image, PREVIEW_W, ph, image::imageops::FilterType::Triangle), prep, canvas_mm / PREVIEW_W as f32);
    photo.shown = Some(prep);
    let ci = egui::ColorImage::from_rgba_unmultiplied([PREVIEW_W as usize, ph as usize], small.as_raw());
    let t = ctx.load_texture("evolve_photo", ci, egui::TextureOptions::LINEAR);
    photo.texture = Some(t.clone());
    t
}

struct Swatch {
    mix: [f32; 16],
    rgb: [f32; 3],
    lab: [f32; 3],
}

fn swatches(set: &PigmentSet, cmf: &[f32]) -> Vec<Swatch> {
    let mut mixes: Vec<[f32; 16]> = Vec::new();
    let n = color::NPIG.min(set.pigments.len());
    let mut bases: Vec<[f32; 16]> = Vec::new();
    for i in 0..n {
        let mut m = [0.0f32; 16];
        m[i] = 1.0;
        bases.push(m);
        for j in i + 1..n {
            for (a, b) in [(3.0, 1.0), (1.0, 1.0), (1.0, 3.0)] {
                let mut m = [0.0f32; 16];
                m[i] = a;
                m[j] = b;
                bases.push(m);
            }
        }
    }
    for b in &bases {
        let total: f32 = b.iter().sum();
        for white in [0.0, 0.05, 0.15, 0.5, 1.5, 4.0, 10.0] {
            let mut m = *b;
            m[WHITE] += white * total;
            mixes.push(m);
        }
        if b[BLACK] == 0.0 {
            for black in [0.08, 0.25, 0.7] {
                let mut m = *b;
                m[BLACK] += black * total;
                mixes.push(m);
            }
        }
    }
    mixes
        .into_par_iter()
        .map(|m| {
            let total: f32 = m.iter().sum();
            let mut mix = [0.0f32; 16];
            for i in 0..n {
                mix[i] = m[i] / total;
            }
            let srgb = color::mix_rgb(set, cmf, &mix);
            Swatch { mix, rgb: srgb.map(decode), lab: lab(srgb.map(decode)) }
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Bezier,
    Flow,
    Iso,
    Ridge,
}

#[derive(Clone)]
struct Cand {
    centre: [f32; 2],
    angle: f32,
    len: f32,
    bend: f32,
    along: f32,
    shape: Shape,
    stop: f32,
    bound: bool,
    structured: bool,
    region: u8,
    size: f64,
    hw: f32,
    hint: [f32; 3],
    pts: Vec<[f32; 2]>,
    swatch: usize,
    score: f32,
}

#[derive(Clone)]
struct Blob {
    centre: [f32; 2],
    axis: f32,
    along: f32,
    across: f32,
    peak: [f32; 2],
    mass: f32,
    count: usize,
    window: [usize; 4],
    tile: usize,
}

#[derive(Clone)]
pub struct Plan {
    pub path: Vec<DVec2>,
    pub lead: f64,
    pub size: f64,
    pub mix: [f32; 16],
    pub speed: f64,
    pub kind: ToolKind,
    pub dry: bool,
    pts: Vec<[f32; 2]>,
    hw: f32,
    lab: [f32; 3],
    predicted: f32,
    width_gain: f64,
    pass: u32,
}

impl Plan {
    fn heading(a: DVec2, b: DVec2) -> DVec2 {
        let d = b - a;
        if d.length() > 1e-9 { d.normalize() } else { DVec2::X }
    }

    pub fn entry(&self) -> (DVec2, DVec2) {
        let n = self.path.len();
        let dir = if n > 1 { Self::heading(self.path[0], self.path[1]) } else { DVec2::X };
        (self.path[0] - dir * self.lead, dir)
    }

    pub fn ahead(&self, s: f64) -> DVec2 {
        let total = self.length();
        if s <= total {
            self.at(s)
        } else {
            *self.path.last().unwrap() + self.exit_dir() * (s - total)
        }
    }

    pub fn exit_dir(&self) -> DVec2 {
        let n = self.path.len();
        if n > 1 { Self::heading(self.path[n - 2], self.path[n - 1]) } else { DVec2::X }
    }

    pub fn length(&self) -> f64 {
        self.path.windows(2).map(|w| (w[1] - w[0]).length()).sum()
    }

    pub fn at(&self, s: f64) -> DVec2 {
        let mut left = s.max(0.0);
        for w in self.path.windows(2) {
            let l = (w[1] - w[0]).length();
            if left <= l && l > 0.0 {
                return w[0] + (w[1] - w[0]) * (left / l);
            }
            left -= l;
        }
        *self.path.last().unwrap()
    }
}

pub enum Phase {
    Wait(u32),
    Observe,
    Plan,
    Lower(Plan, u32),
    Approach(Plan, f64),
    Stroke(Plan, f64),
    Exit(Plan, f64),
    Lift(Plan, u32),
    Settle(Plan, u32),
}

struct Model {
    w: usize,
    h: usize,
    cell: f64,
    target: Vec<[f32; 3]>,
    flow: Vec<[f32; 3]>,
    detail: Vec<f32>,
    label: Vec<u8>,
    kin: Vec<bool>,
    regions: usize,
    weight: Vec<f32>,
    swatches: Arc<Vec<Swatch>>,
}

impl Model {
    fn same_shape(&self, k: usize, region: u8) -> bool {
        self.kin[self.label[k] as usize * self.regions + region as usize]
    }

    fn index(&self, p: [f32; 2]) -> usize {
        (p[1].max(0.0) as usize).min(self.h - 1) * self.w + (p[0].max(0.0) as usize).min(self.w - 1)
    }

    fn bbox(&self, pts: &[[f32; 2]], pad: f32) -> [usize; 4] {
        let mut b = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        for p in pts {
            b[0] = b[0].min(p[0]);
            b[1] = b[1].min(p[1]);
            b[2] = b[2].max(p[0]);
            b[3] = b[3].max(p[1]);
        }
        [
            (b[0] - pad).floor().max(0.0) as usize,
            (b[1] - pad).floor().max(0.0) as usize,
            ((b[2] + pad).ceil().max(0.0) as usize).min(self.w),
            ((b[3] + pad).ceil().max(0.0) as usize).min(self.h),
        ]
    }

    fn footprint(&self, cov: &mut [f32], pts: &[[f32; 2]], hw: f32, tapered: bool) -> Vec<(usize, f32)> {
        self.footprint_at(cov, pts, hw, tapered, 1)
    }

    fn footprint_at(&self, cov: &mut [f32], pts: &[[f32; 2]], hw: f32, tapered: bool, stride: usize) -> Vec<(usize, f32)> {
        let mut touched = Vec::new();
        let segs: Vec<[[f32; 2]; 2]> = if pts.len() < 2 { vec![[pts[0], pts[0]]] } else { pts.windows(2).map(|w| [w[0], w[1]]).collect() };
        let n = segs.len();
        for (si, [a, e]) in segs.into_iter().enumerate() {
            let hw = if tapered && n > 2 { hw * taper((si as f32 + 0.5) / n as f32) } else { hw };
            let bb = self.bbox(&[a, e], hw + 1.0);
            let (ex, ey) = (e[0] - a[0], e[1] - a[1]);
            let l2 = ex * ex + ey * ey;
            let j0 = bb[1].div_ceil(stride) * stride;
            let i0 = bb[0].div_ceil(stride) * stride;
            for j in (j0..bb[3]).step_by(stride) {
                for i in (i0..bb[2]).step_by(stride) {
                    let (x, y) = (i as f32 + 0.5, j as f32 + 0.5);
                    let t = if l2 > 0.0 { (((x - a[0]) * ex + (y - a[1]) * ey) / l2).clamp(0.0, 1.0) } else { 0.0 };
                    let (dx, dy) = (x - a[0] - t * ex, y - a[1] - t * ey);
                    let c = (hw - (dx * dx + dy * dy).sqrt() + 0.5).clamp(0.0, 1.0);
                    if c <= 0.0 {
                        continue;
                    }
                    let k = j * self.w + i;
                    if cov[k] == 0.0 {
                        touched.push(k);
                    }
                    cov[k] = cov[k].max(c);
                }
            }
        }
        let out = touched.iter().map(|&k| (k, cov[k])).collect();
        for k in touched {
            cov[k] = 0.0;
        }
        out
    }

    fn nearest(&self, want: [f32; 3]) -> usize {
        let mut best = (f32::MAX, 0);
        for (k, s) in self.swatches.iter().enumerate() {
            let d = (s.lab[0] - want[0]).powi(2) + (s.lab[1] - want[1]).powi(2) + (s.lab[2] - want[2]).powi(2);
            if d < best.0 {
                best = (d, k);
            }
        }
        best.1
    }

    fn mask(&self, focus: Focus) -> Vec<f32> {
        let (w, h) = (self.w, self.h);
        let aspect = w as f32 / h as f32;
        (0..w * h).into_par_iter().map(|k| focus.weight(((k % w) as f32 + 0.5) / w as f32, 1.0 - ((k / w) as f32 + 0.5) / h as f32, aspect)).collect()
    }
}

#[derive(Clone)]
pub struct Schedule {
    pub size_hi: f64,
    start: f64,
    misses: usize,
}

impl Schedule {
    fn new(size_hi: f64) -> Self {
        Self { size_hi, start: size_hi, misses: 0 }
    }
}

struct Job {
    kind: ToolKind,
    range: ToolRange,
    reset: Option<f64>,
    candidates: usize,
    follow: f64,
    longest: f64,
    focus: Focus,
    paint: [f32; 16],
    width_gain: f64,
    opacity: f32,
    efficiency: f32,
    tile_eff: Arc<Vec<f32>>,
    intent: f32,
    gen: u64,
}

pub struct Batch {
    plans: Vec<Plan>,
    sched: Schedule,
    planned: usize,
    pruned: usize,
    gen: u64,
    reference: Arc<Vec<[f32; 3]>>,
}

struct Region {
    window: [usize; 4],
    cdf: Vec<f32>,
}

struct Sketch {
    m: Arc<Model>,
    mask: Vec<f32>,
    focus: Option<Focus>,
    current: Vec<[f32; 3]>,
    err: Vec<f32>,
    reference: Arc<Vec<[f32; 3]>>,
    tangent: Vec<[f32; 3]>,
    ref_hw: f32,
    smooth_err: Vec<f32>,
    thresh: f32,
    blobs: Vec<Blob>,
    cursor: usize,
    since_blob: usize,
    tiles: Vec<f32>,
    tabu: Vec<f32>,
    tw: usize,
    scratch: Vec<f32>,
    rng: StdRng,
    kind: ToolKind,
    range: ToolRange,
    longest: f32,
    width_gain: f64,
    opacity: f32,
    efficiency: f32,
    tile_eff: Arc<Vec<f32>>,
    sched: Schedule,
    started: bool,
    last_tiles: Vec<usize>,
    intent: f32,
    follow: f32,
    fine: f32,
    gate: Vec<f32>,
}

impl Sketch {
    fn new(m: Arc<Model>, seed: u64, kind: ToolKind, range: ToolRange, size_hi: f64) -> Self {
        let n = m.w * m.h;
        let (tw, th) = (m.w.div_ceil(TILE), m.h.div_ceil(TILE));
        Self {
            mask: vec![1.0; n],
            focus: None,
            current: vec![[0.0; 3]; n],
            err: vec![0.0; n],
            reference: Arc::new(m.target.clone()),
            tangent: Vec::new(),
            ref_hw: -1.0,
            smooth_err: Vec::new(),
            thresh: T_MIN,
            blobs: Vec::new(),
            cursor: 0,
            since_blob: 0,
            tiles: vec![0.0; tw * th],
            tabu: vec![1.0; tw * th],
            tw,
            scratch: vec![0.0; n],
            rng: StdRng::seed_from_u64(seed),
            kind,
            range,
            longest: 0.0,
            width_gain: 1.0,
            opacity: 0.9,
            efficiency: 1.0,
            tile_eff: Arc::new(Vec::new()),
            sched: Schedule::new(size_hi),
            started: false,
            last_tiles: Vec::new(),
            intent: 0.0,
            follow: 0.7,
            fine: 0.0,
            gate: vec![1.0; n],
            m,
        }
    }

    fn refresh_reference(&mut self) {
        let hw = self.hw(self.sched.size_hi);
        if (hw - self.ref_hw).abs() <= 0.05 * hw.max(1.0) {
            return;
        }
        self.ref_hw = hw;
        let m = self.m.clone();
        let (w, h) = (m.w, m.h);
        let sigma = REF_SIGMA * hw;
        self.reference = Arc::new(if sigma < 0.6 { m.target.clone() } else { bilateral(&m.target, w, h, sigma, REF_RANGE + REF_RANGE_WIDE * (hw / REF_RANGE_HW).min(1.0)) });
        let radius = ((0.5 * hw).round() as usize).max(1);
        let (tangent, edges) = structure(&self.reference, w, h, radius);
        let seeded: Vec<[f32; 3]> = tangent.iter().zip(&edges).zip(&m.flow).map(|((t, e), f)| if *e > ETF_SEED_EDGE { *t } else { [f[0], f[1], f[2]] }).collect();
        self.tangent = etf(&seeded, &edges, w, h, radius.max(4), ETF_PASS_ITERS);
        self.err = self.current.par_iter().zip(self.reference.par_iter()).map(|(a, b)| de(*a, *b)).collect();
        self.refresh_tiles();
        self.blobs.clear();
    }

    fn set_focus(&mut self, focus: Focus) {
        if self.focus != Some(focus) {
            self.focus = Some(focus);
            self.mask = self.m.mask(focus);
        }
    }

    fn tile_of(&self, k: usize) -> usize {
        (k / self.m.w / TILE) * self.tw + (k % self.m.w) / TILE
    }

    fn refresh_tiles(&mut self) {
        self.tiles.iter_mut().for_each(|t| *t = 0.0);
        for k in 0..self.err.len() {
            let t = self.tile_of(k);
            self.tiles[t] += self.err[k] * self.m.weight[k] * self.mask[k] * self.gate[k];
        }
        self.tabu.iter_mut().for_each(|t| *t = 1.0);
    }

    fn rebase(&mut self, real: &[[f32; 3]], pending: &[Plan]) {
        let m = self.m.clone();
        self.current = real.to_vec();
        self.err = self.current.par_iter().zip(self.reference.par_iter()).map(|(a, b)| de(*a, *b)).collect();
        self.refresh_tiles();
        self.blobs.clear();
        for p in pending {
            let foot = m.footprint(&mut self.scratch, &p.pts, p.hw, p.kind == ToolKind::Brush);
            self.apply(&foot, p.lab);
        }
    }

    fn apply(&mut self, foot: &[(usize, f32)], c: [f32; 3]) {
        let o = self.opacity;
        for &(k, a) in foot {
            let a = a * o;
            let cur = self.current[k];
            let new = [cur[0] + a * (c[0] - cur[0]), cur[1] + a * (c[1] - cur[1]), cur[2] + a * (c[2] - cur[2])];
            let e = de(new, self.reference[k]);
            let t = self.tile_of(k);
            self.tiles[t] += (e - self.err[k]) * self.m.weight[k] * self.mask[k] * self.gate[k];
            self.current[k] = new;
            self.err[k] = e;
        }
    }

    fn attention(&self, k: usize) -> f32 {
        let g = self.gate[k];
        g + (1.0 - g) * ((self.err[k] - GATE_OPEN_DE) / GATE_FULL_DE).clamp(0.0, 1.0)
    }

    fn min_gain(&self, _hw: f32) -> f32 {
        MIN_GAIN
    }

    fn gain(&self, k: usize, delta: f32) -> f32 {
        if delta < 0.0 { delta * self.mask[k] } else { delta }
    }

    fn size_lo(&self) -> f64 {
        self.range.lo.max(MIN_HW * self.m.cell / (self.range.half_width * self.width_gain))
    }

    fn size_floor(&self) -> f64 {
        self.size_lo().max(SIZE_SPREAD * self.sched.size_hi)
    }

    fn hw(&self, size: f64) -> f32 {
        ((size * self.range.half_width * self.width_gain) / self.m.cell).max(0.6) as f32
    }

    fn len_range(&self, size: f64, hw: f32) -> (f32, f32) {
        let r = self.range;
        let lo = (hw * r.len.0).max(2.0);
        let hi = if r.long { self.longest.max(hw * r.len.1) } else { hw * r.len.1 };
        let hi = hi.min((r.reach * size / self.m.cell) as f32).min(2.0 * LEN_WIDTHS * hw);
        (lo, hi.max(lo * 1.2))
    }

    fn window_region(&self, window: [usize; 4]) -> Option<Region> {
        let m = &self.m;
        let mut cdf = Vec::with_capacity((window[2] - window[0]) * (window[3] - window[1]));
        let mut acc = 0.0f32;
        for y in window[1]..window[3] {
            for x in window[0]..window[2] {
                let k = y * m.w + x;
                acc += self.err[k] * self.err[k] * m.weight[k] * self.mask[k] * self.attention(k);
                cdf.push(acc);
            }
        }
        if acc <= 0.0 {
            return None;
        }
        Some(Region { window, cdf })
    }

    fn region_at_all(&self) -> Option<Region> {
        self.window_region([0, 0, self.m.w, self.m.h])
    }

    fn blob_window(&self, b: &Blob) -> [usize; 4] {
        let pad = (2.0 * self.hw(self.sched.size_hi)).ceil() as usize;
        [b.window[0].saturating_sub(pad), b.window[1].saturating_sub(pad), (b.window[2] + pad).min(self.m.w), (b.window[3] + pad).min(self.m.h)]
    }

    fn blob_stats(&self, cells: &[usize], e: &[f32], window: [usize; 4]) -> Option<Blob> {
        let w = self.m.w;
        let (mut mass, mut sx, mut sy, mut sxx, mut sxy, mut syy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut peak, mut peak_e) = (cells[0], -1.0f32);
        for &k in cells {
            let v = e[k] as f64;
            let (x, y) = ((k % w) as f64 + 0.5, (k / w) as f64 + 0.5);
            mass += v;
            sx += v * x;
            sy += v * y;
            sxx += v * x * x;
            sxy += v * x * y;
            syy += v * y * y;
            if e[k] > peak_e {
                peak_e = e[k];
                peak = k;
            }
        }
        if mass <= 0.0 {
            return None;
        }
        let (cx, cy) = (sx / mass, sy / mass);
        let (vxx, vxy, vyy) = (sxx / mass - cx * cx, sxy / mass - cx * cy, syy / mass - cy * cy);
        let tr = vxx + vyy;
        let det = vxx * vyy - vxy * vxy;
        let disc = (0.25 * tr * tr - det).max(0.0).sqrt();
        let (l1, l2) = (0.5 * tr + disc, (0.5 * tr - disc).max(0.0));
        let axis = if vxy.abs() > 1e-9 { (l1 - vxx).atan2(vxy) } else if vxx >= vyy { 0.0 } else { std::f64::consts::FRAC_PI_2 };
        Some(Blob {
            centre: [cx as f32, cy as f32],
            axis: axis as f32,
            along: (2.0 * (3.0 * l1).sqrt()).max(1.0) as f32,
            across: (2.0 * (3.0 * l2).sqrt()).max(1.0) as f32,
            peak: [(peak % w) as f32 + 0.5, (peak / w) as f32 + 0.5],
            mass: mass as f32,
            count: cells.len(),
            window,
            tile: self.tile_of(peak),
        })
    }

    fn extract_blobs(&mut self) {
        let m = self.m.clone();
        let (w, h) = (m.w, m.h);
        let raw: Vec<f32> = (0..w * h).into_par_iter().map(|k| self.err[k] * m.weight[k] * self.mask[k] * self.attention(k)).collect();
        let e = box_blur(&raw, w, h, 1);
        let mut sample: Vec<f32> = e.iter().step_by(QUANTILE_STRIDE).copied().collect();
        let q = ((sample.len() - 1) as f32 * SEED_QUANTILE) as usize;
        let (_, quant, _) = sample.select_nth_unstable_by(q, |a, b| a.total_cmp(b));
        let thresh = quant.clamp(T_MIN, T_FOCUS * T_MIN);
        self.thresh = thresh;
        let hw = self.hw(self.sched.size_hi);
        let win = ((BLOB_WINDOW_HW * hw) as usize).clamp(BLOB_WINDOW_MIN, BLOB_WINDOW_MAX);
        let mut label = vec![u32::MAX; w * h];
        let mut blobs = Vec::new();
        let mut stack = Vec::new();
        let mut cells = Vec::new();
        let mut next = 0u32;
        for start in 0..w * h {
            if e[start] <= thresh || label[start] != u32::MAX {
                continue;
            }
            let id = next;
            next += 1;
            label[start] = id;
            stack.push(start);
            cells.clear();
            let mut bb = [w, h, 0, 0];
            while let Some(k) = stack.pop() {
                cells.push(k);
                let (x, y) = (k % w, k / w);
                bb = [bb[0].min(x), bb[1].min(y), bb[2].max(x + 1), bb[3].max(y + 1)];
                let nb = [(x > 0, k.wrapping_sub(1)), (x + 1 < w, k + 1), (y > 0, k.wrapping_sub(w)), (y + 1 < h, k + w)];
                for (ok, n) in nb {
                    if ok && e[n] > thresh && label[n] == u32::MAX {
                        label[n] = id;
                        stack.push(n);
                    }
                }
            }
            if cells.len() < MIN_BLOB {
                continue;
            }
            if bb[2] - bb[0] <= win && bb[3] - bb[1] <= win {
                if let Some(b) = self.blob_stats(&cells, &e, bb) {
                    blobs.push(b);
                }
                continue;
            }
            let step = win / 2;
            let mut y0 = bb[1];
            loop {
                let y1 = (y0 + win).min(bb[3]);
                let mut x0 = bb[0];
                loop {
                    let x1 = (x0 + win).min(bb[2]);
                    let part: Vec<usize> = (y0..y1).flat_map(|y| (x0..x1).map(move |x| y * w + x)).filter(|&k| label[k] == id).collect();
                    if part.len() >= MIN_BLOB {
                        if let Some(b) = self.blob_stats(&part, &e, [x0, y0, x1, y1]) {
                            blobs.push(b);
                        }
                    }
                    if x1 >= bb[2] {
                        break;
                    }
                    x0 += step;
                }
                if y1 >= bb[3] {
                    break;
                }
                y0 += step;
            }
        }
        let key = |b: &Blob| b.mass / (b.count as f32).sqrt() * self.tabu[b.tile] * self.tile_eff.get(b.tile).copied().unwrap_or(1.0).clamp(0.2, 1.5);
        blobs.sort_by(|a, b| key(b).total_cmp(&key(a)));
        blobs.truncate(BLOBS_KEEP);
        self.blobs = blobs;
        self.smooth_err = e;
        self.cursor = 0;
        self.since_blob = 0;
    }

    fn sample_in(&mut self, reg: &Region) -> [f32; 2] {
        let total = *reg.cdf.last().unwrap();
        let pick = self.rng.random::<f32>() * total;
        let k = reg.cdf.partition_point(|c| *c < pick).min(reg.cdf.len() - 1);
        let rw = reg.window[2] - reg.window[0];
        [(reg.window[0] + k % rw) as f32 + self.rng.random::<f32>(), (reg.window[1] + k / rw) as f32 + self.rng.random::<f32>()]
    }

    fn means(&self, foot: &[(usize, f32)]) -> ([f32; 3], [f32; 3]) {
        let mut acc = [0.0f32; 4];
        let mut chans: [Vec<f32>; 3] = [Vec::with_capacity(foot.len()), Vec::with_capacity(foot.len()), Vec::with_capacity(foot.len())];
        for &(k, a) in foot {
            let (t, c) = (self.reference[k], self.current[k]);
            for i in 0..3 {
                acc[i] += a * c[i];
            }
            acc[3] += a;
            if a >= MEDIAN_COVER {
                for i in 0..3 {
                    chans[i].push(t[i]);
                }
            }
        }
        if acc[3] <= 0.0 {
            return ([50.0, 0.0, 0.0], [50.0, 0.0, 0.0]);
        }
        if chans[0].is_empty() {
            for &(k, _) in foot {
                let t = self.reference[k];
                for i in 0..3 {
                    chans[i].push(t[i]);
                }
            }
        }
        let n = acc[3];
        let mid = |v: &mut Vec<f32>| {
            let q = v.len() / 2;
            *v.select_nth_unstable_by(q, |a, b| a.total_cmp(b)).1
        };
        let [c0, c1, c2] = &mut chans;
        ([mid(c0), mid(c1), mid(c2)], [acc[0] / n, acc[1] / n, acc[2] / n])
    }

    fn pixel_delta(&self, k: usize, a: f32, c: [f32; 3], region: Option<u8>) -> f32 {
        let cover = a;
        let a = a * self.opacity;
        let cur = self.current[k];
        let new = [cur[0] + a * (c[0] - cur[0]), cur[1] + a * (c[1] - cur[1]), cur[2] + a * (c[2] - cur[2])];
        let d = self.gain(k, self.m.weight[k] * (de(new, self.reference[k]) - self.err[k])) + STROKE_COST * cover;
        match region {
            Some(r) if d < 0.0 && !self.m.same_shape(k, r) => d * (1.0 - OUTSIDE_DISCOUNT * self.intent),
            _ => d,
        }
    }

    fn score(&self, foot: &[(usize, f32)], c: [f32; 3], region: Option<u8>) -> f32 {
        foot.iter().map(|&(k, a)| self.pixel_delta(k, a, c, region)).sum()
    }

    fn stride(&self, hw: f32) -> usize {
        ((hw / STRIDE_HW).floor() as usize).max(1)
    }

    fn trim(&mut self, pts: &[[f32; 2]], hw: f32, c: [f32; 3], region: Option<u8>) -> Vec<[f32; 2]> {
        if pts.len() < 4 {
            return pts.to_vec();
        }
        let st = self.stride(hw);
        let m = self.m.clone();
        let gains: Vec<f32> = pts
            .windows(2)
            .map(|w| m.footprint_at(&mut self.scratch, w, hw, false, st).iter().map(|&(k, a)| self.pixel_delta(k, a, c, region)).sum())
            .collect();
        let (mut best, mut best_range) = (0.0f32, (0, gains.len()));
        let (mut run, mut start) = (0.0f32, 0usize);
        for (i, g) in gains.iter().enumerate() {
            if run > 0.0 {
                run = 0.0;
                start = i;
            }
            run += g;
            if run < best {
                best = run;
                best_range = (start, i + 1);
            }
        }
        if best >= 0.0 {
            return pts.to_vec();
        }
        pts[best_range.0..=best_range.1].to_vec()
    }

    fn bounded(&self, pts: Vec<[f32; 2]>, centre: [f32; 2], region: u8) -> Vec<[f32; 2]> {
        if pts.len() < 2 {
            return pts;
        }
        let m = &self.m;
        let ci = (0..pts.len()).min_by(|a, b| {
            let d = |p: [f32; 2]| (p[0] - centre[0]).powi(2) + (p[1] - centre[1]).powi(2);
            d(pts[*a]).total_cmp(&d(pts[*b]))
        }).unwrap();
        let inside = |p: [f32; 2]| m.same_shape(m.index(p), region);
        let mut a = ci;
        while a > 0 && inside(pts[a - 1]) {
            a -= 1;
        }
        let mut b = ci;
        while b + 1 < pts.len() && inside(pts[b + 1]) {
            b += 1;
        }
        pts[a..=b].to_vec()
    }

    fn pick_colour(&mut self, foot: &[(usize, f32)], jitter: f32) -> (usize, [f32; 3]) {
        let (m, cur) = self.means(foot);
        let o = self.opacity.max(0.65);
        let mut want = [0.0f32; 3];
        for c in 0..3 {
            want[c] = cur[c] + (m[c] - cur[c]) / o + jitter * gauss(&mut self.rng) * if c == 0 { 1.0 } else { 1.4 };
        }
        want[0] = want[0].clamp(0.0, 100.0);
        (self.m.nearest(want), m)
    }

    fn trace(&self, c: &Cand) -> Vec<[f32; 2]> {
        let m = &self.m;
        let step = (0.5 * c.hw).max(c.len / MAX_POINTS).max(1.0);
        let (xmax, ymax) = (m.w as f32 - EDGE_PX, m.h as f32 - EDGE_PX);
        let walk = |sign: f32, dist: f32| -> Vec<[f32; 2]> {
            let mut p = c.centre;
            let mut d = [sign * c.angle.cos(), sign * c.angle.sin()];
            let mut out = Vec::new();
            let mut gone = 0.0;
            while gone + step <= dist {
                let t = m.flow[m.index(p)];
                if t[2] > TRACE_COHERENT {
                    let flip = if t[0] * d[0] + t[1] * d[1] < 0.0 { -1.0 } else { 1.0 };
                    let n = [d[0] + TRACE_TURN * (flip * t[0] - d[0]), d[1] + TRACE_TURN * (flip * t[1] - d[1])];
                    let l = (n[0] * n[0] + n[1] * n[1]).sqrt().max(1e-6);
                    d = [n[0] / l, n[1] / l];
                }
                let q = [p[0] + d[0] * step, p[1] + d[1] * step];
                if q[0] < EDGE_PX || q[1] < EDGE_PX || q[0] > xmax || q[1] > ymax {
                    break;
                }
                p = q;
                out.push(p);
                gone += step;
            }
            out
        };
        let back = walk(-1.0, c.len * c.along);
        let fwd = walk(1.0, c.len * (1.0 - c.along));
        let mut pts: Vec<[f32; 2]> = back.into_iter().rev().collect();
        pts.push([c.centre[0].clamp(EDGE_PX, xmax), c.centre[1].clamp(EDGE_PX, ymax)]);
        pts.extend(fwd);
        pts
    }

    fn seed_colour(&self, p: [f32; 2]) -> usize {
        let k = self.m.index(p);
        let (r, cur) = (self.reference[k], self.current[k]);
        let o = self.opacity.max(0.65);
        let mut want = [0.0f32; 3];
        for c in 0..3 {
            want[c] = cur[c] + (r[c] - cur[c]) / o;
        }
        want[0] = want[0].clamp(0.0, 100.0);
        self.m.nearest(want)
    }

    fn stop_de(&self) -> f32 {
        ISO_STOP_DE * (1.0 - STOP_FINE * self.fine)
    }

    fn start_dir(&self, c: &Cand) -> [f32; 2] {
        let t = self.tangent[self.m.index(c.centre)];
        if t[2] > TRACE_COHERENT { [t[0], t[1]] } else { [c.angle.cos(), c.angle.sin()] }
    }

    fn iso(&self, c: &Cand) -> Vec<[f32; 2]> {
        let m = &self.m;
        let step = c.hw.max(1.0);
        let (xmax, ymax) = (m.w as f32 - EDGE_PX, m.h as f32 - EDGE_PX);
        let d0 = self.start_dir(c);
        let fc = self.follow;
        let walk = |sign: f32, dist: f32| -> Vec<[f32; 2]> {
            let mut p = c.centre;
            let mut d = [sign * d0[0], sign * d0[1]];
            let mut out = Vec::new();
            let mut gone = 0.0;
            while gone + step <= dist && out.len() < MAX_POINTS as usize {
                let q = [p[0] + d[0] * step, p[1] + d[1] * step];
                if q[0] < EDGE_PX || q[1] < EDGE_PX || q[0] > xmax || q[1] > ymax {
                    break;
                }
                let k = m.index(q);
                let r = self.reference[k];
                let away = de(r, c.hint);
                if away > self.stop_de() || (out.len() >= ISO_MIN_STEPS && de(r, self.current[k]) < away) {
                    break;
                }
                let t = self.tangent[k];
                if t[2] > TRACE_COHERENT {
                    let flip = if t[0] * d[0] + t[1] * d[1] < 0.0 { -1.0 } else { 1.0 };
                    let n = [fc * flip * t[0] + (1.0 - fc) * d[0], fc * flip * t[1] + (1.0 - fc) * d[1]];
                    let l = (n[0] * n[0] + n[1] * n[1]).sqrt().max(1e-6);
                    d = [n[0] / l, n[1] / l];
                }
                p = q;
                out.push(p);
                gone += step;
            }
            out
        };
        let back = walk(-1.0, c.len * c.along);
        let fwd = walk(1.0, c.len * (1.0 - c.along));
        let mut pts: Vec<[f32; 2]> = back.into_iter().rev().collect();
        pts.push([c.centre[0].clamp(EDGE_PX, xmax), c.centre[1].clamp(EDGE_PX, ymax)]);
        pts.extend(fwd);
        pts
    }

    fn ridge(&self, c: &Cand) -> Vec<[f32; 2]> {
        let m = &self.m;
        if self.smooth_err.len() != m.w * m.h {
            return self.iso(c);
        }
        let step = c.hw.max(1.0);
        let (xmax, ymax) = (m.w as f32 - EDGE_PX, m.h as f32 - EDGE_PX);
        let d0 = self.start_dir(c);
        let walk = |sign: f32, dist: f32| -> Vec<[f32; 2]> {
            let mut p = c.centre;
            let mut d = [sign * d0[0], sign * d0[1]];
            let mut out = Vec::new();
            let mut gone = 0.0;
            while gone + step <= dist && out.len() < MAX_POINTS as usize {
                let mut best: Option<([f32; 2], f32)> = None;
                for a in RIDGE_FAN {
                    let (s, co) = a.sin_cos();
                    let nd = [co * d[0] - s * d[1], s * d[0] + co * d[1]];
                    let q = [p[0] + nd[0] * step, p[1] + nd[1] * step];
                    if q[0] < EDGE_PX || q[1] < EDGE_PX || q[0] > xmax || q[1] > ymax {
                        continue;
                    }
                    let v = self.smooth_err[m.index(q)];
                    if best.map(|b| v > b.1).unwrap_or(true) {
                        best = Some((nd, v));
                    }
                }
                let Some((nd, v)) = best else { break };
                if v < self.thresh && out.len() >= ISO_MIN_STEPS {
                    break;
                }
                let n = [RIDGE_LAMBDA * nd[0] + (1.0 - RIDGE_LAMBDA) * d[0], RIDGE_LAMBDA * nd[1] + (1.0 - RIDGE_LAMBDA) * d[1]];
                let l = (n[0] * n[0] + n[1] * n[1]).sqrt().max(1e-6);
                d = [n[0] / l, n[1] / l];
                let q = [p[0] + d[0] * step, p[1] + d[1] * step];
                if q[0] < EDGE_PX || q[1] < EDGE_PX || q[0] > xmax || q[1] > ymax {
                    break;
                }
                let k = m.index(q);
                if de(self.reference[k], c.hint) > self.stop_de() {
                    break;
                }
                p = q;
                out.push(p);
                gone += step;
            }
            out
        };
        let back = walk(-1.0, c.len * c.along);
        let fwd = walk(1.0, c.len * (1.0 - c.along));
        let mut pts: Vec<[f32; 2]> = back.into_iter().rev().collect();
        pts.push([c.centre[0].clamp(EDGE_PX, xmax), c.centre[1].clamp(EDGE_PX, ymax)]);
        pts.extend(fwd);
        pts
    }

    fn bezier(&self, c: &Cand) -> Vec<[f32; 2]> {
        let (s, co) = c.angle.sin_cos();
        let start = [c.centre[0] - co * c.len * c.along, c.centre[1] - s * c.len * c.along];
        let end = [start[0] + co * c.len, start[1] + s * c.len];
        let ctrl = [(start[0] + end[0]) * 0.5 - s * c.bend * c.len, (start[1] + end[1]) * 0.5 + co * c.bend * c.len];
        let n = if c.len < 2.0 { 1 } else { (c.len / c.hw.max(2.0)).ceil().clamp(8.0, MAX_POINTS) as usize };
        let (xmax, ymax) = (self.m.w as f32 - EDGE_PX, self.m.h as f32 - EDGE_PX);
        (0..=n)
            .map(|k| {
                let t = k as f32 / n as f32;
                let u = 1.0 - t;
                let x = u * u * start[0] + 2.0 * u * t * ctrl[0] + t * t * end[0];
                let y = u * u * start[1] + 2.0 * u * t * ctrl[1] + t * t * end[1];
                [x.clamp(EDGE_PX, xmax), y.clamp(EDGE_PX, ymax)]
            })
            .collect()
    }

    fn clip(&self, pts: Vec<[f32; 2]>, centre: [f32; 2], stop: f32) -> Vec<[f32; 2]> {
        if !stop.is_finite() || pts.len() < 2 {
            return pts;
        }
        let m = &self.m;
        let ci = (0..pts.len()).min_by(|a, b| {
            let d = |p: [f32; 2]| (p[0] - centre[0]).powi(2) + (p[1] - centre[1]).powi(2);
            d(pts[*a]).total_cmp(&d(pts[*b]))
        }).unwrap();
        let here = self.reference[m.index(pts[ci])];
        let inside = |p: [f32; 2]| de(self.reference[m.index(p)], here) <= stop;
        let mut a = ci;
        while a > 0 && inside(pts[a - 1]) {
            a -= 1;
        }
        let mut b = ci;
        while b + 1 < pts.len() && inside(pts[b + 1]) {
            b += 1;
        }
        pts[a..=b].to_vec()
    }

    fn evaluate(&mut self, mut c: Cand, refit: bool, jitter: f32) -> Cand {
        c.size = c.size.clamp(self.size_floor(), self.sched.size_hi.max(self.size_floor()));
        c.hw = self.hw(c.size);
        let (lo, hi) = self.len_range(c.size, c.hw);
        c.len = c.len.clamp(lo, hi);
        let raw = match c.shape {
            Shape::Bezier => self.bezier(&c),
            Shape::Flow => self.trace(&c),
            Shape::Iso => self.iso(&c),
            Shape::Ridge => self.ridge(&c),
        };
        let m = self.m.clone();
        let tapered = self.kind == ToolKind::Brush;
        c.region = m.label[m.index(c.centre)];
        let region = c.bound.then_some(c.region);
        let stop = c.stop.min(self.stop_de());
        c.pts = if c.bound {
            self.clip(self.bounded(raw, c.centre, c.region), c.centre, stop)
        } else {
            self.clip(raw, c.centre, stop)
        };
        let st = self.stride(c.hw);
        let area = (st * st) as f32;
        let foot = m.footprint_at(&mut self.scratch, &c.pts, c.hw, tapered, st);
        if refit {
            (c.swatch, c.hint) = self.pick_colour(&foot, jitter);
        }
        let colour = m.swatches[c.swatch].lab;
        {
            let trimmed = self.trim(&c.pts, c.hw, colour, region);
            if trimmed.len() != c.pts.len() {
                c.pts = trimmed;
                let foot = m.footprint_at(&mut self.scratch, &c.pts, c.hw, tapered, st);
                c.score = self.score(&foot, colour, region) * area;
                return c;
            }
        }
        c.score = self.score(&foot, colour, region) * area;
        c
    }

    fn random_stop(&mut self) -> f32 {
        if self.rng.random::<f32>() < self.intent {
            STOPS[self.rng.random_range(STOPS.len() - 2..STOPS.len())]
        } else {
            STOPS[self.rng.random_range(0..STOPS.len())]
        }
    }

    fn candidate(&mut self, reg: &Region) -> Cand {
        let centre = self.sample_in(reg);
        let lo = self.size_floor();
        let hi = self.sched.size_hi.max(lo);
        let size = (lo.ln() + self.rng.random::<f64>() * (hi.ln() - lo.ln())).exp();
        let hw = self.hw(size);
        let (llo, lhi) = self.len_range(size, hw);
        let len = (llo.ln() + self.rng.random::<f32>() * (lhi.ln() - llo.ln())).exp();
        let f = self.m.flow[self.m.index(centre)];
        let shaped = self.rng.random::<f32>() < self.intent;
        let shape = if shaped && self.rng.random::<f32>() < self.follow { Shape::Flow } else { Shape::Bezier };
        let angle = f[1].atan2(f[0]) + FLOW_JITTER * (1.0 - self.intent) * gauss(&mut self.rng);
        let stop = self.random_stop();
        let c = Cand { centre, angle, len, bend: 0.12 * gauss(&mut self.rng).clamp(-2.0, 2.0), along: self.rng.random(), shape, stop, bound: shaped, structured: false, region: 0, size, hw, hint: [50.0, 0.0, 0.0], pts: Vec::new(), swatch: 0, score: 0.0 };
        self.evaluate(c, true, 3.0)
    }

    fn structured(&mut self, b: &Blob) -> Vec<Cand> {
        let lo = self.size_floor();
        let hi = self.sched.size_hi.max(lo);
        let size = hi * (STRUCT_SIZE_LO + (1.0 - STRUCT_SIZE_LO) * self.rng.random::<f64>());
        let hw = self.hw(size);
        let (_, lhi) = self.len_range(size, hw);
        let seed = self.seed_colour(b.peak);
        let hint = self.reference[self.m.index(b.peak)];
        let base = Cand { centre: b.peak, angle: b.axis, len: lhi, bend: 0.0, along: 0.5, shape: Shape::Iso, stop: f32::INFINITY, bound: false, structured: true, region: 0, size, hw, hint, pts: Vec::new(), swatch: seed, score: 0.0 };
        let mut out = Vec::with_capacity(5);
        out.push(self.evaluate(base.clone(), false, 0.0));
        out.push(self.evaluate(base.clone(), true, 0.0));
        if hw >= RIDGE_MIN_HW {
            out.push(self.evaluate(Cand { shape: Shape::Ridge, ..base.clone() }, true, 0.0));
        }
        out.push(self.evaluate(Cand { shape: Shape::Flow, ..base.clone() }, true, 0.0));
        let across = (b.across * 0.5 * self.m.cell as f32) as f64 / (self.range.half_width * self.width_gain);
        let ribbon = Cand { centre: b.centre, len: b.along, size: across.clamp(lo, hi), shape: Shape::Bezier, ..base };
        out.push(self.evaluate(ribbon, true, 0.0));
        out
    }

    fn mutate(&mut self, best: &Cand) -> Cand {
        let mut c = best.clone();
        match self.rng.random_range(0..6) {
            0 => {
                c.size *= (0.3 * gauss(&mut self.rng)).exp() as f64;
                self.evaluate(c, true, 0.0)
            }
            1 => {
                let sw = self.m.swatches[c.swatch].lab;
                let want = [sw[0] + 5.0 * gauss(&mut self.rng), sw[1] + 7.0 * gauss(&mut self.rng), sw[2] + 7.0 * gauss(&mut self.rng)];
                c.swatch = self.m.nearest(want);
                self.evaluate(c, false, 0.0)
            }
            2 => {
                c.stop = self.random_stop();
                self.evaluate(c, true, 0.0)
            }
            3 => {
                c.len *= (0.5 * gauss(&mut self.rng)).exp();
                c.along = (c.along + 0.15 * gauss(&mut self.rng)).clamp(0.0, 1.0);
                if c.shape == Shape::Bezier {
                    c.angle += 0.15 * (1.0 - 0.8 * self.intent) * gauss(&mut self.rng);
                }
                self.evaluate(c, true, 0.0)
            }
            4 => {
                c.size *= (0.3 * gauss(&mut self.rng)).exp() as f64;
                self.evaluate(c, true, 4.0)
            }
            _ => {
                c.centre[0] += 0.3 * c.hw * gauss(&mut self.rng);
                c.centre[1] += 0.3 * c.hw * gauss(&mut self.rng);
                self.evaluate(c, true, 0.0)
            }
        }
    }

    fn consider(best: &mut Option<Cand>, c: Cand) {
        if best.as_ref().map(|b| c.score < b.score).unwrap_or(true) {
            *best = Some(c);
        }
    }

    fn plan_one(&mut self, job: &Job) -> Option<(Plan, Vec<(usize, f32)>)> {
        self.refresh_reference();
        if self.blobs.is_empty() || self.cursor >= self.blobs.len() || self.since_blob >= REBLOB {
            self.extract_blobs();
        }
        let n_rand = (job.candidates as f32 * (1.0 - self.intent) * (1.0 - self.fine)).round() as usize;
        let floor = self.min_gain(self.hw(self.sched.size_hi));
        let mut best: Option<Cand> = None;
        let mut tried = Vec::new();
        for _ in 0..BLOB_TRIES {
            if self.cursor >= self.blobs.len() {
                break;
            }
            let b = self.blobs[self.cursor].clone();
            self.cursor += 1;
            tried.push(b.tile);
            for c in self.structured(&b) {
                Self::consider(&mut best, c);
            }
            if n_rand > 0 {
                if let Some(reg) = self.window_region(self.blob_window(&b)) {
                    for _ in 0..n_rand.div_ceil(2) {
                        let c = self.candidate(&reg);
                        Self::consider(&mut best, c);
                    }
                }
            }
            if best.as_ref().map(|b| b.score < -floor).unwrap_or(false) {
                break;
            }
        }
        if n_rand > 0 {
            if let Some(reg) = self.region_at_all() {
                for _ in 0..n_rand / 2 {
                    let c = self.candidate(&reg);
                    Self::consider(&mut best, c);
                }
            }
        }
        self.last_tiles = tried;
        let mut best = best?;
        for _ in 0..REFINEMENTS {
            let c = self.mutate(&best);
            if c.score < best.score {
                best = c;
            }
        }
        if best.score > -floor {
            log::debug!("miss: best {:.0} floor {:.0} blobs {} cursor {} thresh {:.1}", -best.score, floor, self.blobs.len(), self.cursor, self.thresh);
            return None;
        }
        self.since_blob += 1;
        if log::log_enabled!(log::Level::Trace) {
            let (a, b) = (best.pts[0], best.pts[best.pts.len() - 1]);
            let len: f32 = best.pts.windows(2).map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt()).sum();
            log::trace!("stroke shape {:?} struct {} len/hw {:.1} angle {:.0} gain {:.0} hw {:.1} at {:.0},{:.0}", best.shape as u8, best.structured, len / best.hw.max(1.0), (b[1] - a[1]).atan2(b[0] - a[0]).to_degrees(), -best.score, best.hw, best.centre[0], best.centre[1]);
        }
        let m = self.m.clone();
        let mut mix = m.swatches[best.swatch].mix;
        mix[12] = job.paint[12];
        mix[13] = job.paint[13];
        mix[14] = job.paint[14];
        let path = best.pts.iter().map(|p| DVec2::new(p[0] as f64 * m.cell, p[1] as f64 * m.cell)).collect();
        let lead = if self.kind == ToolKind::Brush { (APPROACH_HW * best.hw as f64 * m.cell).clamp(APPROACH_MIN, APPROACH_MAX) } else { 0.0 };
        let foot = m.footprint(&mut self.scratch, &best.pts, best.hw, self.kind == ToolKind::Brush);
        let plan = Plan { path, lead, size: best.size, mix, speed: self.range.speed, kind: self.kind, dry: false, pts: best.pts, hw: best.hw, lab: m.swatches[best.swatch].lab, predicted: best.score, width_gain: self.width_gain, pass: 0 };
        Some((plan, foot))
    }

    fn hit(&mut self) -> Vec<usize> {
        self.sched.misses = 0;
        let tiles = std::mem::take(&mut self.last_tiles);
        for &t in &tiles {
            self.tabu[t] *= VISIT_DECAY;
        }
        tiles
    }

    fn miss(&mut self) -> bool {
        for t in std::mem::take(&mut self.last_tiles) {
            self.tabu[t] *= TABU;
        }
        self.sched.misses += 1;
        self.sched.misses >= MISS_LIMIT
    }

    fn run(&mut self, job: &Job) -> Batch {
        if self.kind != job.kind || self.range.lo != job.range.lo || self.range.hi != job.range.hi {
            self.kind = job.kind;
            self.range = job.range;
        }
        if let Some(s) = job.reset {
            let start = if self.started { self.sched.start.max(s) } else { s };
            self.started = true;
            self.sched = Schedule::new(s);
            self.sched.start = start;
            self.tabu.iter_mut().for_each(|t| *t = 1.0);
            self.blobs.clear();
        }
        self.set_focus(job.focus);
        self.intent = job.intent;
        self.follow = (job.follow as f32).clamp(0.05, 1.0);
        let fine = (1.0 - self.sched.size_hi / self.sched.start.max(1e-9)).clamp(0.0, 1.0) as f32;
        self.fine = fine;
        let a = GATE * fine;
        let m = self.m.clone();
        self.gate = m.detail.par_iter().map(|d| 1.0 - a + a * d).collect();
        self.refresh_tiles();
        self.longest = (job.longest / self.m.cell) as f32;
        self.width_gain = job.width_gain;
        self.opacity = job.opacity;
        self.efficiency = job.efficiency;
        self.tile_eff = job.tile_eff.clone();
        self.refresh_reference();
        self.blobs.clear();
        self.sched.misses = 0;
        let (mut plans, mut feet, mut scheds, mut cooled) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        while plans.len() < LOOKAHEAD {
            match self.plan_one(job) {
                Some((plan, foot)) => {
                    self.apply(&foot, plan.lab);
                    cooled.push(self.hit());
                    plans.push(plan);
                    feet.push(foot);
                    scheds.push(self.sched.clone());
                }
                None => {
                    if self.miss() {
                        break;
                    }
                }
            }
        }
        let planned = plans.len();
        let o = self.opacity;
        let mut keep = vec![false; planned];
        let mut shade = std::mem::take(&mut self.scratch);
        for i in (0..planned).rev() {
            let (mut vis, mut area) = (0.0f32, 0.0f32);
            for &(k, a) in &feet[i] {
                let a = a * o;
                let w = self.m.weight[k];
                vis += a * (1.0 - shade[k]) * w;
                area += a * w;
                shade[k] = 1.0 - (1.0 - shade[k]) * (1.0 - a);
            }
            let f = vis / area.max(1e-6);
            keep[i] = f >= PRUNE_VISIBLE && plans[i].predicted * f < -MIN_GAIN;
        }
        for foot in &feet {
            for &(k, _) in foot {
                shade[k] = 0.0;
            }
        }
        self.scratch = shade;
        let (mut out, mut pruned, mut last) = (Vec::new(), 0, None);
        for (i, mut p) in plans.into_iter().enumerate() {
            if out.len() >= EXECUTE {
                break;
            }
            if keep[i] {
                p.dry = false;
                out.push(p);
                last = Some(i);
            } else {
                pruned += 1;
            }
        }
        if out.len() >= EXECUTE {
            if let Some(i) = last {
                self.sched = scheds[i].clone();
            }
        }
        let executed = last.map(|i| i + 1).unwrap_or(0);
        for (i, tiles) in cooled.iter().enumerate() {
            if i >= executed || !keep[i] {
                for &t in tiles {
                    self.tabu[t] = (self.tabu[t] / VISIT_DECAY).min(1.0);
                }
            }
        }
        Batch { plans: out, sched: self.sched.clone(), planned, pruned, gen: job.gen, reference: self.reference.clone() }
    }
}

pub struct Evolver {
    base: image::RgbaImage,
    first: Vec<[f32; 3]>,
    pub prep: Prep,
    gain: Option<[f32; 3]>,
    offset: [f32; 3],
    base_gain: [f32; 3],
    pub w: usize,
    pub h: usize,
    pub factor: u32,
    cell: f64,
    photo: Vec<[f32; 3]>,
    white: [f32; 3],
    black: [f32; 3],
    pub calibrated: bool,
    palette: Arc<Vec<Swatch>>,
    model: Option<Arc<Model>>,
    mask: Vec<f32>,
    focus: Option<Focus>,
    current: Vec<[f32; 3]>,
    err: Vec<f32>,
    scratch: Vec<f32>,
    rng: StdRng,
    tool: Option<(ToolKind, BrushShape)>,
    range: Option<ToolRange>,
    reset: Option<f64>,
    photo_gen: u64,
    applied: Option<f64>,
    seen: Option<f64>,
    pub size_hi: f64,
    width_gain: f64,
    opacity: f32,
    efficiency: f32,
    tile_eff: Arc<Vec<f32>>,
    planned_ref: Option<Arc<Vec<[f32; 3]>>>,
    sketch: Option<Sketch>,
    job: Option<std::thread::JoinHandle<(Sketch, Batch)>>,
    ready: Option<Batch>,
    queue: VecDeque<Plan>,
    synced: bool,
    gen: u64,
    pub finished: bool,
    dry_next: bool,
    jobs: u32,
    empty_batches: u32,
    retry_at: Option<std::time::Instant>,
    pub tried: u64,
    pub painted: u64,
    pub kept: u64,
    pub pruned: u64,
    pub passes: u32,
    pub start_err: f32,
    pub mean_err: f32,
    pub status: String,
    pub phase: Phase,
}

impl Evolver {
    fn linear(img: &image::RgbaImage, prep: Prep, cell_mm: f32) -> Vec<[f32; 3]> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let img = prepare(img, prep, cell_mm);
        (0..w * h)
            .into_par_iter()
            .map(|k| {
                let p = img.get_pixel((k % w) as u32, (h - 1 - k / w) as u32);
                [decode(p[0] as f32 / 255.0), decode(p[1] as f32 / 255.0), decode(p[2] as f32 / 255.0)]
            })
            .collect()
    }

    pub fn new(photo: &image::RgbaImage, film_w: u32, film_h: u32, dx: f32, set: &PigmentSet, cmf: &[f32], seed: u64, prep: Prep) -> Self {
        let factor = (film_w / EVAL_WIDTH).max(1);
        let (w, h) = ((film_w / factor) as usize, (film_h / factor) as usize);
        let img = image::imageops::resize(&cover_crop(photo, w as f32 / h as f32), w as u32, h as u32, image::imageops::FilterType::Triangle);
        let white = color::mix_rgb(set, cmf, &color::PaintMix::pure(WHITE).frac).map(decode);
        let black = color::mix_rgb(set, cmf, &color::PaintMix::pure(BLACK).frac).map(decode);
        let photo = Self::linear(&img, prep, dx * factor as f32 * 1e3);
        Self {
            base: img,
            first: Vec::new(),
            prep,
            gain: None,
            offset: [0.0; 3],
            base_gain: [1.0; 3],
            w,
            h,
            factor,
            cell: dx as f64 * factor as f64,
            photo,
            white,
            black,
            calibrated: false,
            palette: Arc::new(swatches(set, cmf)),
            model: None,
            mask: vec![1.0; w * h],
            focus: None,
            current: vec![[0.0; 3]; w * h],
            err: vec![0.0; w * h],
            scratch: vec![0.0; w * h],
            rng: StdRng::seed_from_u64(seed),
            tool: None,
            range: None,
            reset: None,
            photo_gen: 0,
            applied: None,
            seen: None,
            size_hi: 0.0,
            width_gain: 1.0,
            opacity: 0.9,
            efficiency: 1.0,
            tile_eff: Arc::new(vec![1.0; w.div_ceil(TILE) * h.div_ceil(TILE)]),
            planned_ref: None,
            sketch: None,
            job: None,
            ready: None,
            queue: VecDeque::new(),
            synced: false,
            gen: 0,
            finished: false,
            empty_batches: 0,
            retry_at: None,
            dry_next: false,
            jobs: 0,
            tried: 0,
            painted: 0,
            kept: 0,
            pruned: 0,
            passes: 0,
            start_err: 0.0,
            mean_err: 0.0,
            status: String::new(),
            phase: Phase::Wait(WARMUP_FRAMES),
        }
    }

    pub fn set_prep(&mut self, prep: Prep) {
        self.prep = prep;
        self.planned_ref = None;
        self.photo = Self::linear(&self.base, prep, (self.cell * 1e3) as f32);
        if self.gain.is_some() {
            let gain = self.base_gain.map(|g| g * (-prep.lift_ev).exp2());
            self.calibrate(gain, self.offset);
        }
        self.sketch = None;
        self.gen += 1;
        self.queue.clear();
        self.ready = None;
        self.synced = false;
        self.finished = false;
        self.start_err = 0.0;
    }

    pub fn calibrate(&mut self, gain: [f32; 3], offset: [f32; 3]) {
        self.gain = Some(gain);
        self.offset = offset;
        let (white, black) = (self.white, self.black);
        let show = self.base_gain;
        let (lb, lw) = (view_lab(black, show)[0], view_lab(white, show)[0]);
        let lift = [0, 1, 2].map(|c| show[c] / gain[c]);
        let shown: Vec<Swatch> = self.palette.iter().map(|s| Swatch { mix: s.mix, rgb: s.rgb, lab: view_lab(s.rgb, show) }).collect();
        let lifted: Vec<[f32; 3]> = self.photo.par_iter().map(|p| lab([0, 1, 2].map(|c| if lift[c] == 1.0 { p[c] } else { tonemap(tonemap_inverse(p[c]) * lift[c]) }))).collect();
        let mut sample: Vec<f32> = lifted.iter().step_by(QUANTILE_STRIDE).map(|v| v[0]).collect();
        sample.sort_by(|a, b| a.total_cmp(b));
        let at = |q: f32| sample[((sample.len() - 1) as f32 * q) as usize];
        let l0 = at(TONE_LOW);
        let lh = at(TONE_HIGH).max(lb + TONE_MIN_SPAN).min(lw);
        let mapped: Vec<[f32; 3]> = lifted
            .into_par_iter()
            .map(|v| {
                let l = if v[0] >= lh || lh <= l0 { v[0].max(lb) } else { lb + (lh - lb) * ((v[0] - l0) / (lh - l0)).clamp(0.0, 1.0).powf(TONE_GAMMA) };
                [l.min(lw), v[1], v[2]]
            })
            .collect();
        let target = gamut_map(mapped, &shown);
        let (tangent, edges) = structure(&target, self.w, self.h, TENSOR_RADIUS);
        let weight = edges.iter().map(|e| 1.0 + EDGE_WEIGHT * e).collect();
        let base = flow_field(&tangent, self.w, self.h);
        let flow = etf(&base, &edges, self.w, self.h, ETF_RADIUS, ETF_ITERS);
        let detail = box_blur(&edges, self.w, self.h, DETAIL_RADIUS);
        let (label, centres) = segment(&target, self.w, self.h, 7);
        let n = centres.len();
        let kin: Vec<bool> = (0..n * n).map(|k| de(centres[k / n], centres[k % n]) < KIN_DE).collect();
        let swatches = Arc::new(shown);
        self.model = Some(Arc::new(Model { w: self.w, h: self.h, cell: self.cell, target, flow, detail, label, kin, regions: n, weight, swatches }));
        self.calibrated = true;
        log::info!("display gain {:.2} {:.2} {:.2}", gain[0], gain[1], gain[2]);
    }

    fn model(&self) -> &Model {
        self.model.as_ref().expect("calibrated")
    }

    pub fn observe(&mut self, lab: Vec<[f32; 3]>) {
        self.current = lab;
        let m = self.model.clone().expect("calibrated");
        self.err = self.current.par_iter().zip(&m.target).map(|(a, b)| de(*a, *b)).collect();
        self.mean_err = self.err.iter().sum::<f32>() / self.err.len().max(1) as f32;
        if self.first.is_empty() {
            self.first = self.current.clone();
        }
        if self.start_err == 0.0 {
            self.start_err = self.first.par_iter().zip(&m.target).map(|(a, b)| de(*a, *b)).sum::<f32>() / self.first.len().max(1) as f32;
        }
        self.synced = true;
    }

    pub fn synced(&self) -> bool {
        self.synced
    }

    pub fn set_tool(&mut self, kind: ToolKind, shape: BrushShape, size: f64) -> Option<f64> {
        let user_changed = self.applied.or(self.seen).map(|a| (size - a).abs() > SIZE_CHANGED * a).unwrap_or(true);
        if self.tool == Some((kind, shape)) && !user_changed {
            return None;
        }
        if self.tool != Some((kind, shape)) {
            self.tool = Some((kind, shape));
            self.range = tool_range(kind, shape);
            self.width_gain = 1.0;
            self.opacity = 0.9;
            self.efficiency = 1.0;
            self.tile_eff = Arc::new(vec![1.0; self.tile_eff.len()]);
        }
        self.applied = None;
        self.seen = Some(size);
        self.gen += 1;
        self.queue.clear();
        self.ready = None;
        self.finished = false;
        self.synced = false;
        let r = self.range?;
        self.size_hi = size.clamp(r.lo, r.hi);
        self.reset = Some(self.size_hi);
        if self.jobs > 0 {
            self.dry_next = true;
            self.passes += 1;
        }
        Some(size)
    }

    pub fn has_tool(&self) -> bool {
        self.range.is_some()
    }

    pub fn applied(&mut self, size: f64) {
        self.applied = Some(size);
    }

    pub fn resume(&mut self) {
        self.phase = Phase::Wait(WARMUP_FRAMES);
        self.tool = None;
        self.applied = None;
        self.seen = None;
        self.queue.clear();
        self.ready = None;
        self.synced = false;
        self.finished = false;
        self.gen += 1;
        self.status.clear();
    }

    pub fn progress(&self) -> f32 {
        let (mut sum, mut total, mut flat) = (0.0f64, 0.0f64, 0.0f64);
        let Some(m) = self.model.as_ref() else { return 0.0 };
        let mut mean = [0.0f64; 3];
        for (t, w) in m.target.iter().zip(&self.mask) {
            for c in 0..3 {
                mean[c] += (t[c] * *w) as f64;
            }
            total += *w as f64;
        }
        if total <= 0.0 {
            return 0.0;
        }
        let mean = mean.map(|v| (v / total) as f32);
        for ((t, e), w) in m.target.iter().zip(&self.err).zip(&self.mask) {
            sum += (*e * *w) as f64;
            flat += (de(*t, mean) * *w) as f64;
        }
        if flat <= 0.0 {
            return 0.0;
        }
        (1.0 - sum / flat).clamp(0.0, 1.0) as f32
    }

    fn set_focus(&mut self, focus: Focus) {
        if self.focus != Some(focus) {
            self.focus = Some(focus);
            self.mask = self.model().mask(focus);
        }
    }

    fn gain(&self, k: usize, delta: f32) -> f32 {
        if delta < 0.0 { delta * self.mask[k] } else { delta }
    }

    pub fn next(&mut self, candidates: usize, follow: f64, intent: f64, longest: f64, focus: Focus, paint: &[f32; 16]) -> Option<Plan> {
        if let Some(p) = self.queue.pop_front() {
            return Some(p);
        }
        if !self.synced {
            return None;
        }
        if self.job.as_ref().map(|j| j.is_finished()).unwrap_or(false) {
            let (sk, batch) = self.job.take().unwrap().join().expect("planner");
            let mut sk = sk;
            if let Some(m) = self.model.as_ref().filter(|m| !Arc::ptr_eq(m, &sk.m)) {
                sk.m = m.clone();
                sk.ref_hw = -1.0;
                sk.blobs.clear();
            }
            self.sketch = Some(sk);
            if batch.gen == self.gen {
                let mut batch = batch;
                if self.dry_next {
                    if let Some(p) = batch.plans.first_mut() {
                        p.dry = true;
                    }
                    self.dry_next = false;
                }
                self.tried += batch.planned as u64;
                self.pruned += batch.pruned as u64;
                self.size_hi = batch.sched.size_hi;
                if batch.plans.is_empty() {
                    self.empty_batches += 1;
                    self.finished = self.empty_batches >= EMPTY_BATCHES;
                    self.retry_at = Some(std::time::Instant::now() + std::time::Duration::from_secs_f32(RETRY_SECS));
                } else {
                    self.empty_batches = 0;
                    self.finished = false;
                }
                log::info!("batch planned {} pruned {} painting {} size {:.1} mm pass {} closer {:.0}%", batch.planned, batch.pruned, batch.plans.len(), batch.sched.size_hi * 1e3, self.passes, self.progress() * 100.0);
                self.planned_ref = Some(batch.reference.clone());
                self.ready = Some(batch);
            }
        }
        if self.job.is_some() || self.retry_at.map(|t| std::time::Instant::now() < t).unwrap_or(false) {
            return None;
        }
        self.retry_at = None;
        self.set_focus(focus);
        let pending = self.ready.take().map(|b| b.plans).unwrap_or_default();
        let range = self.range?;
        let kind = self.tool?.0;
        let m = self.model.clone()?;
        let size_hi = self.size_hi;
        let seed = self.rng.random();
        let mut sk = self.sketch.take().unwrap_or_else(|| Sketch::new(m, seed, kind, range, size_hi));
        let job = Job { kind, range, reset: self.reset.take(), candidates, follow, longest, focus, paint: *paint, width_gain: self.width_gain, opacity: self.opacity, efficiency: self.efficiency, tile_eff: self.tile_eff.clone(), intent: intent as f32, gen: self.gen };
        let real = self.current.clone();
        let ahead = pending.clone();
        self.jobs += 1;
        self.job = Some(std::thread::spawn(move || {
            sk.rebase(&real, &ahead);
            let b = sk.run(&job);
            (sk, b)
        }));
        if pending.is_empty() {
            return None;
        }
        self.synced = false;
        self.queue.extend(pending);
        self.queue.pop_front()
    }

    pub fn region(&self, plan: &Plan) -> [usize; 4] {
        self.model().bbox(&plan.pts, plan.hw * 1.5 + 4.0)
    }

    pub fn judge(&mut self, plan: &Plan, region: [usize; 4], after: &[[f32; 3]]) -> bool {
        let m = self.model.clone().expect("calibrated");
        let rw = region[2] - region[0];
        let mut covs = vec![0.0f32; rw * (region[3] - region[1])];
        for (k, c) in m.footprint(&mut self.scratch, &plan.pts, plan.hw, plan.kind == ToolKind::Brush) {
            let (i, j) = (k % self.w, k / self.w);
            if (region[0]..region[2]).contains(&i) && (region[1]..region[3]).contains(&j) {
                covs[(j - region[1]) * rw + (i - region[0])] = c;
            }
        }
        let judged = self.planned_ref.clone();
        let goal: &[[f32; 3]] = judged.as_deref().map(|v| v.as_slice()).unwrap_or(&m.target);
        let mut delta = 0.0f64;
        let (mut pred, mut changed) = (0u32, 0u32);
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for j in region[1]..region[3] {
            for i in region[0]..region[2] {
                let k = j * self.w + i;
                let a = after[(j - region[1]) * rw + (i - region[0])];
                let b = self.current[k];
                let w = m.weight[k];
                delta += self.gain(k, w * (de(a, goal[k]) - de(b, goal[k]))) as f64;
                let cov = covs[(j - region[1]) * rw + (i - region[0])];
                let reach = de(plan.lab, b);
                if reach > 2.0 * CHANGED_DE {
                    if cov > 0.5 {
                        pred += 1;
                    }
                    if de(a, b) > 0.35 * reach {
                        changed += 1;
                    }
                }
                if cov > 0.9 {
                    let d = [plan.lab[0] - b[0], plan.lab[1] - b[1], plan.lab[2] - b[2]];
                    num += (a[0] - b[0]) * d[0] + (a[1] - b[1]) * d[1] + (a[2] - b[2]) * d[2];
                    den += d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                }
            }
        }
        if pred > 20 {
            let ratio = (changed as f64 / pred as f64).clamp(0.5, 2.0);
            let seen = (plan.width_gain * ratio).clamp(0.15, 3.0);
            self.width_gain = (self.width_gain * (seen / self.width_gain).powf(0.25)).clamp(0.15, 3.0);
        }
        if den > 2000.0 {
            let o = (num / den).clamp(0.15, 1.0);
            self.opacity = 0.85 * self.opacity + 0.15 * o;
        }
        if plan.predicted < -MIN_GAIN {
            let ratio = ((delta as f32) / plan.predicted).clamp(0.0, 1.5);
            self.efficiency += EFFICIENCY_RATE * (ratio - self.efficiency);
            let mid = plan.pts[plan.pts.len() / 2];
            let tile = (mid[1].max(0.0) as usize / TILE).min(self.h.div_ceil(TILE) - 1) * self.w.div_ceil(TILE) + (mid[0].max(0.0) as usize / TILE).min(self.w.div_ceil(TILE) - 1);
            let eff = Arc::make_mut(&mut self.tile_eff);
            if let Some(e) = eff.get_mut(tile) {
                *e += TILE_EFF_RATE * (ratio - *e);
            }
        }
        self.painted += 1;
        let ok = delta < 0.0;
        if ok {
            self.store(region, after);
            self.kept += 1;
        }
        log::debug!("stroke {} predicted {:.0} real {:.0} kept {} gain {:.2} opacity {:.2}", self.painted, plan.predicted, delta, ok, self.width_gain, self.opacity);
        if self.painted % 50 == 0 {
            log::info!("painted {} kept {} planned {} pruned {} pass {} size {:.1} mm closer {:.0}% gain {:.2} opacity {:.2} efficiency {:.2}", self.painted, self.kept, self.tried, self.pruned, self.passes, self.size_hi * 1e3, self.progress() * 100.0, self.width_gain, self.opacity, self.efficiency);
        }
        ok
    }

    pub fn store(&mut self, region: [usize; 4], after: &[[f32; 3]]) {
        let m = self.model.clone().expect("calibrated");
        let rw = region[2] - region[0];
        for j in region[1]..region[3] {
            for i in region[0]..region[2] {
                let k = j * self.w + i;
                let a = after[(j - region[1]) * rw + (i - region[0])];
                let e = de(a, m.target[k]);
                self.mean_err += (e - self.err[k]) / self.err.len() as f32;
                self.current[k] = a;
                self.err[k] = e;
            }
        }
    }
}

fn press_profile(s: f64, total: f64, size: f64) -> f64 {
    let ramp = (TAPER_END as f64 * total).min(PRESS_RAMP_SIZES * size).max(1e-6);
    let t = (s.min(total - s) / ramp).clamp(0.0, 1.0);
    PRESS_START + (1.0 - PRESS_START) * t * t * (3.0 - 2.0 * t)
}

fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let (n, len) = if horizontal { (h, w) } else { (w, h) };
        let at = |line: usize, i: usize| if horizontal { line * w + i } else { i * w + line };
        let lines: Vec<Vec<f32>> = (0..n)
            .into_par_iter()
            .map(|line| {
                let mut pre = vec![0.0f32; len + 1];
                for i in 0..len {
                    pre[i + 1] = pre[i] + src[at(line, i)];
                }
                (0..len)
                    .map(|i| {
                        let (a, b) = (i.saturating_sub(r), (i + r + 1).min(len));
                        (pre[b] - pre[a]) / (b - a) as f32
                    })
                    .collect()
            })
            .collect();
        let mut out = vec![0.0f32; w * h];
        for (line, vals) in lines.into_iter().enumerate() {
            for (i, v) in vals.into_iter().enumerate() {
                out[at(line, i)] = v;
            }
        }
        out
    };
    pass(&pass(src, true), false)
}

fn flow_field(tangent: &[[f32; 3]], w: usize, h: usize) -> Vec<[f32; 3]> {
    let dx: Vec<f32> = tangent.iter().map(|t| t[2] * (t[0] * t[0] - t[1] * t[1])).collect();
    let dy: Vec<f32> = tangent.iter().map(|t| t[2] * 2.0 * t[0] * t[1]).collect();
    let mut fx = vec![0.0f32; w * h];
    let mut fy = vec![0.0f32; w * h];
    for (r, wgt) in [(6usize, 1.0f32), (24, 0.6), (96, 0.4)] {
        let bx = box_blur(&box_blur(&dx, w, h, r), w, h, r);
        let by = box_blur(&box_blur(&dy, w, h, r), w, h, r);
        for k in 0..w * h {
            fx[k] += wgt * bx[k];
            fy[k] += wgt * by[k];
        }
    }
    let mags: Vec<f32> = fx.iter().zip(&fy).map(|(x, y)| (x * x + y * y).sqrt()).collect();
    let mut sorted = mags.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let strong = sorted[sorted.len() * 9 / 10].max(1e-6);
    (0..w * h)
        .into_par_iter()
        .map(|k| {
            let th = 0.5 * fy[k].atan2(fx[k]);
            [th.cos(), th.sin(), (mags[k] / strong).min(1.0)]
        })
        .collect()
}

fn etf(tangent: &[[f32; 3]], mag: &[f32], w: usize, h: usize, r: usize, iters: usize) -> Vec<[f32; 3]> {
    let mut t: Vec<[f32; 2]> = tangent.iter().map(|v| [v[0], v[1]]).collect();
    let r = r as isize;
    for _ in 0..iters {
        for horizontal in [true, false] {
            let prev = t;
            t = (0..w * h)
                .into_par_iter()
                .map(|k| {
                    let (i, j) = ((k % w) as isize, (k / w) as isize);
                    let tx = prev[k];
                    let gx = mag[k];
                    let mut acc = [0.0f32; 2];
                    for d in -r..=r {
                        let (x, y) = if horizontal { ((i + d).clamp(0, w as isize - 1), j) } else { (i, (j + d).clamp(0, h as isize - 1)) };
                        let n = y as usize * w + x as usize;
                        let ty = prev[n];
                        let dot = tx[0] * ty[0] + tx[1] * ty[1];
                        let wm = 0.5 * (1.0 + (ETF_ETA * (mag[n] - gx)).tanh());
                        let wgt = dot.abs() * wm * if dot < 0.0 { -1.0 } else { 1.0 };
                        acc[0] += wgt * ty[0];
                        acc[1] += wgt * ty[1];
                    }
                    let l = (acc[0] * acc[0] + acc[1] * acc[1]).sqrt();
                    if l > 1e-6 { [acc[0] / l, acc[1] / l] } else { tx }
                })
                .collect();
        }
    }
    let near = box_blur(mag, w, h, (r as usize) * iters);
    (0..w * h).into_par_iter().map(|k| [t[k][0], t[k][1], (near[k] * ETF_REACH).min(1.0)]).collect()
}

fn structure(lab: &[[f32; 3]], w: usize, h: usize, radius: usize) -> (Vec<[f32; 3]>, Vec<f32>) {
    let l = |i: isize, j: isize| lab[(j.clamp(0, h as isize - 1) as usize) * w + i.clamp(0, w as isize - 1) as usize];
    let tensor: Vec<[f32; 3]> = (0..w * h)
        .into_par_iter()
        .map(|k| {
            let (i, j) = ((k % w) as isize, (k / w) as isize);
            let mut t = [0.0f32; 3];
            for c in 0..3 {
                let s = if c == 0 { 1.0 } else { CHROMA_WEIGHT };
                let gx = s * ((l(i + 1, j - 1)[c] + 2.0 * l(i + 1, j)[c] + l(i + 1, j + 1)[c]) - (l(i - 1, j - 1)[c] + 2.0 * l(i - 1, j)[c] + l(i - 1, j + 1)[c]));
                let gy = s * ((l(i - 1, j + 1)[c] + 2.0 * l(i, j + 1)[c] + l(i + 1, j + 1)[c]) - (l(i - 1, j - 1)[c] + 2.0 * l(i, j - 1)[c] + l(i + 1, j - 1)[c]));
                t[0] += gx * gx;
                t[1] += gx * gy;
                t[2] += gy * gy;
            }
            t
        })
        .collect();
    let r = radius as isize;
    let blur = |src: &[[f32; 3]], horizontal: bool| -> Vec<[f32; 3]> {
        (0..w * h)
            .into_par_iter()
            .map(|k| {
                let (i, j) = ((k % w) as isize, (k / w) as isize);
                let mut acc = [0.0f32; 3];
                for d in -r..=r {
                    let (x, y) = if horizontal { ((i + d).clamp(0, w as isize - 1), j) } else { (i, (j + d).clamp(0, h as isize - 1)) };
                    let v = src[y as usize * w + x as usize];
                    acc[0] += v[0];
                    acc[1] += v[1];
                    acc[2] += v[2];
                }
                acc
            })
            .collect()
    };
    let t = blur(&blur(&tensor, true), false);
    let mut strength: Vec<f32> = t.iter().map(|v| v[0] + v[2]).collect();
    strength.sort_by(|a, b| a.total_cmp(b));
    let typical = strength[strength.len() * 3 / 4].max(1e-3);
    let strong = strength[strength.len() * 19 / 20].max(typical);
    let tangent = t
        .par_iter()
        .map(|v| {
            let (jxx, jxy, jyy) = (v[0], v[1], v[2]);
            let tr = jxx + jyy;
            let diff = ((jxx - jyy).powi(2) + 4.0 * jxy * jxy).sqrt();
            let coherence = if tr > 1e-6 { diff / tr } else { 0.0 };
            let theta = 0.5 * (2.0 * jxy).atan2(jxx - jyy) + std::f32::consts::FRAC_PI_2;
            [theta.cos(), theta.sin(), coherence * (tr / typical).min(1.0)]
        })
        .collect();
    let edges = t.par_iter().map(|v| ((v[0] + v[2]) / strong).min(1.0)).collect();
    (tangent, edges)
}

impl App {
    pub(crate) fn evolving(&self) -> bool {
        self.evo.active
    }

    pub(crate) fn tool_size(&self, kind: ToolKind) -> f64 {
        match kind {
            ToolKind::Pour => self.pour_radius as f64,
            ToolKind::Brush => self.brush.size(),
            ToolKind::Roller => self.roller.width,
            ToolKind::Sponge => self.sponge.size,
            ToolKind::Stick => self.tool.stick_diameter,
            ToolKind::Bottle => self.tool.bottle_radius,
            ToolKind::Bucket => self.tool.bucket_radius,
            ToolKind::Knife => self.tool.knife_width,
            ToolKind::Rubber => self.rubber_size,
        }
    }

    pub(crate) fn set_tool_size(&mut self, kind: ToolKind, size: f64) {
        match kind {
            ToolKind::Pour => self.pour_radius = size as f32,
            ToolKind::Brush => {
                if (self.brush.size() - size).abs() > 0.02 * size {
                    let (shape, n, len) = (self.brush.shape, self.brush.bristles.len(), self.brush.len);
                    self.rebuild_brush(shape, n, len, size);
                }
            }
            ToolKind::Roller => self.roller.width = size,
            ToolKind::Sponge => self.sponge.size = size,
            ToolKind::Stick => self.tool.stick_diameter = size,
            ToolKind::Bottle => self.tool.bottle_radius = size,
            ToolKind::Bucket => self.tool.bucket_radius = size,
            ToolKind::Knife => self.tool.knife_width = size,
            ToolKind::Rubber => self.rubber_size = size,
        }
    }

    pub(crate) fn load_photo(&mut self, path: &Path) {
        let aspect = self.film.settings.width_m / self.film.settings.height_m;
        match load_photo(path, aspect) {
            Ok(p) => {
                self.notice(&format!("Loaded {}", p.name));
                self.evo.photo = Some(p);
                self.evo.photo_gen += 1;
            }
            Err(e) => self.notice(&format!("Could not open that image: {}", e)),
        }
    }

    pub(crate) fn pick_photo(&mut self) {
        let file = rfd::FileDialog::new().set_title("Choose a photo to paint").add_filter("Images", &["jpg", "jpeg", "png", "webp", "bmp"]).pick_file();
        if let Some(path) = file {
            self.load_photo(&path);
        }
    }

    pub(crate) fn start_evolve(&mut self) {
        let Some(photo) = &self.evo.photo else { return };
        if let Some(p) = self.painter.as_mut() {
            p.stop();
        }
        let gen = self.evo.photo_gen;
        match self.evo.run.as_mut().filter(|ev| ev.photo_gen == gen) {
            Some(ev) => ev.resume(),
            None => {
                let mut ev = Evolver::new(&photo.image, self.film.w, self.film.h, self.film.dx, &self.film.pigments, &self.film.cmf_table, self.auto.as_ref().map(|a| a.seed as u64).unwrap_or_else(rand::random), self.evo.prep);
                ev.photo_gen = gen;
                self.evo.run = Some(ev);
            }
        }
        self.begin_full();
        self.tool.flowing = false;
        self.air.detach();
        self.evo.saved_paint = Some(self.paint.frac);
        self.evo.saved_height = Some(self.tool.height);
        self.evo.saved_len = Some(self.brush.len);
        self.evo.saved_load = Some(self.brush.capacity / self.brush.base_capacity.max(1e-30));
        self.evo.user_sizes.clear();
        self.evo.active = true;
    }

    pub(crate) fn stop_evolve(&mut self) {
        if !self.evo.active {
            return;
        }
        self.evo.active = false;
        self.brush_raise = 0.0;
        self.tool.flowing = false;
        self.tool.vel = DVec3::ZERO;
        self.air.detach();
        if let Some(f) = self.evo.saved_paint.take() {
            self.paint.frac = f;
        }
        let shape = self.brush.shape;
        for (kind, s, size) in std::mem::take(&mut self.evo.user_sizes) {
            if kind != ToolKind::Brush || s == shape {
                self.set_tool_size(kind, size);
            }
        }
        if let Some(len) = self.evo.saved_len.take() {
            if (self.brush.len - len).abs() > 1e-6 {
                let (shape, n, size) = (self.brush.shape, self.brush.bristles.len(), self.brush.size());
                self.rebuild_brush(shape, n, len, size);
            }
        }
        if let Some(k) = self.evo.saved_load.take() {
            self.brush.capacity = self.brush.base_capacity * k;
        }
        if let Some(h) = self.evo.saved_height.take() {
            self.tool.height = h;
        }
    }

    pub(crate) fn restore_exposure(&mut self) {
        self.renderer.exposure = crate::render::DEFAULT_EXPOSURE;
        self.stop_evolve();
        self.evo.run = None;
    }

    pub(crate) fn evolve_tick(&mut self) -> bool {
        if !self.evo.active {
            return false;
        }
        let Some(mut ev) = self.evo.run.take() else { return false };
        if !self.paused {
            let phase = std::mem::replace(&mut ev.phase, Phase::Plan);
            ev.phase = self.evolve_phase(&mut ev, phase);
        }
        self.evo.run = Some(ev);
        true
    }

    fn evolve_phase(&mut self, ev: &mut Evolver, phase: Phase) -> Phase {
        let z = self.tool.height;
        match phase {
            Phase::Wait(n) => {
                self.tool.flowing = false;
                if n == 0 {
                    Phase::Observe
                } else {
                    Phase::Wait(n - 1)
                }
            }
            Phase::Observe => {
                self.tool.flowing = false;
                let rgb = self.read_linear(ev, [0, 0, ev.w, ev.h]);
                if !ev.calibrated {
                    let base = self.display_gain(ev, &rgb).map(|g| g * (-crate::render::DEFAULT_EXPOSURE).exp2());
                    ev.base_gain = base;
                    let gain = base.map(|g| g * (-ev.prep.lift_ev).exp2());
                    log::info!("display gain {:.2} {:.2} {:.2}", gain[0], gain[1], gain[2]);
                    ev.calibrate(gain, [0.0; 3]);
                }
                let show = ev.base_gain;
                ev.observe(rgb.into_par_iter().map(|c| view_lab(c, show)).collect());
                Phase::Plan
            }
            Phase::Plan => {
                self.tool.flowing = false;
                self.tool.vel = DVec3::ZERO;
                let settled = self.evo.prep_touched.map(|t| t.elapsed().as_secs_f32() > PREP_SETTLE).unwrap_or(true);
                if ev.prep != self.evo.prep && settled {
                    ev.set_prep(self.evo.prep);
                    return Phase::Observe;
                }
                let (kind, shape) = (self.tool.kind, self.brush.shape);
                if let Some(size) = ev.set_tool(kind, shape, self.tool_size(kind)) {
                    self.evo.user_sizes.retain(|u| u.0 != kind || u.1 != shape);
                    self.evo.user_sizes.push((kind, shape, size));
                }
                if !ev.has_tool() {
                    ev.status = "This tool cannot paint from a photo; pick another tool to keep painting".to_string();
                    return Phase::Plan;
                }
                ev.status.clear();
                match ev.next(self.evo.candidates, self.evo.follow, self.evo.intent, self.evo.longest, self.evo.focus, &self.paint.frac) {
                    None if !ev.synced() => Phase::Observe,
                    None => {
                        if ev.finished {
                            ev.status = "Nothing found to gain at this size right now; still looking. Lower the tool size for detail, or change the tool or paint".to_string();
                        }
                        Phase::Plan
                    }
                    Some(plan) => {
                        if plan.dry && self.evo.dry_passes {
                            self.begin_full();
                            self.film.dry_now(&self.gpu);
                        }
                        self.prepare_tool(&plan);
                        ev.applied(self.tool_size(plan.kind));
                        self.begin_stroke();
                        let (from, _) = plan.entry();
                        self.tool.pos = DVec3::new(from.x, from.y, z);
                        Phase::Lower(plan, 0)
                    }
                }
            }
            Phase::Lower(plan, n) => {
                let (from, _) = plan.entry();
                self.tool.pos = DVec3::new(from.x, from.y, z);
                let thread = is_thread(plan.kind);
                self.tool.flowing = thread;
                let fall = if thread { ((2.0 * z / 9.81).sqrt() / self.sim_dt.max(1e-6)).ceil() as u32 } else { 0 };
                if n + 1 < LOWER_FRAMES.max(fall) {
                    Phase::Lower(plan, n + 1)
                } else if plan.lead > 0.0 {
                    Phase::Approach(plan, 0.0)
                } else {
                    Phase::Stroke(plan, 0.0)
                }
            }
            Phase::Approach(plan, s) => {
                let s1 = (s + (plan.lead / APPROACH_STEPS).min(plan.speed * self.sim_dt)).min(plan.lead);
                let (from, dir) = plan.entry();
                let p = from + dir * s1;
                self.tool.pos = DVec3::new(p.x, p.y, z);
                self.tool.vel = DVec3::new(dir.x, dir.y, 0.0) * plan.speed;
                self.tool.flowing = true;
                self.brush_raise = (self.brush.len - self.tool.height).max(0.0) + HOVER * (1.0 - s1 / plan.lead);
                if s1 >= plan.lead {
                    self.brush_raise = 0.0;
                    Phase::Stroke(plan, 0.0)
                } else {
                    Phase::Approach(plan, s1)
                }
            }
            Phase::Exit(plan, s) => {
                let s1 = (s + (plan.lead / APPROACH_STEPS).min(plan.speed * self.sim_dt)).min(plan.lead);
                let dir = plan.exit_dir();
                let p = *plan.path.last().unwrap() + dir * s1;
                self.tool.pos = DVec3::new(p.x, p.y, z);
                self.tool.vel = DVec3::new(dir.x, dir.y, 0.0) * plan.speed;
                self.tool.flowing = true;
                self.brush_raise = self.brush_hover() * (s1 / plan.lead);
                if s1 >= plan.lead {
                    Phase::Lift(plan, 0)
                } else {
                    Phase::Exit(plan, s1)
                }
            }
            Phase::Stroke(plan, s) => {
                let total = plan.length();
                let travel = if plan.kind == ToolKind::Brush { (plan.speed * self.sim_dt).min(TRAVEL_PER_SIZE * plan.size) } else { plan.speed * self.sim_dt };
                let s1 = (s + travel).min(total);
                let (p0, p1) = (plan.ahead(s), plan.ahead(s1));
                let z = if plan.kind == ToolKind::Brush {
                    self.tool.height = self.brush.len - self.evo.stroke_press * press_profile(s1, total, plan.size);
                    self.tool.height
                } else {
                    z
                };
                self.tool.pos = DVec3::new(p1.x, p1.y, z);
                self.tool.vel = DVec3::new(p1.x - p0.x, p1.y - p0.y, 0.0) / self.sim_dt.max(1e-6);
                self.tool.flowing = plan.kind != ToolKind::Pour;
                if plan.kind == ToolKind::Pour {
                    let r = plan.size;
                    let pig = color::PaintMix { frac: plan.mix }.normalized().packed();
                    let n = ((p1 - p0).length() / r).ceil().max(1.0) as usize;
                    for k in 0..n {
                        let a = p0 + (p1 - p0) * (k as f64 / n as f64);
                        let b = p0 + (p1 - p0) * ((k + 1) as f64 / n as f64);
                        let seg = (b - a).length().max(if s == 0.0 && k == 0 { r } else { 0.0 });
                        let vol = (seg * 2.0 * r * POUR_FILM) as f32;
                        self.pending_deposits.push(Deposit::stroke([a.x as f32, a.y as f32], [b.x as f32, b.y as f32], r as f32, vol, pig, self.film.dx).laid_by(Method::Pour));
                    }
                }
                if s1 < total {
                    Phase::Stroke(plan, s1)
                } else if plan.lead > 0.0 {
                    Phase::Exit(plan, 0.0)
                } else {
                    Phase::Lift(plan, 0)
                }
            }
            Phase::Lift(plan, n) => {
                self.tool.flowing = false;
                self.tool.vel = DVec3::ZERO;
                self.brush_raise = 0.0;
                if n == 0 && is_thread(plan.kind) {
                    self.air.detach();
                }
                if n + 1 >= LIFT_FRAMES {
                    Phase::Settle(plan, 0)
                } else {
                    Phase::Lift(plan, n + 1)
                }
            }
            Phase::Settle(plan, n) => {
                self.tool.flowing = false;
                let airborne = !self.air.rods.is_empty() || !self.air.drops.is_empty();
                if n + 1 < SETTLE_FRAMES || (airborne && n < THREAD_SETTLE_MAX) {
                    return Phase::Settle(plan, n + 1);
                }
                let region = ev.region(&plan);
                let after = self.read_eval(ev, region);
                if !ev.judge(&plan, region, &after) && self.rollback().is_err() {
                    ev.store(region, &after);
                }
                Phase::Plan
            }
        }
    }

    fn brush_hover(&self) -> f64 {
        (self.brush.len - self.tool.height).max(0.0) + HOVER
    }

    fn prepare_tool(&mut self, plan: &Plan) {
        let mut f = plan.mix;
        f[14] = (self.tool.mu as f32).ln() + 10.0;
        for i in 0..13 {
            self.paint.frac[i] = plan.mix[i];
        }
        match plan.kind {
            ToolKind::Pour => self.pour_radius = plan.size as f32,
            ToolKind::Brush => {
                let user_len = self.evo.saved_len.unwrap_or(self.brush.len);
                let len = (BRISTLE_PER_SIZE * plan.size).clamp(BRISTLE_MIN.min(user_len), user_len);
                if (self.brush.size() - plan.size).abs() > 0.02 * plan.size || (self.brush.len - len).abs() > 0.02 * len {
                    let (shape, n) = (self.brush.shape, self.brush.bristles.len());
                    self.rebuild_brush(shape, n, len, plan.size);
                }
                let press = (PRESS_PER_SIZE * plan.size).clamp(PRESS_MIN, self.brush_press(self.brush_pressure).max(PRESS_MIN));
                let press = press * (1.0 + PRESS_JITTER * (2.0 * rand::random::<f64>() - 1.0));
                self.evo.stroke_press = press;
                self.tool.height = self.brush.len - press * PRESS_START;
                self.brush.capacity = self.brush.base_capacity * (plan.size / LOAD_SIZE).clamp(1.0, LOAD_MAX);
                self.brush.clean();
                self.brush.dip(&f);
            }
            ToolKind::Roller => {
                self.roller.width = plan.size;
                self.roller.clean();
                self.roller.dip(&f);
            }
            ToolKind::Sponge => {
                self.sponge.size = plan.size;
                self.sponge.clean();
                self.sponge.dip(&f);
            }
            ToolKind::Stick => {
                self.tool.stick_diameter = plan.size;
                self.tool.reload();
            }
            ToolKind::Rubber => {}
            ToolKind::Bottle => self.tool.bottle_radius = plan.size,
            ToolKind::Bucket => self.tool.bucket_radius = plan.size,
            ToolKind::Knife => {
                self.tool.knife_width = plan.size;
                self.load_knife();
                let need = KNIFE_LOAD_MARGIN * self.tool.knife_height.max(1e-4) * plan.size * plan.length() + self.tool.knife_width * crate::KNIFE_HOLD_PER_WIDTH;
                let each = (need / self.knife_bands.len() as f64).min(self.knife_bands[0].vol);
                for b in self.knife_bands.iter_mut() {
                    b.vol = each;
                    let t = (2.0 * rand::random::<f32>() - 1.0).powi(3);
                    let total: f32 = b.comp[..12].iter().sum::<f32>().max(1e-6);
                    if t > 0.0 {
                        b.comp[WHITE] += KNIFE_STREAK_LIGHT * t * total;
                    } else {
                        b.comp[BLACK] += KNIFE_STREAK_DARK * -t * total;
                    }
                }
                let d = plan.entry().1;
                self.knife_dir = [d.x, d.y];
                self.knife_track = None;
                self.knife_prev = None;
            }
        }
    }

    fn display_gain(&self, ev: &Evolver, albedo: &[[f32; 3]]) -> [f32; 3] {
        self.refine.run(&self.gpu, &self.film);
        let shown = self.beauty.render(&self.gpu, &self.renderer, &self.cam, &self.film, true, ev.w as u32, ev.h as u32, 1, Some(&self.refine));
        let mut ratios: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for j in (0..ev.h).step_by(3) {
            for i in (0..ev.w).step_by(3) {
                let p = shown.get_pixel(i as u32, (ev.h - 1 - j) as u32);
                let a = albedo[j * ev.w + i];
                for c in 0..3 {
                    let y = decode(p[c] as f32 / 255.0);
                    if y < 0.93 && a[c] > 0.02 {
                        ratios[c].push(tonemap_inverse(y) / a[c]);
                    }
                }
            }
        }
        let mut gain = [1.0f32; 3];
        for c in 0..3 {
            if !ratios[c].is_empty() {
                ratios[c].sort_by(|a, b| a.total_cmp(b));
                gain[c] = ratios[c][ratios[c].len() / 2].clamp(0.2, 20.0);
            }
        }
        gain
    }

    pub(crate) fn dump_observed(&self, path: &str) {
        let Some(ev) = self.evo.run.as_ref() else { return };
        let gain = ev.base_gain;
        let rgb = self.read_linear(ev, [0, 0, ev.w, ev.h]);
        let mut im = image::RgbImage::new(ev.w as u32, ev.h as u32);
        for (k, p) in rgb.iter().enumerate() {
            let px = [0, 1, 2].map(|c| (encode(tonemap(p[c] * gain[c])) * 255.0) as u8);
            im.put_pixel((k % ev.w) as u32, (ev.h - 1 - k / ev.w) as u32, image::Rgb(px));
        }
        let _ = im.save(path);
    }

    fn read_eval(&self, ev: &Evolver, region: [usize; 4]) -> Vec<[f32; 3]> {
        let show = ev.base_gain;
        self.read_linear(ev, region).into_par_iter().map(|c| view_lab(c, show)).collect()
    }

    fn read_linear(&self, ev: &Evolver, region: [usize; 4]) -> Vec<[f32; 3]> {
        let f = ev.factor as usize;
        let (rw, rh) = (region[2] - region[0], region[3] - region[1]);
        if rw == 0 || rh == 0 {
            return Vec::new();
        }
        let (cw, ch) = ((rw * f) as u32, (rh * f) as u32);
        let bpr = ((cw * 8 + 255) / 256) * 256;
        let buf = self.gpu.readback("evolve_read", (bpr * ch) as u64);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("evolve_read") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &self.film.color_tex, mip_level: 0, origin: wgpu::Origin3d { x: (region[0] * f) as u32, y: (region[1] * f) as u32, z: 0 }, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(ch) } },
            wgpu::Extent3d { width: cw, height: ch, depth_or_array_layers: 1 },
        );
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range().expect("map");
        let halves: &[u16] = bytemuck::cast_slice(&data);
        let row = (bpr / 2) as usize;
        let inv = 1.0 / (f * f) as f32;
        let out = (0..rw * rh)
            .into_par_iter()
            .map(|k| {
                let (i, j) = (k % rw, k / rw);
                let mut rgb = [0.0f32; 3];
                for y in j * f..(j + 1) * f {
                    for x in i * f..(i + 1) * f {
                        let o = y * row + x * 4;
                        for c in 0..3 {
                            rgb[c] += half::f16::from_bits(halves[o + c]).to_f32();
                        }
                    }
                }
                rgb.map(|v| v * inv)
            })
            .collect();
        drop(data);
        buf.unmap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn sketch_bench() {
        let path = std::env::var("EVO_PHOTO").unwrap();
        let out = std::env::var("EVO_OUT").unwrap_or_default();
        let counts: Vec<usize> = std::env::var("EVO_N").unwrap_or_else(|_| "900".to_string()).split(',').filter_map(|v| v.parse().ok()).collect();
        let total: usize = counts.iter().sum();
        let set = color::load_pigments();
        let cmf = color::cmf_table();
        let img = image::open(&path).unwrap().to_rgba8();
        let photo = cover_crop(&img, 1.5);
        let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let prep = Prep { contrast: env("EVO_CONTRAST", 1.0), blur_mm: env("EVO_BLUR", 0.0), sharpen: env("EVO_SHARPEN", 0.0), smooth_mm: env("EVO_SMOOTH", 0.0), lift_ev: env("EVO_LIFT", 0.0) };
        let mut ev = Evolver::new(&photo, 3072, 2048, 1.2 / 3072.0, &set, &cmf, 7, prep);
        let gain = [2.11f32 * 1.27, 2.2 * 1.27, 2.27 * 1.27];
        ev.base_gain = gain;
        ev.calibrate(gain.map(|g| g * (-prep.lift_ev).exp2()), [0.0; 3]);
        if let Ok(path) = std::env::var("EVO_TARGET") {
            let mm = ev.model.as_ref().unwrap();
            let mut im = image::RgbImage::new(mm.w as u32, mm.h as u32);
            for (k, l) in mm.target.iter().enumerate() {
                let fy = (l[0] + 16.0) / 116.0;
                let (fx, fz) = (fy + l[1] / 500.0, fy - l[2] / 200.0);
                let inv = |t: f32| if t > 0.2069 { t * t * t } else { (t - 16.0 / 116.0) / 7.787 };
                let (x, y, z) = (0.95047 * inv(fx), inv(fy), 1.08883 * inv(fz));
                let rgb = [3.2406 * x - 1.5372 * y - 0.4986 * z, -0.9689 * x + 1.8758 * y + 0.0415 * z, 0.0557 * x - 0.2040 * y + 1.0570 * z];
                let px = rgb.map(|c| (encode(c.clamp(0.0, 1.0)) * 255.0) as u8);
                im.put_pixel((k % mm.w) as u32, (mm.h - 1 - k / mm.w) as u32, image::Rgb(px));
            }
            im.save(&path).unwrap();
        }
        let m = ev.model.clone().unwrap();
        let _ = env_logger::try_init();
        let kind = ToolKind::Brush;
        let flat = std::env::var("EVO_FLAT").is_ok();
        let range = tool_range(kind, if flat { BrushShape::Flat } else { BrushShape::Round }).unwrap();
        let sizes: Vec<f64> = std::env::var("EVO_SIZE").unwrap_or_else(|_| "16".to_string()).split(',').filter_map(|v| v.parse::<f64>().ok()).map(|v| v * 1e-3).collect();
        let mut stage = 0usize;
        let mut sk = Sketch::new(m.clone(), 7, kind, range, sizes[0]);
        let base_lab = if std::env::var("EVO_BLACK").is_ok() { lab(ev.black) } else { [80.0f32, 1.0, 6.0] };
        let mut base = vec![base_lab; m.w * m.h];
        let err0 = base.iter().zip(&m.target).map(|(a, b)| de(*a, *b)).sum::<f32>();
        let mut focus = Focus { shape: FocusShape::Off, a: [0.5, 0.5], b: [0.75, 0.5], soft: 0.5, outside: 0.0 };
        if let Ok(spec) = std::env::var("EVO_FOCUS") {
            let (kind, nums) = spec.split_once(':').unwrap_or((spec.as_str(), ""));
            let v: Vec<f32> = nums.split(',').filter_map(|x| x.parse().ok()).collect();
            if v.len() >= 4 {
                focus.shape = if kind == "circle" { FocusShape::Circle } else { FocusShape::Rect };
                (focus.a, focus.b) = ([v[0], v[1]], [v[2], v[3]]);
            }
        }
        let opacity: f32 = std::env::var("EVO_OPACITY").ok().and_then(|v| v.parse().ok()).unwrap_or(0.7);
        let (mut done, mut next_report, mut empty) = (0usize, 0usize, 0usize);
        let t0 = std::time::Instant::now();
        while done < total {
            let mut reset = if done == 0 { Some(sizes[0]) } else { None };
            let next: usize = counts[..=stage.min(counts.len() - 1)].iter().sum();
            if done >= next && stage + 1 < sizes.len() {
                stage += 1;
                reset = Some(sizes[stage]);
            }
            let job = Job { kind, range, reset, candidates: 32, follow: env("EVO_FOLLOW", 0.7) as f64, longest: 0.25, focus, paint: [0.0; 16], width_gain: env("EVO_GAIN", 1.85) as f64, opacity, efficiency: env("EVO_EFF", 1.0), tile_eff: Arc::new(Vec::new()), intent: env("EVO_INTENT", 0.7), gen: 0 };
            sk.rebase(&base, &[]);
            let b = sk.run(&job);
            sk.rebase(&base, &b.plans);
            base = sk.current.clone();
            done += b.plans.len();
            if done >= next_report || b.plans.is_empty() || done >= total {
                let e = base.iter().zip(&m.target).map(|(a, b)| de(*a, *b)).sum::<f32>();
                let ed = base.iter().zip(&m.target).zip(&m.weight).map(|((a, b), w)| (w - 1.0) * de(*a, *b)).sum::<f32>() / m.weight.iter().map(|w| w - 1.0).sum::<f32>();
                print!("edge dE {:.1} ", ed);
                println!("strokes {} closer {:.1}% size {:.1} mm stage {} pruned {} {:.1}s", done, 100.0 * (1.0 - e / err0), b.sched.size_hi * 1e3, stage, b.pruned, t0.elapsed().as_secs_f32());
                next_report += 600;
            }
            if b.plans.is_empty() {
                empty += 1;
                if empty >= EMPTY_BATCHES as usize {
                    if stage + 1 >= sizes.len() {
                        break;
                    }
                    println!("stage {} ran dry at {} strokes", stage, done);
                    done = next;
                    empty = 0;
                }
            } else {
                empty = 0;
            }
        }
        if !out.is_empty() {
            let mut im = image::RgbImage::new(m.w as u32, m.h as u32);
            for (k, l) in base.iter().enumerate() {
                let fy = (l[0] + 16.0) / 116.0;
                let (fx, fz) = (fy + l[1] / 500.0, fy - l[2] / 200.0);
                let inv = |t: f32| if t > 0.2069 { t * t * t } else { (t - 16.0 / 116.0) / 7.787 };
                let (x, y, z) = (0.95047 * inv(fx), inv(fy), 1.08883 * inv(fz));
                let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
                let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
                let bl = 0.0557 * x - 0.2040 * y + 1.0570 * z;
                let enc = |c: f32| {
                    let c = c.clamp(0.0, 1.0);
                    ((if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }) * 255.0) as u8
                };
                im.put_pixel((k % m.w) as u32, (m.h - 1 - k / m.w) as u32, image::Rgb([enc(r), enc(g), enc(bl)]));
            }
            im.save(&out).unwrap();
        }
    }
}
