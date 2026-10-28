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
use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::Path;

const EVAL_WIDTH: u32 = 1536;
const EDGE_WEIGHT: f32 = 2.0;
const APPROACH_HW: f64 = 3.0;
const APPROACH_MIN: f64 = 0.004;
const APPROACH_MAX: f64 = 0.03;
const HOVER: f64 = 0.004;
const PRESS_PER_SIZE: f64 = 1.0;
const PRESS_MIN: f64 = 0.0025;
const APPROACH_STEPS: f64 = 4.0;
const BRUSH_REST: f64 = 0.75;
const TRAVEL_PER_SIZE: f64 = 1.0;
const REFINEMENTS: usize = 16;
const WINDOW: usize = 40;
const SHRINK_BELOW: f64 = 0.15;
const SHRINK: f64 = 0.8;
const SIZE_SPREAD: f64 = 0.4;
const SIZE_CHANGED: f64 = 0.01;
const MIN_GAIN: f32 = 2.0;
const CHANGED_DE: f32 = 4.0;
const REFRESH_EVERY: u32 = 60;
const LOWER_FRAMES: u32 = 2;
const WARMUP_FRAMES: u32 = 3;
const LIFT_FRAMES: u32 = 2;
const SETTLE_FRAMES: u32 = 4;
const THREAD_SETTLE_MAX: u32 = 180;
const POUR_FILM: f64 = 2.0e-4;
const PREVIEW_W: u32 = 288;
const EDGE_PX: f32 = 2.0;
const TRACE_TURN: f32 = 0.3;
const TRACE_COHERENT: f32 = 0.1;
const MAX_POINTS: f32 = 96.0;
const TENSOR_RADIUS: usize = 4;
const WHITE: usize = 0;
const BLACK: usize = 1;

pub struct Photo {
    pub name: String,
    pub image: image::RgbaImage,
    pub texture: Option<egui::TextureHandle>,
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
    pub dry_passes: bool,
    pub longest: f64,
    pub active: bool,
    photo_gen: u64,
    saved_paint: Option<[f32; 16]>,
    saved_height: Option<f64>,
    user_sizes: Vec<(ToolKind, BrushShape, f64)>,
}

impl EvolveState {
    pub fn new() -> Self {
        Self { photo: None, focus: Focus { shape: FocusShape::Off, a: [0.5, 0.5], b: [0.75, 0.5], soft: 0.5, outside: 0.0 }, run: None, candidates: 32, follow: 0.7, dry_passes: true, longest: 0.25, active: false, photo_gen: 0, saved_paint: None, saved_height: None, user_sizes: Vec::new() }
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
}

fn tool_range(kind: ToolKind, shape: BrushShape) -> Option<ToolRange> {
    let r = |lo, hi, half_width, len, speed| Some(ToolRange { lo, hi, half_width, len, speed, long: kind != ToolKind::Sponge });
    match kind {
        ToolKind::Pour => r(0.0008, 0.02, 1.15, (2.0, 10.0), 0.5),
        ToolKind::Brush if shape == BrushShape::Flat => r(0.0001, 0.1, 0.5, (2.0, 8.0), 0.3),
        ToolKind::Brush => r(0.0001, 0.03, 0.55, (2.0, 8.0), 0.3),
        ToolKind::Roller => r(0.04, 0.25, 0.5, (1.0, 4.0), 0.35),
        ToolKind::Sponge => r(0.02, 0.15, 0.4, (0.0, 1.2), 0.12),
        ToolKind::Stick => r(0.003, 0.03, 0.2, (6.0, 30.0), 0.4),
        ToolKind::Bottle => r(0.0005, 0.006, 1.5, (6.0, 30.0), 0.2),
        ToolKind::Bucket => r(0.003, 0.03, 1.0, (6.0, 30.0), 0.4),
        ToolKind::Knife => None,
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
    Ok(Photo { name, image: crop, texture: None })
}

pub fn preview(photo: &mut Photo, ctx: &egui::Context) -> egui::TextureHandle {
    if let Some(t) = &photo.texture {
        return t.clone();
    }
    let (w, h) = photo.image.dimensions();
    let ph = ((PREVIEW_W as f32) * h as f32 / w as f32).round().max(1.0) as u32;
    let small = image::imageops::resize(&photo.image, PREVIEW_W, ph, image::imageops::FilterType::Triangle);
    let ci = egui::ColorImage::from_rgba_unmultiplied([PREVIEW_W as usize, ph as usize], small.as_raw());
    let t = ctx.load_texture("evolve_photo", ci, egui::TextureOptions::LINEAR);
    photo.texture = Some(t.clone());
    t
}

struct Swatch {
    mix: [f32; 16],
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
        for white in [0.0, 0.15, 0.5, 1.5, 4.0] {
            let mut m = *b;
            m[WHITE] += white * total;
            mixes.push(m);
        }
        if b[BLACK] == 0.0 {
            for black in [0.08, 0.25] {
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
            Swatch { mix, lab: lab(srgb.map(decode)) }
        })
        .collect()
}

#[derive(Clone)]
struct Cand {
    centre: [f32; 2],
    angle: f32,
    len: f32,
    bend: f32,
    along: f32,
    traced: bool,
    size: f64,
    hw: f32,
    pts: Vec<[f32; 2]>,
    swatch: usize,
    score: f32,
}

pub struct Plan {
    pub path: Vec<DVec2>,
    pub lead: f64,
    pub size: f64,
    pub mix: [f32; 16],
    pub speed: f64,
    pub kind: ToolKind,
    pts: Vec<[f32; 2]>,
    hw: f32,
    lab: [f32; 3],
    predicted: f32,
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

pub struct Evolver {
    pub w: usize,
    pub h: usize,
    pub factor: u32,
    cell: f64,
    photo: Vec<[f32; 3]>,
    white: [f32; 3],
    black: [f32; 3],
    pub calibrated: bool,
    target: Vec<[f32; 3]>,
    tangent: Vec<[f32; 3]>,
    weight: Vec<f32>,
    mask: Vec<f32>,
    focus: Option<Focus>,
    cdf: Vec<f64>,
    current: Vec<[f32; 3]>,
    err: Vec<f32>,
    swatches: Vec<Swatch>,
    rng: StdRng,
    tool: Option<(ToolKind, BrushShape)>,
    range: Option<ToolRange>,
    longest: f32,
    scratch: RefCell<Vec<f32>>,
    photo_gen: u64,
    applied: Option<f64>,
    pub size_hi: f64,
    width_gain: f64,
    opacity: f32,
    recent: VecDeque<bool>,
    pub tried: u64,
    pub painted: u64,
    pub kept: u64,
    pub passes: u32,
    pub start_err: f32,
    pub mean_err: f32,
    pub status: String,
    pub dry_due: bool,
    since_refresh: u32,
    pub phase: Phase,
}

impl Evolver {
    pub fn new(photo: &image::RgbaImage, film_w: u32, film_h: u32, dx: f32, set: &PigmentSet, cmf: &[f32], seed: u64) -> Self {
        let factor = (film_w / EVAL_WIDTH).max(1);
        let (w, h) = ((film_w / factor) as usize, (film_h / factor) as usize);
        let img = image::imageops::resize(&cover_crop(photo, w as f32 / h as f32), w as u32, h as u32, image::imageops::FilterType::Triangle);
        let white = color::mix_rgb(set, cmf, &color::PaintMix::pure(WHITE).frac).map(decode);
        let black = color::mix_rgb(set, cmf, &color::PaintMix::pure(BLACK).frac).map(decode);
        let photo: Vec<[f32; 3]> = (0..w * h)
            .into_par_iter()
            .map(|k| {
                let p = img.get_pixel((k % w) as u32, (h - 1 - k / w) as u32);
                [decode(p[0] as f32 / 255.0), decode(p[1] as f32 / 255.0), decode(p[2] as f32 / 255.0)]
            })
            .collect();
        Self {
            w,
            h,
            factor,
            cell: dx as f64 * factor as f64,
            photo,
            white,
            black,
            calibrated: false,
            target: vec![[0.0; 3]; w * h],
            tangent: vec![[0.0; 3]; w * h],
            weight: vec![1.0; w * h],
            mask: vec![1.0; w * h],
            focus: None,
            cdf: Vec::new(),
            current: vec![[0.0; 3]; w * h],
            err: vec![0.0; w * h],
            swatches: swatches(set, cmf),
            rng: StdRng::seed_from_u64(seed),
            tool: None,
            range: None,
            longest: 0.0,
            scratch: RefCell::new(vec![0.0; w * h]),
            photo_gen: 0,
            applied: None,
            size_hi: 0.0,
            width_gain: 1.0,
            opacity: 0.9,
            recent: VecDeque::new(),
            tried: 0,
            painted: 0,
            kept: 0,
            passes: 0,
            start_err: 0.0,
            mean_err: 0.0,
            status: String::new(),
            dry_due: false,
            since_refresh: 0,
            phase: Phase::Wait(WARMUP_FRAMES),
        }
    }

    pub fn calibrate(&mut self, gain: [f32; 3]) {
        let (white, black) = (self.white, self.black);
        self.target = self
            .photo
            .par_iter()
            .map(|p| {
                let mut rgb = [0.0f32; 3];
                for c in 0..3 {
                    rgb[c] = (tonemap_inverse(p[c]) / gain[c]).clamp(black[c], white[c]);
                }
                lab(rgb)
            })
            .collect();
        let (tangent, edges) = structure(&self.target, self.w, self.h);
        self.tangent = tangent;
        self.weight = edges.iter().map(|e| 1.0 + EDGE_WEIGHT * e).collect();
        self.calibrated = true;
        log::info!("display gain {:.2} {:.2} {:.2}", gain[0], gain[1], gain[2]);
    }

    pub fn observe(&mut self, lab: Vec<[f32; 3]>) {
        self.current = lab;
        self.err = self.current.par_iter().zip(&self.target).map(|(a, b)| de(*a, *b)).collect();
        self.mean_err = self.err.iter().sum::<f32>() / self.err.len().max(1) as f32;
        if self.start_err == 0.0 {
            self.start_err = self.mean_err;
        }
        self.since_refresh = 0;
    }

    pub fn refresh_due(&self) -> bool {
        self.since_refresh >= REFRESH_EVERY
    }

    pub fn set_tool(&mut self, kind: ToolKind, shape: BrushShape, size: f64) -> Option<f64> {
        let user_changed = self.applied.map(|a| (size - a).abs() > SIZE_CHANGED * a).unwrap_or(true);
        if self.tool == Some((kind, shape)) && !user_changed {
            return None;
        }
        if self.tool != Some((kind, shape)) {
            self.tool = Some((kind, shape));
            self.range = tool_range(kind, shape);
            self.width_gain = 1.0;
            self.opacity = 0.9;
        }
        self.recent.clear();
        self.applied = None;
        let r = self.range?;
        self.size_hi = size.clamp(r.lo, r.hi);
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
        self.recent.clear();
        self.status.clear();
    }

    pub fn size_lo(&self) -> f64 {
        self.range.map(|r| r.lo).unwrap_or(0.0)
    }

    fn size_floor(&self) -> f64 {
        self.size_lo().max(SIZE_SPREAD * self.size_hi)
    }

    pub fn progress(&self) -> f32 {
        if self.start_err <= 0.0 {
            return 0.0;
        }
        (1.0 - self.mean_err / self.start_err).clamp(0.0, 1.0)
    }

    fn set_focus(&mut self, focus: Focus) {
        if self.focus == Some(focus) {
            return;
        }
        self.focus = Some(focus);
        let (w, h) = (self.w, self.h);
        let aspect = w as f32 / h as f32;
        self.mask = (0..w * h).into_par_iter().map(|k| focus.weight(((k % w) as f32 + 0.5) / w as f32, 1.0 - ((k / w) as f32 + 0.5) / h as f32, aspect)).collect();
    }

    fn gain(&self, k: usize, delta: f32) -> f32 {
        if delta < 0.0 {
            delta * self.mask[k]
        } else {
            delta
        }
    }

    fn build_cdf(&mut self) {
        let mut acc = 0.0f64;
        self.cdf = self
            .err
            .iter()
            .zip(&self.weight)
            .zip(&self.mask)
            .map(|((e, w), m)| {
                acc += (*e as f64) * (*e as f64) * *w as f64 * *m as f64;
                acc
            })
            .collect();
    }

    fn sample_point(&mut self) -> [f32; 2] {
        let total = self.cdf.last().copied().unwrap_or(0.0);
        if total <= 0.0 {
            return [self.rng.random::<f32>() * self.w as f32, self.rng.random::<f32>() * self.h as f32];
        }
        let pick = self.rng.random::<f64>() * total;
        let k = self.cdf.partition_point(|c| *c < pick).min(self.err.len() - 1);
        [(k % self.w) as f32 + self.rng.random::<f32>(), (k / self.w) as f32 + self.rng.random::<f32>()]
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

    fn footprint(&self, pts: &[[f32; 2]], hw: f32) -> Vec<(usize, f32)> {
        let mut cov = self.scratch.borrow_mut();
        let mut touched = Vec::new();
        let segs: Vec<[[f32; 2]; 2]> = if pts.len() < 2 { vec![[pts[0], pts[0]]] } else { pts.windows(2).map(|w| [w[0], w[1]]).collect() };
        for [a, e] in segs {
            let bb = self.bbox(&[a, e], hw + 1.0);
            let (ex, ey) = (e[0] - a[0], e[1] - a[1]);
            let l2 = ex * ex + ey * ey;
            for j in bb[1]..bb[3] {
                for i in bb[0]..bb[2] {
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

    fn means(&self, foot: &[(usize, f32)]) -> ([f32; 3], [f32; 3]) {
        let mut acc = [0.0f32; 7];
        for &(k, a) in foot {
            let (t, c) = (self.target[k], self.current[k]);
            for i in 0..3 {
                acc[i] += a * t[i];
                acc[3 + i] += a * c[i];
            }
            acc[6] += a;
        }
        if acc[6] <= 0.0 {
            return ([50.0, 0.0, 0.0], [50.0, 0.0, 0.0]);
        }
        let n = acc[6];
        ([acc[0] / n, acc[1] / n, acc[2] / n], [acc[3] / n, acc[4] / n, acc[5] / n])
    }

    fn score(&self, foot: &[(usize, f32)], c: [f32; 3]) -> f32 {
        let o = self.opacity;
        let mut delta = 0.0;
        for &(k, a) in foot {
            let a = a * o;
            let cur = self.current[k];
            let new = [cur[0] + a * (c[0] - cur[0]), cur[1] + a * (c[1] - cur[1]), cur[2] + a * (c[2] - cur[2])];
            delta += self.gain(k, self.weight[k] * (de(new, self.target[k]) - self.err[k]));
        }
        delta
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

    fn pick_colour(&mut self, foot: &[(usize, f32)], jitter: f32) -> usize {
        let (m, cur) = self.means(foot);
        let o = self.opacity.max(0.65);
        let mut want = [0.0f32; 3];
        for c in 0..3 {
            want[c] = cur[c] + (m[c] - cur[c]) / o + jitter * gauss(&mut self.rng) * if c == 0 { 1.0 } else { 1.4 };
        }
        want[0] = want[0].clamp(0.0, 100.0);
        self.nearest(want)
    }

    fn trace(&self, c: &Cand) -> Vec<[f32; 2]> {
        let step = (0.5 * c.hw).max(c.len / MAX_POINTS).max(1.0);
        let (xmax, ymax) = (self.w as f32 - EDGE_PX, self.h as f32 - EDGE_PX);
        let walk = |sign: f32, dist: f32| -> Vec<[f32; 2]> {
            let mut p = c.centre;
            let mut d = [sign * c.angle.cos(), sign * c.angle.sin()];
            let mut out = Vec::new();
            let mut gone = 0.0;
            while gone + step <= dist {
                let k = (p[1] as usize).min(self.h - 1) * self.w + (p[0] as usize).min(self.w - 1);
                let t = self.tangent[k];
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

    fn shape(&self, c: &Cand) -> Vec<[f32; 2]> {
        if c.traced {
            return self.trace(c);
        }
        let (s, co) = c.angle.sin_cos();
        let start = [c.centre[0] - co * c.len * c.along, c.centre[1] - s * c.len * c.along];
        let end = [start[0] + co * c.len, start[1] + s * c.len];
        let ctrl = [(start[0] + end[0]) * 0.5 - s * c.bend * c.len, (start[1] + end[1]) * 0.5 + co * c.bend * c.len];
        let n = if c.len < 2.0 { 1 } else { (c.len / c.hw.max(2.0)).ceil().clamp(8.0, MAX_POINTS) as usize };
        let (xmax, ymax) = (self.w as f32 - EDGE_PX, self.h as f32 - EDGE_PX);
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

    fn evaluate(&mut self, mut c: Cand, refit: bool, jitter: f32) -> Cand {
        let r = self.range.unwrap();
        c.size = c.size.clamp(self.size_floor(), self.size_hi);
        c.hw = ((c.size * r.half_width * self.width_gain) / self.cell).max(0.6) as f32;
        let (lo, hi) = self.len_range(c.hw);
        c.len = c.len.clamp(lo, hi);
        c.pts = self.shape(&c);
        let foot = self.footprint(&c.pts, c.hw);
        if refit {
            c.swatch = self.pick_colour(&foot, jitter);
        }
        c.score = self.score(&foot, self.swatches[c.swatch].lab);
        c
    }

    fn len_range(&self, hw: f32) -> (f32, f32) {
        let r = self.range.unwrap();
        let lo = (hw * r.len.0).max(2.0);
        let hi = if r.long { self.longest.max(hw * r.len.1) } else { hw * r.len.1 };
        (lo, hi.max(lo * 1.2))
    }

    fn random_cand(&mut self, follow: f64) -> Cand {
        let r = self.range.unwrap();
        let centre = self.sample_point();
        let lo = self.size_floor();
        let size = (lo.ln() + self.rng.random::<f64>() * (self.size_hi.ln() - lo.ln())).exp();
        let hw = ((size * r.half_width * self.width_gain) / self.cell).max(0.6) as f32;
        let (lo, hi) = self.len_range(hw);
        let len = (lo.ln() + self.rng.random::<f32>() * (hi.ln() - lo.ln())).exp();
        let k = (centre[1] as usize).min(self.h - 1) * self.w + (centre[0] as usize).min(self.w - 1);
        let t = self.tangent[k];
        let traced = self.rng.random::<f64>() < follow * t[2] as f64;
        let angle = if traced { t[1].atan2(t[0]) } else { self.rng.random::<f32>() * std::f32::consts::TAU };
        let c = Cand { centre, angle, len, bend: 0.12 * gauss(&mut self.rng).clamp(-2.0, 2.0), along: self.rng.random(), traced, size, hw, pts: Vec::new(), swatch: 0, score: 0.0 };
        self.evaluate(c, true, 3.0)
    }

    fn mutate(&mut self, best: &Cand) -> Cand {
        let mut c = best.clone();
        match self.rng.random_range(0..5) {
            0 => {
                c.size *= (0.3 * gauss(&mut self.rng)).exp() as f64;
                self.evaluate(c, true, 0.0)
            }
            1 => {
                let sw = self.swatches[c.swatch].lab;
                let want = [sw[0] + 5.0 * gauss(&mut self.rng), sw[1] + 7.0 * gauss(&mut self.rng), sw[2] + 7.0 * gauss(&mut self.rng)];
                c.swatch = self.nearest(want);
                self.evaluate(c, false, 0.0)
            }
            2 => {
                c.size *= (0.3 * gauss(&mut self.rng)).exp() as f64;
                self.evaluate(c, true, 4.0)
            }
            3 => {
                c.len *= (0.5 * gauss(&mut self.rng)).exp();
                c.along = (c.along + 0.15 * gauss(&mut self.rng)).clamp(0.0, 1.0);
                if !c.traced {
                    c.angle += 0.15 * gauss(&mut self.rng);
                }
                self.evaluate(c, true, 0.0)
            }
            _ => {
                c.centre[0] += 0.3 * c.hw * gauss(&mut self.rng);
                c.centre[1] += 0.3 * c.hw * gauss(&mut self.rng);
                self.evaluate(c, true, 0.0)
            }
        }
    }

    pub fn plan(&mut self, candidates: usize, follow: f64, longest: f64, focus: Focus, paint: &[f32; 16]) -> Option<Plan> {
        let r = self.range?;
        self.set_focus(focus);
        self.longest = (longest / self.cell) as f32;
        self.build_cdf();
        self.tried += 1;
        let mut best: Option<Cand> = None;
        for _ in 0..candidates.max(1) {
            let c = self.random_cand(follow);
            if best.as_ref().map(|b| c.score < b.score).unwrap_or(true) {
                best = Some(c);
            }
        }
        let mut best = best?;
        for _ in 0..REFINEMENTS {
            let c = self.mutate(&best);
            if c.score < best.score {
                best = c;
            }
        }
        if best.score > -MIN_GAIN {
            self.record(false);
            return None;
        }
        let mut mix = self.swatches[best.swatch].mix;
        mix[12] = paint[12];
        mix[13] = paint[13];
        mix[14] = paint[14];
        let path = best.pts.iter().map(|p| DVec2::new(p[0] as f64 * self.cell, p[1] as f64 * self.cell)).collect();
        self.painted += 1;
        self.since_refresh += 1;
        let lead = if self.tool.unwrap().0 == ToolKind::Brush { (APPROACH_HW * best.hw as f64 * self.cell).clamp(APPROACH_MIN, APPROACH_MAX) } else { 0.0 };
        Some(Plan { path, lead, size: best.size, mix, speed: r.speed, kind: self.tool.unwrap().0, pts: best.pts, hw: best.hw, lab: self.swatches[best.swatch].lab, predicted: best.score })
    }

    pub fn region(&self, plan: &Plan) -> [usize; 4] {
        self.bbox(&plan.pts, plan.hw * 1.5 + 4.0)
    }

    pub fn judge(&mut self, plan: &Plan, region: [usize; 4], after: &[[f32; 3]]) -> bool {
        let rw = region[2] - region[0];
        let mut covs = vec![0.0f32; rw * (region[3] - region[1])];
        for (k, c) in self.footprint(&plan.pts, plan.hw) {
            let (i, j) = (k % self.w, k / self.w);
            if (region[0]..region[2]).contains(&i) && (region[1]..region[3]).contains(&j) {
                covs[(j - region[1]) * rw + (i - region[0])] = c;
            }
        }
        let mut delta = 0.0f64;
        let (mut pred, mut changed) = (0u32, 0u32);
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for j in region[1]..region[3] {
            for i in region[0]..region[2] {
                let k = j * self.w + i;
                let a = after[(j - region[1]) * rw + (i - region[0])];
                let b = self.current[k];
                let w = self.weight[k];
                delta += self.gain(k, w * (de(a, self.target[k]) - self.err[k])) as f64;
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
            self.width_gain = (self.width_gain * ratio.powf(0.25)).clamp(0.15, 3.0);
        }
        if den > 2000.0 {
            let o = (num / den).clamp(0.15, 1.0);
            self.opacity = 0.85 * self.opacity + 0.15 * o;
        }
        let ok = delta < 0.0;
        if ok {
            self.store(region, after);
            self.kept += 1;
        }
        log::debug!("stroke {} predicted {:.0} real {:.0} kept {} gain {:.2} opacity {:.2}", self.painted, plan.predicted, delta, ok, self.width_gain, self.opacity);
        self.record(ok);
        if self.painted % 50 == 0 {
            log::info!("painted {} kept {} tried {} pass {} size {:.1} mm closer {:.0}% gain {:.2} opacity {:.2}", self.painted, self.kept, self.tried, self.passes, self.size_hi * 1e3, self.progress() * 100.0, self.width_gain, self.opacity);
        }
        ok
    }

    pub fn store(&mut self, region: [usize; 4], after: &[[f32; 3]]) {
        let rw = region[2] - region[0];
        for j in region[1]..region[3] {
            for i in region[0]..region[2] {
                let k = j * self.w + i;
                let a = after[(j - region[1]) * rw + (i - region[0])];
                let e = de(a, self.target[k]);
                self.mean_err += (e - self.err[k]) / self.err.len() as f32;
                self.current[k] = a;
                self.err[k] = e;
            }
        }
    }

    fn record(&mut self, ok: bool) {
        self.recent.push_back(ok);
        while self.recent.len() > WINDOW {
            self.recent.pop_front();
        }
        if self.recent.len() < WINDOW {
            return;
        }
        let rate = self.recent.iter().filter(|k| **k).count() as f64 / WINDOW as f64;
        let lo = self.size_lo();
        if rate < SHRINK_BELOW && self.size_hi > lo * 1.25 {
            self.size_hi = (self.size_hi * SHRINK).max(lo * 1.25);
            self.passes += 1;
            self.dry_due = true;
            self.recent.clear();
        }
    }
}

fn structure(lab: &[[f32; 3]], w: usize, h: usize) -> (Vec<[f32; 3]>, Vec<f32>) {
    let l = |i: isize, j: isize| lab[(j.clamp(0, h as isize - 1) as usize) * w + i.clamp(0, w as isize - 1) as usize][0];
    let tensor: Vec<[f32; 3]> = (0..w * h)
        .into_par_iter()
        .map(|k| {
            let (i, j) = ((k % w) as isize, (k / w) as isize);
            let gx = (l(i + 1, j - 1) + 2.0 * l(i + 1, j) + l(i + 1, j + 1)) - (l(i - 1, j - 1) + 2.0 * l(i - 1, j) + l(i - 1, j + 1));
            let gy = (l(i - 1, j + 1) + 2.0 * l(i, j + 1) + l(i + 1, j + 1)) - (l(i - 1, j - 1) + 2.0 * l(i, j - 1) + l(i + 1, j - 1));
            [gx * gx, gx * gy, gy * gy]
        })
        .collect();
    let r = TENSOR_RADIUS as isize;
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

    fn tool_size(&self, kind: ToolKind) -> f64 {
        match kind {
            ToolKind::Pour => self.pour_radius as f64,
            ToolKind::Brush => self.brush.size(),
            ToolKind::Roller => self.roller.width,
            ToolKind::Sponge => self.sponge.size,
            ToolKind::Stick => self.tool.stick_diameter,
            ToolKind::Bottle => self.tool.bottle_radius,
            ToolKind::Bucket => self.tool.bucket_radius,
            ToolKind::Knife => self.tool.knife_width,
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
            ToolKind::Knife => {}
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
                let mut ev = Evolver::new(&photo.image, self.film.w, self.film.h, self.film.dx, &self.film.pigments, &self.film.cmf_table, rand::random());
                ev.photo_gen = gen;
                self.evo.run = Some(ev);
            }
        }
        self.begin_full();
        self.tool.flowing = false;
        self.air.detach();
        self.evo.saved_paint = Some(self.paint.frac);
        self.evo.saved_height = Some(self.tool.height);
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
        if let Some(h) = self.evo.saved_height.take() {
            self.tool.height = h;
        }
        let shape = self.brush.shape;
        for (kind, s, size) in std::mem::take(&mut self.evo.user_sizes) {
            if kind != ToolKind::Brush || s == shape {
                self.set_tool_size(kind, size);
            }
        }
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
                    let gain = self.display_gain(ev, &rgb);
                    ev.calibrate(gain);
                }
                ev.observe(rgb.into_par_iter().map(lab).collect());
                Phase::Plan
            }
            Phase::Plan => {
                self.tool.flowing = false;
                self.tool.vel = DVec3::ZERO;
                let (kind, shape) = (self.tool.kind, self.brush.shape);
                if let Some(size) = ev.set_tool(kind, shape, self.tool_size(kind)) {
                    self.evo.user_sizes.retain(|u| u.0 != kind || u.1 != shape);
                    self.evo.user_sizes.push((kind, shape, size));
                }
                if !ev.has_tool() {
                    ev.status = "The knife only moves paint; pick another tool to keep painting".to_string();
                    return Phase::Plan;
                }
                ev.status.clear();
                if ev.dry_due {
                    ev.dry_due = false;
                    if self.evo.dry_passes {
                        self.begin_full();
                        self.film.dry_now(&self.gpu);
                    }
                }
                if ev.refresh_due() {
                    return Phase::Observe;
                }
                let t0 = std::time::Instant::now();
                let planned = ev.plan(self.evo.candidates, self.evo.follow, self.evo.longest, self.evo.focus, &self.paint.frac);
                log::debug!("plan {:.1} ms", t0.elapsed().as_secs_f32() * 1e3);
                match planned {
                    None => Phase::Plan,
                    Some(plan) => {
                        self.prepare_tool(&plan);
                        ev.applied(self.tool_size(plan.kind));
                        self.begin_stroke();
                        let p = plan.entry().0;
                        self.tool.pos = DVec3::new(p.x, p.y, z);
                        Phase::Lower(plan, 0)
                    }
                }
            }
            Phase::Lower(plan, n) => {
                let p = plan.entry().0;
                self.tool.pos = DVec3::new(p.x, p.y, z);
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
                self.brush_raise = self.brush_hover() * (1.0 - s1 / plan.lead);
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
                let (p0, p1) = (plan.at(s), plan.at(s1));
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
                self.set_tool_size(ToolKind::Brush, plan.size);
                let len = self.brush.len;
                let user = self.evo.saved_height.filter(|h| *h < len - PRESS_MIN).unwrap_or(BRUSH_REST * len);
                let press = (PRESS_PER_SIZE * plan.size).clamp(PRESS_MIN, (self.brush.len - user).max(PRESS_MIN));
                self.tool.height = self.brush.len - press;
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
            ToolKind::Bottle => self.tool.bottle_radius = plan.size,
            ToolKind::Bucket => self.tool.bucket_radius = plan.size,
            ToolKind::Knife => {}
        }
    }

    fn display_gain(&self, ev: &Evolver, albedo: &[[f32; 3]]) -> [f32; 3] {
        let shown = self.beauty.render(&self.gpu, &self.renderer, &self.cam, &self.film, true, ev.w as u32, ev.h as u32, 1, None);
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

    fn read_eval(&self, ev: &Evolver, region: [usize; 4]) -> Vec<[f32; 3]> {
        self.read_linear(ev, region).into_par_iter().map(lab).collect()
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
