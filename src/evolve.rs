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
const BRUSH_REST: f64 = 0.75;
const TRAVEL_PER_SIZE: f64 = 1.0;
const REFINEMENTS: usize = 16;
const WINDOW: usize = 40;
const SHRINK: f64 = 0.8;
const TILE: usize = 8;
const REGION_HW: f32 = 1.5;
const MIN_HW: f64 = 0.75;
const STOPS: [f32; 4] = [f32::INFINITY, 30.0, 18.0, 10.0];
const LOOKAHEAD: usize = 160;
const EXECUTE: usize = 48;
const PRUNE_VISIBLE: f32 = 0.25;
const LEVEL_DROP: f32 = 0.5;
const MISS_LIMIT: usize = 24;
const TABU: f32 = 0.5;
const REGIONS: usize = 4;
const AIMED_SHARE: f32 = 0.5;
const BRISTLE_PER_SIZE: f64 = 5.0;
const BRISTLE_MIN: f64 = 0.006;
const SIZE_SPREAD: f64 = 0.4;
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
    saved_load: Option<f64>,
    saved_len: Option<f64>,
    user_sizes: Vec<(ToolKind, BrushShape, f64)>,
}

impl EvolveState {
    pub fn new() -> Self {
        Self { photo: None, focus: Focus { shape: FocusShape::Off, a: [0.5, 0.5], b: [0.75, 0.5], soft: 0.5, outside: 0.0 }, run: None, candidates: 32, follow: 0.7, dry_passes: true, longest: 0.25, active: false, photo_gen: 0, saved_paint: None, saved_height: None, saved_load: None, saved_len: None, user_sizes: Vec::new() }
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
    let r = |lo, hi, half_width, len, speed| Some(ToolRange { lo, hi, half_width, len, speed, long: kind != ToolKind::Sponge, reach });
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
    stop: f32,
    size: f64,
    hw: f32,
    pts: Vec<[f32; 2]>,
    swatch: usize,
    score: f32,
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
    tangent: Vec<[f32; 3]>,
    weight: Vec<f32>,
    swatches: Vec<Swatch>,
}

impl Model {
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

    fn footprint(&self, cov: &mut [f32], pts: &[[f32; 2]], hw: f32) -> Vec<(usize, f32)> {
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
    reference: Option<f32>,
    window: VecDeque<f32>,
    misses: usize,
    pub pass: u32,
    pub done: bool,
}

impl Schedule {
    fn new(size_hi: f64) -> Self {
        Self { size_hi, reference: None, window: VecDeque::new(), misses: 0, pass: 0, done: false }
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
    gen: u64,
}

pub struct Batch {
    plans: Vec<Plan>,
    sched: Schedule,
    planned: usize,
    pruned: usize,
    gen: u64,
}

struct Region {
    window: [usize; 4],
    tile: usize,
    angle: f32,
    elongation: f32,
    grain: f32,
    coherence: f32,
    cdf: Vec<f32>,
}

struct Sketch {
    m: Arc<Model>,
    mask: Vec<f32>,
    focus: Option<Focus>,
    current: Vec<[f32; 3]>,
    err: Vec<f32>,
    tiles: Vec<f32>,
    tabu: Vec<f32>,
    sat: Vec<f32>,
    tw: usize,
    th: usize,
    scratch: Vec<f32>,
    rng: StdRng,
    kind: ToolKind,
    range: ToolRange,
    longest: f32,
    width_gain: f64,
    opacity: f32,
    sched: Schedule,
    last_tiles: Vec<usize>,
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
            tiles: vec![0.0; tw * th],
            tabu: vec![1.0; tw * th],
            sat: vec![0.0; (tw + 1) * (th + 1)],
            tw,
            th,
            scratch: vec![0.0; n],
            rng: StdRng::seed_from_u64(seed),
            kind,
            range,
            longest: 0.0,
            width_gain: 1.0,
            opacity: 0.9,
            sched: Schedule::new(size_hi),
            last_tiles: Vec::new(),
            m,
        }
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
            self.tiles[t] += self.err[k] * self.m.weight[k] * self.mask[k];
        }
        self.tabu.iter_mut().for_each(|t| *t = 1.0);
    }

    fn rebase(&mut self, real: &[[f32; 3]], pending: &[Plan]) {
        let m = self.m.clone();
        self.current = real.to_vec();
        self.err = self.current.par_iter().zip(&m.target).map(|(a, b)| de(*a, *b)).collect();
        self.refresh_tiles();
        for p in pending {
            let foot = m.footprint(&mut self.scratch, &p.pts, p.hw);
            self.apply(&foot, p.lab);
        }
    }

    fn apply(&mut self, foot: &[(usize, f32)], c: [f32; 3]) {
        let o = self.opacity;
        for &(k, a) in foot {
            let a = a * o;
            let cur = self.current[k];
            let new = [cur[0] + a * (c[0] - cur[0]), cur[1] + a * (c[1] - cur[1]), cur[2] + a * (c[2] - cur[2])];
            let e = de(new, self.m.target[k]);
            let t = self.tile_of(k);
            self.tiles[t] += (e - self.err[k]) * self.m.weight[k] * self.mask[k];
            self.current[k] = new;
            self.err[k] = e;
        }
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
        let hi = hi.min((r.reach * size / self.m.cell) as f32);
        (lo, hi.max(lo * 1.2))
    }

    fn regions(&mut self) -> Vec<Region> {
        let (tw, th) = (self.tw, self.th);
        for j in 0..th {
            let mut row = 0.0f32;
            for i in 0..tw {
                row += self.tiles[j * tw + i].max(0.0);
                self.sat[(j + 1) * (tw + 1) + i + 1] = self.sat[j * (tw + 1) + i + 1] + row;
            }
        }
        let hw = self.hw(self.sched.size_hi);
        let r = ((REGION_HW * hw) / TILE as f32).ceil().max(0.0) as usize;
        let mut ranked: Vec<(f32, usize, usize)> = Vec::with_capacity(tw * th);
        for j in 0..th {
            for i in 0..tw {
                let (x0, y0, x1, y1) = (i.saturating_sub(r), j.saturating_sub(r), (i + r + 1).min(tw), (j + r + 1).min(th));
                let s = self.sat[y1 * (tw + 1) + x1] - self.sat[y0 * (tw + 1) + x1] - self.sat[y1 * (tw + 1) + x0] + self.sat[y0 * (tw + 1) + x0];
                let area = ((x1 - x0) * (y1 - y0)) as f32 / ((2 * r + 1) * (2 * r + 1)) as f32;
                let s = s * self.tabu[j * tw + i] / area.max(0.25).sqrt();
                if s > 0.0 {
                    ranked.push((s, i, j));
                }
            }
        }
        let keep = (REGIONS * 64).min(ranked.len());
        if keep == 0 {
            return Vec::new();
        }
        ranked.select_nth_unstable_by(keep - 1, |a, b| b.0.total_cmp(&a.0));
        ranked.truncate(keep);
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut picked: Vec<(usize, usize)> = Vec::new();
        for &(_, i, j) in &ranked {
            if picked.iter().all(|&(pi, pj)| pi.abs_diff(i) > r || pj.abs_diff(j) > r) {
                picked.push((i, j));
                if picked.len() == REGIONS {
                    break;
                }
            }
        }
        picked.into_iter().filter_map(|(i, j)| self.region_at(i, j, r)).collect()
    }

    fn region_at_all(&self) -> Vec<Region> {
        let m = &self.m;
        let mut cdf = Vec::with_capacity(m.w * m.h);
        let mut acc = 0.0f32;
        for k in 0..m.w * m.h {
            acc += self.err[k] * self.err[k] * m.weight[k] * self.mask[k];
            cdf.push(acc);
        }
        vec![Region { window: [0, 0, m.w, m.h], tile: 0, angle: 0.0, elongation: 0.0, grain: 0.0, coherence: -0.3, cdf }]
    }

    fn region_at(&self, i: usize, j: usize, r: usize) -> Option<Region> {
        let m = &self.m;
        let half = r * TILE + TILE / 2;
        let (cx, cy) = (i * TILE + TILE / 2, j * TILE + TILE / 2);
        let window = [cx.saturating_sub(half), cy.saturating_sub(half), (cx + half).min(m.w), (cy + half).min(m.h)];
        let mut cdf = Vec::with_capacity((window[2] - window[0]) * (window[3] - window[1]));
        let (mut acc, mut sx, mut sy, mut sxx, mut sxy, mut syy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut tx, mut ty, mut tn) = (0.0f32, 0.0f32, 0.0f32);
        for y in window[1]..window[3] {
            for x in window[0]..window[2] {
                let k = y * m.w + x;
                let e = self.err[k] * self.err[k] * m.weight[k] * self.mask[k];
                acc += e;
                cdf.push(acc);
                let (fx, fy) = (x as f32, y as f32);
                sx += e * fx;
                sy += e * fy;
                sxx += e * fx * fx;
                sxy += e * fx * fy;
                syy += e * fy * fy;
                let t = m.tangent[k];
                tx += t[2] * (t[0] * t[0] - t[1] * t[1]);
                ty += t[2] * 2.0 * t[0] * t[1];
                tn += 1.0;
            }
        }
        if acc <= 0.0 {
            return None;
        }
        let (mx, my) = (sx / acc, sy / acc);
        let (cxx, cxy, cyy) = (sxx / acc - mx * mx, sxy / acc - mx * my, syy / acc - my * my);
        let angle = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
        let tr = cxx + cyy;
        let elongation = if tr > 1e-6 { ((cxx - cyy).powi(2) + 4.0 * cxy * cxy).sqrt() / tr } else { 0.0 };
        let grain = 0.5 * ty.atan2(tx);
        let coherence = ((tx * tx + ty * ty).sqrt() / tn.max(1.0)).min(1.0);
        Some(Region { window, tile: j * self.tw + i, angle, elongation, grain, coherence, cdf })
    }

    fn sample_in(&mut self, reg: &Region) -> [f32; 2] {
        let total = *reg.cdf.last().unwrap();
        let pick = self.rng.random::<f32>() * total;
        let k = reg.cdf.partition_point(|c| *c < pick).min(reg.cdf.len() - 1);
        let rw = reg.window[2] - reg.window[0];
        [(reg.window[0] + k % rw) as f32 + self.rng.random::<f32>(), (reg.window[1] + k / rw) as f32 + self.rng.random::<f32>()]
    }

    fn means(&self, foot: &[(usize, f32)]) -> ([f32; 3], [f32; 3]) {
        let mut acc = [0.0f32; 7];
        for &(k, a) in foot {
            let (t, c) = (self.m.target[k], self.current[k]);
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
            delta += self.gain(k, self.m.weight[k] * (de(new, self.m.target[k]) - self.err[k]));
        }
        delta
    }

    fn pick_colour(&mut self, foot: &[(usize, f32)], jitter: f32) -> usize {
        let (m, cur) = self.means(foot);
        let o = self.opacity.max(0.65);
        let mut want = [0.0f32; 3];
        for c in 0..3 {
            want[c] = cur[c] + (m[c] - cur[c]) / o + jitter * gauss(&mut self.rng) * if c == 0 { 1.0 } else { 1.4 };
        }
        want[0] = want[0].clamp(0.0, 100.0);
        self.m.nearest(want)
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
                let t = m.tangent[m.index(p)];
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
        let here = m.target[m.index(pts[ci])];
        let inside = |p: [f32; 2]| de(m.target[m.index(p)], here) <= stop;
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
        let raw = if c.traced { self.trace(&c) } else { self.bezier(&c) };
        c.pts = self.clip(raw, c.centre, c.stop);
        let m = self.m.clone();
        let foot = m.footprint(&mut self.scratch, &c.pts, c.hw);
        if refit {
            c.swatch = self.pick_colour(&foot, jitter);
        }
        c.score = self.score(&foot, m.swatches[c.swatch].lab);
        c
    }

    fn random_stop(&mut self) -> f32 {
        STOPS[self.rng.random_range(0..STOPS.len())]
    }

    fn candidate(&mut self, reg: &Region, follow: f64) -> Cand {
        let centre = self.sample_in(reg);
        let lo = self.size_floor();
        let hi = self.sched.size_hi.max(lo);
        let size = (lo.ln() + self.rng.random::<f64>() * (hi.ln() - lo.ln())).exp();
        let hw = self.hw(size);
        let (llo, lhi) = self.len_range(size, hw);
        let len = (llo.ln() + self.rng.random::<f32>() * (lhi.ln() - llo.ln())).exp();
        let t = self.m.tangent[self.m.index(centre)];
        let traced = self.rng.random::<f64>() < follow * t[2] as f64;
        let pick: f32 = self.rng.random();
        let along_grain = (follow as f32 * (0.3 + reg.coherence)).min(0.9);
        let angle = if traced {
            t[1].atan2(t[0])
        } else if pick < along_grain {
            reg.grain + 0.12 * gauss(&mut self.rng)
        } else if pick < along_grain + 0.6 * reg.elongation * (1.0 - along_grain) {
            reg.angle + 0.2 * gauss(&mut self.rng)
        } else {
            self.rng.random::<f32>() * std::f32::consts::TAU
        };
        let stop = self.random_stop();
        let c = Cand { centre, angle, len, bend: 0.12 * gauss(&mut self.rng).clamp(-2.0, 2.0), along: self.rng.random(), traced, stop, size, hw, pts: Vec::new(), swatch: 0, score: 0.0 };
        self.evaluate(c, true, 3.0)
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
                if !c.traced {
                    c.angle += 0.15 * gauss(&mut self.rng);
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

    fn plan_one(&mut self, job: &Job) -> Option<(Plan, Vec<(usize, f32)>)> {
        let mut regs = self.regions();
        self.last_tiles = regs.iter().map(|r| r.tile).collect();
        let focused = regs.len();
        let aimed = if focused > 0 { ((job.candidates as f32 * AIMED_SHARE).round() as usize).div_ceil(focused) } else { 0 };
        let wide = job.candidates.saturating_sub(aimed * focused).max(1);
        regs.extend(self.region_at_all());
        let mut best: Option<Cand> = None;
        for (ri, reg) in regs.iter().enumerate() {
            let n = if ri < focused { aimed } else { wide };
            for _ in 0..n {
                let c = self.candidate(reg, job.follow);
                if best.as_ref().map(|b| c.score < b.score).unwrap_or(true) {
                    best = Some(c);
                }
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
            return None;
        }
        let m = self.m.clone();
        let mut mix = m.swatches[best.swatch].mix;
        mix[12] = job.paint[12];
        mix[13] = job.paint[13];
        mix[14] = job.paint[14];
        let path = best.pts.iter().map(|p| DVec2::new(p[0] as f64 * m.cell, p[1] as f64 * m.cell)).collect();
        let lead = if self.kind == ToolKind::Brush { (APPROACH_HW * best.hw as f64 * m.cell).clamp(APPROACH_MIN, APPROACH_MAX) } else { 0.0 };
        let foot = m.footprint(&mut self.scratch, &best.pts, best.hw);
        let plan = Plan { path, lead, size: best.size, mix, speed: self.range.speed, kind: self.kind, dry: false, pts: best.pts, hw: best.hw, lab: m.swatches[best.swatch].lab, predicted: best.score, width_gain: self.width_gain, pass: self.sched.pass };
        Some((plan, foot))
    }

    fn shrink(&mut self) {
        let lo = self.size_lo();
        let s = &mut self.sched;
        if s.size_hi > lo * 1.25 {
            s.size_hi = (s.size_hi * SHRINK).max(lo * 1.25);
            s.pass += 1;
            s.reference = None;
            s.window.clear();
            s.misses = 0;
            self.tabu.iter_mut().for_each(|t| *t = 1.0);
        } else {
            s.done = true;
        }
    }

    fn hit(&mut self, gain: f32) {
        let s = &mut self.sched;
        s.misses = 0;
        s.window.push_back(gain);
        if s.window.len() > WINDOW {
            s.window.pop_front();
        }
        if s.window.len() < WINDOW {
            return;
        }
        let mean = s.window.iter().sum::<f32>() / WINDOW as f32;
        match s.reference {
            None => {
                s.reference = Some(mean);
                s.window.clear();
            }
            Some(r) if mean < LEVEL_DROP * r => self.shrink(),
            _ => {}
        }
    }

    fn miss(&mut self) {
        for t in std::mem::take(&mut self.last_tiles) {
            self.tabu[t] *= TABU;
        }
        self.sched.misses += 1;
        if self.sched.misses >= MISS_LIMIT {
            self.shrink();
        }
    }

    fn run(&mut self, job: &Job) -> Batch {
        if self.kind != job.kind || self.range.lo != job.range.lo || self.range.hi != job.range.hi {
            self.kind = job.kind;
            self.range = job.range;
        }
        if let Some(s) = job.reset {
            self.sched = Schedule::new(s);
            self.tabu.iter_mut().for_each(|t| *t = 1.0);
        }
        self.set_focus(job.focus);
        self.refresh_tiles();
        self.longest = (job.longest / self.m.cell) as f32;
        self.width_gain = job.width_gain;
        self.opacity = job.opacity;
        let start_pass = self.sched.pass;
        let (mut plans, mut feet, mut scheds) = (Vec::new(), Vec::new(), Vec::new());
        while plans.len() < LOOKAHEAD && !self.sched.done {
            match self.plan_one(job) {
                Some((plan, foot)) => {
                    self.apply(&foot, plan.lab);
                    self.hit(-plan.predicted);
                    plans.push(plan);
                    feet.push(foot);
                    scheds.push(self.sched.clone());
                }
                None => self.miss(),
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
        let (mut out, mut pruned, mut last, mut pass) = (Vec::new(), 0, None, start_pass);
        for (i, mut p) in plans.into_iter().enumerate() {
            if out.len() >= EXECUTE {
                break;
            }
            if keep[i] {
                p.dry = p.pass > pass;
                pass = p.pass;
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
        Batch { plans: out, sched: self.sched.clone(), planned, pruned, gen: job.gen }
    }
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
    palette: Option<Vec<Swatch>>,
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
    sketch: Option<Sketch>,
    job: Option<std::thread::JoinHandle<(Sketch, Batch)>>,
    ready: Option<Batch>,
    queue: VecDeque<Plan>,
    synced: bool,
    gen: u64,
    pub finished: bool,
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
            palette: Some(swatches(set, cmf)),
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
            sketch: None,
            job: None,
            ready: None,
            queue: VecDeque::new(),
            synced: false,
            gen: 0,
            finished: false,
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

    pub fn calibrate(&mut self, gain: [f32; 3]) {
        let (white, black) = (self.white, self.black);
        let target: Vec<[f32; 3]> = self
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
        let (tangent, edges) = structure(&target, self.w, self.h);
        let weight = edges.iter().map(|e| 1.0 + EDGE_WEIGHT * e).collect();
        let swatches = self.palette.take().unwrap_or_default();
        self.model = Some(Arc::new(Model { w: self.w, h: self.h, cell: self.cell, target, tangent, weight, swatches }));
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
        if self.start_err == 0.0 {
            self.start_err = self.mean_err;
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
        if self.start_err <= 0.0 {
            return 0.0;
        }
        (1.0 - self.mean_err / self.start_err).clamp(0.0, 1.0)
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

    pub fn next(&mut self, candidates: usize, follow: f64, longest: f64, focus: Focus, paint: &[f32; 16]) -> Option<Plan> {
        if let Some(p) = self.queue.pop_front() {
            return Some(p);
        }
        if !self.synced {
            return None;
        }
        if self.job.as_ref().map(|j| j.is_finished()).unwrap_or(false) {
            let (sk, batch) = self.job.take().unwrap().join().expect("planner");
            self.sketch = Some(sk);
            if batch.gen == self.gen {
                self.tried += batch.planned as u64;
                self.pruned += batch.pruned as u64;
                self.size_hi = batch.sched.size_hi;
                self.passes = batch.sched.pass;
                if batch.plans.is_empty() && batch.sched.done {
                    self.finished = true;
                }
                log::info!("batch planned {} pruned {} painting {} size {:.1} mm pass {} closer {:.0}%", batch.planned, batch.pruned, batch.plans.len(), batch.sched.size_hi * 1e3, batch.sched.pass, self.progress() * 100.0);
                self.ready = Some(batch);
            }
        }
        if self.job.is_some() || self.finished {
            return None;
        }
        self.set_focus(focus);
        let pending = self.ready.take().map(|b| b.plans).unwrap_or_default();
        let range = self.range?;
        let kind = self.tool?.0;
        let m = self.model.clone()?;
        let size_hi = self.size_hi;
        let seed = self.rng.random();
        let mut sk = self.sketch.take().unwrap_or_else(|| Sketch::new(m, seed, kind, range, size_hi));
        let job = Job { kind, range, reset: self.reset.take(), candidates, follow, longest, focus, paint: *paint, width_gain: self.width_gain, opacity: self.opacity, gen: self.gen };
        let real = self.current.clone();
        let ahead = pending.clone();
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
        for (k, c) in m.footprint(&mut self.scratch, &plan.pts, plan.hw) {
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
                let w = m.weight[k];
                delta += self.gain(k, w * (de(a, m.target[k]) - self.err[k])) as f64;
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
        self.painted += 1;
        let ok = delta < 0.0;
        if ok {
            self.store(region, after);
            self.kept += 1;
        }
        log::debug!("stroke {} predicted {:.0} real {:.0} kept {} gain {:.2} opacity {:.2}", self.painted, plan.predicted, delta, ok, self.width_gain, self.opacity);
        if self.painted % 50 == 0 {
            log::info!("painted {} kept {} planned {} pruned {} pass {} size {:.1} mm closer {:.0}% gain {:.2} opacity {:.2}", self.painted, self.kept, self.tried, self.pruned, self.passes, self.size_hi * 1e3, self.progress() * 100.0, self.width_gain, self.opacity);
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
                match ev.next(self.evo.candidates, self.evo.follow, self.evo.longest, self.evo.focus, &self.paint.frac) {
                    None if !ev.synced() => Phase::Observe,
                    None => {
                        if ev.finished {
                            ev.status = "Finished: no stroke at the smallest size helps any more. Pick a smaller tool size or another tool and press Paint to carry on".to_string();
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
                let user_len = self.evo.saved_len.unwrap_or(self.brush.len);
                let len = (BRISTLE_PER_SIZE * plan.size).clamp(BRISTLE_MIN.min(user_len), user_len);
                if (self.brush.size() - plan.size).abs() > 0.02 * plan.size || (self.brush.len - len).abs() > 0.02 * len {
                    let (shape, n) = (self.brush.shape, self.brush.bristles.len());
                    self.rebuild_brush(shape, n, len, plan.size);
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn sketch_bench() {
        let path = std::env::var("EVO_PHOTO").unwrap();
        let out = std::env::var("EVO_OUT").unwrap_or_default();
        let total: usize = std::env::var("EVO_N").ok().and_then(|v| v.parse().ok()).unwrap_or(900);
        let set = color::load_pigments();
        let cmf = color::cmf_table();
        let img = image::open(&path).unwrap().to_rgba8();
        let photo = cover_crop(&img, 1.5);
        let mut ev = Evolver::new(&photo, 3072, 2048, 1.2 / 3072.0, &set, &cmf, 7);
        ev.calibrate([2.11, 2.2, 2.27]);
        let m = ev.model.clone().unwrap();
        let kind = ToolKind::Brush;
        let range = tool_range(kind, BrushShape::Round).unwrap();
        let mut sk = Sketch::new(m.clone(), 7, kind, range, 0.016);
        let mut base = vec![[80.0f32, 1.0, 6.0]; m.w * m.h];
        let err0 = base.iter().zip(&m.target).map(|(a, b)| de(*a, *b)).sum::<f32>();
        let focus = Focus { shape: FocusShape::Off, a: [0.5, 0.5], b: [0.75, 0.5], soft: 0.5, outside: 0.0 };
        let opacity: f32 = std::env::var("EVO_OPACITY").ok().and_then(|v| v.parse().ok()).unwrap_or(0.7);
        let (mut done, mut next_report) = (0usize, 0usize);
        let t0 = std::time::Instant::now();
        while done < total {
            let job = Job { kind, range, reset: None, candidates: 32, follow: 0.7, longest: 0.25, focus, paint: [0.0; 16], width_gain: 1.85, opacity, gen: 0 };
            sk.rebase(&base, &[]);
            let b = sk.run(&job);
            sk.rebase(&base, &b.plans);
            base = sk.current.clone();
            done += b.plans.len();
            if done >= next_report || b.plans.is_empty() {
                let e = base.iter().zip(&m.target).map(|(a, b)| de(*a, *b)).sum::<f32>();
                let ed = base.iter().zip(&m.target).zip(&m.weight).map(|((a, b), w)| (w - 1.0) * de(*a, *b)).sum::<f32>() / m.weight.iter().map(|w| w - 1.0).sum::<f32>();
                print!("edge dE {:.1} ", ed);
                println!("strokes {} closer {:.1}% size {:.1} mm pass {} pruned {} {:.1}s", done, 100.0 * (1.0 - e / err0), b.sched.size_hi * 1e3, b.sched.pass, b.pruned, t0.elapsed().as_secs_f32());
                next_report += 600;
            }
            if b.plans.is_empty() && b.sched.done {
                break;
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
