use crate::color;
use crate::tools::ToolKind;
use glam::{DVec2, DVec3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct Command {
    pub kind: ToolKind,
    pub pos: DVec3,
    pub vel: DVec3,
    pub flowing: bool,
    pub height: f64,
    pub mu: f64,
    pub sigma: f64,
    pub mix: [f32; 16],
    pub stick_diameter: f64,
    pub bottle_flow: f64,
    pub bottle_radius: f64,
    pub bucket_flow: f64,
    pub bucket_radius: f64,
    pub pour_rate: f64,
    pub pour_radius: f64,
    pub knife_width: f64,
    pub knife_height: f64,
    pub brush_width: f64,
    pub brush_count: usize,
    pub brush_capacity: f64,
    pub brush_release: f64,
    pub tilt: Option<(f64, f64)>,
    pub reload: bool,
    pub cut: bool,
    pub cure_rate: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Style {
    Flood,
    Drip,
    Pour,
    Fling,
    Dots,
    Lasso,
    Snap,
    Dry,
    Wash,
    Scrape,
    Scrub,
    Tilt,
}

#[derive(Clone)]
struct Paint {
    name: String,
    mix: [f32; 16],
    mu: f64,
}

#[derive(Clone)]
pub struct Palette {
    name: String,
    scheme: String,
    ground: Vec<Paint>,
    ground_dark: Option<Paint>,
    main: Paint,
    second: Paint,
    accents: Vec<Paint>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum GroundMethod {
    Auto,
    Brush,
    Bucket,
}

struct Layer {
    name: String,
    mix: [f32; 16],
    mu: f64,
    style: Style,
    duration: f64,
    pace: f64,
    loop_chance: f64,
    height_median: f64,
    stick_diameter: f64,
    flow: f64,
    pour_radius: f64,
    knife_width: f64,
    brush_width: f64,
    shades: Vec<[f32; 16]>,
    tilt: (f64, f64),
    margin: f64,
    edge_bias: bool,
    strokes: usize,
}

struct Gesture {
    t_start: f64,
    t_end: f64,
    speed: f64,
    height: f64,
    turn_sigma: f64,
    flow_until: f64,
    hop_at: f64,
    centre: DVec2,
    radius: f64,
    phase0: f64,
    lissajous: bool,
    drift: DVec2,
    wobble: f64,
}

pub struct Painter {
    rng: StdRng,
    w: f64,
    h: f64,
    t: f64,
    layer: usize,
    layer_t: f64,
    layers: Vec<Layer>,
    gesture: Option<Gesture>,
    pos: DVec2,
    heading: f64,
    ang_vel: f64,
    speed: f64,
    speed_noise: f64,
    height_noise: f64,
    cut_pending: bool,
    reload_pending: bool,
    ground_angle: f64,
    stroke_len: f64,
    layer_strokes: usize,
    dip_len: f64,
    stroke_mix: [f32; 16],
    cover: Vec<f32>,
    ink: Vec<f32>,
    pal: Palette,
    pub round: usize,
    pub palette: String,
    pub ground: String,
    pub seed: u64,
    pub done: bool,
    figured: bool,
}

const RHO: f64 = 1000.0;
const G: f64 = 9.81;
const COVER_W: usize = 16;
const COVER_H: usize = 11;
const CLASSIC_SHARE: f64 = 0.2;
const MIN_GROUND_CONTRAST: f64 = 45.0;
const MIN_LINE_SPREAD: f64 = 26.0;
const MIN_HUE_GAP: f64 = 28.0;
const GROUND_MARGIN_BLEED: f64 = -0.03;
const GROUND_MARGIN_BORDER: f64 = 0.05;
const GROUND_MARGIN_FLOATING: f64 = 0.04;
const EDGE_BAND: f64 = 0.13;
const INITIAL_ROUNDS_MIN: usize = 2;
const EDGE_MARGIN: f64 = 0.03;
const EDGE_LOOKAHEAD: f64 = 0.15;
const EDGE_LOOKAHEAD_MAX: f64 = 0.1;
const EDGE_TURN_GAIN: f64 = 5.0;
const EDGE_STOP: f64 = 0.005;
const EDGE_OVERSHOOT: f64 = 0.06;
const TOOL_EDGE: f64 = 0.02;
const INK_W: usize = 24;
const INK_H: usize = 16;
const START_CANDIDATES: usize = 4;
const START_INSET: f64 = 0.015;
const INWARD_REACH: f64 = 0.25;
const INITIAL_ROUNDS_MAX: usize = 3;

fn mix(parts: &[(usize, f32)], thinner: f32, yield_pa: f32) -> [f32; 16] {
    let mut m = [0.0f32; 16];
    for (i, a) in parts {
        m[*i] += *a;
    }
    m[12] = thinner;
    m[13] = yield_pa;
    m
}

pub fn transition_speed(q: f64, h: f64, nu: f64) -> f64 {
    let hs = h * (G / (nu * nu)).powf(1.0 / 3.0);
    let qs = q * (G / nu.powi(5)).powf(1.0 / 3.0);
    let us = 3.04 * qs.sqrt() * hs.powf(0.75);
    us * (G * nu).powf(1.0 / 3.0)
}

pub fn stick_flow(d: f64, nu: f64) -> f64 {
    0.016 * d * d * (G * nu * 0.5).powf(0.25)
}

fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-12);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn lognormal(rng: &mut StdRng, median: f64, sigma: f64) -> f64 {
    median * (sigma * gauss(rng)).exp()
}

fn pick<'a, T>(rng: &mut StdRng, items: &'a [T]) -> &'a T {
    &items[rng.random_range(0..items.len())]
}

fn weighted<'a, T>(rng: &mut StdRng, items: &'a [(T, f64)]) -> &'a T {
    let total: f64 = items.iter().map(|(_, w)| w).sum();
    let mut u = rng.random::<f64>() * total;
    for (item, w) in items {
        if u < *w {
            return item;
        }
        u -= w;
    }
    &items[items.len() - 1].0
}

fn paint(rng: &mut StdRng, name: &str, parts: &[(usize, f32)], thinner: f32, yield_pa: f32, mu: f64) -> Paint {
    let mut jittered: Vec<(usize, f32)> = Vec::new();
    for (i, a) in parts {
        jittered.push((*i, a * lognormal(rng, 1.0, 0.22) as f32));
    }
    let th = (thinner + 0.06 * gauss(rng) as f32).clamp(0.0, 0.75);
    Paint { name: name.to_string(), mix: mix(&jittered, th, yield_pa), mu: lognormal(rng, mu, 0.2) }
}

const CHROMATIC: [usize; 10] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const DEEP_PIGMENTS: [usize; 7] = [4, 5, 6, 7, 10, 4, 9];

struct ColourModel {
    set: color::PigmentSet,
    cmf: Vec<f32>,
}

impl ColourModel {
    fn load() -> Self {
        Self { set: color::load_pigments(), cmf: color::cmf_table() }
    }

    fn lab(&self, m: &[f32; 16]) -> [f64; 3] {
        let l = color::mix_lab(&self.set, &self.cmf, m);
        [l[0] as f64, l[1] as f64, l[2] as f64]
    }
}

fn delta_e(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn colour_name(lab: [f64; 3]) -> String {
    let (l, a, b) = (lab[0], lab[1], lab[2]);
    let chroma = (a * a + b * b).sqrt();
    if chroma < 9.0 {
        return match l {
            x if x > 88.0 => "white",
            x if x > 68.0 => "light grey",
            x if x > 45.0 => "grey",
            x if x > 25.0 => "charcoal",
            _ => "black",
        }
        .to_string();
    }
    let hue = b.atan2(a).to_degrees().rem_euclid(360.0);
    let word = match hue {
        h if h < 15.0 => "crimson",
        h if h < 45.0 => if l < 45.0 { "brown" } else { "red" },
        h if h < 65.0 => if l < 50.0 { "umber" } else { "orange" },
        h if h < 80.0 => if l < 62.0 { "ochre" } else { "yellow" },
        h if h < 105.0 => if l < 65.0 { "olive" } else { "yellow" },
        h if h < 160.0 => "green",
        h if h < 215.0 => "teal",
        h if h < 285.0 => "blue",
        h if h < 330.0 => "violet",
        _ => "magenta",
    };
    let tone = if l > 80.0 { "pale " } else if l < 32.0 { "deep " } else { "" };
    format!("{}{}", tone, word)
}

fn random_parts(rng: &mut StdRng) -> Vec<(usize, f32)> {
    random_parts_from(rng, &CHROMATIC)
}

fn random_parts_from(rng: &mut StdRng, bases: &[usize]) -> Vec<(usize, f32)> {
    let base = *pick(rng, bases);
    let mut parts = vec![(base, 1.0f32)];
    if rng.random::<f64>() < 0.45 {
        let other = *pick(rng, &CHROMATIC);
        if other != base {
            parts.push((other, rng.random_range(0.1..0.6) as f32));
        }
    }
    parts
}

fn tone_to(cm: &ColourModel, parts: &[(usize, f32)], target_l: f64, darkener: usize) -> [f32; 16] {
    let base = mix(parts, 0.0, 0.0);
    let l0 = cm.lab(&base)[0];
    let toward = if target_l > l0 { 0 } else { darkener };
    let blend_at = |t: f32| {
        let mut m = [0.0f32; 16];
        let total: f32 = base[..12].iter().sum::<f32>().max(1e-6);
        for i in 0..12 {
            m[i] = base[i] / total * (1.0 - t);
        }
        m[toward] += t;
        m
    };
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..22 {
        let mid = 0.5 * (lo + hi);
        let l = cm.lab(&blend_at(mid))[0];
        let reached = if toward == 0 { l >= target_l } else { l <= target_l };
        if reached { hi = mid; } else { lo = mid; }
    }
    blend_at(hi)
}

fn named_paint(cm: &ColourModel, rng: &mut StdRng, m: [f32; 16], thinner: f32, yield_pa: f32, mu: f64) -> Paint {
    let mut m = m;
    m[12] = (thinner + 0.05 * gauss(rng) as f32).clamp(0.0, 0.75);
    m[13] = yield_pa;
    Paint { name: colour_name(cm.lab(&m)), mix: m, mu: lognormal(rng, mu, 0.2) }
}

fn cm() -> &'static ColourModel {
    static CM: std::sync::OnceLock<ColourModel> = std::sync::OnceLock::new();
    CM.get_or_init(ColourModel::load)
}

fn lch(l: f64, c: f64, h_deg: f64) -> [f64; 3] {
    let h = h_deg.to_radians();
    [l, c * h.cos(), c * h.sin()]
}

fn hue_chroma(lab: [f64; 3]) -> (f64, f64) {
    (lab[2].atan2(lab[1]).to_degrees().rem_euclid(360.0), (lab[1] * lab[1] + lab[2] * lab[2]).sqrt())
}

fn srgb_to_lab(rgb: [f32; 3]) -> [f64; 3] {
    let lin = |c: f32| {
        let c = c as f64;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let (r, g, b) = (lin(rgb[0]), lin(rgb[1]), lin(rgb[2]));
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.9505;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.089;
    let f = |t: f64| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn build_mix(i: usize, j: usize, r: f64, white: f64, black: f64) -> [f32; 16] {
    let mut m = [0.0f32; 16];
    m[i] += (1.0 - r) as f32;
    m[j] += r as f32;
    m[0] += white as f32;
    m[1] += black as f32;
    let s: f32 = m[..12].iter().sum::<f32>().max(1e-6);
    for v in m[..12].iter_mut() {
        *v /= s;
    }
    m
}

fn best_mix(target: [f64; 3], lab_of: &dyn Fn(&[f32; 16]) -> [f64; 3]) -> [f32; 16] {
    let n = color::NPIG;
    let mut best = (f64::MAX, 0usize, 0usize, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        for j in i..n {
            let ratios: &[f64] = if i == j { &[0.0] } else { &[0.2, 0.4, 0.6, 0.8] };
            for &r in ratios {
                for &w in &[0.0, 0.15, 0.4, 1.0, 2.5] {
                    for &b in &[0.0, 0.05, 0.15, 0.4] {
                        let e = delta_e(lab_of(&build_mix(i, j, r, w, b)), target);
                        if e < best.0 {
                            best = (e, i, j, r, w, b);
                        }
                    }
                }
            }
        }
    }
    let (mut e, i, j, mut r, mut w, mut b) = best;
    let mut step = 1.0;
    for _ in 0..5 {
        let mut improved = true;
        while improved {
            improved = false;
            let tries = [
                ((r + 0.1 * step).min(1.0), w, b),
                ((r - 0.1 * step).max(0.0), w, b),
                (r, w * (1.0 + 0.4 * step) + 0.05 * step, b),
                (r, (w * (1.0 - 0.3 * step) - 0.03 * step).max(0.0), b),
                (r, w, b * (1.0 + 0.4 * step) + 0.02 * step),
                (r, w, (b * (1.0 - 0.3 * step) - 0.01 * step).max(0.0)),
            ];
            for (r2, w2, b2) in tries {
                let e2 = delta_e(lab_of(&build_mix(i, j, r2, w2, b2)), target);
                if e2 < e - 1e-3 {
                    e = e2;
                    r = r2;
                    w = w2;
                    b = b2;
                    improved = true;
                }
            }
        }
        step *= 0.5;
    }
    build_mix(i, j, r, w, b)
}

pub fn match_rgb(rgb: [f32; 3]) -> [f32; 16] {
    let c = cm();
    best_mix(srgb_to_lab(rgb), &|m| srgb_to_lab(color::mix_rgb(&c.set, &c.cmf, m)))
}

fn tone_mix(cm: &ColourModel, base: &[f32; 16], target_l: f64) -> [f32; 16] {
    let l0 = cm.lab(base)[0];
    let toward = if target_l > l0 { 0 } else if target_l < 40.0 { 1 } else { 9 };
    let total: f32 = base[..12].iter().sum::<f32>().max(1e-6);
    let blend_at = |t: f32| {
        let mut m = *base;
        for i in 0..12 {
            m[i] = base[i] / total * (1.0 - t);
        }
        m[toward] += t;
        m
    };
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..22 {
        let mid = 0.5 * (lo + hi);
        let l = cm.lab(&blend_at(mid))[0];
        let reached = if toward == 0 { l >= target_l } else { l <= target_l };
        if reached { hi = mid; } else { lo = mid; }
    }
    blend_at(hi)
}

const SCHEMES: [(&str, &[f64], f64); 5] = [
    ("complementary", &[180.0, 0.0, 210.0, 150.0], 1.0),
    ("split complementary", &[150.0, 210.0, 0.0, 180.0], 1.2),
    ("triadic", &[120.0, 240.0, 0.0, 180.0], 1.0),
    ("analogous", &[40.0, -40.0, 180.0, 75.0], 0.6),
    ("tetradic", &[90.0, 180.0, 270.0, 0.0], 0.6),
];

fn harmonic_palette(rng: &mut StdRng, cm: &ColourModel) -> Palette {
    let key = *weighted(rng, &[("dark", 1.2), ("mid", 1.3), ("light", 0.6)]);
    let target_l = match key {
        "dark" => rng.random_range(28.0..38.0),
        "mid" => rng.random_range(38.0..56.0),
        _ => rng.random_range(64.0..80.0),
    };
    let darkener = if target_l < 42.0 || rng.random::<f64>() < 0.5 { 1 } else { 9 };
    let ground_parts = match key {
        "dark" => random_parts_from(rng, &DEEP_PIGMENTS),
        _ => random_parts(rng),
    };
    let g0 = tone_to(cm, &ground_parts, target_l, darkener);
    let gl = cm.lab(&g0);
    let (h0, c0) = hue_chroma(gl);
    let h0 = if c0 < 8.0 { rng.random_range(0.0..360.0) } else { h0 };
    let lab_of = |m: &[f32; 16]| cm.lab(m);
    let side = if rng.random::<f64>() < 0.5 { 1.0 } else { -1.0 };
    let companion = if rng.random::<f64>() < 0.6 {
        lch(gl[0] + rng.random_range(-6.0..8.0), (c0 * rng.random_range(0.8..1.2)).max(15.0), h0 + side * rng.random_range(25.0..50.0))
    } else {
        lch(gl[0] + rng.random_range(-10.0..10.0), c0 * rng.random_range(0.3..0.6), h0 + rng.random_range(-15.0..15.0))
    };
    let g1 = best_mix(companion, &lab_of);
    let (scheme, offsets, _) = *weighted(rng, &SCHEMES.iter().map(|s| ((s.0, s.1, s.2), s.2)).collect::<Vec<_>>());
    let wants_light = |k: usize, flip: bool| -> bool {
        if gl[0] < 45.0 {
            true
        } else if gl[0] > 62.0 {
            false
        } else {
            (k % 2 == 0) ^ flip
        }
    };
    let flip = rng.random::<f64>() < 0.5;
    let contrast = |lab: [f64; 3]| (lab[0] - gl[0]).abs() * 1.4 + delta_e(lab, gl);
    let mut fg: Vec<[f32; 16]> = Vec::new();
    let mut labs: Vec<[f64; 3]> = Vec::new();
    let neutral_main = rng.random::<f64>() < 0.6;
    let mut hue_slot = 0usize;
    for k in 0..4 {
        let light = wants_light(k, flip);
        let mut best: Option<([f32; 16], f64)> = None;
        for attempt in 0..10 {
            let push = (attempt % 5) as f64 * 7.0;
            if attempt == 5 && !(k == 0 && neutral_main) {
                hue_slot += 1;
            }
            let m = if k == 0 && neutral_main {
                let t = if light { lch((90.0 - push * 0.5).max(70.0), rng.random_range(2.0..8.0), h0) } else { lch(rng.random_range(16.0..24.0), rng.random_range(0.0..6.0), h0 + 180.0) };
                best_mix(t, &lab_of)
            } else {
                let h = h0 + offsets[hue_slot % offsets.len()] + rng.random_range(-10.0..10.0);
                let l = if light { rng.random_range(62.0..88.0) + push * 0.4 } else { rng.random_range(18.0..42.0) - push * 0.5 };
                best_mix(lch(l.clamp(12.0, 94.0), rng.random_range(40.0..72.0), h), &lab_of)
            };
            let lab = cm.lab(&m);
            let spread = labs.iter().map(|o| delta_e(lab, *o)).fold(200.0f64, f64::min);
            let (hk, ck) = hue_chroma(lab);
            let hue_gap = labs
                .iter()
                .filter(|o| ck > 15.0 && hue_chroma(**o).1 > 15.0)
                .map(|o| {
                    let d = (hue_chroma(*o).0 - hk).rem_euclid(360.0);
                    d.min(360.0 - d)
                })
                .fold(180.0f64, f64::min);
            let name = colour_name(lab);
            let fresh = !labs.iter().any(|o| colour_name(*o) == name) && name != colour_name(gl);
            let score = contrast(lab).min(MIN_GROUND_CONTRAST * 1.5) + spread.min(MIN_LINE_SPREAD * 1.5) + hue_gap.min(MIN_HUE_GAP) + if fresh { 30.0 } else { 0.0 };
            if best.as_ref().map(|b| score > b.1).unwrap_or(true) {
                best = Some((m, score));
            }
            if contrast(lab) >= MIN_GROUND_CONTRAST && spread >= MIN_LINE_SPREAD && hue_gap >= MIN_HUE_GAP && fresh {
                break;
            }
        }
        if !(k == 0 && neutral_main) {
            hue_slot += 1;
        }
        let m = best.unwrap().0;
        labs.push(cm.lab(&m));
        fg.push(m);
    }
    let body = [(0.2, 0.0, 10.0), (0.15, 5.0, 10.0), (0.22, 0.0, 7.5), (0.22, 0.0, 7.5)];
    let mut fgp: Vec<Paint> = fg.iter().zip(body.iter()).map(|(m, &(th, y, mu))| named_paint(cm, rng, *m, th, y, mu)).collect();
    let accents = fgp.split_off(2);
    let mut pal = Palette {
        name: String::new(),
        scheme: format!("{} on {}", scheme, key),
        ground: vec![named_paint(cm, rng, g0, 0.45, 0.0, 4.0), named_paint(cm, rng, g1, 0.45, 0.0, 4.0)],
        ground_dark: None,
        main: fgp[0].clone(),
        second: fgp[1].clone(),
        accents,
    };
    pal.rebuild(cm);
    pal
}

impl Palette {
    pub const SLOTS: usize = 6;

    pub fn random(seed: u64) -> Palette {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15);
        palette(&mut rng)
    }

    fn slot(&self, s: usize) -> &Paint {
        match s {
            0 | 1 => &self.ground[s],
            2 => &self.main,
            3 => &self.second,
            _ => &self.accents[s - 4],
        }
    }

    fn slot_mut(&mut self, s: usize) -> &mut Paint {
        match s {
            0 | 1 => &mut self.ground[s],
            2 => &mut self.main,
            3 => &mut self.second,
            _ => &mut self.accents[s - 4],
        }
    }

    pub fn slot_mix(&self, s: usize) -> [f32; 16] {
        self.slot(s).mix
    }

    pub fn slot_name(&self, s: usize) -> &str {
        &self.slot(s).name
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set_slot(&mut self, s: usize, pigments: &[f32; 16]) {
        let total: f32 = pigments[..12].iter().sum::<f32>().max(1e-6);
        let p = self.slot_mut(s);
        for i in 0..12 {
            p.mix[i] = pigments[i] / total;
        }
        self.scheme = "custom".to_string();
        self.rebuild(cm());
    }

    fn rebuild(&mut self, cm: &ColourModel) {
        while self.ground.len() < 2 {
            let g = self.ground[0].clone();
            self.ground.push(g);
        }
        while self.accents.len() < 2 {
            let w = Paint { name: String::new(), mix: mix(&[(0, 1.0)], 0.22, 0.0), mu: 7.5 };
            self.accents.push(w);
        }
        self.ground.truncate(2);
        self.accents.truncate(2);
        let l0 = cm.lab(&self.ground[0].mix)[0];
        let mut light = self.ground[0].clone();
        light.mix = tone_mix(cm, &self.ground[0].mix, (l0 + 12.0).min(95.0));
        let mut dark = self.ground[0].clone();
        dark.mix = tone_mix(cm, &self.ground[0].mix, (l0 - 15.0).max(8.0));
        let mut seen: Vec<(String, f64)> = Vec::new();
        for s in 0..Self::SLOTS {
            let lab = cm.lab(&self.slot(s).mix);
            let base = colour_name(lab);
            let mut nm = base.clone();
            if let Some((_, l_other)) = seen.iter().find(|(n, _)| *n == base) {
                let stripped = base.trim_start_matches("pale ").trim_start_matches("deep ");
                nm = if stripped == "white" { "off-white".to_string() } else { format!("{} {}", if lab[0] > *l_other { "light" } else { "dark" }, stripped) };
            }
            let mut k = 2;
            let first = nm.clone();
            while seen.iter().any(|(n, _)| *n == nm) {
                nm = format!("{} {}", first, k);
                k += 1;
            }
            seen.push((nm.clone(), lab[0]));
            self.slot_mut(s).name = nm;
        }
        light.name = format!("light {}", self.ground[0].name);
        dark.name = format!("dark {}", self.ground[0].name);
        self.ground.push(light);
        self.ground_dark = Some(dark);
        let fg: Vec<&str> = (2..Self::SLOTS).map(|s| self.slot_name(s)).collect();
        self.name = format!("{}: {} and {} ground, {}", self.scheme, self.ground[0].name, self.ground[1].name, fg.join(", "));
    }
}

fn palette(rng: &mut StdRng) -> Palette {
    if rng.random::<f64>() >= CLASSIC_SHARE {
        return harmonic_palette(rng, cm());
    }
    let mut pal = classic_palette(rng);
    pal.scheme = pal.name.clone();
    pal.rebuild(cm());
    pal
}

fn classic_palette(rng: &mut StdRng) -> Palette {
    let family = rng.random_range(0..6);
    match family {
        0 => Palette {
            name: "autumn rhythm".to_string(),
            ground_dark: None,
            scheme: String::new(),
            ground: vec![
                paint(rng, "umber", &[(9, 0.6), (8, 0.3), (1, 0.1)], 0.45, 0.0, 4.0),
                paint(rng, "white", &[(0, 0.9), (8, 0.1)], 0.4, 0.0, 4.0),
                paint(rng, "teal", &[(10, 0.5), (6, 0.2), (0, 0.3)], 0.45, 0.0, 4.0),
            ],
            main: paint(rng, "black", &[(1, 1.0)], 0.2, 0.0, 10.0),
            second: paint(rng, "white", &[(0, 1.0)], 0.15, 5.0, 10.0),
            accents: vec![
                paint(rng, "umber", &[(9, 0.8), (1, 0.2)], 0.2, 0.0, 8.0),
                paint(rng, "ochre", &[(8, 0.9), (2, 0.1)], 0.2, 0.0, 7.0),
                paint(rng, "silver", &[(0, 0.75), (1, 0.25)], 0.35, 0.0, 1.5),
                paint(rng, "teal", &[(10, 0.5), (6, 0.3), (0, 0.2)], 0.2, 0.0, 8.0),
            ],
        },
        1 => Palette {
            name: "blue poles".to_string(),
            ground_dark: None,
            scheme: String::new(),
            ground: vec![
                paint(rng, "teal", &[(10, 0.4), (6, 0.2), (0, 0.3), (9, 0.1)], 0.5, 0.0, 3.0),
                paint(rng, "pale yellow", &[(2, 0.25), (0, 0.55), (8, 0.2)], 0.5, 0.0, 3.0),
                paint(rng, "white", &[(0, 0.95), (5, 0.05)], 0.45, 0.0, 3.0),
            ],
            main: paint(rng, "deep blue", &[(5, 0.7), (1, 0.3)], 0.2, 0.0, 10.0),
            second: paint(rng, "white", &[(0, 1.0)], 0.15, 5.0, 10.0),
            accents: vec![
                paint(rng, "orange", &[(11, 0.85), (3, 0.15)], 0.25, 0.0, 7.0),
                paint(rng, "yellow", &[(2, 0.75), (11, 0.1), (8, 0.15)], 0.3, 0.0, 6.0),
                paint(rng, "silver", &[(0, 0.75), (1, 0.25)], 0.35, 0.0, 1.5),
                paint(rng, "crimson", &[(4, 0.8), (3, 0.2)], 0.25, 0.0, 7.0),
            ],
        },
        2 => Palette {
            name: "convergence".to_string(),
            ground_dark: None,
            scheme: String::new(),
            ground: vec![
                paint(rng, "black", &[(1, 0.9), (9, 0.1)], 0.4, 0.0, 4.0),
                paint(rng, "umber", &[(9, 0.8), (1, 0.2)], 0.4, 0.0, 4.0),
            ],
            main: paint(rng, "white", &[(0, 1.0)], 0.15, 5.0, 10.0),
            second: paint(rng, "yellow", &[(2, 0.85), (0, 0.15)], 0.25, 0.0, 8.0),
            accents: vec![
                paint(rng, "red", &[(3, 0.8), (11, 0.2)], 0.2, 0.0, 8.0),
                paint(rng, "ultramarine", &[(5, 0.85), (0, 0.15)], 0.2, 0.0, 8.0),
                paint(rng, "orange", &[(11, 0.85), (2, 0.15)], 0.25, 0.0, 7.0),
                paint(rng, "white", &[(0, 1.0)], 0.3, 0.0, 6.0),
            ],
        },
        3 => Palette {
            name: "earth".to_string(),
            ground_dark: None,
            scheme: String::new(),
            ground: vec![
                paint(rng, "ochre", &[(8, 0.8), (0, 0.15), (9, 0.05)], 0.45, 0.0, 4.0),
                paint(rng, "umber", &[(9, 0.7), (1, 0.1), (8, 0.2)], 0.45, 0.0, 4.0),
            ],
            main: paint(rng, "umber black", &[(9, 0.55), (1, 0.45)], 0.2, 0.0, 10.0),
            second: paint(rng, "white", &[(0, 0.95), (8, 0.05)], 0.15, 5.0, 10.0),
            accents: vec![
                paint(rng, "red", &[(3, 0.7), (4, 0.3)], 0.2, 0.0, 8.0),
                paint(rng, "ochre", &[(8, 0.9), (2, 0.1)], 0.2, 0.0, 7.0),
                paint(rng, "orange", &[(11, 0.8), (8, 0.2)], 0.25, 0.0, 7.0),
                paint(rng, "black", &[(1, 1.0)], 0.2, 0.0, 10.0),
            ],
        },
        4 => Palette {
            name: "monochrome".to_string(),
            ground_dark: None,
            scheme: String::new(),
            ground: vec![
                paint(rng, "grey", &[(0, 0.7), (1, 0.3)], 0.45, 0.0, 4.0),
                paint(rng, "white", &[(0, 1.0)], 0.4, 0.0, 4.0),
            ],
            main: paint(rng, "black", &[(1, 1.0)], 0.2, 0.0, 10.0),
            second: paint(rng, "white", &[(0, 1.0)], 0.15, 5.0, 10.0),
            accents: vec![
                paint(rng, "silver", &[(0, 0.75), (1, 0.25)], 0.35, 0.0, 1.5),
                paint(rng, "charcoal", &[(1, 0.85), (0, 0.15)], 0.25, 0.0, 8.0),
                paint(rng, "white", &[(0, 1.0)], 0.3, 0.0, 6.0),
            ],
        },
        _ => Palette {
            name: "lavender mist".to_string(),
            ground_dark: None,
            scheme: String::new(),
            ground: vec![
                paint(rng, "white", &[(0, 1.0)], 0.45, 0.0, 3.0),
                paint(rng, "pink", &[(0, 0.85), (4, 0.15)], 0.5, 0.0, 3.0),
                paint(rng, "lavender", &[(0, 0.8), (5, 0.12), (4, 0.08)], 0.5, 0.0, 3.0),
            ],
            main: paint(rng, "silver grey", &[(0, 0.6), (1, 0.4)], 0.2, 0.0, 10.0),
            second: paint(rng, "black", &[(1, 1.0)], 0.2, 0.0, 10.0),
            accents: vec![
                paint(rng, "crimson", &[(4, 0.85), (0, 0.15)], 0.25, 0.0, 7.0),
                paint(rng, "ochre", &[(8, 0.85), (0, 0.15)], 0.25, 0.0, 7.0),
                paint(rng, "teal", &[(10, 0.5), (6, 0.2), (0, 0.3)], 0.25, 0.0, 7.0),
                paint(rng, "white", &[(0, 1.0)], 0.3, 0.0, 6.0),
            ],
        },
    }
}

fn thinned(p: &Paint, thinner: f32, yield_pa: f32, mu: f64) -> Paint {
    let mut m = p.mix;
    m[12] = thinner;
    m[13] = yield_pa;
    Paint { name: p.name.clone(), mix: m, mu }
}

fn layer(name: String, p: &Paint, style: Style, duration: f64) -> Layer {
    let (pace, loop_chance, height_median, stick_diameter, flow) = match style {
        Style::Flood => (0.15, 0.0, 0.12, 0.01, 4.0e-5),
        Style::Dry | Style::Tilt => (0.0, 0.0, 0.2, 0.01, 0.0),
        Style::Pour => (1.5, 0.12, 0.08, 0.008, 2.5e-6),
        Style::Lasso => (1.4, 0.0, 0.22, 0.014, 3.0e-6),
        Style::Drip => (1.7, 0.1, 0.14, 0.008, 2.0e-6),
        Style::Snap => (4.0, 0.0, 0.3, 0.012, 2.0e-6),
        Style::Fling => (2.8, 0.0, 0.4, 0.01, 2.0e-6),
        Style::Dots => (0.0, 0.0, 0.09, 0.011, 4.0e-7),
        Style::Wash => (0.35, 0.0, 0.1, 0.01, 1.2e-5),
        Style::Scrape => (0.18, 0.0, 0.0, 0.01, 0.0),
        Style::Scrub => (0.2, 0.0, 0.62, 0.01, 0.0),
    };
    Layer { name, mix: p.mix, mu: p.mu, style, duration, pace, loop_chance, height_median, stick_diameter, flow, pour_radius: 0.02, knife_width: 0.08, brush_width: 0.09, shades: Vec::new(), tilt: (0.0, 0.0), margin: GROUND_MARGIN_FLOATING, edge_bias: false, strokes: 0 }
}

fn shade(rng: &mut StdRng, base: &[f32; 16], partner: &[f32; 16], light: f32, dark: f32, cross: f32) -> [f32; 16] {
    let mut m = [0.0f32; 16];
    let total: f32 = base[..12].iter().sum::<f32>().max(1e-6);
    for i in 0..12 {
        m[i] = base[i] / total * lognormal(rng, 1.0, 0.15) as f32 + partner[i] / partner[..12].iter().sum::<f32>().max(1e-6) * cross;
    }
    m[0] += light;
    m[1] += 0.4 * dark;
    m[9] += 0.6 * dark;
    m[12] = base[12];
    m[13] = base[13];
    m
}

fn shades(rng: &mut StdRng, base: &Paint, partner: &Paint, pass: usize) -> Vec<[f32; 16]> {
    let n = rng.random_range(3..=4);
    let mut out = Vec::new();
    for _ in 0..n {
        let cross = rng.random_range(0.0..0.25);
        let (light, dark) = match pass {
            0 => (rng.random_range(0.0..0.1), rng.random_range(0.0..0.08)),
            1 => (rng.random_range(0.12..0.35), rng.random_range(0.0..0.02)),
            _ => (rng.random_range(0.0..0.04), rng.random_range(0.15..0.4)),
        };
        out.push(shade(rng, &base.mix, &partner.mix, light, dark, cross));
    }
    out
}

fn plan(rng: &mut StdRng, pal: &Palette) -> (Vec<Layer>, String) {
    let (mut layers, ground) = ground_plan(rng, pal, GroundMethod::Auto, 0);
    layers.extend(figure(rng, pal));
    (layers, ground)
}

fn ground_plan(rng: &mut StdRng, pal: &Palette, method: GroundMethod, strokes: usize) -> (Vec<Layer>, String) {
    let mut layers: Vec<Layer> = Vec::new();
    let dry = |rng: &mut StdRng, layers: &mut Vec<Layer>, chance: f64| {
        if rng.random::<f64>() < chance {
            layers.push(dry_layer(rng));
        }
    };
    let grounds = [("brushed", 6.0), ("raw canvas", 0.5), ("floods", 1.0), ("washes", 1.0), ("scraped", 0.8), ("tilt run", 0.8)];
    let ground = match method {
        GroundMethod::Auto => *weighted(rng, &grounds),
        GroundMethod::Brush => "brushed",
        GroundMethod::Bucket => "floods",
    };
    let mut gcols: Vec<Paint> = pal.ground.clone();
    if gcols.len() > 1 && rng.random::<f64>() < 0.3 {
        gcols.swap(0, 1);
    }
    let layout = *weighted(rng, &[("bleed", 0.55), ("border", 0.25), ("floating", 0.2)]);
    let margin = match layout {
        "bleed" => GROUND_MARGIN_BLEED,
        "border" => GROUND_MARGIN_BORDER,
        _ => GROUND_MARGIN_FLOATING,
    };
    match ground {
        "brushed" => {
            let base = gcols[0].clone();
            let partner = gcols[1 % gcols.len()].clone();
            let passes = rng.random_range(2..=3);
            for k in 0..passes {
                let (what, dur) = match k {
                    0 => ("ground", rng.random_range(110.0..150.0)),
                    1 => ("lights", rng.random_range(35.0..55.0)),
                    _ => ("darks", rng.random_range(18.0..30.0)),
                };
                let mut l = layer(format!("{} brushed {}", base.name, what), &thinned(&base, 0.12, 20.0, 18.0), Style::Scrub, dur);
                l.shades = shades(rng, &base, &partner, k);
                l.brush_width = rng.random_range(0.08..0.12);
                l.margin = margin;
                layers.push(l);
            }
            let edge_chance = match layout {
                "bleed" => 0.8,
                "border" => 0.45,
                _ => 0.25,
            };
            if rng.random::<f64>() < edge_chance {
                let deep = pal.ground_dark.clone().unwrap_or_else(|| base.clone());
                let mut dark = blend(rng, &deep, &base);
                dark.name = format!("{} edges", base.name);
                let mut l = layer(dark.name.clone(), &thinned(&dark, 0.12, 20.0, 18.0), Style::Scrub, rng.random_range(18.0..28.0));
                l.shades = shades(rng, &dark, &base, 0);
                l.brush_width = rng.random_range(0.07..0.1);
                l.margin = margin;
                l.edge_bias = true;
                layers.push(l);
            }
            if method == GroundMethod::Auto && rng.random::<f64>() < 0.35 {
                let p = &gcols[gcols.len() - 1];
                let d = rng.random_range(6.0..10.0);
                layers.push(layer(format!("{} flood", p.name), p, Style::Flood, d));
            }
            if strokes > 0 {
                layers.truncate(strokes);
                let total: f64 = layers.iter().map(|l| l.duration).sum();
                let mut left = strokes;
                for k in (1..layers.len()).rev() {
                    let n = ((strokes as f64 * layers[k].duration / total).round() as usize).clamp(1, left - k);
                    layers[k].strokes = n;
                    left -= n;
                }
                layers[0].strokes = left;
            }
        }
        "floods" if strokes > 0 => {
            let chunk = strokes.div_ceil(6).max(1);
            let mut k = 0;
            while k < strokes {
                let n = chunk.min(strokes - k);
                let p = &gcols[(k / chunk) % gcols.len()];
                let mut l = layer(format!("{} flood", p.name), p, Style::Flood, 1.0);
                l.strokes = n;
                layers.push(l);
                k += n;
            }
        }
        "floods" => {
            let n = rng.random_range(1..=gcols.len().min(3));
            for p in gcols.iter().take(n) {
                let d = rng.random_range(7.0..14.0);
                layers.push(layer(format!("{} flood", p.name), p, Style::Flood, d));
            }
        }
        "washes" => {
            let n = rng.random_range(2..=gcols.len().min(3));
            for p in gcols.iter().take(n) {
                let d = rng.random_range(8.0..14.0);
                let th = (p.mix[12] + 0.1).min(0.65);
                let mut l = layer(format!("{} sweep", p.name), &thinned(p, th, 0.0, 2.5), Style::Wash, d);
                l.pour_radius = lognormal(rng, 0.02, 0.3).clamp(0.01, 0.035);
                layers.push(l);
            }
        }
        "scraped" => {
            let rounds = rng.random_range(1..=2);
            for k in 0..rounds {
                let p = &gcols[k % gcols.len()];
                let d = rng.random_range(5.0..9.0);
                layers.push(layer(format!("{} pools", p.name), &thinned(p, 0.25, 0.0, 6.0), Style::Flood, d));
                let mut l = layer(format!("{} scraped", p.name), p, Style::Scrape, rng.random_range(10.0..16.0));
                l.knife_width = lognormal(rng, 0.08, 0.3).clamp(0.04, 0.14);
                layers.push(l);
            }
        }
        "tilt run" => {
            let n = rng.random_range(1..=gcols.len().min(2));
            for p in gcols.iter().take(n) {
                let d = rng.random_range(6.0..9.0);
                layers.push(layer(format!("{} pools", p.name), &thinned(p, 0.35, 0.0, 4.0), Style::Flood, d));
            }
            let mut l = layer("tilt run".to_string(), &gcols[0], Style::Tilt, rng.random_range(6.0..10.0));
            l.tilt = (rng.random_range(25.0..40.0), rng.random_range(-180.0..180.0));
            layers.push(l);
            if rng.random::<f64>() < 0.5 {
                let mut l = layer("tilt back".to_string(), &gcols[0], Style::Tilt, rng.random_range(4.0..7.0));
                l.tilt = (rng.random_range(15.0..30.0), layers.last().unwrap().tilt.1 + 180.0);
                layers.push(l);
            }
        }
        _ => {}
    }
    if ground != "raw canvas" {
        dry(rng, &mut layers, 0.85);
    }
    (layers, format!("{} {}", ground, layout))
}

fn dry_layer(rng: &mut StdRng) -> Layer {
    let d = rng.random_range(3.5..6.0);
    layer("drying".to_string(), &Paint { name: String::new(), mix: mix(&[(1, 1.0)], 0.2, 0.0), mu: 10.0 }, Style::Dry, d)
}

fn blend(rng: &mut StdRng, a: &Paint, b: &Paint) -> Paint {
    let t = rng.random_range(0.25..0.75) as f32;
    let sa: f32 = a.mix[..12].iter().sum::<f32>().max(1e-6);
    let sb: f32 = b.mix[..12].iter().sum::<f32>().max(1e-6);
    let mut m = [0.0f32; 16];
    for i in 0..12 {
        m[i] = (1.0 - t) * a.mix[i] / sa + t * b.mix[i] / sb;
    }
    m[12] = (1.0 - t) * a.mix[12] + t * b.mix[12];
    m[13] = (1.0 - t) * a.mix[13] + t * b.mix[13];
    let mu = (a.mu.ln() * (1.0 - t as f64) + b.mu.ln() * t as f64).exp();
    Paint { name: format!("{}-{}", a.name, b.name), mix: m, mu }
}

fn palette_colours(pal: &Palette) -> Vec<Paint> {
    let mut all = vec![pal.main.clone(), pal.second.clone()];
    for p in pal.accents.iter().chain(pal.ground.iter()) {
        if !all.iter().any(|q| q.name == p.name) {
            all.push(p.clone());
        }
    }
    all
}

fn line_layer(rng: &mut StdRng, p: &Paint, styles: &[(Style, f64)], fine: f64) -> Layer {
    let style = *weighted(rng, styles);
    let (name, pp, dur) = match style {
        Style::Snap => (format!("{} snaps", p.name), thinned(p, 0.45, 0.0, 1.5), rng.random_range(10.0..16.0)),
        Style::Fling => (format!("{} flings", p.name), thinned(p, 0.5, 0.0, 1.0), rng.random_range(8.0..14.0)),
        Style::Lasso => (format!("{} lasso", p.name), thinned(p, 0.12, 0.0, 12.0), rng.random_range(10.0..18.0)),
        Style::Dots => (format!("{} dots", p.name), p.clone(), rng.random_range(8.0..13.0)),
        Style::Pour => (format!("{} pours", p.name), p.clone(), rng.random_range(18.0..32.0)),
        _ => (format!("{} lines", p.name), p.clone(), rng.random_range(16.0..28.0)),
    };
    let mut l = layer(name, &pp, style, dur);
    if matches!(style, Style::Drip | Style::Lasso | Style::Snap | Style::Fling) {
        l.stick_diameter = (l.stick_diameter * lognormal(rng, 1.0 - 0.25 * fine, 0.18)).clamp(0.0045, 0.016);
    }
    if style == Style::Pour {
        l.flow *= lognormal(rng, 1.0 - 0.35 * fine, 0.25);
    }
    l.pace *= lognormal(rng, 1.0 + 0.15 * fine, 0.1);
    l
}

fn continuation(rng: &mut StdRng, pal: &Palette, round: usize) -> Vec<Layer> {
    let mut layers: Vec<Layer> = Vec::new();
    if rng.random::<f64>() < 0.75 {
        layers.push(dry_layer(rng));
    }
    let colours = palette_colours(pal);
    let mut theme = pick(rng, &colours).clone();
    if rng.random::<f64>() < 0.35 {
        let other = pick(rng, &colours).clone();
        if other.name != theme.name {
            theme = blend(rng, &theme, &other);
        }
    }
    let fine = (round as f64 / 4.0).min(1.0);
    let n = rng.random_range(4..=7);
    let mut last_name = String::new();
    for _ in 0..n {
        let which = *weighted(rng, &[("theme", 0.35), ("main", 0.2), ("second", 0.15), ("accent", 0.2), ("blend", 0.1)]);
        let p = match which {
            "theme" => theme.clone(),
            "main" => pal.main.clone(),
            "second" => pal.second.clone(),
            "blend" => {
                let a = pick(rng, &colours).clone();
                let b = pick(rng, &colours).clone();
                if a.name == b.name { a } else { blend(rng, &a, &b) }
            }
            _ => pick(rng, &pal.accents).clone(),
        };
        if p.name == last_name && rng.random::<f64>() < 0.7 {
            continue;
        }
        last_name = p.name.clone();
        let l = line_layer(rng, &p, &[(Style::Drip, 3.5), (Style::Pour, 2.0), (Style::Lasso, 1.4), (Style::Snap, 1.0), (Style::Fling, 0.8), (Style::Dots, 0.5)], fine);
        let style = l.style;
        layers.push(l);
        if matches!(style, Style::Pour | Style::Lasso) && rng.random::<f64>() < 0.25 {
            layers.push(dry_layer(rng));
        }
    }
    if rng.random::<f64>() < 0.6 {
        let p = if rng.random::<f64>() < 0.5 { pal.main.clone() } else { theme.clone() };
        let mut l = layer(format!("{} tracery", p.name), &thinned(&p, p.mix[12], 0.0, 6.0), Style::Drip, rng.random_range(16.0..26.0));
        l.stick_diameter = rng.random_range(0.0045..0.006);
        l.pace = 2.1;
        l.loop_chance = 0.06;
        layers.push(l);
    }
    if rng.random::<f64>() < 0.5 {
        let p = pick(rng, &pal.accents).clone();
        let style = *weighted(rng, &[(Style::Dots, 1.0), (Style::Fling, 1.0), (Style::Snap, 0.8)]);
        let (name, pp) = match style {
            Style::Dots => (format!("{} drops", p.name), p.clone()),
            Style::Fling => (format!("{} spatter", p.name), thinned(&p, 0.55, 0.0, 0.6)),
            _ => (format!("{} snaps", p.name), thinned(&p, 0.45, 0.0, 1.2)),
        };
        layers.push(layer(name, &pp, style, rng.random_range(7.0..12.0)));
    }
    for l in layers.iter_mut() {
        if l.style != Style::Dry {
            l.name = format!("r{} {}", round + 1, l.name);
        }
    }
    layers
}

fn figure(rng: &mut StdRng, pal: &Palette) -> Vec<Layer> {
    let mut layers: Vec<Layer> = Vec::new();
    let dry = |rng: &mut StdRng, layers: &mut Vec<Layer>, chance: f64| {
        if rng.random::<f64>() < chance {
            layers.push(dry_layer(rng));
        }
    };
    let skel_style = *weighted(rng, &[(Style::Pour, 1.0), (Style::Drip, 1.0)]);
    let mut skel = layer(format!("{} skeleton", pal.main.name), &pal.main, skel_style, rng.random_range(30.0..50.0));
    skel.loop_chance = rng.random_range(0.08..0.22);
    layers.push(skel);
    if rng.random::<f64>() < 0.6 {
        layers.push(layer(format!("{} lasso", pal.main.name), &thinned(&pal.main, 0.12, 0.0, 12.0), Style::Lasso, rng.random_range(12.0..20.0)));
    }
    dry(rng, &mut layers, 0.5);
    let n_mid = rng.random_range(3..=6);
    let mut last_name = String::new();
    for _ in 0..n_mid {
        let which = *weighted(rng, &[("second", 0.4), ("accent", 0.45), ("main", 0.15)]);
        let p = match which {
            "second" => pal.second.clone(),
            "main" => pal.main.clone(),
            _ => pick(rng, &pal.accents).clone(),
        };
        if p.name == last_name && rng.random::<f64>() < 0.7 {
            continue;
        }
        last_name = p.name.clone();
        let style = *weighted(rng, &[(Style::Drip, 3.0), (Style::Pour, 2.5), (Style::Lasso, 1.2), (Style::Snap, 1.2), (Style::Fling, 1.0), (Style::Dots, 0.6)]);
        let (name, pp, dur) = match style {
            Style::Snap => (format!("{} snaps", p.name), thinned(&p, 0.45, 0.0, 1.5), rng.random_range(10.0..16.0)),
            Style::Fling => (format!("{} flings", p.name), thinned(&p, 0.5, 0.0, 1.0), rng.random_range(8.0..14.0)),
            Style::Lasso => (format!("{} lasso", p.name), thinned(&p, 0.12, 0.0, 12.0), rng.random_range(10.0..18.0)),
            Style::Dots => (format!("{} dots", p.name), p.clone(), rng.random_range(8.0..13.0)),
            Style::Pour => (format!("{} pours", p.name), p.clone(), rng.random_range(18.0..32.0)),
            _ => (format!("{} lines", p.name), p.clone(), rng.random_range(16.0..28.0)),
        };
        layers.push(layer(name, &pp, style, dur));
        if matches!(style, Style::Pour | Style::Drip | Style::Lasso) {
            dry(rng, &mut layers, 0.3);
        }
    }
    let n_acc = rng.random_range(1..=3);
    let mut used: Vec<String> = Vec::new();
    for _ in 0..n_acc {
        let mut p = pick(rng, &pal.accents).clone();
        for _ in 0..4 {
            if !used.contains(&p.name) {
                break;
            }
            p = pick(rng, &pal.accents).clone();
        }
        used.push(p.name.clone());
        let style = *weighted(rng, &[(Style::Dots, 1.0), (Style::Fling, 1.3), (Style::Snap, 1.0)]);
        let (name, pp) = match style {
            Style::Dots => (format!("{} drops", p.name), p.clone()),
            Style::Fling => (format!("{} spatter", p.name), thinned(&p, 0.55, 0.0, 0.6)),
            _ => (format!("{} snaps", p.name), thinned(&p, 0.45, 0.0, 1.2)),
        };
        layers.push(layer(name, &pp, style, rng.random_range(7.0..13.0)));
    }
    if rng.random::<f64>() < 0.7 {
        let p = if rng.random::<f64>() < 0.6 { pal.main.clone() } else { pal.second.clone() };
        let mut l = layer(format!("{} tracery", p.name), &thinned(&p, p.mix[12], 0.0, 6.0), Style::Drip, rng.random_range(18.0..30.0));
        l.stick_diameter = rng.random_range(0.005..0.0065);
        l.pace = 2.1;
        l.loop_chance = 0.06;
        layers.push(l);
        if rng.random::<f64>() < 0.5 {
            let q = if p.name == pal.main.name { pal.second.clone() } else { pal.main.clone() };
            let mut l = layer(format!("{} tracery", q.name), &thinned(&q, q.mix[12], 0.0, 6.0), Style::Drip, rng.random_range(14.0..24.0));
            l.stick_diameter = rng.random_range(0.005..0.0065);
            l.pace = 2.1;
            l.loop_chance = 0.06;
            layers.push(l);
        }
    }
    if rng.random::<f64>() < 0.4 {
        layers.push(layer(format!("{} drops", pal.main.name), &pal.main, Style::Dots, rng.random_range(6.0..10.0)));
    }
    layers
}

impl Painter {
    pub fn new(seed: u64, w: f64, h: f64) -> Self {
        Self::full(seed, w, h, Palette::random(seed))
    }

    pub fn full(seed: u64, w: f64, h: f64, pal: Palette) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let (mut layers, ground) = plan(&mut rng, &pal);
        let rounds = rng.random_range(INITIAL_ROUNDS_MIN..=INITIAL_ROUNDS_MAX);
        for r in 1..=rounds {
            layers.extend(continuation(&mut rng, &pal, r));
        }
        Self::assemble(rng, seed, w, h, pal, layers, ground, rounds, true)
    }

    pub fn background(seed: u64, w: f64, h: f64, pal: Palette, method: GroundMethod, strokes: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let (layers, ground) = ground_plan(&mut rng, &pal, method, strokes);
        Self::assemble(rng, seed, w, h, pal, layers, ground, 0, false)
    }

    pub fn foreground(seed: u64, w: f64, h: f64, pal: Palette) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        let mut p = Self::assemble(rng, seed, w, h, pal.clone(), Vec::new(), "existing canvas".to_string(), 0, false);
        p.paint_foreground(pal);
        p
    }

    fn assemble(mut rng: StdRng, seed: u64, w: f64, h: f64, pal: Palette, layers: Vec<Layer>, ground: String, rounds: usize, figured: bool) -> Self {
        let total: f64 = layers.iter().map(|l| l.duration).sum();
        let names: Vec<String> = layers.iter().map(|l| if l.strokes > 0 { format!("{} x{}", l.name, l.strokes) } else { format!("{} {:.0}s", l.name, l.duration) }).collect();
        log::warn!("pollock seed {} palette {} ground {} total {:.0} s: {}", seed, pal.name, ground, total, names.join(", "));
        let layers_empty = layers.is_empty();
        let pos = DVec2::new(w * rng.random_range(0.2..0.8), h * rng.random_range(0.2..0.8));
        Self {
            rng,
            w,
            h,
            t: 0.0,
            layer: 0,
            layer_t: 0.0,
            layers,
            gesture: None,
            pos,
            heading: 0.0,
            ang_vel: 0.0,
            speed: 0.0,
            speed_noise: 0.0,
            height_noise: 0.0,
            cut_pending: false,
            reload_pending: false,
            ground_angle: 0.0,
            stroke_len: 0.0,
            layer_strokes: 0,
            dip_len: 0.0,
            stroke_mix: [0.0; 16],
            cover: vec![0.0; COVER_W * COVER_H],
            ink: vec![0.0; INK_W * INK_H],
            round: rounds,
            palette: pal.name.to_string(),
            ground,
            seed,
            done: layers_empty,
            figured,
            pal,
        }
    }

    pub fn paint_foreground(&mut self, pal: Palette) {
        self.palette = pal.name.clone();
        self.pal = pal;
        if self.figured {
            self.continue_painting();
            return;
        }
        self.figured = true;
        let added = figure(&mut self.rng, &self.pal);
        self.append(added);
    }

    pub fn continue_painting(&mut self) {
        self.round += 1;
        let added = continuation(&mut self.rng, &self.pal, self.round);
        self.append(added);
    }

    fn append(&mut self, added: Vec<Layer>) {
        let total: f64 = added.iter().map(|l| l.duration).sum();
        let names: Vec<String> = added.iter().map(|l| format!("{} {:.0}s", l.name, l.duration)).collect();
        log::warn!("pollock adds {} layers, palette {} total {:.0} s: {}", added.len(), self.pal.name, total, names.join(", "));
        self.layer = self.layers.len();
        self.layers.extend(added);
        self.layer_t = 0.0;
        self.gesture = None;
        self.cut_pending = true;
        self.reload_pending = true;
        self.stroke_len = 1e9;
        self.done = false;
    }

    pub fn stop(&mut self) {
        self.done = true;
        self.layer = self.layers.len();
        self.gesture = None;
    }

    pub fn skip_to(&mut self, name: &str) {
        if let Some(i) = self.layers.iter().position(|l| l.name == name) {
            self.layer = i;
            self.layer_t = 0.0;
            self.gesture = None;
        }
    }

    pub fn status(&self) -> String {
        if self.done {
            return format!("{} on {}: {} layers in {} rounds, paused", self.palette, self.ground, self.layers.len(), self.round + 1);
        }
        let l = &self.layers[self.layer];
        if l.strokes > 0 {
            return format!("{} on {}  layer {}/{} {}  stroke {}/{}", self.palette, self.ground, self.layer + 1, self.layers.len(), l.name, self.layer_strokes, l.strokes);
        }
        format!("{} on {}  layer {}/{} {}  {:.0}/{:.0} s", self.palette, self.ground, self.layer + 1, self.layers.len(), l.name, self.layer_t, l.duration)
    }

    fn gauss(&mut self) -> f64 {
        gauss(&mut self.rng)
    }

    fn lognormal(&mut self, median: f64, sigma: f64) -> f64 {
        lognormal(&mut self.rng, median, sigma)
    }

    fn layer_nu(&self, li: usize) -> f64 {
        let l = &self.layers[li];
        let thin = (l.mix[12] as f64).clamp(0.0, 0.95);
        l.mu * (1.0 - thin).powi(3) / RHO
    }

    fn layer_flow(&self, li: usize) -> f64 {
        let l = &self.layers[li];
        match l.style {
            Style::Pour | Style::Flood | Style::Lasso | Style::Wash => l.flow,
            _ => stick_flow(l.stick_diameter, self.layer_nu(li)),
        }
    }

    fn teleport(&mut self) {
        let m = START_INSET;
        let mut best = DVec2::new(self.rng.random_range(m..self.w - m), self.rng.random_range(m..self.h - m));
        let mut best_ink = self.ink_at(best);
        for _ in 1..START_CANDIDATES {
            let q = DVec2::new(self.rng.random_range(m..self.w - m), self.rng.random_range(m..self.h - m));
            let k = self.ink_at(q);
            if k < best_ink {
                best_ink = k;
                best = q;
            }
        }
        self.pos = best;
        let dir = DVec2::new(self.heading.cos(), self.heading.sin());
        if self.run_to_border(self.pos, dir, 0.0) < INWARD_REACH {
            let inward = DVec2::new(self.w * 0.5, self.h * 0.5) - self.pos;
            let spread = self.gauss() * 0.9;
            self.heading = inward.y.atan2(inward.x) + spread;
        }
        self.cut_pending = true;
    }

    fn ink_cell(&self, p: DVec2) -> Option<usize> {
        if p.x < 0.0 || p.y < 0.0 || p.x >= self.w || p.y >= self.h {
            return None;
        }
        let i = ((p.x / self.w) * INK_W as f64) as usize;
        let j = ((p.y / self.h) * INK_H as f64) as usize;
        Some(j.min(INK_H - 1) * INK_W + i.min(INK_W - 1))
    }

    fn ink_at(&self, p: DVec2) -> f32 {
        self.ink_cell(p).map(|c| self.ink[c]).unwrap_or(f32::MAX)
    }

    pub fn ink_rings(&self) -> Vec<f32> {
        let rings = INK_H / 2;
        let mut sum = vec![0.0f32; rings];
        let mut count = vec![0usize; rings];
        for j in 0..INK_H {
            for i in 0..INK_W {
                let r = (i.min(INK_W - 1 - i) * INK_H / INK_W).min(j.min(INK_H - 1 - j)).min(rings - 1);
                sum[r] += self.ink[j * INK_W + i];
                count[r] += 1;
            }
        }
        let total: f32 = self.ink.iter().sum::<f32>() / self.ink.len() as f32;
        sum.iter().zip(count.iter()).map(|(s, c)| s / *c as f32 / total.max(1e-6)).collect()
    }

    pub fn ink_balance(&self) -> (f32, f32) {
        let mut edge = (0.0f32, 0usize);
        let mut inner = (0.0f32, 0usize);
        for j in 0..INK_H {
            for i in 0..INK_W {
                let v = self.ink[j * INK_W + i];
                if i < 2 || j < 2 || i >= INK_W - 2 || j >= INK_H - 2 {
                    edge.0 += v;
                    edge.1 += 1;
                } else {
                    inner.0 += v;
                    inner.1 += 1;
                }
            }
        }
        (edge.0 / edge.1 as f32, inner.0 / inner.1 as f32)
    }

    fn edge_start(&mut self, margin: f64) -> (DVec2, f64) {
        let perimeter = 2.0 * (self.w + self.h);
        let u = self.rng.random_range(0.0..perimeter);
        let inset = margin.max(0.0) + EDGE_BAND * self.rng.random::<f64>().powi(2);
        let (p, along) = if u < self.w {
            (DVec2::new(u, inset), 0.0)
        } else if u < self.w + self.h {
            (DVec2::new(self.w - inset, u - self.w), std::f64::consts::FRAC_PI_2)
        } else if u < 2.0 * self.w + self.h {
            (DVec2::new(u - self.w - self.h, self.h - inset), 0.0)
        } else {
            (DVec2::new(inset, u - 2.0 * self.w - self.h), std::f64::consts::FRAC_PI_2)
        };
        let flip = if self.rng.random::<f64>() < 0.5 { std::f64::consts::PI } else { 0.0 };
        (p, along + flip)
    }

    fn cover_at(&self, p: DVec2) -> f32 {
        let i = ((p.x / self.w) * COVER_W as f64).floor().clamp(0.0, COVER_W as f64 - 1.0) as usize;
        let j = ((p.y / self.h) * COVER_H as f64).floor().clamp(0.0, COVER_H as f64 - 1.0) as usize;
        self.cover[j * COVER_W + i]
    }

    fn cover_path(&mut self, start: DVec2, dir: DVec2, len: f64) {
        let n = (len / 0.03).ceil().max(1.0) as usize;
        for k in 0..n {
            let p = start + dir * (len * (k as f64 + 0.5) / n as f64);
            let i = ((p.x / self.w) * COVER_W as f64).floor().clamp(0.0, COVER_W as f64 - 1.0) as usize;
            let j = ((p.y / self.h) * COVER_H as f64).floor().clamp(0.0, COVER_H as f64 - 1.0) as usize;
            self.cover[j * COVER_W + i] += 1.0;
        }
    }

    fn run_to_border(&self, from: DVec2, dir: DVec2, margin: f64) -> f64 {
        let tx = if dir.x > 1e-9 { (self.w - margin - from.x) / dir.x } else if dir.x < -1e-9 { (margin - from.x) / dir.x } else { f64::INFINITY };
        let ty = if dir.y > 1e-9 { (self.h - margin - from.y) / dir.y } else if dir.y < -1e-9 { (margin - from.y) / dir.y } else { f64::INFINITY };
        tx.min(ty).max(0.0)
    }

    fn new_gesture(&mut self) {
        let li = self.layer;
        let style = self.layers[li].style;
        let pace = self.layers[li].pace;
        let hm = self.layers[li].height_median;
        let loop_chance = self.layers[li].loop_chance;
        let nu = self.layer_nu(li);
        let q = self.layer_flow(li);
        let base = Gesture { t_start: self.t, t_end: self.t, speed: 0.0, height: hm, turn_sigma: 0.0, flow_until: self.t, hop_at: 0.0, centre: self.pos, radius: 0.0, phase0: 0.0, lissajous: false, drift: DVec2::ZERO, wobble: 0.0 };
        let g = match style {
            Style::Drip | Style::Pour => {
                let dur = if style == Style::Drip { self.rng.random_range(1.5..5.0) } else { self.rng.random_range(1.5..4.0) };
                let height = self.lognormal(hm, 0.3).clamp(0.04, 0.3);
                let ustar = transition_speed(q, height, nu);
                let is_loop = self.rng.random::<f64>() < loop_chance;
                let factor = if is_loop { self.lognormal(0.45, 0.3).clamp(0.15, 0.75) } else { self.lognormal(pace, 0.22).max(1.15) };
                let speed = (ustar * factor).clamp(0.06, 2.2);
                let r_turn = self.lognormal(0.18, 0.5).clamp(0.05, 0.6);
                Gesture { t_end: self.t + dur, speed, height, turn_sigma: speed / r_turn, flow_until: self.t + dur, ..base }
            }
            Style::Lasso => {
                let speed = self.lognormal(pace, 0.2).clamp(0.8, 4.0);
                let height = self.lognormal(hm, 0.25).clamp(0.12, 0.6);
                let radius = self.lognormal(0.14, 0.4).clamp(0.06, 0.3);
                let turns = self.rng.random_range(1.2..2.6);
                let dur = turns * std::f64::consts::TAU * radius / speed;
                let lissajous = self.rng.random::<f64>() < 0.4;
                let m = START_INSET;
                let centre = DVec2::new(self.pos.x.clamp(m, self.w - m), self.pos.y.clamp(m, self.h - m));
                let phase0 = self.rng.random_range(0.0..6.28);
                let da: f64 = self.rng.random_range(0.0..6.28);
                let ds = self.lognormal(0.25, 0.4).clamp(0.05, 0.6);
                let drift = DVec2::new(da.cos(), da.sin()) * ds;
                let wobble = self.rng.random_range(0.0..6.28);
                self.cut_pending = true;
                Gesture { t_end: self.t + dur, speed, height, flow_until: self.t + dur, centre, radius, phase0, lissajous, drift, wobble, ..base }
            }
            Style::Dry | Style::Tilt => {
                let d = self.layers[li].duration;
                self.cut_pending = true;
                Gesture { t_end: self.t + d, flow_until: self.t, ..base }
            }
            Style::Snap => {
                let speed = self.lognormal(pace, 0.2).clamp(2.5, 6.0);
                let height = self.lognormal(hm, 0.25).clamp(0.2, 0.5);
                self.heading = self.rng.random_range(0.0..6.28);
                self.cut_pending = true;
                Gesture { t_end: self.t + 1.0, speed, height, flow_until: self.t + 0.32, ..base }
            }
            Style::Fling => {
                let dur = 0.8;
                let speed = self.lognormal(pace, 0.25).clamp(1.5, 4.0);
                let height = self.lognormal(hm, 0.3).clamp(0.2, 0.6);
                self.heading = self.rng.random_range(0.0..6.28);
                Gesture { t_end: self.t + dur, speed, height, flow_until: self.t + 0.3, ..base }
            }
            Style::Dots => {
                let hold = self.rng.random_range(0.15..0.7);
                let height = self.lognormal(hm, 0.3).clamp(0.04, 0.15);
                Gesture { t_end: self.t + hold + 0.5, height, flow_until: self.t + hold, hop_at: self.t + hold, ..base }
            }
            Style::Flood => {
                let hold = self.rng.random_range(0.8..1.6);
                let height = self.lognormal(hm, 0.3).clamp(0.06, 0.2);
                self.pos = DVec2::new(self.w * self.rng.random_range(0.15..0.85), self.h * self.rng.random_range(0.15..0.85));
                self.cut_pending = true;
                Gesture { t_end: self.t + hold + 0.6, speed: pace, height, flow_until: self.t + hold, hop_at: self.t + hold, ..base }
            }
            Style::Wash => {
                let dur = self.rng.random_range(2.5..6.0);
                let speed = self.lognormal(pace, 0.25).clamp(0.15, 0.7);
                let r_turn = self.lognormal(0.3, 0.4).clamp(0.1, 0.8);
                let radius = self.lognormal(self.layers[li].pour_radius, 0.3).clamp(0.008, 0.04);
                Gesture { t_end: self.t + dur, speed, height: hm, turn_sigma: speed / r_turn, flow_until: self.t + dur, radius, ..base }
            }
            Style::Scrub => {
                let speed = self.lognormal(pace, 0.2).clamp(0.08, 0.35);
                let m = self.layers[li].margin;
                let mut a = if self.rng.random::<f64>() < 0.25 { self.rng.random_range(0.0..6.28) } else { self.ground_angle + self.gauss() * 0.3 };
                let mut start = DVec2::new(self.rng.random_range(m..self.w - m), self.rng.random_range(m..self.h - m));
                if self.layers[li].edge_bias {
                    let (p, along) = self.edge_start(m);
                    start = p;
                    a = along + self.gauss() * 0.25;
                } else {
                    let mut best = self.cover_at(start);
                    for _ in 0..3 {
                        let q = DVec2::new(self.rng.random_range(m..self.w - m), self.rng.random_range(m..self.h - m));
                        let c = self.cover_at(q);
                        if c < best {
                            best = c;
                            start = q;
                        }
                    }
                }
                let mut dir = DVec2::new(a.cos(), a.sin());
                if self.run_to_border(start, dir, m) < 0.1 {
                    a += std::f64::consts::PI;
                    dir = -dir;
                }
                let reach = self.run_to_border(start, dir, m);
                let len = self.lognormal(0.3, 0.35).clamp(0.12, 0.55).min(reach.max(0.05));
                self.cover_path(start, dir, len);
                let curv = self.gauss() * 0.5;
                let period = self.rng.random_range(0.25..0.8);
                let phase = self.rng.random_range(0.0..6.28);
                let dur = len / speed + 0.25;
                self.stroke_len += len;
                if self.stroke_len > self.dip_len {
                    self.reload_pending = true;
                    self.stroke_len = 0.0;
                    self.dip_len = self.rng.random_range(0.35..0.7);
                    let shades = &self.layers[li].shades;
                    if !shades.is_empty() {
                        self.stroke_mix = shades[self.rng.random_range(0..shades.len())];
                    } else {
                        self.stroke_mix = self.layers[li].mix;
                    }
                }
                self.pos = start;
                self.heading = a;
                self.cut_pending = true;
                Gesture { t_end: self.t + dur, speed, height: self.lognormal(hm, 0.08).clamp(0.5, 0.75), flow_until: self.t + len / speed, centre: start, drift: dir * speed, radius: period, phase0: curv, wobble: phase, ..base }
            }
            Style::Scrape => {
                let speed = self.lognormal(pace, 0.25).clamp(0.08, 0.35);
                let m = 0.06;
                let start = DVec2::new(self.rng.random_range(m..self.w - m), self.rng.random_range(m..self.h - m));
                let a: f64 = self.rng.random_range(0.0..6.28);
                let dir = DVec2::new(a.cos(), a.sin());
                let reach = self.run_to_border(start, dir, m);
                let len = self.lognormal(0.45, 0.4).clamp(0.15, 0.9).min(reach);
                let dur = len / speed + 0.4;
                self.pos = start;
                self.heading = a;
                self.cut_pending = true;
                Gesture { t_end: self.t + dur, speed, height: 0.0, flow_until: self.t + len / speed, centre: start, drift: dir * speed, ..base }
            }
        };
        if !matches!(style, Style::Fling | Style::Snap | Style::Scrape | Style::Scrub) {
            self.heading = self.heading + self.gauss() * 0.8;
        }
        self.gesture = Some(g);
    }

    fn steer(&mut self, dt: f64, turn_sigma: f64) {
        let dir = DVec2::new(self.heading.cos(), self.heading.sin());
        let ahead = self.pos + dir * (self.speed * EDGE_LOOKAHEAD).min(EDGE_LOOKAHEAD_MAX);
        let m = EDGE_MARGIN;
        let push = DVec2::new(
            (m - ahead.x).max(0.0) - (ahead.x - (self.w - m)).max(0.0),
            (m - ahead.y).max(0.0) - (ahead.y - (self.h - m)).max(0.0),
        ) / m;
        let out = push.length();
        if out > 0.0 {
            let n = push / out;
            let toward = dir.dot(n);
            let target = if toward < 0.0 { dir - 2.0 * toward * n } else { dir };
            let mut d = target.y.atan2(target.x) - self.heading;
            while d > std::f64::consts::PI {
                d -= std::f64::consts::TAU;
            }
            while d < -std::f64::consts::PI {
                d += std::f64::consts::TAU;
            }
            self.heading += d * (out * EDGE_TURN_GAIN * dt).min(1.0);
        }
        let tau = 0.35;
        let g = self.gauss();
        self.ang_vel += -self.ang_vel * dt / tau + turn_sigma * (2.0 * dt / tau).sqrt() * g;
        self.heading += self.ang_vel * dt;
    }

    fn noise(&mut self, dt: f64) {
        let g1 = self.gauss();
        let g2 = self.gauss();
        let ts = 0.45;
        let th = 0.9;
        self.speed_noise += -self.speed_noise * dt / ts + 0.22 * (2.0 * dt / ts).sqrt() * g1;
        self.height_noise += -self.height_noise * dt / th + 0.15 * (2.0 * dt / th).sqrt() * g2;
    }

    pub fn step(&mut self, dt: f64) -> Option<Command> {
        if self.done {
            return None;
        }
        self.t += dt;
        self.layer_t += dt;
        let cur = &self.layers[self.layer];
        let finished = if cur.strokes > 0 { self.layer_strokes >= cur.strokes && self.gesture.as_ref().map(|g| self.t >= g.t_end).unwrap_or(true) } else { self.layer_t >= cur.duration };
        if finished {
            self.layer += 1;
            self.layer_t = 0.0;
            self.layer_strokes = 0;
            self.gesture = None;
            self.cut_pending = true;
            self.reload_pending = true;
            self.ground_angle = self.rng.random_range(0.0..6.28);
            self.stroke_len = 1e9;
            if self.layer >= self.layers.len() {
                self.done = true;
                let (edge, inner) = self.ink_balance();
                log::warn!("pollock ink per cell: edge band {:.2} s, interior {:.2} s, ratio {:.2}", edge, inner, edge / inner.max(1e-6));
                return None;
            }
        }
        let mut reload = false;
        if self.gesture.as_ref().map(|g| self.t >= g.t_end).unwrap_or(true) {
            let style = self.layers[self.layer].style;
            if self.gesture.is_some() && !matches!(style, Style::Scrape | Style::Scrub) && self.rng.random::<f64>() < 0.35 {
                self.teleport();
            }
            self.new_gesture();
            self.layer_strokes += 1;
            reload = true;
        }
        self.noise(dt);
        let li = self.layer;
        let layer = &self.layers[li];
        let (mut mix, mu, sd, flow, pour_radius, knife_width, brush_width, tilt) = (layer.mix, layer.mu, layer.stick_diameter, layer.flow, layer.pour_radius, layer.knife_width, layer.brush_width, layer.tilt);
        let style = layer.style;
        let g = self.gesture.as_ref().unwrap();
        let (gt_start, gt_end, gspeed, gheight, turn_sigma, flow_until, hop_at) = (g.t_start, g.t_end, g.speed, g.height, g.turn_sigma, g.flow_until, g.hop_at);
        let (centre, radius, phase0, lissajous, drift, wobble) = (g.centre, g.radius, g.phase0, g.lissajous, g.drift, g.wobble);
        let mut flowing = self.t < flow_until;
        let mut forced: Option<DVec2> = None;
        match style {
            Style::Lasso => {
                let om = gspeed / radius;
                let tl = self.t - gt_start;
                let ph = phase0 + om * tl;
                let rr = radius * (1.0 + 0.22 * (0.37 * ph + wobble).sin());
                let c = centre + drift * tl;
                let p = if lissajous {
                    c + DVec2::new(rr * ph.cos(), 0.6 * rr * (2.0 * ph).sin())
                } else {
                    c + DVec2::new(rr * ph.cos(), rr * ph.sin())
                };
                forced = Some(p);
            }
            Style::Dry | Style::Tilt => {
                self.speed = 0.0;
                flowing = false;
            }
            Style::Snap => {
                let tl = self.t - gt_start;
                let s = if tl < 0.25 {
                    let u = tl / 0.25;
                    u * u * (3.0 - 2.0 * u)
                } else if tl < 0.3 {
                    1.0 - (tl - 0.25) / 0.05
                } else {
                    0.0
                };
                self.speed = gspeed * s;
            }
            Style::Drip | Style::Pour => {
                self.steer(dt, turn_sigma);
                let kappa = self.ang_vel.abs() / gspeed.max(0.05);
                let cf = (3.0 / kappa.max(0.3)).powf(1.0 / 3.0).clamp(0.4, 1.0);
                let target = gspeed * self.speed_noise.exp() * cf;
                self.speed += (target - self.speed) * (8.0 * dt).min(1.0);
            }
            Style::Wash => {
                self.steer(dt, turn_sigma);
                self.speed = gspeed * (0.5 * self.speed_noise).exp();
            }
            Style::Fling => {
                let u = ((gt_end - self.t) / 0.8).clamp(0.0, 1.0);
                let ramp = 1.0 - u;
                let s = ramp * ramp * (3.0 - 2.0 * ramp);
                self.speed = gspeed * s.max(0.05);
                flowing = flowing && self.t > gt_end - 0.75;
            }
            Style::Dots => {
                if self.t >= hop_at {
                    if self.t - dt < hop_at {
                        self.cut_pending = true;
                    }
                    self.speed = 0.9;
                    self.steer(dt, 4.0);
                    flowing = false;
                } else {
                    self.speed = 0.02;
                }
            }
            Style::Flood => {
                if self.t >= hop_at {
                    self.speed = 0.0;
                    flowing = false;
                } else {
                    self.steer(dt, 1.5);
                    self.speed = gspeed;
                }
            }
            Style::Scrape => {
                let tl = (self.t - gt_start).min((flow_until - gt_start).max(0.0));
                forced = Some(centre + drift * tl);
            }
            Style::Scrub => {
                let tl = (self.t - gt_start).min((flow_until - gt_start).max(0.0));
                let along = drift * tl;
                let s = along.length();
                let perp = DVec2::new(-drift.y, drift.x).normalize_or_zero();
                forced = Some(centre + along + perp * phase0 * s * s);
            }
        }
        let dir = DVec2::new(self.heading.cos(), self.heading.sin());
        let prev = self.pos;
        match forced {
            Some(p) => {
                self.pos = p;
            }
            None => {
                self.pos += dir * self.speed * dt;
            }
        }
        let reach = match style {
            Style::Scrub => (-self.layers[li].margin).max(-TOOL_EDGE),
            Style::Scrape | Style::Flood | Style::Wash => -TOOL_EDGE,
            _ => EDGE_OVERSHOOT,
        };
        self.pos.x = self.pos.x.clamp(-reach, self.w + reach);
        self.pos.y = self.pos.y.clamp(-reach, self.h + reach);
        let off = self.pos.x < EDGE_STOP || self.pos.y < EDGE_STOP || self.pos.x > self.w - EDGE_STOP || self.pos.y > self.h - EDGE_STOP;
        if off && reach > 0.0 && style != Style::Scrub {
            if flowing {
                self.cut_pending = true;
            }
            flowing = false;
        }
        if flowing && matches!(style, Style::Drip | Style::Pour | Style::Lasso | Style::Snap | Style::Fling | Style::Dots) {
            if let Some(c) = self.ink_cell(self.pos) {
                self.ink[c] += dt as f32;
            }
        }
        let v = (self.pos - prev) / dt.max(1e-6);
        let hmod = if style == Style::Drip || style == Style::Pour { self.height_noise.exp() } else { 1.0 };
        let height = match style {
            Style::Scrape => 0.0,
            Style::Scrub => (gheight * (1.0 + 0.1 * (std::f64::consts::TAU * (self.t - gt_start) / radius.max(0.05) + wobble).sin())).clamp(0.45, 0.85),
            _ => (gheight * hmod).clamp(0.03, 0.6),
        };
        if style == Style::Scrub {
            mix = self.stroke_mix;
        }
        let kind = match style {
            Style::Pour | Style::Lasso => ToolKind::Bottle,
            Style::Flood => ToolKind::Bucket,
            Style::Wash => ToolKind::Pour,
            Style::Scrape => ToolKind::Knife,
            Style::Scrub => ToolKind::Brush,
            _ => ToolKind::Stick,
        };
        if matches!(style, Style::Lasso | Style::Snap | Style::Scrape | Style::Scrub) {
            self.speed = v.length();
        }
        let cut = self.cut_pending;
        self.cut_pending = false;
        if self.reload_pending {
            reload = true;
            self.reload_pending = false;
        }
        let cure_rate = if style == Style::Dry { 0.6 } else { 0.0 };
        let tilt_cmd = if style == Style::Tilt { Some(tilt) } else { None };
        Some(Command {
            kind,
            pos: DVec3::new(self.pos.x, self.pos.y, height),
            vel: DVec3::new(v.x, v.y, 0.0),
            flowing,
            height,
            mu,
            sigma: 0.04,
            mix,
            stick_diameter: sd,
            bottle_flow: flow,
            bottle_radius: 0.0025,
            bucket_flow: flow,
            bucket_radius: 0.008,
            pour_rate: flow,
            pour_radius: if style == Style::Wash { radius } else { pour_radius },
            knife_width,
            knife_height: 0.0004,
            brush_width: if style == Style::Scrub { brush_width } else { 0.0 },
            brush_count: 400,
            brush_capacity: 3.0e-8,
            brush_release: 0.5,
            tilt: tilt_cmd,
            reload,
            cut,
            cure_rate,
        })
    }
}
