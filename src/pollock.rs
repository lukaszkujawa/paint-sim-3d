mod colour;
mod palette;
mod layers;
pub use colour::*;
pub use palette::*;
pub use layers::*;
use crate::color;
use crate::tools::ToolKind;
use glam::{DVec2, DVec3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const RHO: f64 = 1000.0;
const G: f64 = 9.81;
const COVER_W: usize = 16;
const COVER_H: usize = 11;
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
