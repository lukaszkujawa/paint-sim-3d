use crate::color::PaintMix;
use crate::film::{Deposit, Method, DEPOSIT_CAP_END, DEPOSIT_CAP_START};
use crate::rod::{drop_step, impact_spread_factor, splash_count, Drop, Fluid, Nozzle, Rod, RodOut, Surface};
use rand::Rng;
use glam::DVec3;
use rayon::prelude::*;
use std::collections::HashMap;

const MAX_DROPS: usize = 20000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToolKind {
    Stick,
    Bottle,
    Bucket,
    Pour,
    Knife,
    Brush,
    Roller,
    Sponge,
}

#[derive(Clone, Copy)]
pub struct RecentCap {
    pub p0: [f64; 2],
    pub p1: [f64; 2],
    pub r: f64,
    pub top: f64,
    pub base: f64,
    pub age: u32,
}

pub struct CoarseSurface {
    pub w: usize,
    pub h: usize,
    pub cell: f64,
    pub data: Vec<f32>,
    pub sub: Vec<f32>,
    pub caps: Vec<RecentCap>,
    grid: HashMap<u32, Vec<u32>>,
    flat: bool,
    no_recent: bool,
}

const RECENT_FRAMES: u32 = 6;
const JITTER: f64 = 0.003;

fn seg_dist2(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let l2 = abx * abx + aby * aby;
    let mut t = 0.0;
    if l2 > 0.0 {
        t = (((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / l2).clamp(0.0, 1.0);
    }
    let dx = p[0] - (a[0] + abx * t);
    let dy = p[1] - (a[1] + aby * t);
    dx * dx + dy * dy
}

impl CoarseSurface {
    pub fn new(w: usize, h: usize, cell: f64) -> Self {
        Self { w, h, cell, data: vec![0.0; w * h], sub: vec![0.0; w * h], caps: Vec::new(), grid: HashMap::new(), flat: std::env::var("PAINT_FLAT_SURFACE").is_ok(), no_recent: std::env::var("PAINT_NO_RECENT").is_ok() }
    }

    fn sample(&self, arr: &[f32], x: f64, y: f64) -> f64 {
        if !(x.is_finite() && y.is_finite()) || x < 0.0 || y < 0.0 {
            return 0.0;
        }
        let gx = x / self.cell - 0.5;
        let gy = y / self.cell - 0.5;
        if gx >= (self.w - 1) as f64 || gy >= (self.h - 1) as f64 {
            return 0.0;
        }
        let i0 = gx.floor().max(0.0) as usize;
        let j0 = gy.floor().max(0.0) as usize;
        let tx = (gx - i0 as f64).clamp(0.0, 1.0);
        let ty = (gy - j0 as f64).clamp(0.0, 1.0);
        let s = |i: usize, j: usize| arr[j * self.w + i] as f64;
        let a = s(i0, j0) * (1.0 - tx) + s(i0 + 1, j0) * tx;
        let b = s(i0, j0 + 1) * (1.0 - tx) + s(i0 + 1, j0 + 1) * tx;
        a * (1.0 - ty) + b * ty
    }

    pub fn brush_floor(&self, x: f64, y: f64, sink: f64) -> f64 {
        let s = self.sample(&self.data, x, y);
        let sub = self.sample(&self.sub, x, y);
        sub + (s - sub).max(0.0) * (1.0 - sink)
    }

    fn cell_of(&self, x: f64, y: f64) -> Option<u32> {
        if !(x.is_finite() && y.is_finite()) || x < 0.0 || y < 0.0 {
            return None;
        }
        let i = (x / self.cell) as usize;
        let j = (y / self.cell) as usize;
        if i >= self.w || j >= self.h {
            return None;
        }
        Some((j * self.w + i) as u32)
    }

    fn recent_at(&self, x: f64, y: f64) -> f64 {
        let Some(c) = self.cell_of(x, y) else { return 0.0 };
        let Some(list) = self.grid.get(&c) else { return 0.0 };
        let mut best = 0.0f64;
        for &k in list {
            let cap = &self.caps[k as usize];
            let d2 = seg_dist2([x, y], cap.p0, cap.p1);
            if d2 < cap.r * cap.r {
                let v = cap.base + (cap.top - cap.base) * (1.0 - d2 / (cap.r * cap.r));
                best = best.max(v);
            }
        }
        best
    }

    fn insert_grid(&mut self, k: usize) {
        let cap = self.caps[k];
        let minx = ((cap.p0[0].min(cap.p1[0]) - cap.r) / self.cell).floor().max(0.0) as usize;
        let maxx = ((cap.p0[0].max(cap.p1[0]) + cap.r) / self.cell).floor().max(0.0) as usize;
        let miny = ((cap.p0[1].min(cap.p1[1]) - cap.r) / self.cell).floor().max(0.0) as usize;
        let maxy = ((cap.p0[1].max(cap.p1[1]) + cap.r) / self.cell).floor().max(0.0) as usize;
        for j in miny..=maxy.min(self.h - 1) {
            for i in minx..=maxx.min(self.w - 1) {
                self.grid.entry((j * self.w + i) as u32).or_default().push(k as u32);
            }
        }
    }

    pub fn add_recent(&mut self, p0: [f32; 2], p1: [f32; 2], r: f32, h0: f32) {
        let h0 = h0.min(2.0 * r).max(0.0) as f64;
        if !h0.is_finite() || !(r > 0.0) {
            return;
        }
        let p0 = [p0[0] as f64, p0[1] as f64];
        let p1 = [p1[0] as f64, p1[1] as f64];
        let mid = [0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])];
        let base = self.height(mid[0], mid[1]);
        let k = self.caps.len();
        self.caps.push(RecentCap { p0, p1, r: r as f64, top: base + h0, base, age: 0 });
        self.insert_grid(k);
    }

    pub fn decay_recent(&mut self) {
        for c in self.caps.iter_mut() {
            c.age += 1;
        }
        self.caps.retain(|c| c.age < RECENT_FRAMES);
        self.grid.clear();
        for k in 0..self.caps.len() {
            self.insert_grid(k);
        }
    }
}

impl Surface for CoarseSurface {
    fn height(&self, x: f64, y: f64) -> f64 {
        if self.flat { return 0.0; }
        let s = self.sample(&self.data, x, y);
        if self.no_recent { return s; }
        s.max(self.recent_at(x, y))
    }
}

pub struct Tool {
    pub kind: ToolKind,
    pub height: f64,
    pub pos: DVec3,
    pub vel: DVec3,
    pub flowing: bool,
    pub paint: PaintMix,
    pub stick_load: f64,
    pub stick_capacity: f64,
    pub infinite: bool,
    pub stick_diameter: f64,
    pub bottle_flow: f64,
    pub bottle_radius: f64,
    pub bucket_flow: f64,
    pub bucket_radius: f64,
    pub mu: f64,
    pub sigma: f64,
    pub rho: f64,
    pub knife_width: f64,
    pub knife_height: f64,
    pub flow_scale: f64,
}

impl Default for Tool {
    fn default() -> Self {
        Self {
            kind: ToolKind::Stick,
            height: 0.15,
            pos: DVec3::ZERO,
            vel: DVec3::ZERO,
            flowing: false,
            paint: PaintMix::pure(1),
            stick_load: 3.0e-6,
            stick_capacity: 3.0e-6,
            infinite: false,
            stick_diameter: 0.01,
            bottle_flow: 1.5e-6,
            bottle_radius: 1.5e-3,
            bucket_flow: 4.0e-5,
            bucket_radius: 8.0e-3,
            mu: 4.0,
            sigma: 0.04,
            rho: 1000.0,
            knife_width: 0.04,
            knife_height: 0.0003,
            flow_scale: 1.0,
        }
    }
}

impl Tool {
    pub fn fluid(&self) -> Fluid {
        let thin = self.paint.frac[12].clamp(0.0, 0.95) as f64;
        Fluid { mu: self.mu * (1.0 - thin).powi(3), rho: self.rho, sigma: self.sigma }
    }

    pub fn stick_flow(&self) -> f64 {
        if self.stick_load <= 0.0 {
            return 0.0;
        }
        let nu = self.fluid().mu / self.rho;
        let u_load = 0.5;
        let q0 = 0.016 * self.stick_diameter * self.stick_diameter * (9.81 * nu * u_load).powf(0.25);
        q0 * (self.stick_load / self.stick_capacity).sqrt()
    }

    pub fn nozzle(&self) -> Option<Nozzle> {
        if !self.flowing {
            return None;
        }
        let (flow, radius) = match self.kind {
            ToolKind::Stick => (self.stick_flow(), (self.stick_diameter * 0.15).max(5e-4)),
            ToolKind::Bottle => (self.bottle_flow * self.flow_scale, self.bottle_radius),
            ToolKind::Bucket => (self.bucket_flow * self.flow_scale, self.bucket_radius),
            ToolKind::Pour | ToolKind::Knife | ToolKind::Brush | ToolKind::Roller | ToolKind::Sponge => return None,
        };
        if flow <= 0.0 {
            return None;
        }
        Some(Nozzle { pos: self.pos, vel: self.vel, dir: DVec3::new(0.0, 0.0, -1.0), radius, flow })
    }

    pub fn reload(&mut self) {
        self.stick_load = self.stick_capacity;
    }
}

pub struct Airborne {
    pub rods: Vec<Rod>,
    pub drops: Vec<Drop>,
    pub next_id: u64,
    pub landed_volume: f64,
    pub emitted_volume: f64,
    pub prof: [f32; 3],
    pub rod_steps: u32,
    prev_nozzle: Option<(u64, DVec3)>,
}

impl Airborne {
    pub fn new() -> Self {
        Self { rods: Vec::new(), drops: Vec::new(), next_id: 1, landed_volume: 0.0, emitted_volume: 0.0, prof: [0.0; 3], rod_steps: 0, prev_nozzle: None }
    }

    pub fn detach(&mut self) {
        for r in self.rods.iter_mut() {
            r.nozzle = None;
        }
    }

    pub fn airborne_volume(&self) -> f64 {
        self.rods.iter().map(|r| r.total_volume()).sum::<f64>() + self.drops.iter().map(|d| d.vol).sum::<f64>()
    }

    pub fn update(&mut self, tool: &mut Tool, frame_dt: f64, surface: &mut CoarseSurface, gravity: DVec3, dx: f32, deposits: &mut Vec<Deposit>, bounds: [f64; 2]) {
        let nozzle = tool.nozzle();
        let attached = self.rods.iter().position(|r| r.nozzle.is_some());
        match (nozzle, attached) {
            (Some(nz), Some(i)) => {
                let pig = tool.paint.normalized().packed();
                let fl = tool.fluid();
                let same = self.rods[i].pig == pig && (self.rods[i].fluid.mu - fl.mu).abs() < 1e-9 && (self.rods[i].fluid.sigma - fl.sigma).abs() < 1e-9;
                if same {
                    self.prev_nozzle = self.rods[i].nozzle.as_ref().map(|n| (self.rods[i].id, n.pos));
                    self.rods[i].nozzle = Some(nz);
                } else {
                    self.rods[i].nozzle = None;
                    let rod = Rod::from_nozzle(nz, fl, pig, self.next_id);
                    self.next_id += 1;
                    self.emitted_volume += rod.total_volume();
                    self.rods.push(rod);
                }
            }
            (Some(nz), None) => {
                let fluid = tool.fluid();
                let rod = Rod::from_nozzle(nz, fluid, tool.paint.normalized().packed(), self.next_id);
                self.next_id += 1;
                self.emitted_volume += rod.total_volume();
                self.rods.push(rod);
            }
            (None, Some(i)) => {
                self.rods[i].nozzle = None;
            }
            (None, None) => {}
        }
        if let Some(i) = self.rods.iter().position(|r| r.nozzle.is_some()) {
            let q = self.rods[i].nozzle.as_ref().unwrap().flow;
            self.emitted_volume += q * frame_dt;
            if tool.kind == ToolKind::Stick && !tool.infinite {
                tool.stick_load = (tool.stick_load - q * frame_dt).max(0.0);
            }
        }
        let n_rods = self.rods.len();
        let t_rods = std::time::Instant::now();
        let outs: Vec<RodOut> = self
            .rods
            .par_iter_mut()
            .map(|rod| {
                let mut out = RodOut { landings: vec![], drops: vec![], fragments: vec![] };
                let mut t = 0.0;
                let mut guard = 0;
                let nozzle_end = rod.nozzle.as_ref().map(|n| (n.pos, n.vel));
                let nozzle_start = self.prev_nozzle.filter(|(id, _)| *id == rod.id).map(|(_, p)| p);
                while t < frame_dt && guard < 4000 {
                    guard += 1;
                    let speed = rod.tip_speed().max(0.05);
                    let spacing = rod.min_spacing().max(1e-4);
                    let dt_cfl = spacing / speed;
                    let n = rod.n();
                    let floor = if rod.nozzle.is_none() {
                        3.0e-4
                    } else if n > 1000 {
                        5.0e-4
                    } else if n > 600 {
                        3.0e-4
                    } else if n > 300 {
                        2.0e-4
                    } else {
                        1.0e-4
                    };
                    let dt = (frame_dt - t).min(dt_cfl.clamp(floor, 1.0e-3));
                    if let (Some((pe, ve)), Some(nz)) = (nozzle_end, rod.nozzle.as_mut()) {
                        let k = rod.step_count.wrapping_add(rod.id.wrapping_mul(7919));
                        let jx = (k.wrapping_mul(2654435761) % 1000003) as f64 / 500001.5 - 1.0;
                        let jy = (k.wrapping_mul(40503).wrapping_add(7) % 999983) as f64 / 499991.5 - 1.0;
                        let along = match nozzle_start {
                            Some(p0) => p0 + (pe - p0) * ((t + dt) / frame_dt),
                            None => pe - ve * (frame_dt - (t + dt)),
                        };
                        nz.pos = along + DVec3::new(jx, jy, 0.0) * (JITTER * nz.radius);
                    }
                    rod.step(dt, surface as &dyn Surface, &mut out, gravity);
                    t += dt;
                    if rod.is_empty() {
                        break;
                    }
                    rod.drain_landings(dx as f64, false, &mut out.landings);
                    let nonfinite = rod.x.iter().any(|x| !x.is_finite()) || rod.v.iter().any(|v| !v.is_finite());
                    let outside = rod.x.iter().any(|x| x.x < -0.5 || x.y < -0.5 || x.x > bounds[0] + 0.5 || x.y > bounds[1] + 0.5 || x.z > 3.0);
                    if outside && !nonfinite {
                        rod.x.clear();
                        rod.v.clear();
                        rod.vol.clear();
                        rod.frozen.clear();
                        rod.vimp.clear();
                        rod.m1.clear();
                        rod.tan.clear();
                        break;
                    }
                    if nonfinite {
                        let vmax = rod.v.iter().map(|v| v.length()).fold(0.0, f64::max);
                        let k = rod.n().min(10);
                        let vols: Vec<String> = rod.vol.iter().take(k).map(|v| format!("{:.2e}", v)).collect();
                        let lens: Vec<String> = (0..k.saturating_sub(1)).map(|e| format!("{:.2e}", (rod.x[e + 1] - rod.x[e]).length())).collect();
                        let zs: Vec<String> = rod.x.iter().take(k).map(|x| format!("{:.2e}", x.z)).collect();
                        log::warn!("rod {} diverged: n {} dt {:.2e} spacing {:.2e} rmin {:.2e} vmax {:.2e} nozzle {} frozen {:?} vol {:?} len {:?} z {:?}", rod.id, rod.n(), dt, rod.min_spacing(), rod.min_free_radius(), vmax, rod.nozzle.is_some(), &rod.frozen[..k], vols, lens, zs);
                        rod.x.clear();
                        rod.v.clear();
                        rod.vol.clear();
                        rod.frozen.clear();
                        rod.m1.clear();
                        rod.tan.clear();
                        break;
                    }
                }
                if rod.is_empty() {
                    rod.drain_landings(dx as f64, true, &mut out.landings);
                }
                out
            })
            .collect();
        let _ = n_rods;
        self.prof[0] = t_rods.elapsed().as_secs_f32() * 1e3;
        self.rod_steps = self.rods.iter().map(|r| r.step_count as u32).sum();
        let t_land = std::time::Instant::now();
        let mut new_rods = Vec::new();
        let mut rng = rand::rng();
        let mut splashes: Vec<Drop> = Vec::new();
        for out in outs {
            for l in out.landings {
                let g = l.seg;
                let mut vol = g.vol;
                let n = splash_count(g.speed, 2.0 * g.thread_r, &l.fluid);
                if n > 0 {
                    let share = (0.3 * vol).min(vol);
                    let each = share / n as f64;
                    let mid = 0.5 * (g.p0 + g.p1);
                    for _ in 0..n {
                        let az: f64 = rng.random::<f64>() * std::f64::consts::TAU;
                        let el: f64 = (30.0 + 30.0 * rng.random::<f64>()).to_radians();
                        let sp = g.speed * (0.1 + 0.2 * rng.random::<f64>());
                        let v = DVec3::new(sp * el.cos() * az.cos(), sp * el.cos() * az.sin(), sp * el.sin());
                        let frac = 0.5 + rng.random::<f64>();
                        splashes.push(Drop { x: DVec3::new(mid.x, mid.y, 2.0 * g.thread_r), v, vol: each * frac, pig: l.pig, fluid: l.fluid });
                    }
                    let spent: f64 = splashes.iter().rev().take(n).map(|d| d.vol).sum();
                    vol = (vol - spent).max(0.0);
                }
                let caps = if g.cap_start { DEPOSIT_CAP_START } else { 0 } | if g.cap_end { DEPOSIT_CAP_END } else { 0 };
                let d = Deposit::ribbon(g.p0.as_vec2().to_array(), g.p1.as_vec2().to_array(), g.n0.as_vec2().to_array(), g.n1.as_vec2().to_array(), g.r0 as f32, g.r1 as f32, caps, vol as f32, l.pig, dx).laid_by(Method::Thread);
                surface.add_recent(d.p0, d.p1, d.max_radius(), d.max_height());
                self.landed_volume += vol;
                deposits.push(d);
            }
            for d in out.drops {
                self.drops.push(d);
            }
            for f in out.fragments {
                new_rods.push(f);
            }
        }
        self.rods.retain(|r| !r.is_empty());
        self.rods.extend(new_rods);
        self.drops.extend(splashes);
        self.drops.retain(|d| d.x.is_finite() && d.x.x > -0.5 && d.x.y > -0.5 && d.x.x < bounds[0] + 0.5 && d.x.y < bounds[1] + 0.5 && d.x.z < 3.0);
        self.prof[1] = t_land.elapsed().as_secs_f32() * 1e3;
        let t_drops = std::time::Instant::now();
        let mut i = 0;
        let mut secondary: Vec<Drop> = Vec::new();
        while i < self.drops.len() {
            let mut t = 0.0;
            let mut landed = false;
            while t < frame_dt {
                let dt = (frame_dt - t).min(5e-4);
                drop_step(&mut self.drops[i], dt, gravity);
                t += dt;
                let d = &self.drops[i];
                let zs = surface.height(d.x.x, d.x.y);
                let r = (3.0 * d.vol / (4.0 * std::f64::consts::PI)).cbrt();
                if d.x.z <= zs + r {
                    let u = (-d.v.z).max(0.0);
                    let f = impact_spread_factor(u, 2.0 * r, &d.fluid);
                    let rd = (r * f) as f32;
                    let mut vol = d.vol;
                    let n = splash_count(u, 2.0 * r, &d.fluid);
                    if n > 0 && d.vol > 1e-12 && self.drops.len() + secondary.len() + n < MAX_DROPS {
                        let mut rng = rand::rng();
                        let each = 0.3 * vol / n as f64;
                        for _ in 0..n {
                            let az: f64 = rng.random::<f64>() * std::f64::consts::TAU;
                            let el: f64 = (30.0 + 30.0 * rng.random::<f64>()).to_radians();
                            let sp = u * (0.1 + 0.2 * rng.random::<f64>());
                            let v = DVec3::new(sp * el.cos() * az.cos(), sp * el.cos() * az.sin(), sp * el.sin());
                            let sv = each * (0.5 + rng.random::<f64>());
                            secondary.push(Drop { x: d.x + DVec3::new(0.0, 0.0, r), v, vol: sv, pig: d.pig, fluid: d.fluid });
                            vol -= sv;
                        }
                        vol = vol.max(0.0);
                    }
                    let dep = Deposit::capsule([d.x.x as f32, d.x.y as f32], [d.x.x as f32, d.x.y as f32], rd, vol as f32, d.pig, dx).laid_by(Method::Drop);
                    surface.add_recent(dep.p0, dep.p1, dep.max_radius(), dep.max_height());
                    self.landed_volume += vol;
                    deposits.push(dep);
                    landed = true;
                    break;
                }
            }
            if landed {
                self.drops.swap_remove(i);
            } else {
                i += 1;
            }
        }
        self.drops.extend(secondary);
        self.prof[2] = t_drops.elapsed().as_secs_f32() * 1e3;
    }
}

pub fn push_tube(verts: &mut Vec<TubeVertex>, idx: &mut Vec<u32>, pts: &[DVec3], r: f64, col: [f32; 3]) {
    let n = pts.len();
    if n < 2 {
        return;
    }
    let segs = 6u32;
    let base = verts.len() as u32;
    let mut m1 = DVec3::X;
    for i in 0..n {
        let t = if i == 0 { pts[1] - pts[0] } else if i == n - 1 { pts[n - 1] - pts[n - 2] } else { pts[i + 1] - pts[i - 1] };
        let mut t = t.normalize_or_zero();
        if t == DVec3::ZERO {
            t = DVec3::Z;
        }
        m1 = (m1 - t * m1.dot(t)).normalize_or_zero();
        if m1 == DVec3::ZERO {
            m1 = t.any_orthonormal_vector();
        }
        let m2 = t.cross(m1);
        for s in 0..segs {
            let a = s as f64 / segs as f64 * std::f64::consts::TAU;
            let nrm = m1 * a.cos() + m2 * a.sin();
            let p = pts[i] + nrm * r;
            verts.push(TubeVertex { pos: [p.x as f32, p.y as f32, p.z as f32], nrm: [nrm.x as f32, nrm.y as f32, nrm.z as f32], col });
        }
    }
    for i in 0..(n as u32 - 1) {
        for s in 0..segs {
            let s1 = (s + 1) % segs;
            let a = base + i * segs + s;
            let b = base + i * segs + s1;
            let c = base + (i + 1) * segs + s;
            let d = base + (i + 1) * segs + s1;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
}

pub fn tube_mesh(rods: &[Rod], drops: &[Drop], color_of: &dyn Fn(&[u32; 8]) -> [f32; 3]) -> (Vec<TubeVertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut idx = Vec::new();
    let segs = 10u32;
    for rod in rods {
        let n = rod.n();
        if n < 2 {
            continue;
        }
        let col = color_of(&rod.pig);
        let base = verts.len() as u32;
        for i in 0..n {
            let t = if i == 0 { rod.tan[0] } else if i == n - 1 { rod.tan[n - 2] } else { (rod.tan[i - 1] + rod.tan[i]).normalize_or_zero() };
            let m1 = if i == 0 { rod.m1[0] } else if i == n - 1 { rod.m1[n - 2] } else { (rod.m1[i - 1] + rod.m1[i]).normalize_or_zero() };
            let m1 = (m1 - t * m1.dot(t)).normalize_or_zero();
            let m2 = t.cross(m1);
            let r = rod.vertex_radius(i).max(2e-5);
            for s in 0..segs {
                let a = s as f64 / segs as f64 * std::f64::consts::TAU;
                let nrm = m1 * a.cos() + m2 * a.sin();
                let p = rod.x[i] + nrm * r;
                verts.push(TubeVertex { pos: [p.x as f32, p.y as f32, p.z as f32], nrm: [nrm.x as f32, nrm.y as f32, nrm.z as f32], col });
            }
        }
        for i in 0..(n as u32 - 1) {
            for s in 0..segs {
                let s1 = (s + 1) % segs;
                let a = base + i * segs + s;
                let b = base + i * segs + s1;
                let c = base + (i + 1) * segs + s;
                let d = base + (i + 1) * segs + s1;
                idx.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
    }
    for d in drops {
        let r = (3.0 * d.vol / (4.0 * std::f64::consts::PI)).cbrt().max(2e-5);
        let col = color_of(&d.pig);
        let base = verts.len() as u32;
        let rings = 6u32;
        for j in 0..=rings {
            let phi = j as f64 / rings as f64 * std::f64::consts::PI;
            for s in 0..segs {
                let a = s as f64 / segs as f64 * std::f64::consts::TAU;
                let nrm = DVec3::new(phi.sin() * a.cos(), phi.sin() * a.sin(), phi.cos());
                let p = d.x + nrm * r;
                verts.push(TubeVertex { pos: [p.x as f32, p.y as f32, p.z as f32], nrm: [nrm.x as f32, nrm.y as f32, nrm.z as f32], col });
            }
        }
        for j in 0..rings {
            for s in 0..segs {
                let s1 = (s + 1) % segs;
                let a = base + j * segs + s;
                let b = base + j * segs + s1;
                let c = base + (j + 1) * segs + s;
                let dd = base + (j + 1) * segs + s1;
                idx.extend_from_slice(&[a, b, c, b, dd, c]);
            }
        }
    }
    (verts, idx)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TubeVertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    pub col: [f32; 3],
}

