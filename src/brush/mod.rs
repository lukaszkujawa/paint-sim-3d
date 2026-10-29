use crate::film::Contact;
use crate::tools::{push_tube, TubeVertex};
use glam::DVec3;
use rayon::prelude::*;

const BASE_CAPACITY: f64 = 6.0e-9;
const BASE_AREA: f64 = std::f64::consts::PI * 0.008 * 0.008;
const BASE_LEN: f64 = 0.035;
const TUFT: f64 = 0.4;
const STEER_MIN_SPEED: f64 = 0.002;
const STEER_DIST: f64 = 0.35;
const BELLY: f64 = 2.2;
const POINT: f64 = 0.8;
const TAPER: f64 = 0.45;
const CLUMP: f64 = 0.03;
const PACK: f64 = 0.9;
const FILL: f64 = 0.6;
const HAIR_MIN: f64 = 2.0e-5;
const COHESION: f64 = 0.5;
const SPREAD: f64 = 2.5;
const SPLAY: f64 = 4.0;
const FILM: f64 = 1.2e-4;
const RELEASE_REF: f64 = 1.5;
const SWEEP_MIN: f64 = 0.25;
const CONTACT_BODY: u32 = 3;
const OVERFLOW: f64 = 8.0;
const LUBE: f64 = 0.6;
const BEAD: f64 = 3.0e-4;

pub struct Bristle {
    pub pts: Vec<DVec3>,
    pub prev: Vec<DVec3>,
    pub root: DVec3,
    pub rest: DVec3,
    pub dirs: Vec<DVec3>,
    pub seg: f64,
    pub rho: Vec<f64>,
    pub rr: Vec<f64>,
    pub rn: f64,
    pub out: DVec3,
    pub load: f64,
    pub comp: [f32; 16],
    pub touch: Vec<bool>,
    pub last: Vec<DVec3>,
    pub was: Vec<bool>,
    pub thick: f64,
}

fn hash01(k: usize, salt: u64) -> f64 {
    let mut h = (k as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(salt.wrapping_mul(0xD1B54A32D192ED03));
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 32;
    (h & 0xffff) as f64 / 65535.0
}

pub struct Release {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub r: f32,
    pub vol: f64,
    pub comp: [f32; 16],
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BrushShape {
    Round,
    Flat,
}

pub struct Brush {
    pub bristles: Vec<Bristle>,
    pub shape: BrushShape,
    pub width: f64,
    pub angle: f64,
    pub len: f64,
    pub segs: usize,
    pub ferrule_r: f64,
    pub radius: f64,
    pub hair: f64,
    pub tuft: f64,
    pub stiffness: f64,
    pub friction: f64,
    pub capacity: f64,
    pub base_capacity: f64,
    pub release: f64,
    pub placed: bool,
    pub handle: DVec3,
    pub touching: usize,
}

impl Brush {
    pub fn new(n: usize, len: f64, ferrule_r: f64, radius: f64) -> Self {
        Self::build(BrushShape::Round, n, len, 2.0 * ferrule_r, radius)
    }

    pub fn build(shape: BrushShape, n: usize, len: f64, size: f64, radius: f64) -> Self {
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        let segs = 8;
        let mut bristles = Vec::with_capacity(n);
        let blank = |k: usize, root: DVec3, rest: DVec3| {
            let thick = 0.65 + 0.7 * hash01(k, 11);
            let clump = (hash01(k / 3, 23) - 0.5) * 0.22;
            let lean = DVec3::new(rest.x + clump * (hash01(k / 3, 29) - 0.5), rest.y + clump * (hash01(k / 3, 31) - 0.5), rest.z).normalize();
            Bristle { pts: vec![DVec3::ZERO; segs + 1], prev: vec![DVec3::ZERO; segs + 1], root, rest: lean, dirs: vec![lean; segs], seg: len / segs as f64, rho: Vec::new(), rr: Vec::new(), rn: 0.0, out: DVec3::ZERO, load: 0.0, comp: [0.0; 16], touch: vec![false; segs + 1], last: vec![DVec3::ZERO; segs + 1], was: vec![false; segs + 1], thick }
        };
        let lemon = |k: usize, rj: f64, a: f64, ferrule_r: f64| {
            let rn = (rj / ferrule_r).min(1.0);
            let reach = (1.0 - TAPER * rn * rn) * (1.0 + 0.01 * (hash01(k, 17) - 0.5));
            let (cx, cy) = (CLUMP * (hash01(k / 3, 29) - 0.5), CLUMP * (hash01(k / 3, 31) - 0.5));
            let node = |i: usize| {
                let u = reach * i as f64 / segs as f64;
                let rho = rj * (1.0 + BELLY * u) * (1.0 - u).max(0.0).powf(POINT);
                DVec3::new(rho * a.cos() + cx * u * len, rho * a.sin() + cy * u * len, -u * len)
            };
            let nodes: Vec<DVec3> = (0..=segs).map(node).collect();
            let dirs: Vec<DVec3> = nodes.windows(2).map(|w| (w[1] - w[0]).normalize()).collect();
            let seg = nodes.windows(2).map(|w| (w[1] - w[0]).length()).sum::<f64>() / segs as f64;
            let mut b = blank(k, DVec3::new(rj * a.cos(), rj * a.sin(), 0.0), dirs[0]);
            b.rest = dirs[0];
            b.dirs = dirs;
            b.seg = seg;
            b.rr = nodes.iter().map(|p| (p.x * p.x + p.y * p.y).sqrt()).collect();
            b.rho = b.rr.iter().map(|r| PACK * r).collect();
            b.out = DVec3::new(a.cos(), a.sin(), 0.0);
            b.rn = rn;
            b
        };
        let (ferrule_r, width, area) = match shape {
            BrushShape::Round => {
                let ferrule_r = 0.5 * size;
                for k in 0..n {
                    let r = ferrule_r * ((k as f64 + 0.5) / n as f64).sqrt();
                    let a = k as f64 * golden + (hash01(k, 3) - 0.5) * 0.5;
                    let rj = r * (1.0 + 0.12 * (hash01(k, 5) - 0.5));
                    bristles.push(lemon(k, rj, a, ferrule_r));
                }
                (ferrule_r, size, std::f64::consts::PI * ferrule_r * ferrule_r)
            }
            BrushShape::Flat => {
                let width = size;
                let thick = (0.18 * width).clamp(1.5e-3, 6.0e-3).min(width);
                let cols = ((n as f64 * width / thick).sqrt().ceil() as usize).max(2);
                let rows = n.div_ceil(cols).max(1);
                for k in 0..n {
                    let c = (k % cols) as f64;
                    let r = (k / cols) as f64;
                    let jx = hash01(k, 7) - 0.5;
                    let jy = hash01(k, 13) - 0.5;
                    let x = (-0.5 + (c + 0.5 + 0.9 * jx) / cols as f64) * width;
                    let y = (-0.5 + (r + 0.5 + 0.9 * jy) / rows as f64) * thick;
                    let root = DVec3::new(x, y, 0.0);
                    let rest = DVec3::new(0.3 * x / (0.5 * width), 0.15 * y / (0.5 * thick), -1.0).normalize();
                    bristles.push(blank(k, root, rest));
                }
                (0.5 * width, width, width * (0.18 * width).max(thick))
            }
        };
        let tuft = TUFT * (area / n.max(1) as f64).sqrt();
        let hair = radius;
        let radius = hair.min(ferrule_r * (FILL / n.max(1) as f64).sqrt()).max(HAIR_MIN);
        let capacity = BASE_CAPACITY * (area / BASE_AREA).sqrt() * (len / BASE_LEN);
        Self { bristles, shape, width, angle: 0.0, len, segs, ferrule_r, radius, hair, tuft, stiffness: 0.35, friction: 0.7, capacity, base_capacity: capacity, release: 1.5, placed: false, handle: DVec3::ZERO, touching: 0 }
    }

    pub fn size(&self) -> f64 {
        self.width
    }

    fn rooted(&self, root: DVec3) -> DVec3 {
        let (s, c) = self.angle.sin_cos();
        DVec3::new(c * root.x - s * root.y, s * root.x + c * root.y, root.z)
    }

    pub fn total_load(&self) -> f64 {
        self.bristles.iter().map(|b| b.load).sum()
    }

    pub fn total_capacity(&self) -> f64 {
        self.capacity * self.bristles.len() as f64
    }

    pub fn dip(&mut self, frac: &[f32; 16]) {
        for b in &mut self.bristles {
            b.load = self.capacity;
            b.comp = *frac;
        }
    }

    pub fn top_up(&mut self, frac: &[f32; 16]) {
        let cap = self.capacity;
        for b in &mut self.bristles {
            if b.load >= cap {
                continue;
            }
            let w = (b.load / cap) as f32;
            for i in 0..16 {
                b.comp[i] = w * b.comp[i] + (1.0 - w) * frac[i];
            }
            b.load = cap;
        }
    }

    pub fn dip_uneven(&mut self, frac: &[f32; 16], seed: u64) {
        for (k, b) in self.bristles.iter_mut().enumerate() {
            let mut h = (k as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(seed.wrapping_mul(0xD1B54A32D192ED03));
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58476D1CE4E5B9);
            h ^= h >> 32;
            let u = (h & 0xffff) as f64 / 65535.0;
            b.load = self.capacity * (0.35 + 0.65 * u);
            b.comp = *frac;
        }
    }

    pub fn clean(&mut self) {
        for b in &mut self.bristles {
            b.load = 0.0;
            b.comp = [0.0; 16];
        }
    }

    pub fn absorb(&mut self, res: &[[u32; 17]]) -> f64 {
        let mut total = 0.0;
        for (b, r) in self.bristles.iter_mut().zip(res) {
            let vol = r[0] as f64 * 1e-15;
            if vol <= 0.0 {
                continue;
            }
            let newload = b.load + vol;
            for i in 0..16 {
                let sc = if i >= 13 { 100.0 } else { 1.0 };
                let add = r[1 + i] as f64 * 1e-15 * sc;
                b.comp[i] = ((b.comp[i] as f64 * b.load + add) / newload) as f32;
            }
            b.load = newload;
            total += vol;
        }
        total
    }

    pub fn update(&mut self, handle: DVec3, dt: f64, floor: &(dyn Fn(f64, f64) -> f64 + Sync), dx: f64) -> (Vec<Contact>, Vec<Release>) {
        let l = self.len / self.segs as f64;
        let travel = if self.placed { (handle - self.handle).length() } else { 0.0 };
        if !self.placed || travel > 0.1 {
            return self.substep(handle, dt, floor, dx);
        }
        let k = ((travel / (0.4 * l)).ceil() as usize).clamp(1, 12);
        let start = self.handle;
        let mut contacts = Vec::new();
        let mut releases = Vec::new();
        for s in 1..=k {
            let hk = start.lerp(handle, s as f64 / k as f64);
            let (c, r) = self.substep(hk, dt / k as f64, floor, dx);
            contacts.extend(c);
            releases.extend(r);
        }
        (contacts, releases)
    }

    fn substep(&mut self, handle: DVec3, dt: f64, floor: &(dyn Fn(f64, f64) -> f64 + Sync), dx: f64) -> (Vec<Contact>, Vec<Release>) {
        let segs = self.segs;
        let radius = self.radius;
        let stiffness = self.stiffness;
        let friction = self.friction;
        let capacity = self.capacity;
        let ferrule = self.ferrule_r;
        let vel = if self.placed { (handle - self.handle) / dt.max(1e-6) } else { DVec3::ZERO };
        let speed = vel.truncate().length();
        if self.shape == BrushShape::Flat && speed > STEER_MIN_SPEED {
            let want = vel.y.atan2(vel.x) + std::f64::consts::FRAC_PI_2;
            let mut d = want - self.angle;
            while d > std::f64::consts::FRAC_PI_2 {
                d -= std::f64::consts::PI;
            }
            while d < -std::f64::consts::FRAC_PI_2 {
                d += std::f64::consts::PI;
            }
            self.angle += d * (speed * dt / (STEER_DIST * self.width)).min(1.0);
        }
        let roots: Vec<DVec3> = self.bristles.iter().map(|b| self.rooted(b.root)).collect();
        let (sa, ca) = self.angle.sin_cos();
        let turn = move |v: DVec3| DVec3::new(ca * v.x - sa * v.y, sa * v.x + ca * v.y, v.z);
        if !self.placed || (handle - self.handle).length() > 0.1 {
            for (b, root) in self.bristles.iter_mut().zip(&roots) {
                b.pts[0] = handle + *root;
                b.prev[0] = b.pts[0];
                for i in 1..=segs {
                    b.pts[i] = b.pts[i - 1] + turn(b.dirs[i - 1]) * b.seg;
                    b.prev[i] = b.pts[i];
                }
            }
            self.placed = true;
        }
        self.handle = handle;
        let packed = !self.bristles.is_empty() && !self.bristles[0].rho.is_empty();
        let press = ((self.len + floor(handle.x, handle.y) - handle.z) / self.len).clamp(0.0, 1.0);
        let inv = 1.0 / self.bristles.len().max(1) as f64;
        for b in self.bristles.iter_mut().zip(roots.iter()) {
            b.0.pts[0] = handle + *b.1;
        }
        for _ in 0..8 {
            let axis: Vec<DVec3> = if packed { (0..=segs).map(|i| self.bristles.iter().map(|b| b.pts[i]).sum::<DVec3>() * inv).collect() } else { Vec::new() };
            self.bristles.par_iter_mut().for_each(|b| {
                let l = b.seg;
                let out = turn(b.out);
                let wet = (b.load / capacity.max(1e-30)).min(1.0);
                let friction = friction * (1.0 - LUBE * wet);
                let hold = COHESION * (0.3 + 0.7 * wet);
                for i in 1..=segs {
                    let dir_prev = if i == 1 { turn(b.dirs[0]) } else { ((b.pts[i - 1] - b.pts[i - 2]).normalize_or_zero() + turn(b.dirs[i - 1] - b.dirs[i - 2])).normalize_or_zero() };
                    let target = b.pts[i - 1] + dir_prev * l;
                    let cur = b.pts[i];
                    b.pts[i] = cur + (target - cur) * stiffness;
                    let zs = floor(b.pts[i].x, b.pts[i].y) + radius;
                    if b.pts[i].z < zs {
                        b.pts[i].z = zs;
                        let d = b.prev[i] - b.pts[i];
                        let cur = b.pts[i];
                        b.pts[i] = DVec3::new(cur.x + d.x * friction, cur.y + d.y * friction, cur.z);
                    }
                    if packed {
                        let off = DVec3::new(b.pts[i].x - axis[i].x, b.pts[i].y - axis[i].y, 0.0);
                        let radial = off.dot(out);
                        let floor_r = b.rho[i] + SPLAY * press * ferrule * b.rn * (i as f64 / segs as f64);
                        if radial < floor_r {
                            b.pts[i] += out * (floor_r - radial);
                        }
                        let off = DVec3::new(b.pts[i].x - axis[i].x, b.pts[i].y - axis[i].y, 0.0);
                        let r = off.length();
                        let rmax = b.rr[i] + SPREAD * press * ferrule + 2.0 * radius;
                        if r > rmax {
                            b.pts[i] -= off * ((r - rmax) / r * hold);
                        }
                    }
                    let seg = b.pts[i] - b.pts[i - 1];
                    let sl = seg.length();
                    b.pts[i] = b.pts[i - 1] + if sl > 1e-9 { seg / sl * l } else { dir_prev * l };
                }
            });
        }
        self.bristles.par_iter_mut().for_each(|b| {
            let bead = BEAD * (b.load / capacity.max(1e-30)).min(1.0);
            for i in 0..=segs {
                let zs = floor(b.pts[i].x, b.pts[i].y) + radius;
                b.last[i] = b.prev[i];
                b.was[i] = b.touch[i];
                b.touch[i] = i > 0 && b.pts[i].z <= zs + 0.5 * radius + bead;
                b.prev[i] = b.pts[i];
            }
        });
        let mut contacts = Vec::new();
        let mut releases = Vec::new();
        let vxy = DVec3::new(vel.x, vel.y, 0.0);
        let sp = vxy.length();
        let dir = if sp > 1e-6 { vxy / sp } else { DVec3::ZERO };
        let mut touching = 0;
        let across = if sp > 1e-6 { DVec3::new(-dir.y, dir.x, 0.0) } else { DVec3::X };
        let (mut lo, mut hi, mut lanes, mut count) = (f64::MAX, f64::MIN, 0.0f64, 0usize);
        for b in &self.bristles {
            let mut hit = false;
            for i in 1..=segs {
                if b.touch[i] {
                    for p in [b.pts[i - 1], b.pts[i]] {
                        let q = p.dot(across);
                        lo = lo.min(q);
                        hi = hi.max(q);
                    }
                    hit = true;
                }
            }
            if hit {
                lanes += 2.0 * (radius * 2.0 * b.thick).max(self.tuft * b.thick).max(0.6 * dx);
                count += 1;
            }
        }
        let overlap = if count > 0 { (lanes / (hi - lo + lanes / count as f64)).max(1.0) } else { 1.0 };
        for (bi, b) in self.bristles.iter_mut().enumerate() {
            let segs_touching: Vec<usize> = (1..=segs).filter(|&i| b.touch[i]).collect();
            if segs_touching.is_empty() {
                continue;
            }
            touching += 1;
            let rc = (radius * 2.0 * b.thick).max(self.tuft * b.thick).max(0.6 * dx) as f32;
            let nseg = segs_touching.len() as f64;
            let over = (b.load - self.capacity).max(0.0) * (OVERFLOW * dt).min(1.0) / nseg;
            let film = FILM * (b.load / self.capacity).min(1.0) * (self.release / RELEASE_REF) * b.thick / overlap;
            let laid: Vec<(usize, usize, f64, f64)> = segs_touching
                .iter()
                .map(|&i| {
                    let tip = if b.pts[i].z <= b.pts[i - 1].z { i } else { i - 1 };
                    let swept = (b.pts[tip] - b.last[tip]).truncate().length();
                    (i, tip, swept, 2.0 * rc as f64 * swept.max(rc as f64) * film)
                })
                .collect();
            let want: f64 = laid.iter().map(|l| l.3).sum();
            let scale = if want > 0.0 { (b.load.min(self.capacity) / want).min(1.0) } else { 0.0 };
            let mut spent = 0.0;
            for (i, tip, swept, vol) in laid {
                let rel = vol * scale + over;
                spent += rel;
                let p0 = [b.pts[i - 1].x as f32, b.pts[i - 1].y as f32];
                let p1 = [b.pts[i].x as f32, b.pts[i].y as f32];
                let zb = (b.pts[i].z.min(b.pts[i - 1].z) - radius) as f32;
                contacts.push(Contact { p0, p1, r: rc, zb, bristle: bi as u32, pad: CONTACT_BODY });
                for j in [i - 1, i] {
                    if j == i - 1 && i > 1 && b.touch[i - 1] {
                        continue;
                    }
                    let from = b.last[j];
                    if b.was[i] && (b.pts[j] - from).truncate().length() > SWEEP_MIN * rc as f64 {
                        let zt = (b.pts[j].z.max(from.z) - radius) as f32;
                        contacts.push(Contact { p0: [from.x as f32, from.y as f32], p1: [b.pts[j].x as f32, b.pts[j].y as f32], r: rc, zb: zt, bristle: bi as u32, pad: 0 });
                    }
                }
                if rel > 1e-15 {
                    let from = b.last[tip];
                    if b.was[i] && swept > SWEEP_MIN * rc as f64 {
                        let q0 = [from.x as f32, from.y as f32];
                        let q1 = [b.pts[tip].x as f32, b.pts[tip].y as f32];
                        releases.push(Release { p0: q0, p1: q1, r: rc, vol: rel, comp: b.comp });
                    } else {
                        releases.push(Release { p0, p1, r: rc, vol: rel, comp: b.comp });
                    }
                }
            }
            b.load = (b.load - spent).max(0.0);
        }
        self.touching = touching;
        (contacts, releases)
    }

    pub fn mesh(&self, verts: &mut Vec<TubeVertex>, idx: &mut Vec<u32>, paint_col: &dyn Fn(&[f32; 16]) -> [f32; 3], rmin: f64) {
        for b in &self.bristles {
            let f = (b.load / self.capacity).clamp(0.0, 1.0) as f32;
            let pc = if f > 0.0 { paint_col(&b.comp) } else { [0.0; 3] };
            let base = [0.42f32, 0.3, 0.17];
            let col = [base[0] + (pc[0] - base[0]) * f, base[1] + (pc[1] - base[1]) * f, base[2] + (pc[2] - base[2]) * f];
            push_tube(verts, idx, &b.pts, (self.radius * b.thick).max(rmin), col);
        }
    }
}
