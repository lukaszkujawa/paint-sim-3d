use crate::film::Contact;
use crate::tools::{push_tube, TubeVertex};
use glam::DVec3;
use rayon::prelude::*;

pub struct Bristle {
    pub pts: Vec<DVec3>,
    pub prev: Vec<DVec3>,
    pub root: DVec3,
    pub rest: DVec3,
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
    pub stiffness: f64,
    pub friction: f64,
    pub capacity: f64,
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
            Bristle { pts: vec![DVec3::ZERO; segs + 1], prev: vec![DVec3::ZERO; segs + 1], root, rest: lean, load: 0.0, comp: [0.0; 16], touch: vec![false; segs + 1], last: vec![DVec3::ZERO; segs + 1], was: vec![false; segs + 1], thick }
        };
        let (ferrule_r, width) = match shape {
            BrushShape::Round => {
                let ferrule_r = 0.5 * size;
                for k in 0..n {
                    let r = ferrule_r * ((k as f64 + 0.5) / n as f64).sqrt();
                    let a = k as f64 * golden + (hash01(k, 3) - 0.5) * 0.5;
                    let rj = r * (1.0 + 0.12 * (hash01(k, 5) - 0.5));
                    let root = DVec3::new(rj * a.cos(), rj * a.sin(), 0.0);
                    let splay = 0.35 * r / ferrule_r;
                    let rest = DVec3::new(a.cos() * splay, a.sin() * splay, -1.0).normalize();
                    bristles.push(blank(k, root, rest));
                }
                (ferrule_r, size)
            }
            BrushShape::Flat => {
                let width = size;
                let thick = (0.18 * width).clamp(1.5e-3, 6.0e-3);
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
                (0.5 * width, width)
            }
        };
        Self { bristles, shape, width, angle: 0.0, len, segs, ferrule_r, radius, stiffness: 0.35, friction: 0.7, capacity: 6.0e-9, release: 1.5, placed: false, handle: DVec3::ZERO, touching: 0 }
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
        let l = self.len / segs as f64;
        let radius = self.radius;
        let stiffness = self.stiffness;
        let friction = self.friction;
        let vel = if self.placed { (handle - self.handle) / dt.max(1e-6) } else { DVec3::ZERO };
        if self.shape == BrushShape::Flat && vel.truncate().length() > 0.03 {
            let want = vel.y.atan2(vel.x) + std::f64::consts::FRAC_PI_2;
            let mut d = want - self.angle;
            while d > std::f64::consts::FRAC_PI_2 {
                d -= std::f64::consts::PI;
            }
            while d < -std::f64::consts::FRAC_PI_2 {
                d += std::f64::consts::PI;
            }
            self.angle += d * (6.0 * dt).min(1.0);
        }
        let roots: Vec<DVec3> = self.bristles.iter().map(|b| self.rooted(b.root)).collect();
        if !self.placed || (handle - self.handle).length() > 0.1 {
            for (b, root) in self.bristles.iter_mut().zip(&roots) {
                for i in 0..=segs {
                    b.pts[i] = handle + *root + b.rest * (l * i as f64);
                    b.prev[i] = b.pts[i];
                }
            }
            self.placed = true;
        }
        self.handle = handle;
        self.bristles.par_iter_mut().zip(roots.par_iter()).for_each(|(b, root)| {
            b.pts[0] = handle + *root;
            for _ in 0..8 {
                for i in 1..=segs {
                    let dir_prev = if i == 1 { b.rest } else { (b.pts[i - 1] - b.pts[i - 2]).normalize_or_zero() };
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
                    let seg = b.pts[i] - b.pts[i - 1];
                    let sl = seg.length();
                    b.pts[i] = b.pts[i - 1] + if sl > 1e-9 { seg / sl * l } else { dir_prev * l };
                }
            }
            for i in 0..=segs {
                let zs = floor(b.pts[i].x, b.pts[i].y) + radius;
                b.last[i] = b.prev[i];
                b.was[i] = b.touch[i];
                b.touch[i] = i > 0 && b.pts[i].z <= zs + 0.5 * radius;
                b.prev[i] = b.pts[i];
            }
        });
        let mut contacts = Vec::new();
        let mut releases = Vec::new();
        let vxy = DVec3::new(vel.x, vel.y, 0.0);
        let sp = vxy.length();
        let dir = if sp > 1e-6 { vxy / sp } else { DVec3::ZERO };
        let mut touching = 0;
        for (bi, b) in self.bristles.iter_mut().enumerate() {
            let segs_touching: Vec<usize> = (1..=segs).filter(|&i| b.touch[i]).collect();
            if segs_touching.is_empty() {
                continue;
            }
            touching += 1;
            let rc = (radius * 2.0 * b.thick).max(0.6 * dx) as f32;
            let shift = dir * (2.0 * rc as f64 + sp * dt);
            let nseg = segs_touching.len() as f64;
            let over = (b.load - self.capacity).max(0.0) / nseg;
            let rel = b.load.min(self.capacity) * b.thick * (self.release * dt).min(1.0) / nseg;
            for &i in &segs_touching {
                let p0 = [b.pts[i - 1].x as f32, b.pts[i - 1].y as f32];
                let p1 = [b.pts[i].x as f32, b.pts[i].y as f32];
                let zb = (b.pts[i].z.min(b.pts[i - 1].z) - radius) as f32;
                contacts.push(Contact { p0, p1, r: rc, zb, bristle: bi as u32, pad: 0 });
                if rel > 1e-15 {
                    let tip = if b.pts[i].z <= b.pts[i - 1].z { i } else { i - 1 };
                    let from = b.last[tip];
                    let swept = (b.pts[tip] - from).truncate().length();
                    if b.was[i] && swept > rc as f64 {
                        let q0 = [from.x as f32, from.y as f32];
                        let q1 = [b.pts[tip].x as f32, b.pts[tip].y as f32];
                        releases.push(Release { p0, p1, r: rc, vol: 0.5 * rel, comp: b.comp });
                        releases.push(Release { p0: q0, p1: q1, r: rc, vol: 0.5 * rel, comp: b.comp });
                    } else {
                        releases.push(Release { p0, p1, r: rc, vol: rel, comp: b.comp });
                    }
                }
                if over > 1e-15 {
                    let q0 = [(b.pts[i - 1].x + shift.x) as f32, (b.pts[i - 1].y + shift.y) as f32];
                    let q1 = [(b.pts[i].x + shift.x) as f32, (b.pts[i].y + shift.y) as f32];
                    releases.push(Release { p0: q0, p1: q1, r: 1.5 * rc, vol: over, comp: b.comp });
                }
            }
            b.load = (b.load - (rel + over) * nseg).max(0.0);
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
