use crate::film::{Contact, Deposit, Method};
use crate::tools::TubeVertex;
use glam::{DVec2, DVec3};
use std::f64::consts::TAU;

const CELLS: usize = 16;
const FINE: usize = 64;
const PORE_JITTER: f64 = 0.45;
const CAPACITY_PER_AREA: f64 = 1.3e-3;
const TRANSFER: f64 = 1.0;
const RELEASE_RATE: f64 = 2.0;
const SQUASH_TIME: f64 = 0.15;
const CONTACT_MIN: f64 = 0.55;
const ANGLE_DRIFT: f64 = 0.6;
const SATURATED_GAP: f64 = 2.0e-3;
const STICK_LEN: f64 = 0.004;
const STICK_DUTY: f64 = 0.15;
const STICK_PULSE: f64 = 0.025;
const SLIDE_RELEASE: f64 = 0.35;
const SLIDE_MIN: f64 = 2.0e-4;
const BLOT_GAP: f32 = 1.2e-5;
const CONTACT_SPONGE: u32 = 2;
const PORE_SCALE: f64 = 7.0;
const EDGE_WOBBLE: f64 = 0.32;
const DOME_RINGS: usize = 10;
const DOME_SEGS: usize = 28;
const SQUASH_DEPTH: f64 = 0.4;

#[derive(Clone, Copy)]
struct Cell {
    vol: f64,
    comp: [f32; 16],
}

pub struct Sponge {
    pub size: f64,
    pub density: f64,
    pub pressure: f64,
    pub transfer: f64,
    cells: Vec<Cell>,
    seed: u64,
    angle: f64,
    last: Option<DVec2>,
    pub pressing: bool,
    dabs: u64,
    squash: f64,
    travel: f64,
}

fn hash(a: i64, b: i64, seed: u64) -> f64 {
    let mut h = (a as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (b as u64).wrapping_mul(0xD1B54A32D192ED03) ^ seed.wrapping_mul(0x94D049BB133111EB);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 32;
    (h & 0xffff) as f64 / 65535.0
}

fn noise(x: f64, y: f64, seed: u64) -> f64 {
    let (xi, yi) = (x.floor(), y.floor());
    let (fx, fy) = (x - xi, y - yi);
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(fx), s(fy));
    let (xi, yi) = (xi as i64, yi as i64);
    let a = hash(xi, yi, seed) * (1.0 - sx) + hash(xi + 1, yi, seed) * sx;
    let b = hash(xi, yi + 1, seed) * (1.0 - sx) + hash(xi + 1, yi + 1, seed) * sx;
    a * (1.0 - sy) + b * sy
}

fn blend(a: &mut Cell, vol: f64, comp: &[f32; 16]) {
    let total = a.vol + vol;
    if total <= 0.0 {
        return;
    }
    for i in 0..16 {
        a.comp[i] = ((a.comp[i] as f64 * a.vol + comp[i] as f64 * vol) / total) as f32;
    }
    a.vol = total;
}

impl Sponge {
    pub fn new() -> Self {
        Self { size: 0.06, density: 0.55, pressure: 0.5, transfer: TRANSFER, cells: vec![Cell { vol: 0.0, comp: [0.0; 16] }; CELLS * CELLS], seed: 7, angle: 0.0, last: None, pressing: false, dabs: 0, squash: 0.0, travel: 0.0 }
    }

    pub fn capacity(&self) -> f64 {
        CAPACITY_PER_AREA * self.size * self.size
    }

    fn cell_capacity(&self) -> f64 {
        self.capacity() / (CELLS * CELLS) as f64
    }

    pub fn load(&self) -> f64 {
        self.cells.iter().map(|c| c.vol).sum()
    }

    pub fn mix(&self) -> [f32; 16] {
        let mut m = Cell { vol: 0.0, comp: [0.0; 16] };
        for c in &self.cells {
            blend(&mut m, c.vol, &c.comp);
        }
        m.comp
    }

    pub fn dip(&mut self, frac: &[f32; 16]) {
        let cap = self.cell_capacity();
        for c in &mut self.cells {
            c.vol = cap;
            c.comp = *frac;
        }
    }

    pub fn dab_colour(&mut self, frac: &[f32; 16]) {
        let cap = self.cell_capacity();
        self.seed = self.seed.wrapping_add(1);
        let seed = self.seed;
        for (k, c) in self.cells.iter_mut().enumerate() {
            let (i, j) = ((k % CELLS) as f64, (k / CELLS) as f64);
            if noise(i * 0.35, j * 0.35, seed ^ 0xC0105) > 0.5 {
                let add = (cap - c.vol).max(cap * 0.6);
                blend(c, add, frac);
                c.vol = c.vol.min(cap * 1.4);
            }
        }
    }

    pub fn clean(&mut self) {
        for c in &mut self.cells {
            c.vol = 0.0;
            c.comp = [0.0; 16];
        }
    }

    pub fn absorb(&mut self, res: &[[u32; 17]]) -> f64 {
        let mut total = 0.0;
        for (k, r) in res.iter().enumerate().take(CELLS * CELLS) {
            let vol = r[0] as f64 * 1e-15;
            if vol <= 0.0 {
                continue;
            }
            let mut comp = [0.0f32; 16];
            for i in 0..16 {
                let sc = if i >= 13 { 100.0 } else { 1.0 };
                comp[i] = (r[1 + i] as f64 * 1e-15 * sc / vol) as f32;
            }
            blend(&mut self.cells[k], vol, &comp);
            total += vol;
        }
        total
    }

    fn edge(&self, th: f64) -> f64 {
        let (c, s) = (th.cos(), th.sin());
        let coarse = noise(c * 1.6 + 5.0, s * 1.6 + 5.0, self.seed);
        let fine = noise(c * 4.5 + 17.0, s * 4.5 + 9.0, self.seed ^ 0x3C);
        1.0 - EDGE_WOBBLE * (0.65 * coarse + 0.35 * fine)
    }

    fn solid(&self, u: f64, v: f64, pressure: f64) -> bool {
        let r = (u * u + v * v).sqrt();
        let th = v.atan2(u);
        let edge = self.edge(th);
        if r > edge {
            return false;
        }
        let n = 0.65 * noise(u * PORE_SCALE + 11.0, v * PORE_SCALE + 3.0, self.seed) + 0.35 * noise(u * PORE_SCALE * 2.7, v * PORE_SCALE * 2.7, self.seed ^ 0x55);
        let rim = (r / edge).powi(3) * 0.25;
        n > 1.0 - self.density * (0.6 + 0.8 * pressure) + rim
    }

    pub fn lift(&mut self) {
        self.last = None;
        self.pressing = false;
        self.squash = 0.0;
    }

    pub fn update(&mut self, p: DVec2, pressing: bool, pressure: f64, dt: f64, dx: f32) -> (Vec<Contact>, Vec<Deposit>) {
        if !pressing {
            self.lift();
            return (Vec::new(), Vec::new());
        }
        let prev = match self.last {
            Some(q) => q,
            None => {
                self.dabs += 1;
                self.angle = hash(self.dabs as i64, 3, self.seed) * TAU;
                p
            }
        };
        self.pressing = true;
        self.squash = (self.squash + dt / SQUASH_TIME).min(1.0);
        let moved = (p - prev).length();
        let prev_angle = self.angle;
        self.angle += ANGLE_DRIFT * (moved / self.size) * (noise(self.dabs as f64 * 1.7, self.angle * 3.0, self.seed) - 0.5);
        self.last = Some(p);
        self.travel += moved;
        let sliding = moved > SLIDE_MIN;
        let pulse = noise(self.travel / STICK_PULSE, self.dabs as f64 * 0.31, self.seed ^ 0x99);
        let duty = ((STICK_DUTY + 0.35 * pressure) * (0.35 + 1.1 * pulse)).clamp(0.05, 0.95);
        let reach = (CONTACT_MIN + (1.0 - CONTACT_MIN) * self.squash) * (0.75 + 0.25 * pressure);
        let half = 0.5 * self.size;
        let fine_w = self.size / FINE as f64;
        let (s1, c1) = self.angle.sin_cos();
        let (s0, c0) = prev_angle.sin_cos();
        let mut points: Vec<(DVec2, DVec2, usize, f64)> = Vec::new();
        let mut per_cell = vec![0usize; CELLS * CELLS];
        for j in 0..FINE {
            for i in 0..FINE {
                let ju = PORE_JITTER * (hash(i as i64, j as i64, self.seed ^ 0xA1) - 0.5);
                let jv = PORE_JITTER * (hash(i as i64, j as i64, self.seed ^ 0xB2) - 0.5);
                let u = (i as f64 + 0.5 + ju) / FINE as f64 * 2.0 - 1.0;
                let v = (j as f64 + 0.5 + jv) / FINE as f64 * 2.0 - 1.0;
                let r = (u * u + v * v).sqrt();
                if r > reach * self.edge(v.atan2(u)) || !self.solid(u, v, pressure) {
                    continue;
                }
                if sliding {
                    let pore = (j * FINE + i) as f64;
                    let phase = self.travel / STICK_LEN + 9.0 * hash(i as i64, j as i64, self.seed ^ 0x51);
                    if noise(phase, pore * 0.37, self.seed ^ 0x77) < 1.0 - duty {
                        continue;
                    }
                }
                let now = p + DVec2::new(u * c1 - v * s1, u * s1 + v * c1) * half;
                let was = prev + DVec2::new(u * c0 - v * s0, u * s0 + v * c0) * half;
                let cell = (j * CELLS / FINE) * CELLS + i * CELLS / FINE;
                per_cell[cell] += 1;
                points.push((was, now, cell, 0.55 + 0.5 * hash(j as i64, i as i64, self.seed ^ 0xC3)));
            }
        }
        let cap = self.cell_capacity();
        let rate = self.transfer * RELEASE_RATE * dt * (0.3 + 1.4 * pressure) * if sliding { SLIDE_RELEASE } else { 1.0 };
        let mut out = vec![0.0f64; CELLS * CELLS];
        let mut gap = vec![BLOT_GAP; CELLS * CELLS];
        for (k, cell) in self.cells.iter_mut().enumerate() {
            let sat = (cell.vol / cap).clamp(0.0, 1.5);
            gap[k] = (BLOT_GAP as f64 + sat * SATURATED_GAP) as f32;
            if per_cell[k] == 0 || cell.vol <= 0.0 {
                continue;
            }
            let v = cell.vol * (rate * sat.powf(1.5)).min(0.5);
            cell.vol -= v;
            out[k] = v / per_cell[k] as f64;
        }
        let r = (fine_w * 0.85) as f32;
        let mut contacts = Vec::with_capacity(points.len());
        let mut deposits = Vec::new();
        for (was, now, cell, size) in &points {
            let r = r * *size as f32;
            contacts.push(Contact { p0: [was.x as f32, was.y as f32], p1: [now.x as f32, now.y as f32], r, zb: gap[*cell], bristle: *cell as u32, pad: CONTACT_SPONGE });
            let v = out[*cell];
            if v <= 0.0 {
                continue;
            }
            let pig = crate::color::PaintMix { frac: self.cells[*cell].comp }.normalized().packed();
            deposits.push(Deposit::stroke([was.x as f32, was.y as f32], [now.x as f32, now.y as f32], r, v as f32, pig, dx).laid_by(Method::Brush));
        }
        (contacts, deposits)
    }

    pub fn mesh(&self, verts: &mut Vec<TubeVertex>, idx: &mut Vec<u32>, at: DVec3, paint_col: &dyn Fn(&[f32; 16]) -> [f32; 3]) {
        let half = 0.5 * self.size;
        let height = self.size * 0.45 * (1.0 - SQUASH_DEPTH * self.squash);
        let lift = if self.pressing { 0.0 } else { 0.012 };
        let base = [0.86f32, 0.72, 0.42];
        let fill = (self.load() / self.capacity()).clamp(0.0, 1.0) as f32;
        let pc = if fill > 0.0 { paint_col(&self.mix()) } else { base };
        let k = if fill > 0.0 { 0.35 + 0.65 * fill } else { 0.0 };
        let col = [base[0] + (pc[0] - base[0]) * k, base[1] + (pc[1] - base[1]) * k, base[2] + (pc[2] - base[2]) * k];
        let start = verts.len() as u32;
        let (s, c) = self.angle.sin_cos();
        for ri in 0..=DOME_RINGS {
            let phi = ri as f64 / DOME_RINGS as f64 * std::f64::consts::FRAC_PI_2;
            for si in 0..DOME_SEGS {
                let th = si as f64 / DOME_SEGS as f64 * TAU;
                let wob = self.edge(th) + 0.06 * (noise(th * 3.0, phi * 4.0, self.seed ^ 9) - 0.5);
                let rr = half * wob * phi.cos();
                let z = height * phi.sin() * (0.9 + 0.2 * noise(th * 2.0, phi * 3.0, self.seed ^ 3));
                let lx = rr * th.cos();
                let ly = rr * th.sin();
                let x = at.x + lx * c - ly * s;
                let y = at.y + lx * s + ly * c;
                let n = DVec3::new(th.cos() * phi.cos(), th.sin() * phi.cos(), phi.sin()).normalize();
                let shade = 0.82 + 0.18 * noise(th * 9.0, phi * 9.0, self.seed ^ 77) as f32;
                verts.push(TubeVertex { pos: [x as f32, y as f32, (z + lift) as f32], nrm: [n.x as f32, n.y as f32, n.z as f32], col: [col[0] * shade, col[1] * shade, col[2] * shade] });
            }
        }
        let segs = DOME_SEGS as u32;
        for ri in 0..DOME_RINGS as u32 {
            for si in 0..segs {
                let s1 = (si + 1) % segs;
                let a = start + ri * segs + si;
                let b = start + ri * segs + s1;
                let cc = start + (ri + 1) * segs + si;
                let d = start + (ri + 1) * segs + s1;
                idx.extend_from_slice(&[a, b, cc, b, d, cc]);
            }
        }
    }
}
