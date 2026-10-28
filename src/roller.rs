use crate::film::{Contact, Deposit, Method};
use crate::tools::{push_tube, TubeVertex};
use glam::{DVec2, DVec3};

const CAPACITY_PER_WIDTH: f64 = 3.0e-4;
const FILM_MAX: f64 = 1.6e-4;
const LOAD_EXPONENT: f64 = 0.7;
const NIP_HALF: f32 = 0.004;
const MIN_GAP: f64 = 4.0e-5;
const LANE_WIDTH: f64 = 0.0025;
const LANE_OVERLAP: f64 = 3.2;
const EDGE_RIDGE: f64 = 0.6;
const STIPPLE_CELLS: f64 = 9.0;
const CYL_SEGS: u32 = 28;
pub const CONTACT_ROLLER: u32 = 1;

pub struct Roller {
    pub width: f64,
    pub radius: f64,
    pub pressure: f64,
    pub nap: f64,
    pub load: f64,
    pub comp: [f32; 16],
    axis: DVec2,
    pos: Option<DVec2>,
    phase: f64,
    pub pressing: bool,
}

fn hash(a: u64, b: u64) -> f64 {
    let mut h = a.wrapping_mul(0x9E3779B97F4A7C15) ^ b.wrapping_mul(0xD1B54A32D192ED03);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 32;
    (h & 0xffff) as f64 / 65535.0
}

impl Roller {
    pub fn new() -> Self {
        Self { width: 0.1, radius: 0.02, pressure: 0.5, nap: 0.5, load: 0.0, comp: [0.0; 16], axis: DVec2::new(0.0, 1.0), pos: None, phase: 0.0, pressing: false }
    }

    pub fn capacity(&self) -> f64 {
        self.width * CAPACITY_PER_WIDTH
    }

    pub fn dip(&mut self, frac: &[f32; 16]) {
        self.load = self.capacity();
        self.comp = *frac;
    }

    pub fn clean(&mut self) {
        self.load = 0.0;
        self.comp = [0.0; 16];
    }

    pub fn lift(&mut self) {
        self.pos = None;
    }

    pub fn film(&self, pressure: f64) -> f64 {
        let f = (self.load / self.capacity()).clamp(0.0, 1.0);
        FILM_MAX * f.powf(LOAD_EXPONENT) * (0.5 + pressure)
    }

    pub fn absorb(&mut self, res: &[[u32; 17]]) -> f64 {
        let Some(r) = res.first() else { return 0.0 };
        let vol = r[0] as f64 * 1e-15;
        if vol <= 0.0 {
            return 0.0;
        }
        let newload = self.load + vol;
        for i in 0..16 {
            let sc = if i >= 13 { 100.0 } else { 1.0 };
            let add = r[1 + i] as f64 * 1e-15 * sc;
            self.comp[i] = ((self.comp[i] as f64 * self.load + add) / newload) as f32;
        }
        self.load = newload;
        vol
    }

    pub fn update(&mut self, p: DVec2, pressing: bool, pressure: f64, dx: f32) -> (Vec<Contact>, Vec<Deposit>) {
        self.pressing = pressing;
        let Some(prev) = self.pos else {
            self.pos = Some(p);
            return (Vec::new(), Vec::new());
        };
        let mv = p - prev;
        let dist = mv.length();
        let prev_axis = self.axis;
        if dist > 1e-9 {
            let dir = mv / dist;
            let mut target = DVec2::new(-dir.y, dir.x);
            if target.dot(self.axis) < 0.0 {
                target = -target;
            }
            let k = (dist / (0.5 * self.width)).min(1.0);
            self.axis = (self.axis + (target - self.axis) * k).normalize_or(self.axis);
        }
        let normal = DVec2::new(self.axis.y, -self.axis.x);
        let roll = mv.dot(normal).abs();
        self.phase += roll / self.radius;
        self.pos = Some(p);
        if !pressing {
            return (Vec::new(), Vec::new());
        }
        let t = self.film(pressure);
        let gap = (t * (1.6 - pressure)).max(MIN_GAP);
        let half = self.axis * (0.5 * self.width);
        let steps = ((dist / NIP_HALF as f64).ceil() as usize).clamp(1, 64);
        let mut contacts = Vec::with_capacity(steps);
        for s in 1..=steps {
            let c = prev + mv * (s as f64 / steps as f64);
            let a = c - half;
            let b = c + half;
            contacts.push(Contact { p0: [a.x as f32, a.y as f32], p1: [b.x as f32, b.y as f32], r: NIP_HALF, zb: gap as f32, bristle: 0, pad: CONTACT_ROLLER });
        }
        let mut deposits = Vec::new();
        let vol = (t * self.width * roll).min(self.load);
        if vol > 0.0 && roll > 1e-6 {
            self.load -= vol;
            let lanes = ((self.width / LANE_WIDTH).round() as usize).clamp(8, 100);
            let lane_w = self.width / lanes as f64;
            let fill = (self.load / self.capacity()).clamp(0.0, 1.0);
            let u = self.phase * STIPPLE_CELLS / std::f64::consts::TAU;
            let cell = u.floor() as i64;
            let f = u - u.floor();
            let f = f * f * (3.0 - 2.0 * f);
            let weights: Vec<f64> = (0..lanes)
                .map(|i| {
                    let g = i as f64 / 3.0;
                    let gi = g.floor() as u64;
                    let gf = g - g.floor();
                    let gf = gf * gf * (3.0 - 2.0 * gf);
                    let at = |a: u64, c: i64| hash(a, c as u64) * (1.0 - f) + hash(a, (c + 1) as u64) * f;
                    let n = at(gi, cell) * (1.0 - gf) + at(gi + 1, cell) * gf - 0.5;
                    let edge = if i < 2 || i + 2 >= lanes { 1.0 + EDGE_RIDGE * fill } else { 1.0 };
                    (1.0 + 1.2 * self.nap * n).max(0.1) * edge
                })
                .collect();
            let total: f64 = weights.iter().sum();
            let pig = crate::color::PaintMix { frac: self.comp }.normalized().packed();
            let along = |ax: DVec2| {
                let n = DVec2::new(ax.y, -ax.x);
                let n = if n.dot(mv) < 0.0 { -n } else { n };
                [n.x as f32, n.y as f32]
            };
            let (n0, n1) = (along(prev_axis), along(self.axis));
            for (i, w) in weights.iter().enumerate() {
                let r_lane = 0.5 * lane_w * LANE_OVERLAP;
                let span = self.width - 2.0 * r_lane * 0.6;
                let off = (i as f64 + 0.5) / lanes as f64 * span - 0.5 * span;
                let a = prev + prev_axis * off;
                let b = p + self.axis * off;
                let r = r_lane as f32;
                deposits.push(Deposit::ribbon([a.x as f32, a.y as f32], [b.x as f32, b.y as f32], n0, n1, r, r, 0, (vol * w / total) as f32, pig, dx).laid_by(Method::Roller));
            }
        }
        (contacts, deposits)
    }

    pub fn mesh(&self, verts: &mut Vec<TubeVertex>, idx: &mut Vec<u32>, at: DVec3, paint_col: &dyn Fn(&[f32; 16]) -> [f32; 3]) {
        let lift = if self.pressing { 0.0 } else { 0.015 };
        let c = DVec3::new(at.x, at.y, self.radius + lift);
        let ax = DVec3::new(self.axis.x, self.axis.y, 0.0);
        let side = DVec3::new(self.axis.y, -self.axis.x, 0.0);
        let f = (self.load / self.capacity()).clamp(0.0, 1.0) as f32;
        let nap = [0.86f32, 0.82, 0.74];
        let pc = if f > 0.0 { paint_col(&self.comp) } else { nap };
        let k = (0.25 + 0.75 * f).min(1.0) * if f > 0.0 { 1.0 } else { 0.0 };
        let col = [nap[0] + (pc[0] - nap[0]) * k, nap[1] + (pc[1] - nap[1]) * k, nap[2] + (pc[2] - nap[2]) * k];
        let base = verts.len() as u32;
        let hw = 0.5 * self.width;
        for end in 0..2 {
            let x = if end == 0 { -hw } else { hw };
            for s in 0..CYL_SEGS {
                let a = s as f64 / CYL_SEGS as f64 * std::f64::consts::TAU + self.phase;
                let n = side * a.cos() + DVec3::Z * a.sin();
                let p = c + ax * x + n * self.radius;
                let shade = 0.92 + 0.08 * ((s as f64 * 2.3).sin() as f32);
                verts.push(TubeVertex { pos: [p.x as f32, p.y as f32, p.z as f32], nrm: [n.x as f32, n.y as f32, n.z as f32], col: [col[0] * shade, col[1] * shade, col[2] * shade] });
            }
        }
        for s in 0..CYL_SEGS {
            let s1 = (s + 1) % CYL_SEGS;
            let (a0, a1, b0, b1) = (base + s, base + s1, base + CYL_SEGS + s, base + CYL_SEGS + s1);
            idx.extend_from_slice(&[a0, b0, a1, a1, b0, b1]);
        }
        for end in 0..2 {
            let x = if end == 0 { -hw } else { hw };
            let n = if end == 0 { -ax } else { ax };
            let centre = verts.len() as u32;
            let pc = c + ax * x;
            let cap = [0.35f32, 0.35, 0.37];
            verts.push(TubeVertex { pos: [pc.x as f32, pc.y as f32, pc.z as f32], nrm: [n.x as f32, n.y as f32, n.z as f32], col: cap });
            for s in 0..CYL_SEGS {
                let a = s as f64 / CYL_SEGS as f64 * std::f64::consts::TAU;
                let p = pc + (side * a.cos() + DVec3::Z * a.sin()) * self.radius * 0.97;
                verts.push(TubeVertex { pos: [p.x as f32, p.y as f32, p.z as f32], nrm: [n.x as f32, n.y as f32, n.z as f32], col: cap });
            }
            for s in 0..CYL_SEGS {
                let s1 = (s + 1) % CYL_SEGS;
                if end == 0 {
                    idx.extend_from_slice(&[centre, centre + 1 + s1, centre + 1 + s]);
                } else {
                    idx.extend_from_slice(&[centre, centre + 1 + s, centre + 1 + s1]);
                }
            }
        }
        let axle_end = c + ax * (hw + 0.006);
        let frame = [
            c - ax * (hw * 0.6),
            axle_end,
            axle_end + DVec3::Z * (self.radius + 0.025),
            c + DVec3::Z * (self.radius + 0.05) - side * 0.04,
            c + DVec3::Z * (self.radius + 0.17) - side * 0.16,
        ];
        push_tube(verts, idx, &frame, 0.0025, [0.62, 0.63, 0.66]);
        let grip = [c + DVec3::Z * (self.radius + 0.17) - side * 0.16, c + DVec3::Z * (self.radius + 0.27) - side * 0.26];
        push_tube(verts, idx, &grip, 0.011, [0.12, 0.12, 0.13]);
    }
}
