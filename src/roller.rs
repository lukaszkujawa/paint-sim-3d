use crate::film::{Contact, Deposit, Method};
use crate::tools::{push_tube, TubeVertex};
use glam::{DVec2, DVec3};
use std::f64::consts::{FRAC_PI_2, TAU};

const CAPACITY_PER_WIDTH: f64 = 3.0e-4;
const FILM_MAX: f64 = 1.6e-4;
const LOAD_EXPONENT: f64 = 0.7;
const NIP_HALF: f32 = 0.004;
const MIN_GAP: f64 = 4.0e-5;
const NAP_LANES: usize = 24;
const NAP_SECTORS: usize = 24;
const SUB_LANES: usize = 2;
const LANE_OVERLAP: f64 = 3.2;
const EDGE_RIDGE: f64 = 0.6;
const STIPPLE_CELLS: f64 = 9.0;
const MIX_PER_REV: f64 = 0.14;
const EDGE_BLEND: f64 = 0.3;
const ROLL_SMOOTH: f64 = 0.65;
const PICKUP_KERNEL: [(i64, i64, f64); 5] = [(0, 0, 0.4), (0, -1, 0.2), (0, 1, 0.2), (-1, 0, 0.1), (1, 0, 0.1)];
const MIX_MAX: f64 = 0.4;
const MAX_SECTORS_PER_FRAME: i64 = 48;
pub const CONTACT_ROLLER: u32 = 1;

#[derive(Clone, Copy)]
struct Cell {
    vol: f64,
    comp: [f32; 16],
}

pub struct Roller {
    pub width: f64,
    pub radius: f64,
    pub pressure: f64,
    pub nap: f64,
    cells: Vec<Cell>,
    axis: DVec2,
    pos: Option<DVec2>,
    phase: f64,
    pub pressing: bool,
    trail: Vec<[f32; 16]>,
}

fn hash(a: u64, b: u64) -> f64 {
    let mut h = a.wrapping_mul(0x9E3779B97F4A7C15) ^ b.wrapping_mul(0xD1B54A32D192ED03);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 32;
    (h & 0xffff) as f64 / 65535.0
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

impl Roller {
    pub fn new() -> Self {
        Self {
            width: 0.1,
            radius: 0.02,
            pressure: 0.5,
            nap: 0.5,
            cells: vec![Cell { vol: 0.0, comp: [0.0; 16] }; NAP_LANES * NAP_SECTORS],
            axis: DVec2::new(0.0, 1.0),
            pos: None,
            phase: 0.0,
            pressing: false,
            trail: vec![[0.0; 16]; NAP_LANES],
        }
    }

    pub fn capacity(&self) -> f64 {
        self.width * CAPACITY_PER_WIDTH
    }

    fn cell_capacity(&self) -> f64 {
        self.capacity() / (NAP_LANES * NAP_SECTORS) as f64
    }

    pub fn load(&self) -> f64 {
        self.cells.iter().map(|c| c.vol).sum()
    }

    pub fn dip(&mut self, frac: &[f32; 16]) {
        let cap = self.cell_capacity();
        for c in &mut self.cells {
            c.vol = cap;
            c.comp = *frac;
        }
    }

    pub fn clean(&mut self) {
        for c in &mut self.cells {
            c.vol = 0.0;
            c.comp = [0.0; 16];
        }
    }

    pub fn lift(&mut self) {
        self.pos = None;
        for t in &mut self.trail {
            *t = [0.0; 16];
        }
    }

    fn thickness(&self, vol: f64, pressure: f64) -> f64 {
        let f = (vol / self.cell_capacity()).clamp(0.0, 1.0);
        FILM_MAX * f.powf(LOAD_EXPONENT) * (0.5 + pressure)
    }

    pub fn film(&self, pressure: f64) -> f64 {
        let n = (NAP_LANES * NAP_SECTORS) as f64;
        self.thickness(self.load() / n, pressure)
    }

    fn sector_at(&self, phase: f64) -> usize {
        ((phase / TAU * NAP_SECTORS as f64).floor() as i64).rem_euclid(NAP_SECTORS as i64) as usize
    }

    pub fn absorb(&mut self, res: &[[u32; 17]]) -> f64 {
        let k = self.sector_at(self.phase);
        let mut total = 0.0;
        for (lane, r) in res.iter().enumerate().take(NAP_LANES) {
            let vol = r[0] as f64 * 1e-15;
            if vol <= 0.0 {
                continue;
            }
            let mut comp = [0.0f32; 16];
            for i in 0..16 {
                let sc = if i >= 13 { 100.0 } else { 1.0 };
                comp[i] = (r[1 + i] as f64 * 1e-15 * sc / vol) as f32;
            }
            for (dl, ds, w) in PICKUP_KERNEL {
                let l = lane as i64 + dl;
                if l < 0 || l >= NAP_LANES as i64 {
                    continue;
                }
                let sec = (k as i64 + ds).rem_euclid(NAP_SECTORS as i64) as usize;
                blend(&mut self.cells[l as usize * NAP_SECTORS + sec], vol * w, &comp);
            }
            total += vol;
        }
        total
    }

    fn mix_nap(&mut self, revolutions: f64) {
        let m = (MIX_PER_REV * revolutions).min(MIX_MAX);
        if m <= 0.0 {
            return;
        }
        let old = self.cells.clone();
        for l in 0..NAP_LANES {
            for s in 0..NAP_SECTORS {
                let mut nb: Vec<usize> = vec![l * NAP_SECTORS + (s + 1) % NAP_SECTORS, l * NAP_SECTORS + (s + NAP_SECTORS - 1) % NAP_SECTORS];
                if l > 0 {
                    nb.push((l - 1) * NAP_SECTORS + s);
                }
                if l + 1 < NAP_LANES {
                    nb.push((l + 1) * NAP_SECTORS + s);
                }
                let me = &old[l * NAP_SECTORS + s];
                let avg_vol = nb.iter().map(|&j| old[j].vol).sum::<f64>() / nb.len() as f64;
                let mut c = Cell { vol: me.vol * (1.0 - m), comp: me.comp };
                for &j in &nb {
                    blend(&mut c, old[j].vol * m / nb.len() as f64, &old[j].comp);
                }
                c.vol = me.vol * (1.0 - m) + avg_vol * m;
                self.cells[l * NAP_SECTORS + s] = c;
            }
        }
    }

    fn weight(&self, lane: usize, lanes: usize, phase: f64, fill: f64) -> f64 {
        let u = phase * STIPPLE_CELLS / TAU;
        let cell = u.floor() as i64;
        let f = u - u.floor();
        let f = f * f * (3.0 - 2.0 * f);
        let g = lane as f64 / 3.0;
        let gi = g.floor() as u64;
        let gf = g - g.floor();
        let gf = gf * gf * (3.0 - 2.0 * gf);
        let at = |a: u64, c: i64| hash(a, c as u64) * (1.0 - f) + hash(a, (c + 1) as u64) * f;
        let n = at(gi, cell) * (1.0 - gf) + at(gi + 1, cell) * gf - 0.5;
        let edge = if lane < 2 || lane + 2 >= lanes { 1.0 + EDGE_RIDGE * fill } else { 1.0 };
        (1.0 + 1.2 * self.nap * n).max(0.1) * edge
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
        let phase0 = self.phase;
        self.phase += roll / self.radius;
        self.pos = Some(p);
        if !pressing {
            for t in &mut self.trail {
                *t = [0.0; 16];
            }
            return (Vec::new(), Vec::new());
        }
        let lane_w = self.width / NAP_LANES as f64;
        let bottom = self.sector_at(phase0);
        let axis_at = |f: f64| (prev_axis * (1.0 - f) + self.axis * f).normalize_or(self.axis);
        let steps = ((dist / NIP_HALF as f64).ceil() as usize).clamp(1, 32);
        let mut contacts = Vec::with_capacity(steps * NAP_LANES);
        for s in 1..=steps {
            let f = s as f64 / steps as f64;
            let c = prev + mv * f;
            let ax = axis_at(f);
            for lane in 0..NAP_LANES {
                let t = self.thickness(self.cells[lane * NAP_SECTORS + bottom].vol, pressure);
                let gap = (t * (1.6 - pressure)).max(MIN_GAP);
                let a = c + ax * (lane as f64 * lane_w - 0.5 * self.width);
                let b = a + ax * lane_w;
                contacts.push(Contact { p0: [a.x as f32, a.y as f32], p1: [b.x as f32, b.y as f32], r: NIP_HALF, zb: gap as f32, bristle: lane as u32, pad: CONTACT_ROLLER });
            }
        }
        let mut deposits = Vec::new();
        if roll > 1e-6 {
            let fine = NAP_LANES * SUB_LANES;
            let r_lane = 0.5 * (self.width / fine as f64) * LANE_OVERLAP;
            let span = self.width - 2.0 * r_lane * 0.6;
            let fill = (self.load() / self.capacity()).clamp(0.0, 1.0);
            let sector_angle = TAU / NAP_SECTORS as f64;
            let k0 = (phase0 / sector_angle).floor() as i64;
            let k1 = ((self.phase / sector_angle).floor() as i64).min(k0 + MAX_SECTORS_PER_FRAME);
            let mut laid = vec![Cell { vol: 0.0, comp: [0.0; 16] }; NAP_LANES];
            for ks in k0..=k1 {
                let a_phase = (ks as f64 * sector_angle).max(phase0);
                let b_phase = ((ks + 1) as f64 * sector_angle).min(self.phase);
                if b_phase <= a_phase {
                    continue;
                }
                let k = ks.rem_euclid(NAP_SECTORS as i64) as usize;
                let seg = (b_phase - a_phase) * self.radius;
                for (lane, out) in laid.iter_mut().enumerate() {
                    let idx = lane * NAP_SECTORS + k;
                    let cell = self.cells[idx];
                    let vol = (self.thickness(cell.vol, pressure) * lane_w * seg).min(cell.vol);
                    if vol > 0.0 {
                        self.cells[idx].vol -= vol;
                        blend(out, vol, &cell.comp);
                    }
                }
            }
            let along = |ax: DVec2| {
                let n = DVec2::new(ax.y, -ax.x);
                let n = if n.dot(mv) < 0.0 { -n } else { n };
                [n.x as f32, n.y as f32]
            };
            let (n0, n1) = (along(prev_axis), along(self.axis));
            for (lane, cell) in laid.iter_mut().enumerate() {
                if cell.vol <= 0.0 {
                    continue;
                }
                let prior = self.trail[lane];
                if prior.iter().any(|v| *v > 0.0) {
                    let mut t = Cell { vol: 1.0 - ROLL_SMOOTH, comp: cell.comp };
                    blend(&mut t, ROLL_SMOOTH, &prior);
                    cell.comp = t.comp;
                }
                self.trail[lane] = cell.comp;
            }
            for (lane, cell) in laid.iter().enumerate() {
                if cell.vol <= 0.0 {
                    continue;
                }
                let ws: Vec<f64> = (0..SUB_LANES).map(|j| self.weight(lane * SUB_LANES + j, fine, phase0, fill)).collect();
                let wsum: f64 = ws.iter().sum::<f64>() / SUB_LANES as f64;
                for (j, w) in ws.iter().enumerate() {
                    let i = lane * SUB_LANES + j;
                    let off = (i as f64 + 0.5) / fine as f64 * span - 0.5 * span;
                    let a = prev + prev_axis * off;
                    let b = p + self.axis * off;
                    let v = cell.vol / SUB_LANES as f64 * w / wsum;
                    let side = if j < SUB_LANES / 2 { lane.checked_sub(1) } else { Some(lane + 1).filter(|&n| n < NAP_LANES) };
                    let mut c = Cell { vol: cell.vol * (1.0 - EDGE_BLEND), comp: cell.comp };
                    if let Some(n) = side {
                        if laid[n].vol > 0.0 {
                            blend(&mut c, cell.vol * EDGE_BLEND, &laid[n].comp);
                        }
                    }
                    let pig = crate::color::PaintMix { frac: c.comp }.normalized().packed();
                    deposits.push(Deposit::ribbon([a.x as f32, a.y as f32], [b.x as f32, b.y as f32], n0, n1, r_lane as f32, r_lane as f32, 0, v as f32, pig, dx).laid_by(Method::Roller));
                }
            }
            self.mix_nap(roll / (TAU * self.radius));
        }
        (contacts, deposits)
    }

    pub fn mesh(&self, verts: &mut Vec<TubeVertex>, idx: &mut Vec<u32>, at: DVec3, paint_col: &dyn Fn(&[f32; 16]) -> [f32; 3]) {
        let lift = if self.pressing { 0.0 } else { 0.015 };
        let c = DVec3::new(at.x, at.y, self.radius + lift);
        let ax = DVec3::new(self.axis.x, self.axis.y, 0.0);
        let side = DVec3::new(self.axis.y, -self.axis.x, 0.0);
        let nap = [0.86f32, 0.82, 0.74];
        let cap = self.cell_capacity();
        let colours: Vec<[f32; 3]> = self
            .cells
            .iter()
            .map(|cell| {
                if cell.vol <= 0.0 {
                    return nap;
                }
                let f = (cell.vol / cap).clamp(0.0, 1.0) as f32;
                let pc = paint_col(&cell.comp);
                let k = (0.3 + 0.7 * f).min(1.0);
                [nap[0] + (pc[0] - nap[0]) * k, nap[1] + (pc[1] - nap[1]) * k, nap[2] + (pc[2] - nap[2]) * k]
            })
            .collect();
        let hw = 0.5 * self.width;
        let lane_w = self.width / NAP_LANES as f64;
        let segs = NAP_SECTORS as u32;
        let turn = self.phase / TAU * NAP_SECTORS as f64;
        for lane in 0..NAP_LANES {
            for edge in 0..2 {
                let x = -hw + (lane + edge) as f64 * lane_w;
                let base = verts.len() as u32;
                for s in 0..=segs {
                    let a = -FRAC_PI_2 - (s as f64 - turn) * TAU / NAP_SECTORS as f64;
                    let n = side * a.cos() + DVec3::Z * a.sin();
                    let p = c + ax * x + n * self.radius;
                    let col = colours[lane * NAP_SECTORS + (s as usize % NAP_SECTORS)];
                    verts.push(TubeVertex { pos: [p.x as f32, p.y as f32, p.z as f32], nrm: [n.x as f32, n.y as f32, n.z as f32], col });
                }
                if edge == 1 {
                    let left = base - (segs + 1);
                    for s in 0..segs {
                        let (a0, a1, b0, b1) = (left + s, left + s + 1, base + s, base + s + 1);
                        idx.extend_from_slice(&[a0, a1, b0, a1, b1, b0]);
                    }
                }
            }
        }
        for end in 0..2 {
            let x = if end == 0 { -hw } else { hw };
            let n = if end == 0 { -ax } else { ax };
            let centre = verts.len() as u32;
            let pc = c + ax * x;
            let capc = [0.35f32, 0.35, 0.37];
            verts.push(TubeVertex { pos: [pc.x as f32, pc.y as f32, pc.z as f32], nrm: [n.x as f32, n.y as f32, n.z as f32], col: capc });
            for s in 0..segs {
                let a = s as f64 / segs as f64 * TAU;
                let p = pc + (side * a.cos() + DVec3::Z * a.sin()) * self.radius * 0.97;
                verts.push(TubeVertex { pos: [p.x as f32, p.y as f32, p.z as f32], nrm: [n.x as f32, n.y as f32, n.z as f32], col: capc });
            }
            for s in 0..segs {
                let s1 = (s + 1) % segs;
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
