pub mod banded;
pub mod ribbon;

use banded::Banded;
use glam::{DMat3, DVec3};
pub use ribbon::{Ribbon, Segment};
use rayon::prelude::*;
use std::f64::consts::PI;

pub const RHO_AIR: f64 = 1.2;
pub const G_ACC: f64 = 9.81;
#[cfg(test)]
pub const G: f64 = 9.81;

#[derive(Clone, Copy, Debug)]
pub struct Fluid {
    pub mu: f64,
    pub rho: f64,
    pub sigma: f64,
}

#[derive(Clone, Debug)]
pub struct Nozzle {
    pub pos: DVec3,
    pub vel: DVec3,
    pub dir: DVec3,
    pub radius: f64,
    pub flow: f64,
}

#[derive(Clone, Debug)]
pub struct Rod {
    pub x: Vec<DVec3>,
    pub v: Vec<DVec3>,
    pub vol: Vec<f64>,
    pub m1: Vec<DVec3>,
    pub tan: Vec<DVec3>,
    pub frozen: Vec<bool>,
    pub vimp: Vec<f64>,
    pub scratch: Vec<f64>,
    pub step_count: u64,
    pub stall: u32,
    pub fluid: Fluid,
    pub pig: [u32; 8],
    pub nozzle: Option<Nozzle>,
    pub feed_len: f64,
    pub landed: Ribbon,
    pub landed_done: Vec<Ribbon>,
    pub id: u64,
}

#[derive(Clone, Debug)]
pub struct Landing {
    pub seg: Segment,
    pub pig: [u32; 8],
    pub fluid: Fluid,
}

#[derive(Clone, Debug)]
pub struct Drop {
    pub x: DVec3,
    pub v: DVec3,
    pub vol: f64,
    pub pig: [u32; 8],
    pub fluid: Fluid,
}

pub trait Surface {
    fn height(&self, x: f64, y: f64) -> f64;
}

pub struct RodOut {
    pub landings: Vec<Landing>,
    pub drops: Vec<Drop>,
    pub fragments: Vec<Rod>,
}

fn skew(a: DVec3) -> DMat3 {
    DMat3::from_cols(DVec3::new(0.0, a.z, -a.y), DVec3::new(-a.z, 0.0, a.x), DVec3::new(a.y, -a.x, 0.0))
}

fn outer(a: DVec3, b: DVec3) -> DMat3 {
    DMat3::from_cols(a * b.x, a * b.y, a * b.z)
}

fn rotate_to(from: DVec3, to: DVec3, v: DVec3) -> DVec3 {
    let c = from.dot(to);
    if c > 1.0 - 1e-12 {
        return v;
    }
    if c < -1.0 + 1e-12 {
        return -v;
    }
    let axis = from.cross(to);
    let s2 = axis.length_squared();
    let k = 1.0 / (1.0 + c);
    let _ = s2;
    v * c + axis.cross(v) + axis * (axis.dot(v) * k)
}

fn coarsen_from_arc(s: f64) -> f64 {
    1.0 + 5.0 * ((s - 0.04) / 0.1).clamp(0.0, 1.0)
}

fn target_spacing(r: f64) -> f64 {
    (3.0 * r).clamp(3.0e-4, 1.5e-3)
}

const R_MIN: f64 = 3.0e-5;
const MAX_VERTS: usize = 1500;
const V_MAX: f64 = 15.0;

impl Rod {
    pub fn from_nozzle(nozzle: Nozzle, fluid: Fluid, pig: [u32; 8], id: u64) -> Self {
        let p = nozzle.pos;
        let d = nozzle.dir.normalize();
        let hang = (0.15 * nozzle.pos.z).clamp(0.005, 0.03);
        let n = 6usize;
        let seg = hang / (n - 1) as f64;
        let ue = nozzle.exit_speed();
        let mut x = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);
        let mut vol = Vec::with_capacity(n - 1);
        for k in 0..n {
            let s = hang - seg * k as f64;
            x.push(p + d * s);
            let fall = (2.0 * G_ACC * s).sqrt();
            v.push(nozzle.vel + d * (ue + fall));
            if k + 1 < n {
                let stretch = 1.0 + fall / ue.max(1e-3);
                vol.push(PI * nozzle.radius * nozzle.radius * seg / stretch.max(1.0).min(20.0));
            }
        }
        let mut tan = Vec::with_capacity(n - 1);
        let mut m1 = Vec::with_capacity(n - 1);
        for e in 0..n - 1 {
            let t = (x[e + 1] - x[e]).normalize();
            tan.push(t);
            m1.push(any_perp(t));
        }
        Self { x, v, vol, m1, tan, frozen: vec![false; n], vimp: vec![0.0; n], scratch: Vec::new(), step_count: 0, stall: 0, fluid, pig, nozzle: Some(nozzle), feed_len: 0.0, landed: Ribbon::default(), landed_done: Vec::new(), id }
    }

    pub fn n(&self) -> usize {
        self.x.len()
    }

    pub fn edge_radius(&self, e: usize) -> f64 {
        let l = (self.x[e + 1] - self.x[e]).length().max(1e-9);
        let cyl = (self.vol[e] / (PI * l)).sqrt();
        let blob = 1.2 * (3.0 * self.vol[e].max(0.0) / (4.0 * PI)).cbrt();
        cyl.min(blob)
    }

    pub fn vertex_radius(&self, i: usize) -> f64 {
        let n = self.n();
        if n < 2 {
            return 0.0;
        }
        if i == 0 {
            self.edge_radius(0)
        } else if i == n - 1 {
            self.edge_radius(n - 2)
        } else {
            0.5 * (self.edge_radius(i - 1) + self.edge_radius(i))
        }
    }

    pub fn total_volume(&self) -> f64 {
        self.vol.iter().sum()
    }

    fn update_frames(&mut self) {
        let n = self.n();
        for e in 0..n - 1 {
            let t = (self.x[e + 1] - self.x[e]).normalize_or_zero();
            if t.length_squared() < 0.5 {
                continue;
            }
            let mut m = rotate_to(self.tan[e], t, self.m1[e]);
            m -= t * m.dot(t);
            if m.length_squared() < 1e-12 {
                m = any_perp(t);
            }
            self.m1[e] = m.normalize();
            self.tan[e] = t;
        }
    }

    pub fn curvature_rate_jacobian(&self, i: usize) -> [[f64; 9]; 2] {
        let a = self.x[i] - self.x[i - 1];
        let b = self.x[i + 1] - self.x[i];
        let la = a.length().max(1e-12);
        let lb = b.length().max(1e-12);
        let ta = a / la;
        let tb = b / lb;
        let d = (la * lb + a.dot(b)).max(0.02 * la * lb);
        let kb = a.cross(b) * (2.0 / d);
        let dkb_db = (skew(a) * 2.0 - outer(kb, ta * la + a)) * (1.0 / d);
        let dkb_da = (skew(b) * -2.0 - outer(kb, tb * lb + b)) * (1.0 / d);
        let m1a = self.m1[i - 1];
        let m1b = self.m1[i];
        let mut m1 = (m1a + m1b) * 0.5;
        let tv = (ta + tb).normalize_or_zero();
        m1 -= tv * m1.dot(tv);
        if m1.length_squared() < 1e-14 {
            m1 = any_perp(tv);
        }
        let m1 = m1.normalize();
        let m2 = tv.cross(m1);
        let pa = DMat3::IDENTITY - outer(ta, ta);
        let pb = DMat3::IDENTITY - outer(tb, tb);
        let mut j = [[0.0f64; 9]; 2];
        for col in 0..9 {
            let mut v = [DVec3::ZERO; 3];
            v[col / 3][col % 3] = 1.0;
            let ad = v[1] - v[0];
            let bd = v[2] - v[1];
            let kbd = dkb_da * ad + dkb_db * bd;
            let tad = pa * ad / la;
            let tbd = pb * bd / lb;
            let w1v = ta.cross(tad);
            let w2v = tb.cross(tbd);
            let c = ta.dot(tb);
            let den = 1.0 - c * c;
            let s = if den > 1e-8 { (w2v.dot(ta) + c * tb.dot(w1v)) / den } else { 0.0 };
            let wf = w1v + ta * s;
            let w1 = kbd.dot(m2) + kb.dot(wf.cross(m2));
            let w2 = -(kbd.dot(m1) + kb.dot(wf.cross(m1)));
            j[0][col] = w1;
            j[1][col] = w2;
        }
        j
    }

    fn surface_energy_local(&self, i: usize, xi: DVec3) -> f64 {
        let n = self.n();
        let sigma = self.fluid.sigma;
        let lo = i.saturating_sub(2);
        let hi = (i + 2).min(n - 1);
        let xs = |k: usize| if k == i { xi } else { self.x[k] };
        let er = |e: usize| {
            let l = (xs(e + 1) - xs(e)).length().max(1e-9);
            ((self.vol[e] / (PI * l)).sqrt(), l)
        };
        let vr = |k: usize| {
            if k == 0 {
                er(0).0
            } else if k == n - 1 {
                er(n - 2).0
            } else {
                0.5 * (er(k - 1).0 + er(k).0)
            }
        };
        let mut e_sum = 0.0;
        for e in lo..hi {
            let (_, l) = er(e);
            let r0 = vr(e);
            let r1 = vr(e + 1);
            let rbar = 0.5 * (r0 + r1);
            let dr = r1 - r0;
            e_sum += sigma * 2.0 * PI * rbar * (l * l + dr * dr).sqrt();
        }
        e_sum
    }

    fn surface_force(&self, i: usize, mass: f64, dt: f64) -> DVec3 {
        let n = self.n();
        let lref = if i == 0 {
            (self.x[1] - self.x[0]).length()
        } else if i == n - 1 {
            (self.x[n - 1] - self.x[n - 2]).length()
        } else {
            0.5 * ((self.x[i] - self.x[i - 1]).length() + (self.x[i + 1] - self.x[i]).length())
        };
        let eps = (1e-4 * lref).max(1e-9);
        let mut f = DVec3::ZERO;
        for k in 0..3 {
            let mut d = DVec3::ZERO;
            d[k] = eps;
            let ep = self.surface_energy_local(i, self.x[i] + d);
            let em = self.surface_energy_local(i, self.x[i] - d);
            f[k] = -(ep - em) / (2.0 * eps);
        }
        if !f.is_finite() {
            return DVec3::ZERO;
        }
        let r = self.vertex_radius(i).max(1e-5);
        let vcap = (self.fluid.sigma / (self.fluid.rho * r)).sqrt();
        let fmax = mass * vcap / dt;
        let fl = f.length();
        if fl > fmax {
            f *= fmax / fl;
        }
        f
    }

    pub fn step(&mut self, dt: f64, surface: &dyn Surface, out: &mut RodOut, gravity: DVec3) {
        let n = self.n();
        if n < 2 {
            return;
        }
        self.update_frames();
        let mu = self.fluid.mu;
        let rho = self.fluid.rho;
        let mut mass = vec![0.0f64; n];
        for e in 0..n - 1 {
            let m = rho * self.vol[e];
            mass[e] += 0.5 * m;
            mass[e + 1] += 0.5 * m;
        }
        let mut sys = Banded::from_scratch(std::mem::take(&mut self.scratch), 3 * n, 8);
        let mut rhs = vec![0.0f64; 3 * n];
        let vertex_terms = |i: usize| -> (DVec3, f64, [f64; 45]) {
            let mut f = gravity * mass[i];
            f += self.surface_force(i, mass[i], dt);
            let t = if i == 0 { self.tan[0] } else if i == n - 1 { self.tan[n - 2] } else { (self.tan[i - 1] + self.tan[i]).normalize_or_zero() };
            let vn = self.v[i] - t * self.v[i].dot(t);
            let r = self.vertex_radius(i);
            let l = if i == 0 {
                0.5 * (self.x[1] - self.x[0]).length()
            } else if i == n - 1 {
                0.5 * (self.x[n - 1] - self.x[n - 2]).length()
            } else {
                0.5 * ((self.x[i] - self.x[i - 1]).length() + (self.x[i + 1] - self.x[i]).length())
            };
            let cdrag = 0.5 * RHO_AIR * 1.2 * 2.0 * r * l * vn.length();
            let mut blk = [0.0f64; 45];
            if i > 0 && i + 1 < n {
                let inertia = PI * r.powi(4) / 4.0;
                let kb = dt * 3.0 * mu * inertia / l.max(1e-9);
                let j = self.curvature_rate_jacobian(i);
                let mut q = 0;
                for a in 0..9 {
                    for b in 0..=a {
                        blk[q] = kb * (j[0][a] * j[0][b] + j[1][a] * j[1][b]);
                        q += 1;
                    }
                }
            }
            (f, cdrag, blk)
        };
        let terms: Vec<(DVec3, f64, [f64; 45])> = if n > 128 {
            (0..n).into_par_iter().map(vertex_terms).collect()
        } else {
            (0..n).map(vertex_terms).collect()
        };
        for i in 0..n {
            let (f, cdrag, _) = &terms[i];
            for k in 0..3 {
                rhs[3 * i + k] = mass[i] * self.v[i][k] + dt * f[k];
                sys.add(3 * i + k, 3 * i + k, mass[i] + dt * cdrag);
            }
        }
        let neck_cap = self.nozzle.as_ref().map(|nz| 4.0 * PI * nz.radius * nz.radius).unwrap_or(f64::MAX);
        for e in 0..n - 1 {
            let l = (self.x[e + 1] - self.x[e]).length().max(1e-9);
            let re = self.edge_radius(e);
            let mut area = (self.vol[e] / l).min(PI * re * re * 1.5);
            if e + 2 >= n {
                area = area.min(neck_cap);
            }
            let k = dt * 3.0 * mu * area / l;
            let t = self.tan[e];
            for a in 0..3 {
                for b in 0..3 {
                    let v = k * t[a] * t[b];
                    sys.add(3 * e + a, 3 * e + b, v);
                    sys.add(3 * (e + 1) + a, 3 * (e + 1) + b, v);
                    sys.add(3 * (e + 1) + a, 3 * e + b, -v);
                }
            }
        }
        for i in 1..n - 1 {
            let blk = &terms[i].2;
            let base = 3 * (i - 1);
            let mut q = 0;
            for a in 0..9 {
                for b in 0..=a {
                    let v = blk[q];
                    q += 1;
                    if v != 0.0 {
                        sys.add(base + a, base + b, v);
                    }
                }
            }
        }
        for i in 0..n {
            if self.frozen[i] {
                for k in 0..3 {
                    sys.fix_dof(3 * i + k, 0.0, &mut rhs);
                }
            }
        }
        if let Some(nz) = &self.nozzle {
            let ve = nz.vel + nz.dir.normalize() * nz.exit_speed();
            for k in 0..3 {
                sys.fix_dof(3 * (n - 1) + k, ve[k], &mut rhs);
            }
        }
        sys.factor();
        sys.solve(&mut rhs);
        self.scratch = sys.into_scratch();
        let ok = rhs.iter().all(|v| v.is_finite());
        for i in 0..n {
            if self.frozen[i] {
                self.v[i] = DVec3::ZERO;
                continue;
            }
            if ok {
                self.v[i] = DVec3::new(rhs[3 * i], rhs[3 * i + 1], rhs[3 * i + 2]);
            } else {
                self.v[i] += gravity * dt;
            }
            let sp = self.v[i].length();
            if sp > V_MAX {
                self.v[i] *= V_MAX / sp;
            }
            self.x[i] += self.v[i] * dt;
        }
        if let Some(nz) = self.nozzle.clone() {
            let moved = (self.x[n - 1] - nz.pos).length();
            self.x[n - 1] = nz.pos;
            self.feed_len += moved;
            self.vol[n - 2] += nz.flow * dt;
        }
        self.contact(surface);
        self.step_count += 1;
        if self.step_count % 3 == 0 || self.nozzle.is_some() && self.feed_len > 0.0 {
            self.remesh();
        }
        self.pinch(out);
    }

    fn contact(&mut self, surface: &dyn Surface) {
        let n = self.n();
        for i in 0..n {
            if self.frozen[i] {
                let r = self.vertex_radius(i);
                let zs = surface.height(self.x[i].x, self.x[i].y);
                let target = zs + r;
                if self.x[i].z > target + 1e-4 {
                    self.x[i].z = (self.x[i].z - 0.01).max(target);
                }
                continue;
            }
            if self.nozzle.is_some() && i == n - 1 {
                continue;
            }
            let r = self.vertex_radius(i);
            let zs = surface.height(self.x[i].x, self.x[i].y);
            let mut zc = zs + r;
            if i > 0 && self.frozen[i - 1] {
                zc = zc.max(self.x[i - 1].z);
            }
            if i + 1 < n && self.frozen[i + 1] {
                zc = zc.max(self.x[i + 1].z);
            }
            if self.x[i].z <= zc {
                self.x[i].z = self.x[i].z.max(zs + r);
                self.frozen[i] = true;
                self.vimp[i] = self.v[i].length();
                self.v[i] = DVec3::ZERO;
            }
        }
        let nn = self.n();
        let frozen_idx: Vec<usize> = (0..nn).filter(|k| self.frozen[*k]).collect();
        if !frozen_idx.is_empty() {
            for i in 0..nn {
                if self.frozen[i] || (self.nozzle.is_some() && i == nn - 1) {
                    continue;
                }
                let ri = self.vertex_radius(i);
                let xi = self.x[i];
                let zs = surface.height(xi.x, xi.y);
                if xi.z > zs + 0.03 {
                    continue;
                }
                let mut rest = false;
                for &j in &frozen_idx {
                    if i.abs_diff(j) < 8 {
                        continue;
                    }
                    let rj = self.vertex_radius(j);
                    let d = xi - self.x[j];
                    let reach = ri + rj;
                    if d.length() <= reach && d.z >= 0.0 {
                        rest = true;
                        break;
                    }
                }
                if rest {
                    self.frozen[i] = true;
                    self.vimp[i] = self.v[i].length();
                    self.v[i] = DVec3::ZERO;
                }
            }
        }
        let mut i = 0;
        while i < nn {
            if !self.frozen[i] && (i == 0 || self.frozen[i - 1]) {
                let mut j = i + 1;
                let mut found = None;
                while j < nn && j < i + 12 {
                    if self.frozen[j] {
                        found = Some(j);
                        break;
                    }
                    j += 1;
                }
                if let Some(j) = found {
                    for k in i..j {
                        if self.nozzle.is_some() && k == nn - 1 {
                            continue;
                        }
                        let r = self.vertex_radius(k);
                        let zs = surface.height(self.x[k].x, self.x[k].y);
                        self.x[k].z = self.x[k].z.max(zs + r);
                        self.frozen[k] = true;
                        self.vimp[k] = self.v[k].length();
                        self.v[k] = DVec3::ZERO;
                    }
                    i = j;
                    continue;
                }
            }
            i += 1;
        }
        if self.x.len() >= 3 && self.frozen[0] && self.frozen[1] && !self.frozen[2] {
            self.stall += 1;
        } else {
            self.stall = 0;
        }
        if self.stall > 150 && self.x.len() >= 3 {
            self.emit_front();
            self.stall = 0;
        }
        let mut deposited = 0usize;
        while self.x.len() >= 3 && self.frozen[0] && self.frozen[1] && self.frozen[2] {
            self.emit_front();
            deposited += 1;
        }
        let _ = deposited;
        if self.nozzle.is_none() {
            while self.x.len() >= 2 && self.frozen.iter().all(|f| *f) {
                self.emit_front();
            }
        }
    }

    fn emit_front(&mut self) {
        let r = self.edge_radius(0);
        let speed = self.vimp[0].max(self.vimp[1]);
        let spread = impact_spread_factor(speed, 2.0 * r, &self.fluid);
        if let Some(done) = self.landed.push(self.x[0].truncate(), self.x[1].truncate(), self.vol[0], speed, spread) {
            self.landed_done.push(done);
        }
        self.remove_vertex_front();
    }

    pub fn drain_landings(&mut self, step: f64, flush: bool, out: &mut Vec<Landing>) {
        let mut segs = Vec::new();
        for mut done in self.landed_done.drain(..) {
            done.drain(step, true, &mut segs);
        }
        self.landed.drain(step, flush, &mut segs);
        out.extend(segs.into_iter().map(|seg| Landing { seg, pig: self.pig, fluid: self.fluid }));
    }

    fn remove_vertex_front(&mut self) {
        self.x.remove(0);
        self.v.remove(0);
        self.frozen.remove(0);
        self.vimp.remove(0);
        self.vol.remove(0);
        self.m1.remove(0);
        self.tan.remove(0);
    }

    fn arc_lengths(&self) -> Vec<f64> {
        let n = self.x.len();
        let mut s = vec![0.0; n];
        for k in 1..n {
            s[k] = s[k - 1] + (self.x[k] - self.x[k - 1]).length();
        }
        s
    }

    fn coarsen_factor(&self, i: usize) -> f64 {
        let s: f64 = (0..i.min(self.x.len().saturating_sub(1))).map(|k| (self.x[k + 1] - self.x[k]).length()).sum();
        coarsen_from_arc(s)
    }

    fn remesh(&mut self) {
        let mut arc = self.arc_lengths();
        let mut e = 0;
        while e + 1 < self.x.len() {
            let l = (self.x[e + 1] - self.x[e]).length();
            let r = self.edge_radius(e);
            let lt = target_spacing(r) * coarsen_from_arc(arc[e]);
            let is_feed = self.nozzle.is_some() && e + 2 == self.x.len();
            if l > 2.0 * lt && !(self.frozen[e] && self.frozen[e + 1]) && r > 2.0 * R_MIN && self.x.len() < MAX_VERTS {
                let mid = 0.5 * (self.x[e] + self.x[e + 1]);
                let vm = 0.5 * (self.v[e] + self.v[e + 1]);
                let half = 0.5 * self.vol[e];
                self.x.insert(e + 1, mid);
                self.v.insert(e + 1, vm);
                self.frozen.insert(e + 1, false);
                self.vimp.insert(e + 1, 0.0);
                self.vol[e] = half;
                self.vol.insert(e + 1, half);
                self.m1.insert(e + 1, self.m1[e]);
                self.tan.insert(e + 1, self.tan[e]);
                arc.insert(e + 1, arc[e] + 0.5 * l);
                if is_feed {
                    self.feed_len = 0.0;
                }
                continue;
            }
            e += 1;
        }
        let mut i = 1;
        while i + 1 < self.x.len() {
            if self.frozen[i] || self.frozen[i - 1] || self.frozen[i + 1] {
                i += 1;
                continue;
            }
            if self.nozzle.is_some() && i + 2 >= self.x.len() {
                break;
            }
            let la = (self.x[i] - self.x[i - 1]).length();
            let lb = (self.x[i + 1] - self.x[i]).length();
            let r = 0.5 * (self.edge_radius(i - 1) + self.edge_radius(i));
            let lt = target_spacing(r) * coarsen_from_arc(arc[i]);
            if la + lb < 0.8 * lt {
                self.vol[i - 1] += self.vol[i];
                self.vol.remove(i);
                self.x.remove(i);
                self.v.remove(i);
                self.frozen.remove(i);
                self.vimp.remove(i);
                self.m1.remove(i);
                self.tan.remove(i);
                arc.remove(i);
                continue;
            }
            i += 1;
        }
    }

    fn edge_to_drops(&self, e: usize, out: &mut RodOut) {
        let a = self.x[e];
        let b = self.x[e + 1];
        let l = (b - a).length();
        let r = self.edge_radius(e);
        let k = ((l / (10.0 * r.max(1e-6))).floor() as usize).clamp(1, 12);
        let each = self.vol[e] / k as f64;
        for q in 0..k {
            let t = (q as f64 + 0.5) / k as f64;
            let x = a + (b - a) * t;
            let v = self.v[e] * (1.0 - t) + self.v[e + 1] * t;
            out.drops.push(Drop { x, v, vol: each, pig: self.pig, fluid: self.fluid });
        }
    }

    fn pinch(&mut self, out: &mut RodOut) {
        let r_min = R_MIN;
        let mut e = 0;
        while e + 1 < self.x.len() {
            let r = self.edge_radius(e);
            let l = (self.x[e + 1] - self.x[e]).length();
            let feed = self.nozzle.is_some() && e + 3 >= self.x.len();
            let both_frozen = self.frozen[e] && self.frozen[e + 1];
            let slender = l > (30.0 * r).max(5.0e-3) && r < 5.0e-4;
            if (r < r_min || slender) && !feed && !both_frozen {
                if slender {
                    self.edge_to_drops(e, out);
                    self.vol[e] = 0.0;
                }
                let lower = self.split_at(e);
                if let Some(l) = lower {
                    l.emit_or_keep(out);
                }
                continue;
            }
            e += 1;
        }
        if self.nozzle.is_none() && self.x.len() >= 2 {
            let len: f64 = (0..self.x.len() - 1).map(|k| (self.x[k + 1] - self.x[k]).length()).sum();
            let vol = self.total_volume();
            let rr = (vol / (PI * len.max(1e-9))).sqrt();
            let free = self.frozen.iter().all(|f| !*f);
            if free && (len < 2.0 * PI * rr || self.x.len() <= 2) {
                let c = self.x.iter().copied().sum::<DVec3>() / self.x.len() as f64;
                let v = self.v.iter().copied().sum::<DVec3>() / self.v.len() as f64;
                out.drops.push(Drop { x: c, v, vol, pig: self.pig, fluid: self.fluid });
                self.x.clear();
                self.v.clear();
                self.vol.clear();
                self.frozen.clear();
                self.vimp.clear();
                self.m1.clear();
                self.tan.clear();
            }
        }
    }

    fn split_at(&mut self, e: usize) -> Option<Rod> {
        let lost = self.vol[e];
        let n = self.x.len();
        let lower_x: Vec<DVec3> = self.x[..=e].to_vec();
        let lower_v: Vec<DVec3> = self.v[..=e].to_vec();
        let lower_vol: Vec<f64> = self.vol[..e].to_vec();
        let lower_frozen: Vec<bool> = self.frozen[..=e].to_vec();
        let lower_vimp: Vec<f64> = self.vimp[..=e].to_vec();
        let lower_m1: Vec<DVec3> = self.m1[..e].to_vec();
        let lower_tan: Vec<DVec3> = self.tan[..e].to_vec();
        self.x.drain(..=e);
        self.v.drain(..=e);
        self.vol.drain(..=e);
        self.frozen.drain(..=e);
        self.vimp.drain(..=e);
        self.m1.drain(..=e);
        self.tan.drain(..=e);
        if !self.vol.is_empty() {
            self.vol[0] += 0.5 * lost;
        }
        let _ = n;
        if lower_x.len() < 2 {
            return None;
        }
        let mut lower = Rod {
            x: lower_x,
            v: lower_v,
            vol: lower_vol,
            m1: lower_m1,
            tan: lower_tan,
            frozen: lower_frozen,
            vimp: lower_vimp,
            scratch: Vec::new(),
            step_count: 0,
            stall: 0,
            fluid: self.fluid,
            pig: self.pig,
            nozzle: None,
            feed_len: 0.0,
            landed: std::mem::take(&mut self.landed),
            landed_done: std::mem::take(&mut self.landed_done),
            id: self.id + 1,
        };
        let last = lower.vol.len() - 1;
        lower.vol[last] += 0.5 * lost;
        Some(lower)
    }

    fn emit_or_keep(self, out: &mut RodOut) {
        out.fragments.push(self);
    }

    pub fn is_empty(&self) -> bool {
        self.x.len() < 2
    }

    pub fn tip_speed(&self) -> f64 {
        let mut s: f64 = 0.0;
        for (i, v) in self.v.iter().enumerate() {
            if !self.frozen[i] {
                s = s.max(v.length());
            }
        }
        s
    }

    pub fn min_spacing(&self) -> f64 {
        let mut m = f64::MAX;
        for e in 0..self.x.len().saturating_sub(1) {
            m = m.min((self.x[e + 1] - self.x[e]).length());
        }
        m
    }

    pub fn min_free_radius(&self) -> f64 {
        let mut m = f64::MAX;
        for e in 0..self.x.len().saturating_sub(1) {
            if !self.frozen[e] {
                m = m.min(self.edge_radius(e));
            }
        }
        m
    }
}

impl Nozzle {
    pub fn exit_speed(&self) -> f64 {
        self.flow / (PI * self.radius * self.radius)
    }
}

pub fn any_perp(t: DVec3) -> DVec3 {
    let a = if t.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    (a - t * a.dot(t)).normalize()
}

pub fn drop_step(d: &mut Drop, dt: f64, gravity: DVec3) {
    let r = (3.0 * d.vol / (4.0 * PI)).cbrt();
    let speed = d.v.length();
    let re = RHO_AIR * speed * 2.0 * r / 1.8e-5;
    let cd = if re > 1e-6 { 24.0 / re * (1.0 + 0.15 * re.powf(0.687)) } else { 0.0 };
    let m = d.fluid.rho * d.vol;
    let drag = -d.v * (0.5 * RHO_AIR * cd * PI * r * r * speed);
    d.v += (gravity + drag / m) * dt;
    d.x += d.v * dt;
}

pub fn splash_count(u: f64, d0: f64, f: &Fluid) -> usize {
    if u <= 0.0 || d0 <= 0.0 {
        return 0;
    }
    let re = f.rho * u * d0 / f.mu;
    let we = f.rho * u * u * d0 / f.sigma;
    let k = we.sqrt() * re.powf(0.25);
    if k <= 57.7 {
        return 0;
    }
    (((k - 57.7) / 12.0).round() as usize).min(40)
}

pub fn impact_spread_factor(u: f64, d0: f64, f: &Fluid) -> f64 {
    if u <= 0.0 || d0 <= 0.0 {
        return 1.0;
    }
    let re = f.rho * u * d0 / f.mu;
    let we = f.rho * u * u * d0 / f.sigma;
    let r = 0.87 * re.powf(0.2) - 0.40 * re.powf(0.4) / we.sqrt().max(1e-6);
    r.clamp(1.0, 4.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Flat;
    impl Surface for Flat {
        fn height(&self, _x: f64, _y: f64) -> f64 {
            0.0
        }
    }

    fn bent_rod() -> Rod {
        let fluid = Fluid { mu: 4.0, rho: 1000.0, sigma: 0.04 };
        let n = 12;
        let mut x = Vec::new();
        for i in 0..n {
            let s = i as f64 * 0.002;
            x.push(DVec3::new(0.3 * s * s, s, 0.2 + 0.1 * s));
        }
        let v = vec![DVec3::ZERO; n];
        let vol = vec![PI * 0.5e-3 * 0.5e-3 * 0.002; n - 1];
        let mut tan = Vec::new();
        let mut m1 = Vec::new();
        for e in 0..n - 1 {
            let t = (x[e + 1] - x[e]).normalize();
            tan.push(t);
            m1.push(any_perp(t));
        }
        Rod { x, v, vol, m1, tan, frozen: vec![false; n], vimp: vec![0.0; n], scratch: Vec::new(), step_count: 0, stall: 0, fluid, pig: [0; 8], nozzle: None, feed_len: 0.0, landed: Ribbon::default(), landed_done: Vec::new(), id: 0 }
    }

    #[test]
    fn bending_rate_is_objective() {
        let rod = bent_rod();
        let omega = DVec3::new(0.3, -1.1, 0.7);
        for i in 1..rod.n() - 1 {
            let j = rod.curvature_rate_jacobian(i);
            let mut v = [0.0f64; 9];
            for k in 0..3 {
                let vk = omega.cross(rod.x[i - 1 + k]);
                v[3 * k] = vk.x;
                v[3 * k + 1] = vk.y;
                v[3 * k + 2] = vk.z;
            }
            let mut w = [0.0f64; 2];
            let mut scale = 0.0f64;
            for c in 0..9 {
                w[0] += j[0][c] * v[c];
                w[1] += j[1][c] * v[c];
                scale = scale.max(j[0][c].abs() * v[c].abs()).max(j[1][c].abs() * v[c].abs());
            }
            assert!(w[0].abs() < 1e-6 * scale.max(1.0), "w1 {} at {}", w[0], i);
            assert!(w[1].abs() < 1e-6 * scale.max(1.0), "w2 {} at {}", w[1], i);
        }
    }

    #[test]
    fn nozzle_jet_stays_finite() {
        let fluid = Fluid { mu: 4.0, rho: 1000.0, sigma: 0.04 };
        let nz = Nozzle { pos: DVec3::new(0.5, 0.4, 0.15), vel: DVec3::new(0.6, 0.0, 0.0), dir: DVec3::new(0.0, 0.0, -1.0), radius: 1.5e-3, flow: 1.5e-7 };
        let mut rod = Rod::from_nozzle(nz.clone(), fluid, [0; 8], 1);
        let initial_volume = rod.total_volume();
        let mut out = RodOut { landings: vec![], drops: vec![], fragments: vec![] };
        let mut t = 0.0;
        let mut steps = 0;
        while t < 2.0 {
            let speed = rod.tip_speed().max(0.05);
            let spacing = rod.min_spacing().max(1e-4);
            let dt = (spacing / speed).clamp(1.0e-4, 1.0e-3);
            let mut n = rod.nozzle.clone().unwrap();
            n.pos += n.vel * dt;
            rod.nozzle = Some(n);
            rod.step(dt, &Flat, &mut out, DVec3::new(0.0, 0.0, -G));
            t += dt;
            steps += 1;
            if steps % 100000 == 0 {
                let zmin = rod.x.iter().map(|x| x.z).fold(1.0f64, f64::min);
                let vmin = rod.v.iter().map(|v| v.z).fold(1.0f64, f64::min);
                println!("t {:.4} n {} zmin {:.4} vzmin {:.3} r0 {:.2e} spacing {:.2e} dt {:.2e} vol {:.3e}", t, rod.n(), zmin, vmin, rod.edge_radius(0), rod.min_spacing(), dt, rod.total_volume());
            }
            let finite = rod.x.iter().all(|x| x.is_finite()) && rod.v.iter().all(|v| v.is_finite());
            if !finite {
                let r = rod.min_free_radius();
                panic!("non-finite at t={} step={} n={} spacing={} r={}", t, steps, rod.n(), rod.min_spacing(), r);
            }
        }
        rod.drain_landings(4e-4, true, &mut out.landings);
        for f in out.fragments.iter_mut() {
            f.drain_landings(4e-4, true, &mut out.landings);
        }
        for (k, l) in out.landings.iter().enumerate().skip(20).take(25) {
            let g = &l.seg;
            println!("landing {} p0 ({:.4},{:.4}) p1 ({:.4},{:.4}) len {:.2e} r {:.2e} vol {:.2e}", k, g.p0.x, g.p0.y, g.p1.x, g.p1.y, g.p1.distance(g.p0), g.thread_r, g.vol);
        }
        let frag: f64 = out.fragments.iter().map(|r| r.total_volume()).sum::<f64>() + out.drops.iter().map(|d| d.vol).sum::<f64>();
        let total = rod.total_volume() + out.landings.iter().map(|l| l.seg.vol).sum::<f64>() + frag;
        println!("steps {} n {} landed {} airborne {} fragments {} total {}", steps, rod.n(), out.landings.len(), rod.total_volume(), frag, total);
        assert!(!out.landings.is_empty());
        let emitted = nz.flow * t + initial_volume;
        assert!((total - emitted).abs() < 1e-3 * emitted, "total {} emitted {}", total, emitted);
    }

    #[test]
    fn fragments_stay_finite() {
        let fluid = Fluid { mu: 1.37, rho: 1000.0, sigma: 0.04 };
        let nz = Nozzle { pos: DVec3::new(0.3, 0.4, 0.15), vel: DVec3::new(0.6, 0.0, 0.0), dir: DVec3::new(0.0, 0.0, -1.0), radius: 1.5e-3, flow: 4.6e-7 };
        let mut rods = vec![Rod::from_nozzle(nz.clone(), fluid, [0; 8], 1)];
        let mut t: f64 = 0.0;
        let mut steps = 0;
        let mut landed = 0.0;
        while t < 3.0 {
            let dt = 1.0e-4;
            let mut new_rods = Vec::new();
            for rod in rods.iter_mut() {
                let mut out = RodOut { landings: vec![], drops: vec![], fragments: vec![] };
                if let Some(n) = rod.nozzle.as_mut() {
                    n.pos += n.vel * dt;
                    n.pos.y = 0.4 + 0.08 * (t * 2.5).sin();
                    n.vel = DVec3::new(0.6, 0.2 * (t * 2.5).cos(), 0.0);
                }
                let before: Vec<f64> = rod.vol.clone();
                rod.step(dt, &Flat, &mut out, DVec3::new(0.0, 0.0, -G));
                let finite = rod.x.iter().all(|x| x.is_finite()) && rod.v.iter().all(|v| v.is_finite());
                if !finite {
                    let ls: Vec<f64> = (0..rod.n().saturating_sub(1)).map(|e| (rod.x[e + 1] - rod.x[e]).length()).collect();
                    panic!("non-finite rod {} nozzle {} at t={} n={} vol={:?} len={:?} frozen={:?} before={:?}", rod.id, rod.nozzle.is_some(), t, rod.n(), rod.vol, ls, rod.frozen, before);
                }
                rod.drain_landings(4e-4, rod.is_empty(), &mut out.landings);
                landed += out.landings.iter().map(|l| l.seg.vol).sum::<f64>();
                new_rods.extend(out.fragments);
                landed += out.drops.iter().map(|d| d.vol).sum::<f64>();
            }
            rods.retain(|r| !r.is_empty());
            rods.extend(new_rods);
            t += dt;
            steps += 1;
            if steps % 2500 == 0 {
                let r = &rods[0];
                let zmin = r.x.iter().map(|x| x.z).fold(1.0f64, f64::min);
                let xs = r.x.iter().map(|x| x.x).fold(1.0f64, f64::min);
                println!("t {:.3} rods {} n {} zmin {:.4} xmin {:.3} nozzle_x {:.3} r0 {:.2e} vol {:.2e} frozen {}", t, rods.len(), r.n(), zmin, xs, r.nozzle.as_ref().map(|n| n.pos.x).unwrap_or(-1.0), r.edge_radius(0), r.total_volume(), r.frozen.iter().filter(|f| **f).count());
                let lmax = (0..r.n() - 1).map(|e| (r.x[e + 1] - r.x[e]).length()).fold(0.0f64, f64::max);
                let zmean = r.x.iter().map(|x| x.z).sum::<f64>() / r.n() as f64;
                let xmax = r.x.iter().map(|x| x.x).fold(0.0f64, f64::max);
                println!("   lmax {:.2e} zmean {:.4} xmax {:.3} len {:.3}", lmax, zmean, xmax, (0..r.n() - 1).map(|e| (r.x[e + 1] - r.x[e]).length()).sum::<f64>());
                let k = r.n().min(6);
                println!("   frozen {:?} z {:?} r {:?}", &r.frozen[..k], r.x[..k].iter().map(|x| (x.z * 1e3 * 100.0).round() / 100.0).collect::<Vec<_>>(), (0..k).map(|i| (r.vertex_radius(i) * 1e5).round() / 100.0).collect::<Vec<_>>());
            }
        }
        println!("steps {} rods {} landed {}", steps, rods.len(), landed);
    }

    #[test]
    fn bottle_hold_trace() {
        let fluid = Fluid { mu: 2.7, rho: 1000.0, sigma: 0.0585 };
        let nz = Nozzle { pos: DVec3::new(0.6, 0.4, 0.15), vel: DVec3::ZERO, dir: DVec3::new(0.0, 0.0, -1.0), radius: 1.5e-3, flow: 1.5e-6 };
        let mut rods = vec![Rod::from_nozzle(nz.clone(), fluid, [0; 8], 1)];
        let mut t: f64 = 0.0;
        let mut landed = 0.0;
        let mut next_print = 0.25;
        while t < 6.0 {
            let dt = 2.0e-4;
            let mut new_rods = Vec::new();
            for rod in rods.iter_mut() {
                let mut out = RodOut { landings: vec![], drops: vec![], fragments: vec![] };
                rod.step(dt, &Flat, &mut out, DVec3::new(0.0, 0.0, -G));
                rod.drain_landings(4e-4, rod.is_empty(), &mut out.landings);
                landed += out.landings.iter().map(|l| l.seg.vol).sum::<f64>() + out.drops.iter().map(|d| d.vol).sum::<f64>();
                new_rods.extend(out.fragments);
            }
            rods.retain(|r| !r.is_empty());
            rods.extend(new_rods);
            t += dt;
            if t >= next_print {
                next_print += 0.25;
                let r = &rods[0];
                let n = r.n();
                let air: f64 = rods.iter().map(|r| r.total_volume()).sum();
                let zmin = r.x.iter().map(|x| x.z).fold(1.0f64, f64::min);
                let top: Vec<String> = (n.saturating_sub(4)..n.saturating_sub(1)).map(|e| format!("L{:.1}mm r{:.2}mm v{:.3}", (r.x[e + 1] - r.x[e]).length() * 1e3, r.edge_radius(e) * 1e3, r.vol[e] * 1e6)).collect();
                println!("t {:.2} rods {} n {} air {:.3} mL landed {:.3} mL zmin {:.3} frozen {} top {:?}", t, rods.len(), n, air * 1e6, landed * 1e6, zmin, r.frozen.iter().filter(|f| **f).count(), top);
            }
        }
    }

    #[test]
    fn bench_long_rod_step() {
        let fluid = Fluid { mu: 0.05, rho: 1000.0, sigma: 0.04 };
        let n = 1500;
        let mut x = Vec::new();
        for i in 0..n {
            let s = i as f64 * 0.0003;
            x.push(DVec3::new(0.3 + 0.02 * (s * 40.0).sin(), 0.4, 0.01 + s));
        }
        let v = vec![DVec3::new(0.0, 0.0, -1.0); n];
        let vol = vec![PI * 2e-4 * 2e-4 * 0.0003; n - 1];
        let mut tan = Vec::new();
        let mut m1 = Vec::new();
        for e in 0..n - 1 {
            let t = (x[e + 1] - x[e]).normalize();
            tan.push(t);
            m1.push(any_perp(t));
        }
        let mut rod = Rod { x, v, vol, m1, tan, frozen: vec![false; n], vimp: vec![0.0; n], scratch: Vec::new(), step_count: 0, stall: 0, fluid, pig: [0; 8], nozzle: None, feed_len: 0.0, landed: Ribbon::default(), landed_done: Vec::new(), id: 0 };
        let mut out = RodOut { landings: vec![], drops: vec![], fragments: vec![] };
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            rod.step(1e-4, &Flat, &mut out, DVec3::new(0.0, 0.0, -G));
        }
        println!("bench: {:.2} ms per step for n={}", t0.elapsed().as_secs_f64() * 1e3 / 20.0, rod.n());
        let n = rod.n();
        let t1 = std::time::Instant::now();
        let mut acc = DVec3::ZERO;
        for i in 0..n { acc += rod.surface_force(i, 1e-7, 1e-4); }
        let ms_sigma = t1.elapsed().as_secs_f64() * 1e3;
        let t2 = std::time::Instant::now();
        let mut acc2 = 0.0;
        for i in 1..n - 1 { let j = rod.curvature_rate_jacobian(i); acc2 += j[0][0]; }
        let ms_jac = t2.elapsed().as_secs_f64() * 1e3;
        let t3 = std::time::Instant::now();
        let mut sys = Banded::new(3 * n, 8);
        for i in 0..3 * n { sys.add(i, i, 1.0); if i > 0 { sys.add(i, i - 1, 0.1); } }
        let mut rhs = vec![1.0; 3 * n];
        sys.factor();
        sys.solve(&mut rhs);
        let ms_solve = t3.elapsed().as_secs_f64() * 1e3;
        let t4 = std::time::Instant::now();
        rod.remesh();
        let ms_remesh = t4.elapsed().as_secs_f64() * 1e3;
        let t5 = std::time::Instant::now();
        let mut acc3 = 0.0;
        for i in 0..n { acc3 += rod.vertex_radius(i) + rod.coarsen_factor(i); }
        let ms_misc = t5.elapsed().as_secs_f64() * 1e3;
        println!("breakdown n={} sigma {:.3} ms  jac {:.3} ms  solve {:.3} ms  remesh {:.3} ms  radius+coarsen {:.3} ms  ({} {} {})", n, ms_sigma, ms_jac, ms_solve, ms_remesh, ms_misc, acc.x, acc2, acc3);
    }

    #[test]
    fn free_fall_conserves_volume_and_accelerates() {
        let mut rod = bent_rod();
        let v0 = rod.total_volume();
        let mut out = RodOut { landings: vec![], drops: vec![], fragments: vec![] };
        for _ in 0..200 {
            rod.step(2e-4, &Flat, &mut out, DVec3::new(0.0, 0.0, -G));
        }
        rod.drain_landings(4e-4, true, &mut out.landings);
        let landed: f64 = out.landings.iter().map(|l| l.seg.vol).sum();
        assert!(((rod.total_volume() + landed) - v0).abs() < 1e-12 * v0.max(1e-12) + 1e-15);
        let vz = rod.v.iter().map(|v| v.z).fold(0.0f64, |a, b| a.min(b));
        assert!(vz < -0.3, "vz {}", vz);
    }
}

