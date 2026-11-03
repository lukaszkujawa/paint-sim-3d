use crate::app::*;
use crate::film::{Deposit, Method};
use crate::script::*;
use crate::tools::ToolKind;
use crate::color;
use glam::DVec3;

impl App {
    pub(super) fn scene_layers(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let w = self.film.settings.width_m as f64;
        let hgt = self.film.settings.height_m as f64;
        let dt = 1.0 / 60.0;
        self.tool.kind = ToolKind::Pour;
        self.tool.flowing = false;
        if f < 60 {
            let t0 = f as f64 / 60.0;
            let mut paint = color::PaintMix::pure(2);
            paint.frac[12] = 0.1;
            self.knife_test_deposit(t0, w, hgt, dt, paint);
        }
        if (200..260).contains(&f) {
            let t0 = (f - 200) as f64 / 60.0;
            let t1 = (f - 199) as f64 / 60.0;
            let line = |t: f64| [(w * 0.5 - 0.06 + 0.12 * t) as f32, (hgt * 0.5 - 0.03 + 0.06 * t) as f32];
            let mut paint = color::PaintMix::pure(5);
            paint.frac[12] = 0.2;
            let d = Deposit::stroke(line(t0), line(t1), 0.003, (1.5e-6 * dt) as f32, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour);
            self.pending_deposits.push(d);
        }
        true
    }

    pub(super) fn scene_sponge(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        self.tool.kind = ToolKind::Sponge;
        if f == 1 {
            let mut blue = color::PaintMix::pure(5);
            blue.frac[12] = 0.2;
            blue.frac[14] = (4f32).ln() + 10.0;
            for k in 0..12 {
                let y = 0.18 + 0.012 * k as f32;
                self.pending_deposits.push(Deposit::stroke([0.82, y], [1.08, y], 0.008, 2.4e-6, blue.normalized().packed(), self.film.dx).laid_by(Method::Pour));
            }
            let mut yellow = color::PaintMix::pure(2);
            yellow.frac[12] = 0.2;
            yellow.frac[14] = (6f32).ln() + 10.0;
            self.sponge.dip(&yellow.frac);
        }
        self.tool.flowing = false;
        let press = |at: (f64, f64), k0: u64, len: u64, pos: &mut DVec3, flowing: &mut bool| {
            if (k0..k0 + len + 12).contains(&f) {
                *pos = DVec3::new(at.0, at.1, 0.0);
                *flowing = (k0 + 6..k0 + 6 + len).contains(&f);
            }
        };
        let dabs = [(0.15, 0.6), (0.24, 0.62), (0.2, 0.52), (0.31, 0.55), (0.12, 0.45)];
        for (k, at) in dabs.iter().enumerate() {
            press(*at, 10 + k as u64 * 26, 12, &mut self.tool.pos, &mut self.tool.flowing);
        }
        press((0.33, 0.43), 140, 60, &mut self.tool.pos, &mut self.tool.flowing);
        if f == 215 {
            let mut red = color::PaintMix::pure(3);
            red.frac[12] = 0.2;
            red.frac[14] = (6f32).ln() + 10.0;
            self.sponge.dab_colour(&red.frac);
        }
        let dabs2 = [(0.45, 0.62), (0.53, 0.58), (0.48, 0.5), (0.57, 0.48)];
        for (k, at) in dabs2.iter().enumerate() {
            press(*at, 220 + k as u64 * 26, 12, &mut self.tool.pos, &mut self.tool.flowing);
        }
        if f == 325 {
            let mut yellow = color::PaintMix::pure(2);
            yellow.frac[12] = 0.2;
            yellow.frac[14] = (6f32).ln() + 10.0;
            self.sponge.dip(&yellow.frac);
        }
        if (330..430).contains(&f) {
            let t = (f - 330) as f64 / 100.0;
            self.tool.pos = DVec3::new(0.12 + 0.25 * t, 0.3 + 0.03 * (t * 6.0).sin(), 0.0);
            self.tool.flowing = f > 332;
        }
        if (440..480).contains(&f) {
            let t = (f - 440) as f64 / 40.0;
            self.tool.pos = DVec3::new(0.45 + 0.35 * t, 0.3 - 0.04 * t, 0.0);
            self.tool.flowing = f > 442;
        }
        if f == 485 {
            self.sponge.clean();
        }
        let blots = [(0.88, 0.25), (0.97, 0.27), (1.04, 0.22)];
        for (k, at) in blots.iter().enumerate() {
            press(*at, 490 + k as u64 * 26, 12, &mut self.tool.pos, &mut self.tool.flowing);
        }
        true
    }

    pub(super) fn scene_tiltoil(&mut self, a: &AutoScript) -> bool {
        let f = self.frame_index;
        let w = self.film.settings.width_m as f64;
        let hgt = self.film.settings.height_m as f64;
        self.tool.kind = ToolKind::Bucket;
        self.tool.mu = if a.mu > 0.0 { a.mu } else { 40.0 };
        self.paint = color::PaintMix::pure(if a.pig >= 0 { a.pig as usize } else { 3 });
        self.paint.frac[12] = 0.0;
        self.paint.frac[13] = if a.yield_pa >= 0.0 { a.yield_pa } else { 250.0 };
        let t = f as f64 / 60.0;
        let span = 3.0 * a.seed.max(1) as f64;
        let tilt_at = (60.0 * span) as u64 + 60;
        if a.seed > 0 {
            self.tool.pos = DVec3::new(w * (0.25 + 0.5 * (t / span).min(1.0)), hgt * (0.6 + 0.08 * (t * 7.5 / span).sin()), 0.15);
            self.tool.flowing = f > 5 && (f as f64) < 5.0 + 60.0 * span;
        } else {
            self.tool.pos = DVec3::new(w * 0.5, hgt * 0.6, 0.15);
            self.tool.flowing = f > 5 && f < 90;
        }
        self.tool.vel = DVec3::ZERO;
        if f < tilt_at {
            self.film.settings.tilt_deg = 0.0;
        }
        if f == tilt_at {
            self.film.settings.tilt_deg = if a.tilt > 0.0 { a.tilt } else { 90.0 };
            self.film.settings.tilt_dir_deg = -90.0;
        }
        true
    }

    pub(super) fn scene_mixing(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let dt = 1.0 / 60.0;
        let lnmu = (4f32).ln() + 10.0;
        let paint = |parts: &[(usize, f32)]| {
            let mut m = color::PaintMix { frac: [0.0; 16] };
            for (i, v) in parts {
                m.frac[*i] = *v;
            }
            m.frac[12] = 0.25;
            m.frac[14] = lnmu;
            m.normalized().packed()
        };
        let dx = self.film.dx;
        let bar = |x: f64, y0: f64, y1: f64, pig: [u32; 8]| Deposit::stroke([x as f32, y0 as f32], [x as f32, y1 as f32], 0.012, 3.5e-6, pig, dx).laid_by(Method::Pour);
        if f == 0 {
            let pairs: [(usize, usize, f32); 8] = [(2, 5, 0.5), (2, 6, 0.5), (3, 2, 0.5), (4, 5, 0.5), (3, 7, 0.5), (0, 6, 0.85), (1, 2, 0.2), (0, 8, 0.5)];
            for (k, (a_pig, b_pig, wa)) in pairs.iter().enumerate() {
                let x = (0.1 + 0.143 * k as f64) as f32;
                for (row, pig) in [paint(&[(*a_pig, 1.0)]), paint(&[(*a_pig, *wa), (*b_pig, 1.0 - *wa)]), paint(&[(*b_pig, 1.0)])].iter().enumerate() {
                    let y = (0.72 - 0.08 * row as f64) as f32;
                    self.pending_deposits.push(Deposit::capsule([x, y], [x, y], 0.026, 1.0e-6, *pig, dx).laid_by(Method::Pour));
                }
            }
            self.pending_deposits.push(Deposit::stroke([0.05, 0.25], [0.27, 0.25], 0.015, 2.5e-6, paint(&[(2, 1.0)]), dx).laid_by(Method::Pour));
            for x0 in [0.43, 0.73, 1.02] {
                self.pending_deposits.push(bar(x0, 0.12, 0.38, paint(&[(2, 1.0)])));
                self.pending_deposits.push(bar(x0 + 0.06, 0.12, 0.38, paint(&[(5, 1.0)])));
            }
        }
        if f == 20 {
            self.pending_deposits.push(Deposit::stroke([0.16, 0.13], [0.16, 0.37], 0.015, 2.5e-6, paint(&[(5, 1.0)]), dx).laid_by(Method::Pour));
        }
        self.tool.flowing = false;
        if (60..380).contains(&f) {
            self.tool.kind = ToolKind::Knife;
            let pass = (f - 60) / 80;
            let t = ((f - 60) % 80) as f64 / 80.0;
            let ys = [0.2, 0.3, 0.25, 0.25];
            let right = pass % 2 == 0;
            let x = if right { 0.36 + 0.3 * t } else { 0.66 - 0.3 * t };
            self.knife_dir = if right { [1.0, 0.0] } else { [-1.0, 0.0] };
            self.tool.pos = DVec3::new(x, ys[pass as usize], 0.0);
            self.tool.flowing = t > 0.04 && t < 0.96;
        } else if (400..760).contains(&f) {
            self.tool.kind = ToolKind::Brush;
            self.prev_kind = ToolKind::Brush;
            self.tool.height = 0.024;
            let t = (f - 400) as f64 * dt;
            self.tool.pos = DVec3::new(0.76 + 0.09 * (std::f64::consts::TAU * t / 0.8).sin(), 0.13 + 0.04 * t, 0.024);
            self.tool.flowing = true;
        } else if (780..1080).contains(&f) {
            self.tool.kind = ToolKind::Roller;
            let t = (f - 780) as f64 * dt;
            self.tool.pos = DVec3::new(1.05 + 0.1 * (std::f64::consts::TAU * t / 1.5).sin(), 0.22 + 0.012 * t, 0.0);
            self.tool.flowing = true;
        } else {
            self.tool.kind = ToolKind::Pour;
        }
        true
    }

    pub(super) fn scene_roller(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let w = self.film.settings.width_m as f64;
        let hgt = self.film.settings.height_m as f64;
        let dt = 1.0 / 60.0;
        if f < 60 {
            self.tool.kind = ToolKind::Pour;
            self.tool.flowing = false;
            for (i, pig) in [2usize, 3, 5, 7, 11].iter().enumerate() {
                let x = (w * 0.5 - 0.04 + 0.02 * i as f64) as f32;
                let t0 = f as f64 / 60.0;
                let t1 = (f + 1) as f64 / 60.0;
                let y = |t: f64| (hgt * 0.5 + 0.1 + 0.15 * t) as f32;
                let mut paint = color::PaintMix::pure(*pig);
                paint.frac[12] = 0.25;
                self.pending_deposits.push(Deposit::stroke([x, y(t0)], [x, y(t1)], 0.006, 2.5e-8, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour));
            }
            return true;
        }
        self.tool.kind = ToolKind::Roller;
        let t = (f as f64 - 120.0).max(0.0) * dt;
        let p = DVec3::new(w * 0.5, hgt * 0.5 + 0.28 - 0.15 * t, 0.0);
        self.tool.pos = p;
        self.tool.flowing = f > 121 && p.y > hgt * 0.5 - 0.32;
        true
    }
}
