use crate::app::*;
use crate::film::{Deposit, Method};
use crate::script::*;
use crate::tools::ToolKind;
use crate::color;
use glam::DVec3;

impl App {
    pub(super) fn scene_knifeart(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let (w, hgt) = (self.film.settings.width_m as f64, self.film.settings.height_m as f64);
        let args: Vec<String> = std::env::args().collect();
        let arg = |k: &str, d: f64| args.windows(2).find(|p| p[0] == k).and_then(|p| p[1].parse::<f64>().ok()).unwrap_or(d);
        self.film.settings.cure_rate = 0.0;
        if f < 40 {
            self.tool.kind = ToolKind::Pour;
            self.tool.flowing = false;
            if f == 2 {
                use rand::{Rng, SeedableRng};
                let mut rng = rand::rngs::StdRng::seed_from_u64(5);
                let mut p = color::PaintMix::pure(1);
                p.frac[3] = 0.6;
                p.frac[12] = 0.0;
                p.frac[13] = 150.0;
                p.frac[14] = (60f32).ln() + 10.0;
                for _ in 0..400 {
                    let cx = w * 0.5 + (rng.random::<f64>() - 0.5) * 0.22;
                    let cy = hgt * 0.5 + (rng.random::<f64>() - 0.5) * 0.16;
                    let ang = rng.random::<f64>() * std::f64::consts::TAU;
                    let len = 0.004 + rng.random::<f64>() * 0.02;
                    let r = 0.002 + rng.random::<f64>() * 0.005;
                    let th = 1e-4 + rng.random::<f64>() * 4e-4;
                    let a = [(cx - 0.5 * len * ang.cos()) as f32, (cy - 0.5 * len * ang.sin()) as f32];
                    let b = [(cx + 0.5 * len * ang.cos()) as f32, (cy + 0.5 * len * ang.sin()) as f32];
                    let vol = (th * (len + 2.0 * r) * 2.0 * r * 0.6) as f32;
                    self.pending_deposits.push(Deposit::capsule(a, b, r as f32, vol, p.normalized().packed(), self.film.dx).laid_by(Method::Pour));
                }
            }
            if f == 30 {
                self.film.dry_now(&self.gpu);
            }
            return true;
        }
        let per = 90u64;
        let k = (f - 40) / per;
        let t = (f - 40) % per;
        let colours: [[f32; 2]; 6] = [[0.0, 0.0], [5.0, 0.0], [0.0, 4.0], [9.0, 0.0], [0.0, 0.0], [5.0, 4.0]];
        self.tool.kind = ToolKind::Knife;
        if k >= 6 {
            self.tool.flowing = false;
            return true;
        }
        if t == 0 {
            let mut p = color::PaintMix::pure(0);
            let c = colours[k as usize];
            if c[0] > 0.0 { p.frac[c[0] as usize] = 0.35; }
            if c[1] > 0.0 { p.frac[c[1] as usize] = 0.25; }
            p.frac[12] = 0.0;
            p.frac[13] = 120.0;
            p.frac[14] = (40f32).ln() + 10.0;
            self.paint = p;
            self.tool.knife_width = 0.04;
            self.tool.knife_height = arg("--gap", 0.3) * 1e-3;
            self.load_knife();
            let fill = arg("--load", 1.0);
            for b in &mut self.knife_bands {
                b.vol *= fill;
            }
            self.knife_dir = [0.0, -1.0];
            self.knife_track = None;
            self.knife_prev = None;
        }
        let x = w * 0.5 - 0.09 + 0.036 * k as f64;
        let s = ((t as f64 - 5.0) / 75.0).clamp(0.0, 1.0);
        self.tool.flowing = (5..80).contains(&t);
        self.tool.pos = DVec3::new(x, hgt * 0.5 + 0.07 - 0.14 * s, 0.0);
        true
    }

    pub(super) fn scene_knife(&mut self, a: &AutoScript) -> bool {
        let f = self.frame_index;
        let w = self.film.settings.width_m as f64;
        let hgt = self.film.settings.height_m as f64;
        let dt = 1.0 / 60.0;
        if f < 90 {
            self.tool.kind = ToolKind::Pour;
            let t0 = f as f64 / 90.0;
            let mut paint = color::PaintMix::pure(3);
            paint.frac[12] = 0.2;
            if a.yield_pa >= 0.0 {
                paint.frac[13] = a.yield_pa;
                paint.frac[12] = 0.0;
            }
            self.knife_test_deposit(t0, w, hgt, dt, paint);
            self.tool.flowing = false;
            return true;
        }
        self.tool.kind = ToolKind::Knife;
        let t = (f - 90) as f64 * dt;
        let speed = 0.12;
        let path = |t: f64| DVec3::new(w * 0.5 - 0.1 + speed * t, hgt * 0.5, 0.0);
        let p = path(t);
        self.tool.vel = (path(t + dt) - p) / dt;
        self.tool.pos = p;
        self.tool.flowing = f > 100 && f < 175;
        true
    }
}
