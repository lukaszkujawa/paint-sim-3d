use crate::app::*;
use crate::script::*;
use crate::tools::ToolKind;
use glam::DVec3;

impl App {
    pub(super) fn scene_lines(&mut self, a: &AutoScript) -> bool {
        let f = self.frame_index;
        let w = self.film.settings.width_m as f64;
        let hgt = self.film.settings.height_m as f64;
        let dt = 1.0 / 60.0;
        self.tool.kind = ToolKind::Bottle;
        self.tool.height = 0.1;
        self.tool.mu = if a.mu > 0.0 { a.mu } else { 10.0 };
        self.tool.bottle_flow = 1.8e-6;
        self.tool.bottle_radius = 0.0025;
        self.paint.frac = [0.0; 16];
        self.paint.frac[1] = 1.0;
        self.paint.frac[12] = 0.2;
        let speeds = [0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0];
        let len = w - 0.3;
        let pause = 0.4;
        let t = f as f64 * dt;
        let mut t0 = 0.5;
        let mut cur = None;
        for (i, s) in speeds.iter().enumerate() {
            let dur = len / s;
            if t >= t0 && t < t0 + dur + pause {
                cur = Some((i, *s, t - t0, dur));
                break;
            }
            t0 += dur + pause;
        }
        let was_flowing = self.tool.flowing;
        match cur {
            Some((i, s, tl, dur)) => {
                let y = hgt * (0.08 + 0.14 * i as f64);
                let on = tl < dur;
                let x = 0.15 + s * tl.min(dur);
                self.tool.pos = DVec3::new(x, y, 0.1);
                self.tool.vel = DVec3::new(if on { s } else { 0.0 }, 0.0, 0.0);
                self.tool.flowing = on;
            }
            None => {
                self.tool.flowing = false;
            }
        }
        if was_flowing && !self.tool.flowing {
            self.air.detach();
        }
        true
    }

    pub(super) fn scene_flick(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let w = self.film.settings.width_m as f64;
        let hgt = self.film.settings.height_m as f64;
        let dt = 1.0 / 60.0;
        self.tool.kind = ToolKind::Stick;
        self.tool.height = 0.3;
        self.tool.mu = 0.3;
        self.paint.frac[12] = 0.7;
        self.paint.frac[13] = 0.0;
        let t = f as f64 * dt;
        let swing = |t: f64| {
            let u = (t / 0.35).min(1.0);
            let s = u * u * (3.0 - 2.0 * u);
            DVec3::new(w * 0.3 + 0.25 * s, hgt * 0.5, 0.3)
        };
        let p = swing(t);
        self.tool.vel = (swing(t + dt) - p) / dt;
        self.tool.pos = p;
        self.tool.flowing = f > 2 && t < 0.4;
        true
    }

    pub(super) fn scene_trajectory(&mut self, a: &AutoScript) -> bool {
        if a.script != "jet" && a.script != "coil" && a.script != "bucket" {
            return false;
        }
        let f = self.frame_index;
        let w = self.film.settings.width_m as f64;
        let hgt = self.film.settings.height_m as f64;
        let dt = 1.0 / 60.0;
        let (kind, height, speed) = match a.script.as_str() {
            "jet" => (ToolKind::Stick, 0.15, 0.6),
            "coil" => (ToolKind::Bottle, 0.12, 0.02),
            _ => (ToolKind::Bucket, 0.25, 0.15),
        };
        self.tool.kind = kind;
        self.tool.height = height;
        let t = f as f64 * dt;
        let amp = (0.45 * w).min(speed * 0.6);
        let om = speed / amp.max(1e-3);
        let path = |t: f64| DVec3::new(w * 0.5 + amp * (om * t).sin(), hgt * 0.5 + 0.3 * hgt * (2.0 * om * t).sin(), height);
        let p = path(t);
        self.tool.vel = (path(t + dt) - p) / dt;
        self.tool.pos = p;
        self.tool.flowing = f > 5 && f + 1 < a.frames;
        true
    }
}
