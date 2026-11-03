use crate::app::*;
use crate::film::{Deposit, Method};
use crate::script::*;
use crate::tools::ToolKind;
use crate::{brush, color};
use glam::DVec3;

impl App {
    pub(super) fn scene_brushlong(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let (w, hgt) = (self.film.settings.width_m as f64, self.film.settings.height_m as f64);
        let args: Vec<String> = std::env::args().collect();
        let arg = |k: &str| args.windows(2).find(|p| p[0] == k).and_then(|p| p[1].parse::<f64>().ok());
        let per = 90u64;
        let k = f.saturating_sub(1) / per;
        let t = f.saturating_sub(1) % per;
        let strokes = arg("--strokes").unwrap_or(40.0) as u64;
        if f < 1 || k >= strokes {
            self.tool.flowing = false;
            return true;
        }
        if f == 1 {
            let n = self.brush.requested;
            let len = self.brush.len;
            self.rebuild_brush(brush::BrushShape::Round, n, len, arg("--size").unwrap_or(16.0) * 1e-3);
            self.tool.kind = ToolKind::Brush;
            self.prev_kind = ToolKind::Brush;
        }
        if t == 0 && k > 0 && args.iter().any(|a| a == "--resize") {
            let size = if (k / 3) % 2 == 0 { arg("--size").unwrap_or(16.0) } else { 6.0 } * 1e-3;
            let (n, len) = (self.brush.requested, self.brush.len);
            self.rebuild_brush(brush::BrushShape::Round, n, len, size);
        }
        let reload = arg("--reload").unwrap_or(5.0) as u64;
        if t == 0 && k % reload == 0 {
            let mut p = color::PaintMix::pure(1);
            p.frac[13] = 40.0;
            p.frac[14] = (20f32).ln() + 10.0;
            self.brush.clean();
            self.brush.dip(&p.frac);
            self.paint.frac = p.frac;
        }
        self.tool.height = self.brush.len - self.brush_press(arg("--pressure").unwrap_or(0.5));
        let ang = (k as f64 * 2.399963).rem_euclid(std::f64::consts::TAU);
        let cx = w * 0.5 + 0.25 * ((k % 7) as f64 / 6.0 - 0.5);
        let cy = hgt * 0.5 + 0.2 * ((k / 7 % 6) as f64 / 5.0 - 0.5);
        let s = ((t as f64 - 10.0) / 70.0).clamp(0.0, 1.0) - 0.5;
        self.tool.flowing = (10..80).contains(&t);
        self.tool.pos = DVec3::new(cx + 0.08 * s * ang.cos(), cy + 0.08 * s * ang.sin(), self.tool.height);
        if t == 45 {
            let b = &self.brush;
            let shape: f64 = b.bristles.iter().map(|br| (br.pts[b.segs] - br.pts[0]).truncate().length()).sum::<f64>() / b.bristles.len() as f64;
            log::warn!("LONG stroke {} touching {} load {:.2} hold {:.2} mm tip spread {:.2} mm", k, b.touching, b.total_load() / b.total_capacity(), self.brush_hold * 1e3, shape * 1e3);
        }
        if t == 5 {
            let b = &self.brush;
            let rest: f64 = b.bristles.iter().map(|br| {
                let mut q = br.pts[0];
                let mut d = 0.0f64;
                for i in 1..=b.segs {
                    q += br.dirs[i - 1] * br.seg;
                    d = d.max((br.pts[i] - q).length());
                }
                d
            }).fold(0.0, f64::max);
            let bad = b.bristles.iter().any(|br| br.pts.iter().any(|p| !p.is_finite()));
            log::warn!("LONG lifted {} worst rest error {:.2} mm nan {}", k, rest * 1e3, bad);
        }
        true
    }

    pub(super) fn scene_brushsizes(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let (w, hgt) = (self.film.settings.width_m as f64, self.film.settings.height_m as f64);
        let args: Vec<String> = std::env::args().collect();
        let arg = |k: &str| args.windows(2).find(|p| p[0] == k).and_then(|p| p[1].parse::<f64>().ok());
        let sizes = [1.0, 2.0, 4.0, 8.0];
        let presses = [0.05, 0.15, 0.3, 0.15];
        let per = 120;
        let k = (f.saturating_sub(1) / per) as usize;
        let t = f.saturating_sub(1) % per;
        let (si, pi) = (k / presses.len(), k % presses.len());
        if f < 1 || si >= sizes.len() {
            self.tool.flowing = false;
            return true;
        }
        let size = sizes[si] * 1e-3;
        if t == 0 {
            let len = arg("--len").map(|v| v * 1e-3).unwrap_or(self.brush.len);
            let n = self.brush.requested;
            if (self.brush.size() - size).abs() > 1e-6 || (self.brush.len - len).abs() > 1e-6 {
                self.rebuild_brush(brush::BrushShape::Round, n, len, size);
            }
            self.tool.kind = ToolKind::Brush;
            self.prev_kind = ToolKind::Brush;
            let mut p = color::PaintMix::pure(1);
            p.frac[13] = 40.0;
            p.frac[14] = (20f32).ln() + 10.0;
            self.brush.clean();
            self.brush.dip(&p.frac);
            self.paint.frac = p.frac;
        }
        self.tool.height = self.brush.len - self.brush_press(presses[pi] / PEN_BRUSH_PRESS);
        let mut y0 = hgt * 0.5 - 0.045;
        for s in &sizes[..si] {
            y0 += 4.0 * (s * 1e-3 * 2.5).max(0.004);
        }
        let y = y0 + pi as f64 * (size * 2.5).max(0.004);
        let s = ((t as f64 - 10.0) / 100.0).clamp(0.0, 1.0);
        self.tool.flowing = (10..110).contains(&t);
        let x = w * 0.5 - 0.035 + 0.07 * s;
        let wave = if pi == 3 { (s * std::f64::consts::TAU).sin() * (size * 1.5).max(0.003) } else { 0.0 };
        self.tool.pos = DVec3::new(x, y + wave, self.tool.height);
        true
    }

    pub(super) fn scene_brushlab(&mut self, _a: &AutoScript) -> bool {
        const SPECS: [(brush::BrushShape, f64); 6] = [(brush::BrushShape::Round, 3.0), (brush::BrushShape::Round, 8.0), (brush::BrushShape::Round, 20.0), (brush::BrushShape::Flat, 5.0), (brush::BrushShape::Flat, 20.0), (brush::BrushShape::Flat, 70.0)];
        const PRESSURES: [f64; 3] = [0.15, 0.5, 1.0];
        let f = self.frame_index;
        let period = 200u64;
        let k = (f.saturating_sub(1) / period) as usize;
        let t = f.saturating_sub(1) % period;
        let (row, col) = (k / PRESSURES.len(), k % PRESSURES.len());
        if row >= SPECS.len() {
            self.tool.flowing = false;
            self.tool.pos = DVec3::new(0.02, 0.02, 0.2);
            return true;
        }
        self.tool.kind = ToolKind::Brush;
        self.prev_kind = ToolKind::Brush;
        let (shape, mut mm) = SPECS[row];
        if row == SPECS.len() - 1 {
            mm = std::env::args().collect::<Vec<_>>().windows(2).find(|p| p[0] == "--widest").and_then(|p| p[1].parse().ok()).unwrap_or(mm);
        }
        if t == 0 {
            let (n, len) = (self.brush.requested, self.brush.len);
            self.rebuild_brush(shape, n, len, mm * 1e-3);
            let mut p = color::PaintMix::pure(3);
            p.frac[0] = 0.6;
            if std::env::args().any(|a| a == "--black") {
                p = color::PaintMix::pure(1);
            }
            p.frac[13] = 40.0;
            p.frac[14] = (20f32).ln() + 10.0;
            if std::env::args().any(|a| a == "--thinpaint") {
                p.frac[12] = 0.3;
                p.frac[13] = 0.0;
                p.frac[14] = (self.tool.mu as f32).ln() + 10.0;
            }
            let dry = std::env::args().collect::<Vec<_>>().windows(2).find(|p| p[0] == "--dry").and_then(|p| p[1].parse::<f64>().ok()).unwrap_or(1.0);
            self.brush.capacity = self.brush.base_capacity * (mm * 1e-3 / 0.016).clamp(1.0, 10.0) * dry;
            self.brush.clean();
            self.brush.dip(&p.frac);
            self.paint.frac = p.frac;
        }
        self.tool.height = self.brush.len - self.brush_press(PRESSURES[col]);
        let (x0, y0) = (0.07 + col as f64 * 0.38, 0.07 + row as f64 * 0.125);
        let (x, y, down) = match t {
            0..=9 => (x0 - 0.03 + 0.003 * t as f64, y0, false),
            10..=59 => (x0, y0, true),
            60..=79 => (x0 + 0.06 * (t - 60) as f64 / 20.0, y0, false),
            80..=179 => {
                let s = (t - 80) as f64 / 100.0;
                let bend = if s > 0.6 { 0.03 * ((s - 0.6) / 0.4) * ((s - 0.6) / 0.4) } else { 0.0 };
                (x0 + 0.06 + 0.22 * s, y0 + bend, true)
            }
            _ => (x0 + 0.28, y0 + 0.03, false),
        };
        self.tool.flowing = down;
        self.tool.pos = DVec3::new(x, y, self.tool.height);
        true
    }

    pub(super) fn scene_brushpress(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let (w, hgt) = (self.film.settings.width_m as f64, self.film.settings.height_m as f64);
        let args: Vec<String> = std::env::args().collect();
        let arg = |k: &str| args.windows(2).find(|p| p[0] == k).and_then(|p| p[1].parse::<f64>().ok());
        if f == 1 {
            let (n, len) = (self.brush.requested, arg("--len").map(|v| v * 1e-3).unwrap_or(self.brush.len));
            self.rebuild_brush(brush::BrushShape::Round, n, len, arg("--size").unwrap_or(16.0) * 1e-3);
            self.tool.kind = ToolKind::Brush;
            self.prev_kind = ToolKind::Brush;
        }
        let k = (f / 100) as usize;
        let t = f % 100;
        let presses = [0.5, 1.0, 2.0, 4.0, 7.0, 11.0];
        if k >= presses.len() || f < 1 {
            self.tool.flowing = false;
            return true;
        }
        let press = presses[k] * 1e-3 * self.brush.len / 0.035;
        if t == 0 {
            let mut p = color::PaintMix::pure(1);
            p.frac[13] = 40.0;
            p.frac[14] = (20f32).ln() + 10.0;
            self.brush.clean();
            self.brush.dip(&p.frac);
            self.paint.frac = p.frac;
        }
        self.tool.height = self.brush.len - press;
        let y = hgt * 0.5 + (k as f64 - 2.5) * 0.04;
        let s = ((t as f64 - 10.0) / 80.0).clamp(0.0, 1.0);
        self.tool.flowing = (10..90).contains(&t);
        self.tool.pos = DVec3::new(w * 0.5 - 0.1 + 0.2 * s, y, self.tool.height);
        true
    }

    pub(super) fn scene_brushcurve(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let (w, hgt) = (self.film.settings.width_m as f64, self.film.settings.height_m as f64);
        let args: Vec<String> = std::env::args().collect();
        let arg = |k: &str| args.windows(2).find(|p| p[0] == k).and_then(|p| p[1].parse::<f64>().ok());
        if f == 1 {
            let shape = if args.iter().any(|a| a == "--flat") { brush::BrushShape::Flat } else { brush::BrushShape::Round };
            let (n, len) = (self.brush.requested, self.brush.len);
            self.rebuild_brush(shape, n, len, arg("--size").unwrap_or(30.0) * 1e-3);
            self.tool.kind = ToolKind::Brush;
            self.prev_kind = ToolKind::Brush;
            self.tool.height = self.brush.len * arg("--rest").unwrap_or(0.75);
            let mut p = color::PaintMix::pure(1);
            p.frac[13] = 40.0;
            p.frac[14] = (20f32).ln() + 10.0;
            self.brush.dip(&p.frac);
            self.paint.frac = p.frac;
            self.tool.infinite = args.iter().any(|a| a == "--infinite");
        }
        let t = (f as f64 - 20.0) / 60.0;
        self.tool.flowing = f >= 20 && t < 4.0;
        let a = t.clamp(0.0, 4.0) * std::f64::consts::FRAC_PI_2 * arg("--turns").unwrap_or(1.0);
        let h = self.tool.height;
        let pos = DVec3::new(w * 0.5 + 0.2 * (a - std::f64::consts::PI * 0.5) / std::f64::consts::PI, hgt * 0.5 + 0.15 * a.sin(), h);
        if f % 20 == 0 {
            let v = (pos - self.tool.pos) * 60.0;
            log::warn!("curve f {} dir {:.0} angle {:.0} load {:.3}", f, (v.y.atan2(v.x) + std::f64::consts::FRAC_PI_2).to_degrees(), self.brush.angle.to_degrees(), self.brush.total_load() / self.brush.total_capacity());
        }
        self.tool.pos = pos;
        true
    }

    pub(super) fn scene_brush4(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let w = self.film.settings.width_m as f64;
        let hgt = self.film.settings.height_m as f64;
        let cx = w * 0.5;
        let cy = hgt * 0.5;
        if f < 200 {
            self.tool.kind = ToolKind::Bucket;
            self.prev_kind = ToolKind::Bucket;
            self.tool.height = 0.15;
            self.tool.pos = DVec3::new(cx, cy, 0.15);
            self.tool.vel = DVec3::ZERO;
            self.tool.flowing = f > 10 && f < 130;
        } else {
            self.tool.kind = ToolKind::Brush;
            if self.prev_kind != ToolKind::Brush { self.tool.height = self.brush.len * 0.75; }
            self.prev_kind = ToolKind::Brush;
            let t = (f as f64 - 220.0) / 60.0;
            self.tool.flowing = f >= 220 && t < 3.0;
            self.tool.pos = DVec3::new(cx - 0.15 + 0.1 * t.max(0.0), cy, self.tool.height);
        }
        true
    }

    pub(super) fn scene_brush3(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let w = self.film.settings.width_m as f64;
        let hgt = self.film.settings.height_m as f64;
        let cx = w * 0.5;
        let cy = hgt * 0.5;
        if f < 240 {
            self.tool.kind = ToolKind::Bottle;
            self.prev_kind = ToolKind::Bottle;
            self.tool.height = 0.15;
            self.tool.mu = 8.0;
            self.tool.sigma = 0.0585;
            self.paint.frac[12] = 0.3;
            self.paint.frac[13] = 35.0;
            self.tool.pos = DVec3::new(cx, cy, 0.15);
            self.tool.vel = DVec3::ZERO;
            self.tool.flowing = f > 10 && f < 200;
        } else {
            self.tool.kind = ToolKind::Brush;
            if self.prev_kind != ToolKind::Brush { self.tool.height = self.brush.len * 0.6; }
            self.prev_kind = ToolKind::Brush;
            self.tool.mu = 20.0;
            if f == 260 {
                let mut p = color::PaintMix::pure(0);
                p.frac[13] = 40.0;
                p.frac[14] = (20f32).ln() + 10.0;
                self.brush.dip(&p.frac);
            }
            let t = (f as f64 - 270.0) / 60.0;
            self.tool.flowing = f >= 270 && t < 2.0;
            self.tool.pos = DVec3::new(cx - 0.08 + 0.08 * t.max(0.0), cy, self.tool.height);
        }
        true
    }

    pub(super) fn scene_brush2(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let w = self.film.settings.width_m as f64;
        let hgt = self.film.settings.height_m as f64;
        self.tool.kind = ToolKind::Brush;
        self.prev_kind = ToolKind::Brush;
        self.tool.height = 0.021;
        self.tool.mu = 20.0;
        if f == 10 {
            let mut p = color::PaintMix::pure(1);
            p.frac[13] = 40.0;
            p.frac[14] = (20f32).ln() + 10.0;
            self.brush.dip(&p.frac);
        }
        let t = (f as f64 - 20.0) / 60.0;
        self.tool.flowing = f >= 20 && t < 2.0;
        self.tool.pos = DVec3::new(w * 0.5 - 0.08 + 0.08 * t.max(0.0), hgt * 0.5, 0.021);
        true
    }

    pub(super) fn scene_brush(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let w = self.film.settings.width_m as f64;
        let hgt = self.film.settings.height_m as f64;
        self.tool.kind = ToolKind::Brush;
        self.prev_kind = ToolKind::Brush;
        self.tool.height = 0.021;
        self.tool.mu = 20.0;
        let lnmu = (20f32).ln() + 10.0;
        if f < 40 {
            let mut paint = color::PaintMix::pure(1);
            paint.frac[12] = 0.0;
            paint.frac[13] = 40.0;
            paint.frac[14] = lnmu;
            let y0 = (hgt * 0.5 - 0.05 + f as f64 * 0.0015) as f32;
            let d = Deposit::stroke([(w * 0.5 - 0.05) as f32, y0], [(w * 0.5 + 0.05) as f32, y0], 0.006, 2.4e-7, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour);
            self.pending_deposits.push(d);
            self.tool.flowing = false;
            self.tool.pos = DVec3::new(w * 0.5, hgt * 0.5 - 0.1, 0.021);
        } else {
            if f == 60 {
                let mut p = color::PaintMix::pure(0);
                p.frac[13] = 40.0;
                p.frac[14] = lnmu;
                self.brush.dip(&p.frac);
            }
            let t = (f as f64 - 60.0) / 60.0;
            self.tool.flowing = f >= 60 && t < 2.0;
            let lane = if t < 2.0 { 0.0 } else { 0.02 };
            let tt = if t < 2.0 { t.max(0.0) } else { (t - 2.3).max(0.0) };
            self.tool.pos = DVec3::new(w * 0.5 + lane * 2.0, hgt * 0.5 - 0.09 + 0.08 * tt, 0.021);
            self.tool.flowing = self.tool.flowing || (t >= 2.3 && t < 4.3);
        }
        true
    }
}
