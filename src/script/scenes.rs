mod photo;
mod pollock_scenes;
mod surface;
mod brushes;
mod tools;
mod knives;
mod threads;
use crate::film::{Deposit, Method};
use crate::{brush, color};
use crate::app::*;

impl App {
    pub(crate) fn scripted_deposits(&mut self, deposits: &mut Vec<Deposit>) {
        let f = self.frame_index;
        let dt = 1.0 / 60.0;
        let w = self.film.settings.width_m;
        let hgt = self.film.settings.height_m;
        if (10..130).contains(&f) {
            let t0 = (f - 10) as f32 / 120.0;
            let t1 = (f - 9) as f32 / 120.0;
            let curve = |t: f32| [w * (0.15 + 0.7 * t), hgt * (0.3 + 0.35 * (t * 6.0).sin() * 0.5 + 0.2 * t)];
            let mut paint = color::PaintMix::pure(1);
            paint.frac[12] = 0.3;
            paint.frac[13] = 0.0;
            deposits.push(Deposit::capsule(curve(t0), curve(t1), 0.0012, 1.5e-7 * dt, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour));
        }
        if (40..100).contains(&f) {
            let t0 = (f - 40) as f32 / 60.0;
            let t1 = (f - 39) as f32 / 60.0;
            let line = |t: f32| [w * (0.2 + 0.6 * t), hgt * (0.7 - 0.3 * t)];
            let paint = color::PaintMix::pure(3);
            deposits.push(Deposit::capsule(line(t0), line(t1), 0.0025, 1.2e-6 * dt, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour));
        }
        if (60..75).contains(&f) {
            let mut paint = color::PaintMix::pure(2);
            paint.frac[0] = 0.3;
            paint.frac[12] = 0.3;
            paint.frac[13] = 0.0;
            deposits.push(Deposit::capsule([w * 0.5, hgt * 0.5], [w * 0.5, hgt * 0.5], 0.02, 6.0e-6 * dt * 4.0, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour));
        }
        if (150..170).contains(&f) {
            let mut paint = color::PaintMix::pure(5);
            paint.frac[12] = 0.1;
            deposits.push(Deposit::capsule([w * 0.5, hgt * 0.5], [w * 0.52, hgt * 0.51], 0.008, 4.0e-6 * dt, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour));
        }
    }

    pub(crate) fn scripted_tool(&mut self) -> bool {
        if self.auto.as_ref().map(|a| a.script == "evolving").unwrap_or(false) {
            static THEN_STEP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let args: Vec<String> = std::env::args().collect();
            if let Some(spec) = args.windows(2).find(|p| p[0] == "--then").map(|p| p[1].clone()) {
                let steps: Vec<&str> = spec.split(',').collect();
                let step = THEN_STEP.load(std::sync::atomic::Ordering::Relaxed);
                if let Some(item) = steps.get(step) {
                    let (shape, rest) = if let Some(r) = item.strip_prefix("round") { (Some(brush::BrushShape::Round), r) } else if let Some(r) = item.strip_prefix("flat") { (Some(brush::BrushShape::Flat), r) } else { (None, *item) };
                    let (mm, n) = rest.split_once('@').map(|(a, b)| (a.parse::<f64>().unwrap_or(0.0), b.parse::<u64>().unwrap_or(u64::MAX))).unwrap_or((0.0, u64::MAX));
                    if mm > 0.0 && self.evo.run.as_ref().map(|ev| ev.painted >= n).unwrap_or(false) {
                        THEN_STEP.store(step + 1, std::sync::atomic::Ordering::Relaxed);
                        if let Some(sh) = shape.filter(|sh| *sh != self.brush.shape) {
                            let (nb, len) = (self.brush.requested, self.brush.len);
                            self.rebuild_brush(sh, nb, len, mm * 1e-3);
                        }
                        let kind = self.tool.kind;
                        self.set_tool_size(kind, mm * 1e-3);
                        log::warn!("then: {:?} {} mm after {} strokes", shape, mm, n);
                        if let Some(a) = self.auto.as_ref() {
                            self.shot_request = Some(a.shot.replace(".png", &format!("_{}.png", step)));
                        }
                    }
                }
            }
        }
        if self.evolve_tick() {
            return true;
        }
        if self.apply_painter() {
            return true;
        }
        let Some(a) = self.auto.clone() else { return false };
        match a.script.as_str() {
            "evolve" => self.scene_evolve(&a),
            "groundbrush" | "groundbucket" => self.scene_groundbrush(&a),
            "pollock" => self.scene_pollock(&a),
            "canvas" => self.scene_canvas(&a),
            "physics" => self.scene_physics(&a),
            "brushlong" => self.scene_brushlong(&a),
            "brushsizes" => self.scene_brushsizes(&a),
            "brushlab" => self.scene_brushlab(&a),
            "brushpress" => self.scene_brushpress(&a),
            "brushcurve" => self.scene_brushcurve(&a),
            "brush4" => self.scene_brush4(&a),
            "brush3" => self.scene_brush3(&a),
            "brush2" => self.scene_brush2(&a),
            "brush" => self.scene_brush(&a),
            "layers" => self.scene_layers(&a),
            "sponge" => self.scene_sponge(&a),
            "tiltoil" => self.scene_tiltoil(&a),
            "mixing" => self.scene_mixing(&a),
            "roller" => self.scene_roller(&a),
            "knifeart" => self.scene_knifeart(&a),
            "knife" => self.scene_knife(&a),
            "lines" => self.scene_lines(&a),
            "flick" => self.scene_flick(&a),
            _ => self.scene_trajectory(&a),
        }
    }

    pub(crate) fn knife_test_deposit(&mut self, t0: f64, w: f64, hgt: f64, dt: f64, paint: color::PaintMix) {
        let c = [w * 0.5 + 0.02 * (t0 * 12.0).sin(), hgt * 0.5 + 0.02 * (t0 * 12.0).cos()];
        let d = Deposit::capsule([c[0] as f32, c[1] as f32], [c[0] as f32, c[1] as f32], 0.012, (8.0e-6 * dt) as f32, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour);
        self.pending_deposits.push(d);
    }
}
