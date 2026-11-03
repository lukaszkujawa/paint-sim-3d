use crate::session;
use std::time::Instant;
use crate::app::*;
use crate::tools::ToolKind;
use crate::{brush, color};

impl App {
    pub(crate) fn scripted_input(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.input_eyedrop(size);
        self.input_undo(size);
        self.input_cursor(size);
    }

    fn input_eyedrop(&mut self, _size: winit::dpi::PhysicalSize<u32>) {
        if self.auto.as_ref().map(|a| a.script == "eyedrop").unwrap_or(false) {
            const WANTED: [[f32; 3]; 5] = [[0.5, 0.5, 0.5], [0.6, 0.3, 0.2], [0.2, 0.35, 0.6], [0.85, 0.8, 0.7], [0.25, 0.25, 0.25]];
            let size = self.window.inner_size();
            let f = self.frame_index;
            let y_of = |k: usize| 0.2 + 0.13 * k as f32;
            let span = 230u64;
            self.tool.kind = ToolKind::Brush;
            let at = |y: f32, s: f32| [size.width as f32 * (0.3 + 0.35 * s), size.height as f32 * y];
            let mut down = false;
            if f == 5 {
                let n = self.brush.requested;
                let len = self.brush.len;
                self.rebuild_brush(brush::BrushShape::Round, n, len, 0.03);
            }
            if f >= 10 && f < 10 + span * 5 {
                let k = ((f - 10) / span) as usize;
                let t = (f - 10) % span;
                if t == 0 {
                    self.use_colour(WANTED[k]);
                    log::warn!("eyedrop row {} wanted {:?} paint {:?} shown {:?} at {:?}", k, WANTED[k], color::mix_rgb(&self.film.pigments, &self.film.cmf_table, &self.paint.frac), self.shown(color::mix_rgb(&self.film.pigments, &self.film.cmf_table, &self.paint.frac)), at(y_of(k), 0.5));
                }
                if t < 210 {
                    if t % 70 == 0 {
                        self.reload_tool();
                    }
                    self.cursor = at(y_of(k), (t % 70) as f32 / 70.0);
                    down = t % 70 < 62;
                }
            }
            let done = 10 + span * 5;
            if f == done + 10 {
                self.paint.frac = color::PaintMix::pure(0).frac;
                self.reload_tool();
                self.eyedropper = true;
                self.cursor = at(y_of(1), 0.5);
            }
            if f == done + 40 {
                let seen = self.canvas_colour_at_cursor();
                self.eyedrop_click();
                log::warn!("eyedrop sampled {:?} mixed {:?} still picking {} view {:?}", seen, color::mix_rgb(&self.film.pigments, &self.film.cmf_table, &self.paint.frac), self.eyedropper, self.view_rect);
            }
            if f > done + 50 && f <= done + 120 {
                self.cursor = at(0.87, (f - done - 50) as f32 / 70.0);
                down = true;
            }
            if f == done + 130 {
                self.cursor = at(y_of(2), 0.5);
                self.eyedropper = true;
            }
            self.mouse_down[0] = down;
        }
    }

    fn input_undo(&mut self, _size: winit::dpi::PhysicalSize<u32>) {
        if let Some(script) = self.auto.as_ref().map(|a| a.script.clone()).filter(|s| s.starts_with("undo")) {
            let size = self.window.inner_size();
            let f = self.frame_index;
            self.tool.kind = ToolKind::Pour;
            self.pour_radius = 0.01;
            self.pour_rate = 2.0e-5;
            let t = |a: u64, b: u64| (f.saturating_sub(a)) as f32 / (b - a) as f32;
            if (10..60).contains(&f) {
                self.paint = color::PaintMix::pure(3);
                self.cursor = [size.width as f32 * (0.35 + 0.1 * t(10, 60)), size.height as f32 * 0.45];
            } else if (100..150).contains(&f) {
                self.paint = color::PaintMix::pure(5);
                self.cursor = [size.width as f32 * (0.3 + 0.4 * t(100, 150)), size.height as f32 * (0.38 + 0.14 * t(100, 150))];
            }
            self.mouse_down[0] = (10..60).contains(&f) || (100..150).contains(&f);
            if script == "undosave" {
                if f == 199 {
                    self.paused = true;
                    let t0 = Instant::now();
                    self.save_painting("roundtrip");
                    log::warn!("save took {:.2} s: {}", t0.elapsed().as_secs_f32(), self.last_beauty);
                }
                if f == 210 {
                    self.clear_canvas();
                }
                if f == 220 {
                    let t0 = Instant::now();
                    self.open_painting(&session::painting_path("roundtrip"));
                    self.paused = true;
                    log::warn!("open took {:.2} s: {}", t0.elapsed().as_secs_f32(), self.last_beauty);
                }
                if f == 225 {
                    self.save_painting("roundtrip2");
                }
            }
            if script.starts_with("undorubber") {
                if f == 5 {
                    let args: Vec<String> = std::env::args().collect();
                    if let Some(v) = args.windows(2).find(|p| p[0] == "--canvas").map(|p| p[1].clone()) {
                        let c: Vec<f32> = v.split(',').filter_map(|t| t.parse().ok()).collect();
                        self.start_canvas(CanvasChoice { res: self.film.w, colour: (c.len() >= 3).then(|| [c[0], c[1], c[2]]), weave: true, custom: [0.9; 3], first: false });
                    }
                }
                if f == 160 && script == "undorubber" {
                    self.film.dry_now(&self.gpu);
                }
                if f >= 175 {
                    self.tool.kind = ToolKind::Rubber;
                    self.rubber_size = 0.06;
                }
                if (180..240).contains(&f) {
                    let s = (f - 180) as f32 / 60.0;
                    self.cursor = [size.width as f32 * (0.25 + 0.5 * s), size.height as f32 * 0.45];
                    self.mouse_down[0] = true;
                }
            }
            if script == "undoclear" && f == 200 {
                self.begin_full();
                self.clear_canvas();
            }
            if f == 260 && script != "undosave" {
                self.undo();
            }
            if script == "undotest2" && f == 300 {
                self.undo();
            }
        }
    }

    fn input_cursor(&mut self, _size: winit::dpi::PhysicalSize<u32>) {
        if let Some(a) = &self.auto {
            if a.script == "hold" {
                let size = self.window.inner_size();
                let f = self.frame_index;
                let phase = (f / 420) % 3;
                let off = phase as f32 * 0.06;
                self.cursor = [size.width as f32 * (0.55 + off), size.height as f32 * 0.5];
                self.mouse_down[0] = f > 10 && (f % 420) < 360;
                self.tool.kind = ToolKind::Bottle;
                self.tool.height = 0.15;
                self.tool.mu = 8.0;
                self.tool.sigma = 0.0585;
                self.paint.frac[12] = 0.3;
                self.paint.frac[13] = 0.0;
                if a.yield_pa >= 0.0 { self.paint.frac[13] = a.yield_pa; self.paint.frac[12] = 0.0; }
                if a.pig >= 0 { let y = self.paint.frac[13]; let th = self.paint.frac[12]; self.paint = color::PaintMix::pure(a.pig as usize); self.paint.frac[13] = y; self.paint.frac[12] = th; }
                if a.mu > 0.0 { self.tool.mu = a.mu; }
            }
            if a.script == "pour" {
                let size = self.window.inner_size();
                let f = self.frame_index;
                let t = f as f32 / 60.0;
                let x = if f < 130 { 0.5 } else if f < 400 { 0.5 + 0.05 * (t - 130.0 / 60.0) } else { 0.5 + 0.05 * 4.5 + 0.3 * (t - 400.0 / 60.0) };
                let y = if f < 400 { 0.5 } else { 0.5 + 0.06 * ((t - 400.0 / 60.0) * 3.0).sin() };
                self.cursor = [size.width as f32 * x.min(0.95), size.height as f32 * y];
                self.mouse_down[0] = f > 10 && f < 560;
                self.tool.kind = ToolKind::Pour;
                self.tool.mu = if a.mu > 0.0 { a.mu } else { 4.0 };
                self.paint.frac[12] = 0.3;
                self.paint.frac[13] = if a.yield_pa >= 0.0 { a.yield_pa } else { 0.0 };
            }
            if a.script == "settle" {
                let size = self.window.inner_size();
                let f = self.frame_index;
                self.cursor = [size.width as f32 * 0.55, size.height as f32 * 0.5];
                self.mouse_down[0] = f > 10 && f < 300;
                self.tool.kind = ToolKind::Bottle;
                self.tool.height = 0.15;
                self.tool.mu = 8.0;
                self.tool.sigma = 0.0585;
                self.paint.frac[12] = 0.3;
                self.paint.frac[13] = 0.0;
                if a.yield_pa >= 0.0 { self.paint.frac[13] = a.yield_pa; self.paint.frac[12] = 0.0; }
                if a.mu > 0.0 { self.tool.mu = a.mu; }
            }
            if a.script == "stress2" {
                let size = self.window.inner_size();
                let f = self.frame_index;
                let t = f as f32 / 60.0;
                let seed = ((f as u64).wrapping_mul(2654435761) >> 7) as f32 / 4294967296.0 * 32.0;
                let jump = (f % 17 == 0) as u32 as f32;
                self.cursor = [
                    size.width as f32 * (0.55 + 0.3 * (t * 2.1 + seed).sin() + 0.2 * jump * (seed * 3.0).sin()),
                    size.height as f32 * (0.5 + 0.35 * (t * 3.7 + seed * 0.5).cos()),
                ];
                self.mouse_down[0] = f > 10 && (f / 70) % 3 != 2;
                self.tool.kind = if (f / 300) % 2 == 0 { ToolKind::Stick } else { ToolKind::Bottle };
                self.tool.height = 0.05 + 0.4 * ((t * 0.9).sin() * 0.5 + 0.5) as f64;
                self.tool.mu = 0.05;
                self.paint.frac[12] = 0.8;
                self.paint.frac[13] = 0.0;
                if f % 200 == 0 { self.tool.reload(); }
            }
            if a.script == "stress" {
                let size = self.window.inner_size();
                let f = self.frame_index;
                let t = f as f32 / 60.0;
                let jerk = ((t * 7.3).sin() * (t * 3.1).cos()).abs();
                self.cursor = [size.width as f32 * (0.55 + 0.3 * (t * 2.1).sin() * jerk), size.height as f32 * (0.5 + 0.35 * (t * 3.7).cos())];
                self.mouse_down[0] = f > 10 && (f / 90) % 4 != 3;
                self.tool.kind = ToolKind::Stick;
                self.tool.mu = 0.5;
                self.paint.frac[12] = 0.6;
                self.paint.frac[13] = 0.0;
                if f % 240 == 0 { self.tool.reload(); }
                if f % 500 == 250 { self.tool.kind = ToolKind::Bottle; }
            }
            if a.script == "mouse" {
                let size = self.window.inner_size();
                let f = self.frame_index as f32;
                let t = (f / 60.0).min(4.0);
                self.cursor = [size.width as f32 * (0.45 + 0.08 * (t * 1.5).sin()), size.height as f32 * (0.5 + 0.1 * (t * 2.0).cos())];
                self.mouse_down[0] = f > 10.0 && f < 230.0;
            }
        }
    }
}
