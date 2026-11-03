use crate::film::{Deposit, Method};
use crate::tools::ToolKind;
use crate::brush;
use crate::app::*;

const PEN_PRESS_DIAMETERS: f64 = 3.0;

impl App {
    pub(crate) fn brush_press(&self, pressure: f64) -> f64 {
        self.brush.touch_depth() + pressure * (PEN_BRUSH_PRESS * self.brush.len).min(PEN_PRESS_DIAMETERS * self.brush.size())
    }

    pub(crate) fn rebuild_brush(&mut self, shape: brush::BrushShape, n: usize, len: f64, size: f64) {
        let old = &self.brush;
        let (st, fr, rl, radius) = (old.stiffness, old.friction, old.release, old.hair);
        let load = old.total_load();
        let cap = old.total_capacity();
        let mut comp = [0.0f32; 16];
        if load > 0.0 {
            for b in &old.bristles {
                for i in 0..16 {
                    comp[i] += b.comp[i] * (b.load / load) as f32;
                }
            }
        }
        let press = (old.len - self.tool.height).max(0.0);
        let mut b = brush::Brush::build(shape, n, len, size, radius);
        b.stiffness = st;
        b.angle = old.angle;
        b.lean = old.lean;
        b.friction = fr;
        b.release = rl;
        if load > 0.0 && cap > 0.0 {
            b.dip(&comp);
            let each = load / b.bristles.len().max(1) as f64;
            for br in &mut b.bristles {
                br.load = each;
            }
        }
        self.brush = b;
        if self.tool.kind == ToolKind::Brush {
            self.tool.height = (len - press).clamp(0.002, len - 0.0002);
        }
    }

    pub(crate) fn apply_painter(&mut self) -> bool {
        let Some(p) = self.painter.as_mut() else { return false };
        if self.paused {
            return true;
        }
        match p.step(self.sim_dt) {
            Some(c) => {
                self.painter_active = true;
                if self.tool.kind != c.kind {
                    self.tool.reload();
                }
                self.tool.kind = c.kind;
                self.prev_kind = c.kind;
                if c.kind == ToolKind::Brush && c.brush_width > 0.0 {
                    let b = &self.brush;
                    if b.shape != brush::BrushShape::Flat || (b.width - c.brush_width).abs() > 1e-4 || b.requested != c.brush_count {
                        self.rebuild_brush(brush::BrushShape::Flat, c.brush_count, 0.05, c.brush_width);
                    }
                    self.brush.capacity = c.brush_capacity;
                    self.brush.release = c.brush_release;
                }
                self.tool.pos = c.pos;
                self.tool.vel = c.vel;
                self.tool.flowing = c.flowing;
                self.tool.height = if c.kind == ToolKind::Brush { self.brush.len * c.height } else { c.height };
                self.tool.mu = c.mu;
                self.tool.sigma = c.sigma;
                self.tool.stick_diameter = c.stick_diameter;
                self.tool.bottle_flow = c.bottle_flow;
                self.tool.bottle_radius = c.bottle_radius;
                self.tool.bucket_flow = c.bucket_flow;
                self.tool.bucket_radius = c.bucket_radius;
                self.tool.knife_width = c.knife_width;
                self.tool.knife_height = c.knife_height;
                self.paint.frac = c.mix;
                if c.reload {
                    if c.kind == ToolKind::Brush {
                        let mut f = c.mix;
                        f[14] = (c.mu as f32).ln() + 10.0;
                        if c.brush_width > 0.0 {
                            self.brush.dip_uneven(&f, self.frame_index as u64);
                        } else {
                            self.brush.dip(&f);
                        }
                    } else {
                        self.tool.reload();
                    }
                }
                if c.cut {
                    self.air.detach();
                }
                if c.kind == ToolKind::Pour && c.flowing {
                    let p = [c.pos.x as f32, c.pos.y as f32];
                    let p0 = self.painter_last_pour.unwrap_or(p);
                    let vol = (c.pour_rate * self.sim_dt) as f32;
                    let pig = self.paint.normalized().packed();
                    self.pending_deposits.push(Deposit::stroke(p0, p, c.pour_radius as f32, vol, pig, self.film.dx).laid_by(Method::Pour));
                    self.painter_last_pour = Some(p);
                } else {
                    self.painter_last_pour = None;
                }
                if c.cure_rate > 0.0 {
                    if self.painter_cure_saved.is_none() {
                        self.painter_cure_saved = Some(self.film.settings.cure_rate);
                    }
                    self.film.settings.cure_rate = c.cure_rate as f32;
                } else if let Some(v) = self.painter_cure_saved.take() {
                    self.film.settings.cure_rate = v;
                }
                match c.tilt {
                    Some((deg, dir)) => {
                        if self.painter_tilt_saved.is_none() {
                            self.painter_tilt_saved = Some((self.film.settings.tilt_deg, self.film.settings.tilt_dir_deg));
                        }
                        self.film.settings.tilt_deg = deg as f32;
                        self.film.settings.tilt_dir_deg = dir as f32;
                    }
                    None => {
                        if let Some((deg, dir)) = self.painter_tilt_saved.take() {
                            self.film.settings.tilt_deg = deg;
                            self.film.settings.tilt_dir_deg = dir;
                        }
                    }
                }
                true
            }
            None => {
                if self.painter_active {
                    self.painter_active = false;
                    self.tool.flowing = false;
                    self.painter_last_pour = None;
                    if let Some(v) = self.painter_cure_saved.take() {
                        self.film.settings.cure_rate = v;
                    }
                    if let Some((deg, dir)) = self.painter_tilt_saved.take() {
                        self.film.settings.tilt_deg = deg;
                        self.film.settings.tilt_dir_deg = dir;
                    }
                }
                false
            }
        }
    }

    pub(crate) fn pen(&self) -> Option<f64> {
        if self.pen_down && self.pen_pressure {
            Some((self.pen_raw.clamp(0.0, 1.0) as f64).powf(self.pen_curve as f64))
        } else {
            None
        }
    }
}
