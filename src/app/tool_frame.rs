use crate::app::*;
use crate::film::{self, Deposit, Method};
use crate::tools::ToolKind;
use crate::color;
use glam::DVec3;
use rayon::prelude::*;

const BRUSH_CLEAR: f64 = 0.001;
const BRUSH_DESCENT: f64 = 0.15;
const HOLD_GAIN: f64 = 8.0;
const HOLD_EXP: f64 = 1.5;
const HOLD_SPAN: f64 = 0.6;
const HOLD_MIN: f64 = 0.02;
const RUBBER_PEN_LIGHT: f64 = 0.35;
impl App {
    pub(crate) fn rubber_frame(&mut self, pen: Option<f64>) {
        if self.tool.kind == ToolKind::Rubber && self.tool.flowing {
            if !self.paused {
                let here = DVec3::new(self.tool.pos.x, self.tool.pos.y, 0.0);
                let from = self.rubber_prev.unwrap_or(here);
                let width = pen.map(|p| RUBBER_PEN_LIGHT + (1.0 - RUBBER_PEN_LIGHT) * p).unwrap_or(1.0);
                self.rubber_seg = Some(([from.x as f32, from.y as f32], [here.x as f32, here.y as f32], (0.5 * self.rubber_size * width) as f32));
                self.rubber_prev = Some(here);
            }
        } else {
            self.rubber_prev = None;
        }
    }

    pub(crate) fn absorb_reservoir(&mut self) {
        let mut picked: [Vec<[u32; 17]>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        self.film.poll_reservoir(&self.gpu, &mut picked);
        if !picked[0].is_empty() {
            self.tool_scraped += self.brush.absorb(&picked[0]);
        }
        if !picked[1].is_empty() {
            self.tool_scraped += self.roller.absorb(&picked[1]);
        }
        if !picked[2].is_empty() {
            self.tool_scraped += self.sponge.absorb(&picked[2]);
        }
    }

    pub(crate) fn brush_frame(&mut self, pen: Option<f64>, frame_dt: f32, deposits: &mut Vec<Deposit>, contacts: &mut Vec<film::Contact>) {
        if self.tool.kind == ToolKind::Brush {
            if self.prev_kind != ToolKind::Brush {
                self.tool.height = self.brush.len * 0.75;
            }
            if self.auto.is_none() && !self.evolving() {
                self.tool.height = self.brush.len - self.brush_press(self.brush_pressure);
            }
            self.tool.height = self.tool.height.clamp(0.002, self.brush.len - 0.0002);
            if self.tool.infinite && !self.evolving() {
                self.brush.top_up(&self.paint.frac);
            }
            if !self.paused {
                let lift = if self.tool.flowing { self.brush_raise } else { 0.04 };
                let h = match pen {
                    Some(p) => (self.brush.len - self.brush_press(p)).clamp(0.002, self.brush.len - 0.0002),
                    None => self.tool.height,
                };
                let reach = self.brush_press(1.0);
                let geo = (self.brush.len - h).max(0.0);
                let landed = self.brush_z.map(|z| z <= h - self.brush_hold + 1e-4).unwrap_or(false);
                if self.tool.flowing && self.brush_raise == 0.0 && landed {
                    let p = (geo / reach.max(1e-6)).clamp(0.0, 1.0);
                    let target = HOLD_MIN + HOLD_SPAN * p.powf(HOLD_EXP);
                    let have = self.brush.touching as f64 / self.brush.bristles.len().max(1) as f64;
                    self.brush_hold = (self.brush_hold + HOLD_GAIN * (target - have).max(0.0) * reach * frame_dt as f64).clamp(0.0, (reach - geo).max(0.0));
                } else if !self.tool.flowing {
                    self.brush_hold = 0.0;
                }
                let want = h - self.brush_hold + lift;
                let z = match self.brush_z {
                    Some(prev) if want < prev => {
                        let from = prev.min(want.max(self.brush.len + self.surface.brush_floor(self.tool.pos.x, self.tool.pos.y, self.brush_sink) + BRUSH_CLEAR));
                        want.max(from - BRUSH_DESCENT * frame_dt as f64)
                    }
                    _ => want,
                };
                self.brush_z = Some(z);
                let handle = DVec3::new(self.tool.pos.x, self.tool.pos.y, z);
                let surf = &self.surface;
                let brush = &mut self.brush;
                let sink = self.brush_sink;
                let (c, rel) = brush.update(handle, frame_dt as f64, &|x, y| surf.brush_floor(x, y, sink), self.film.dx as f64);
                *contacts = c;
                let (dx, w, hgt) = (self.film.dx, self.film.settings.width_m, self.film.settings.height_m);
                let laid: Vec<(Deposit, f64)> = rel
                    .par_iter()
                    .map(|r| {
                        let pig = color::PaintMix { frac: r.comp }.normalized().packed();
                        let d = Deposit::stroke(r.p0, r.p1, r.r, r.vol as f32, pig, dx).laid_by(Method::Brush).recycled().strands(r.seed, r.offset, r.fill, dx);
                        let inside = r.vol * d.inside_fraction(w, hgt) as f64;
                        (d, inside)
                    })
                    .collect();
                for (d, inside) in laid {
                    self.tool_released += inside;
                    deposits.push(d);
                }
            }
        }
    }

    pub(crate) fn roller_and_sponge_frame(&mut self, pen: Option<f64>, frame_dt: f32, deposits: &mut Vec<Deposit>, contacts: &mut Vec<film::Contact>) {
        if self.tool.kind == ToolKind::Roller {
            if !self.paused {
                let pressure = pen.unwrap_or(self.roller.pressure);
                let at = glam::DVec2::new(self.tool.pos.x, self.tool.pos.y);
                let (c, d) = self.roller.update(at, self.tool.flowing, pressure, self.film.dx);
                *contacts = c;
                self.lay_recycled(d, deposits);
            }
        } else {
            self.roller.lift();
        }
        if self.tool.kind == ToolKind::Sponge {
            if !self.paused {
                let pressure = pen.unwrap_or(self.sponge.pressure);
                let at = glam::DVec2::new(self.tool.pos.x, self.tool.pos.y);
                let (c, d) = self.sponge.update(at, self.tool.flowing, pressure, frame_dt as f64, self.film.dx);
                *contacts = c;
                self.lay_recycled(d, deposits);
            }
        } else {
            self.sponge.lift();
        }
    }
}
