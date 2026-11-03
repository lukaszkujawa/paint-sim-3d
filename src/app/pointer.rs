use crate::evolve;
use crate::app::*;

const PICK_RADIUS: i64 = 1;

impl App {
    pub(crate) fn canvas_colour_at_cursor(&self) -> Option<[f32; 3]> {
        if self.cursor_over_panel() {
            return None;
        }
        let p = self.cam.plane_hit(self.ndc(), 0.0)?;
        let dx = self.film.dx as f64;
        let (w, h) = (self.film.w as i64, self.film.h as i64);
        let (i, j) = ((p.x as f64 / dx).floor() as i64, (p.y as f64 / dx).floor() as i64);
        if i < 0 || j < 0 || i >= w || j >= h {
            return None;
        }
        let (x0, y0) = ((i - PICK_RADIUS).max(0) as u32, (j - PICK_RADIUS).max(0) as u32);
        let (x1, y1) = ((i + PICK_RADIUS + 1).min(w) as u32, (j + PICK_RADIUS + 1).min(h) as u32);
        Some(self.film.read_mean_colour(&self.gpu, [x0, y0, x1, y1]).map(evolve::encode))
    }

    pub(crate) fn eyedrop_click(&mut self) {
        if let Some(rgb) = self.canvas_colour_at_cursor() {
            self.use_colour(self.shown(rgb));
            self.eyedropper = false;
        }
    }

    pub(crate) fn cursor_over_panel(&self) -> bool {
        let [x, y, w, h] = self.view_rect;
        let (cx, cy) = (self.cursor[0], self.cursor[1]);
        if cx < x || cy < y || cx >= x + w || cy >= y + h || self.menu_open {
            return true;
        }
        let ppp = self.ui.ctx.pixels_per_point().max(0.01);
        let p = egui::pos2(cx / ppp, cy / ppp);
        self.block_rects.iter().any(|r| r.expand(4.0).contains(p))
    }

    pub(crate) fn ndc(&self) -> [f32; 2] {
        let [x, y, w, h] = self.view_rect;
        [(self.cursor[0] - x) / w.max(1.0) * 2.0 - 1.0, 1.0 - (self.cursor[1] - y) / h.max(1.0) * 2.0]
    }
}
