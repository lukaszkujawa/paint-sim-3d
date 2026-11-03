use crate::app::*;
use crate::script::*;
use crate::pollock;

impl App {
    pub(super) fn scene_groundbrush(&mut self, a: &AutoScript) -> bool {
        if self.painter.is_none() {
            let (m, n) = if a.script == "groundbrush" { (pollock::GroundMethod::Brush, a.rounds.max(1) as usize) } else { (pollock::GroundMethod::Bucket, a.rounds.max(1) as usize) };
            self.painter = Some(pollock::Painter::background(a.seed, self.film.settings.width_m as f64, self.film.settings.height_m as f64, self.pollock_pal.clone(), m, n));
        }
        self.apply_painter()
    }

    pub(super) fn scene_pollock(&mut self, a: &AutoScript) -> bool {
        if self.painter.is_none() {
            let mut p = pollock::Painter::new(a.seed, self.film.settings.width_m as f64, self.film.settings.height_m as f64);
            if !a.layer.is_empty() {
                p.skip_to(&a.layer);
            }
            self.painter = Some(p);
            self.painter_rounds_left = a.rounds;
        }
        if self.painter_rounds_left > 0 && self.painter.as_ref().map(|p| p.done).unwrap_or(false) {
            self.painter_rounds_left -= 1;
            if let Some(p) = self.painter.as_mut() {
                p.continue_painting();
            }
        }
        self.apply_painter()
    }
}
