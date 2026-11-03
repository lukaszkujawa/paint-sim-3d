use crate::film::{Deposit, Film};
use crate::render::{Beauty, Refine, Renderer, TubeRenderer};
use crate::tools::{Airborne, CoarseSurface};
use crate::app::*;

impl App {
    pub(crate) fn lay_recycled(&mut self, laid: Vec<Deposit>, deposits: &mut Vec<Deposit>) {
        let (w, h) = (self.film.settings.width_m, self.film.settings.height_m);
        for d in laid {
            let d = d.recycled();
            self.tool_released += d.vol as f64 * d.inside_fraction(w, h) as f64;
            deposits.push(d);
        }
    }

    pub(crate) fn rebuild_canvas(&mut self, w: u32) {
        if w == self.film.w {
            return;
        }
        let settings = self.film.settings.clone();
        let h = ((w as f32) * settings.height_m / settings.width_m) as u32;
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        self.film = Film::new(&self.gpu, w, h, settings);
        let old = std::mem::replace(&mut self.renderer, Renderer::new(&self.gpu, &self.film));
        let r = &mut self.renderer;
        r.light_dir = old.light_dir;
        r.light_intensity = old.light_intensity;
        r.sky = old.sky;
        r.ground = old.ground;
        r.aperture_mm = old.aperture_mm;
        r.exposure = old.exposure;
        r.bloom = old.bloom;
        r.grain = old.grain;
        r.vignette = old.vignette;
        r.sharpen = old.sharpen;
        r.agx = old.agx;
        r.key_elevation_deg = old.key_elevation_deg;
        r.key_azimuth_deg = old.key_azimuth_deg;
        r.key_power = old.key_power;
        r.fill = old.fill;
        self.tubes = TubeRenderer::new(&self.gpu, &self.renderer.cam_buf);
        self.beauty = Beauty::new(&self.gpu, &self.renderer);
        self.refine = Refine::new(&self.gpu, &self.film, &self.renderer.layout, &self.renderer.cam_buf);
        self.surface = CoarseSurface::new(self.film.pool_dims[0], self.film.pool_dims[1], 4.0 * self.film.dx as f64);
        self.air = Airborne::new();
        self.pending_deposits.clear();
        self.held_deposits.clear();
        self.tool_scraped = 0.0;
        self.tool_released = 0.0;
        self.sim_time = 0.0;
        self.history.levels.clear();
        if let Some(p) = self.painter.as_mut() {
            p.stop();
        }
        self.stop_evolve();
        self.evo.run = None;
    }

    pub(crate) fn start_canvas(&mut self, choice: CanvasChoice) {
        self.restore_exposure();
        self.film.settings.raw_canvas = choice.weave;
        self.film.settings.canvas_rgb = choice.colour;
        if choice.res != self.film.w {
            self.rebuild_canvas(choice.res);
        } else {
            self.begin_full();
        }
        self.clear_canvas();
        self.history.name.clear();
    }
}
