use std::time::Instant;
use crate::app::*;

impl App {
    pub(crate) fn finish_auto(&mut self) -> bool {
        let mut exit = false;
        let export_dir = self.auto.as_ref().filter(|a| self.frame_index >= a.frames).map(|a| a.export.clone()).unwrap_or_default();
        if !export_dir.is_empty() {
            self.export_blender(&export_dir);
        }
        if let Some(auto) = &self.auto {
            if self.frame_index >= auto.frames {
                self.shot_request = Some(auto.shot.clone());
                if !auto.snap_save.is_empty() {
                    let sp = auto.snap_save.clone();
                    self.save_snapshot(&sp);
                }
                let auto = self.auto.as_ref().unwrap();
                let spp = auto.spp;
                let vres = auto.vres;
                if !auto.beauty.is_empty() {
                    let bp = auto.beauty.clone();
                    self.refine.run(&self.gpu, &self.film);
                    let t0 = Instant::now();
                    let img = self.beauty.render(&self.gpu, &self.renderer, &self.cam, &self.film, true, self.film.w * 2, self.film.h * 2, spp, Some(&self.refine));
                    log::warn!("beauty took {:.1} s", t0.elapsed().as_secs_f32());
                    let _ = img.save(&bp);
                    log::warn!("saved beauty {}", bp);
                    let args: Vec<String> = std::env::args().collect();
                    if let Some(dp) = args.windows(2).find(|p| p[0] == "--dump").map(|p| p[1].clone()) {
                        self.dump_observed(&dp);
                    }
                }
                if !auto.view.is_empty() {
                    let vp = auto.view.clone();
                    self.refine.run(&self.gpu, &self.film);
                    let t0 = Instant::now();
                    let img = self.beauty.render(&self.gpu, &self.renderer, &self.cam, &self.film, false, vres[0], vres[1], spp, Some(&self.refine));
                    log::warn!("view took {:.1} s", t0.elapsed().as_secs_f32());
                    let _ = img.save(&vp);
                    log::warn!("saved view {}", vp);
                }
                exit = true;
            }
        }
        exit
    }
}
