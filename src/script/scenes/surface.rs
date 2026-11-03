use crate::app::*;
use crate::film::{Deposit, Method};
use crate::script::*;
use crate::tools::ToolKind;
use crate::color;

impl App {
    pub(super) fn scene_canvas(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let args: Vec<String> = std::env::args().collect();
        let arg = |k: &str| args.windows(2).find(|p| p[0] == k).map(|p| p[1].clone());
        if f == 20 {
            let res = arg("--newres").and_then(|v| v.parse().ok()).unwrap_or(self.film.w);
            let colour = arg("--canvas").map(|v| {
                let c: Vec<f32> = v.split(',').filter_map(|t| t.parse().ok()).collect();
                [c[0], c[1], c[2]]
            });
            self.start_canvas(CanvasChoice { res, colour, weave: arg("--noweave").is_none(), custom: [0.9; 3], first: false });
            log::warn!("CANVAS now {}x{} dx {:.3} mm", self.film.w, self.film.h, self.film.dx * 1e3);
        }
        self.tool.kind = ToolKind::Pour;
        self.tool.flowing = false;
        if (40..160).contains(&f) {
            let (w, hgt) = (self.film.settings.width_m, self.film.settings.height_m);
            let t = (f - 40) as f32 / 120.0;
            let mut p = color::PaintMix::pure(3);
            p.frac[13] = 20.0;
            p.frac[14] = (10f32).ln() + 10.0;
            let a = [w * (0.3 + 0.4 * t), hgt * 0.5];
            let b = [w * (0.3 + 0.4 * (t + 1.0 / 120.0)), hgt * 0.5];
            self.pending_deposits.push(Deposit::capsule(a, b, 0.008, 3.0e-7, p.normalized().packed(), self.film.dx).laid_by(Method::Pour));
        }
        true
    }

    pub(super) fn scene_physics(&mut self, _a: &AutoScript) -> bool {
        let f = self.frame_index;
        let (w, hgt) = (self.film.settings.width_m as f64, self.film.settings.height_m as f64);
        let args: Vec<String> = std::env::args().collect();
        let arg = |k: &str, d: f64| args.windows(2).find(|p| p[0] == k).and_then(|p| p[1].parse::<f64>().ok()).unwrap_or(d);
        self.tool.kind = ToolKind::Pour;
        self.tool.flowing = false;
        if f == 1 {
            let st = &mut self.film.settings;
            st.cure_rate = 0.0;
            st.contact_angle_deg = arg("--angle", 5.0) as f32;
            st.hysteresis_deg = 0.0;
            st.tilt_deg = arg("--tilt", 0.0) as f32;
            st.tilt_dir_deg = -90.0;
            if let Some(n) = args.windows(2).find(|p| p[0] == "--maxsub").and_then(|p| p[1].parse::<u32>().ok()) {
                st.max_substeps = n;
                st.min_substeps = st.min_substeps.min(n);
            }
            self.tool.sigma = arg("--sigma", 0.03);
            self.tool.mu = arg("--mu", 10.0);
            let mut p = color::PaintMix::pure(1);
            p.frac[12] = 0.0;
            p.frac[13] = arg("--yield", 0.0) as f32;
            p.frac[14] = (self.tool.mu as f32).ln() + 10.0;
            let vol = arg("--vol", 5.0) * 1e-6;
            let r = arg("--r0", 15.0) * 1e-3;
            let c = [(w * 0.5) as f32, (hgt * 0.5) as f32];
            self.pending_deposits.push(Deposit::capsule(c, c, r as f32, vol as f32, p.normalized().packed(), self.film.dx).laid_by(Method::Pour));
        }
        let every = arg("--every", 60.0) as u64;
        if f > 2 && f % every == 0 {
            let hv: Vec<f32> = bytemuck::cast_slice(&self.read_buffer_raw(&self.film.h_buf)).to_vec();
            let (fw, fh, dx) = (self.film.w as usize, self.film.h as usize, self.film.dx as f64);
            let thr = arg("--thr", 2.0) * 1e-6;
            let (mut vol, mut cx, mut cy, mut area, mut hmax) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (k, v) in hv.iter().enumerate() {
                let v = *v as f64;
                if v <= 0.0 {
                    continue;
                }
                let (x, y) = (((k % fw) as f64 + 0.5) * dx, ((k / fw) as f64 + 0.5) * dx);
                vol += v * dx * dx;
                cx += v * x * dx * dx;
                cy += v * y * dx * dx;
                hmax = hmax.max(v);
                if v > thr {
                    area += dx * dx;
                }
            }
            cx /= vol.max(1e-30);
            cy /= vol.max(1e-30);
            let mut radii = Vec::new();
            for d in 0..8 {
                let a = d as f64 * std::f64::consts::FRAC_PI_4;
                let mut r = 0.0;
                loop {
                    let (x, y) = (cx + (r + dx) * a.cos(), cy + (r + dx) * a.sin());
                    let (i, j) = ((x / dx) as i64, (y / dx) as i64);
                    if i < 0 || j < 0 || i >= fw as i64 || j >= fh as i64 || (hv[j as usize * fw + i as usize] as f64) <= thr {
                        break;
                    }
                    r += dx;
                }
                radii.push(r * 1e3);
            }
            let mean = (area / std::f64::consts::PI).sqrt();
            let rs: Vec<String> = radii.iter().map(|r| format!("{:.2}", r)).collect();
            log::warn!("PHYS t {:.2} s vol {:.4} mL R {:.3} mm hmax {:.3} mm centroid {:.3} {:.3} mm radii {}", f as f64 / 60.0, vol * 1e6, mean * 1e3, hmax * 1e3, (cx - w * 0.5) * 1e3, (cy - hgt * 0.5) * 1e3, rs.join(" "));
        }
        true
    }
}
