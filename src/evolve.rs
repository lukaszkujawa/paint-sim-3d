mod colour;
mod filters;
mod focus;
mod photo;
mod plan;
mod sketch;
mod evolver;
mod driver;
pub(crate) use colour::*;
pub use filters::*;
pub use focus::*;
pub use photo::*;
pub use plan::*;
pub use sketch::*;
pub use evolver::*;
pub use driver::*;
use crate::brush::BrushShape;
use crate::color::{self, PigmentSet};
use crate::film::{Deposit, Method};
use crate::tools::ToolKind;
use crate::App;
use glam::{DVec2, DVec3};
use image::ImageDecoder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

const TILE: usize = 8;
const LOOKAHEAD: usize = 160;
const EMPTY_BATCHES: u32 = 3;
const PRESS_START: f64 = 0.35;
const QUANTILE_STRIDE: usize = 7;
const TAPER_END: f32 = 0.2;
const MIN_GAIN: f32 = 2.0;
const WHITE: usize = 0;
const BLACK: usize = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn sketch_bench() {
        let path = std::env::var("EVO_PHOTO").unwrap();
        let out = std::env::var("EVO_OUT").unwrap_or_default();
        let counts: Vec<usize> = std::env::var("EVO_N").unwrap_or_else(|_| "900".to_string()).split(',').filter_map(|v| v.parse().ok()).collect();
        let total: usize = counts.iter().sum();
        let set = color::load_pigments();
        let cmf = color::cmf_table();
        let img = image::open(&path).unwrap().to_rgba8();
        let photo = cover_crop(&img, 1.5);
        let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let prep = Prep { contrast: env("EVO_CONTRAST", 1.0), blur_mm: env("EVO_BLUR", 0.0), sharpen: env("EVO_SHARPEN", 0.0), smooth_mm: env("EVO_SMOOTH", 0.0), lift_ev: env("EVO_LIFT", 0.0) };
        let mut ev = Evolver::new(&photo, 3072, 2048, 1.2 / 3072.0, &set, &cmf, 7, prep);
        let gain = [2.11f32 * 1.27, 2.2 * 1.27, 2.27 * 1.27];
        ev.base_gain = gain;
        ev.calibrate(gain.map(|g| g * (-prep.lift_ev).exp2()), [0.0; 3]);
        if let Ok(path) = std::env::var("EVO_TARGET") {
            let mm = ev.model.as_ref().unwrap();
            let mut im = image::RgbImage::new(mm.w as u32, mm.h as u32);
            for (k, l) in mm.target.iter().enumerate() {
                let fy = (l[0] + 16.0) / 116.0;
                let (fx, fz) = (fy + l[1] / 500.0, fy - l[2] / 200.0);
                let inv = |t: f32| if t > 0.2069 { t * t * t } else { (t - 16.0 / 116.0) / 7.787 };
                let (x, y, z) = (0.95047 * inv(fx), inv(fy), 1.08883 * inv(fz));
                let rgb = [3.2406 * x - 1.5372 * y - 0.4986 * z, -0.9689 * x + 1.8758 * y + 0.0415 * z, 0.0557 * x - 0.2040 * y + 1.0570 * z];
                let px = rgb.map(|c| (encode(c.clamp(0.0, 1.0)) * 255.0) as u8);
                im.put_pixel((k % mm.w) as u32, (mm.h - 1 - k / mm.w) as u32, image::Rgb(px));
            }
            im.save(&path).unwrap();
        }
        let m = ev.model.clone().unwrap();
        let _ = env_logger::try_init();
        let kind = ToolKind::Brush;
        let flat = std::env::var("EVO_FLAT").is_ok();
        let range = tool_range(kind, if flat { BrushShape::Flat } else { BrushShape::Round }).unwrap();
        let sizes: Vec<f64> = std::env::var("EVO_SIZE").unwrap_or_else(|_| "16".to_string()).split(',').filter_map(|v| v.parse::<f64>().ok()).map(|v| v * 1e-3).collect();
        let mut stage = 0usize;
        let mut sk = Sketch::new(m.clone(), 7, kind, range, sizes[0]);
        let base_lab = if std::env::var("EVO_BLACK").is_ok() { lab(ev.black) } else { [80.0f32, 1.0, 6.0] };
        let mut base = vec![base_lab; m.w * m.h];
        let err0 = base.iter().zip(&m.target).map(|(a, b)| de(*a, *b)).sum::<f32>();
        let mut focus = Focus { shape: FocusShape::Off, a: [0.5, 0.5], b: [0.75, 0.5], soft: 0.5, outside: 0.0 };
        if let Ok(spec) = std::env::var("EVO_FOCUS") {
            let (kind, nums) = spec.split_once(':').unwrap_or((spec.as_str(), ""));
            let v: Vec<f32> = nums.split(',').filter_map(|x| x.parse().ok()).collect();
            if v.len() >= 4 {
                focus.shape = if kind == "circle" { FocusShape::Circle } else { FocusShape::Rect };
                (focus.a, focus.b) = ([v[0], v[1]], [v[2], v[3]]);
            }
        }
        let opacity: f32 = std::env::var("EVO_OPACITY").ok().and_then(|v| v.parse().ok()).unwrap_or(0.7);
        let (mut done, mut next_report, mut empty) = (0usize, 0usize, 0usize);
        let t0 = std::time::Instant::now();
        while done < total {
            let mut reset = if done == 0 { Some(sizes[0]) } else { None };
            let next: usize = counts[..=stage.min(counts.len() - 1)].iter().sum();
            if done >= next && stage + 1 < sizes.len() {
                stage += 1;
                reset = Some(sizes[stage]);
            }
            let job = Job { kind, range, reset, candidates: 32, follow: env("EVO_FOLLOW", 0.7) as f64, longest: 0.25, focus, paint: [0.0; 16], width_gain: env("EVO_GAIN", 1.85) as f64, opacity, efficiency: env("EVO_EFF", 1.0), tile_eff: Arc::new(Vec::new()), intent: env("EVO_INTENT", 0.7), gen: 0 };
            sk.rebase(&base, &[]);
            let b = sk.run(&job);
            sk.rebase(&base, &b.plans);
            base = sk.current.clone();
            done += b.plans.len();
            if done >= next_report || b.plans.is_empty() || done >= total {
                let e = base.iter().zip(&m.target).map(|(a, b)| de(*a, *b)).sum::<f32>();
                let ed = base.iter().zip(&m.target).zip(&m.weight).map(|((a, b), w)| (w - 1.0) * de(*a, *b)).sum::<f32>() / m.weight.iter().map(|w| w - 1.0).sum::<f32>();
                print!("edge dE {:.1} ", ed);
                println!("strokes {} closer {:.1}% size {:.1} mm stage {} pruned {} {:.1}s", done, 100.0 * (1.0 - e / err0), b.sched.size_hi * 1e3, stage, b.pruned, t0.elapsed().as_secs_f32());
                next_report += 600;
            }
            if b.plans.is_empty() {
                empty += 1;
                if empty >= EMPTY_BATCHES as usize {
                    if stage + 1 >= sizes.len() {
                        break;
                    }
                    println!("stage {} ran dry at {} strokes", stage, done);
                    done = next;
                    empty = 0;
                }
            } else {
                empty = 0;
            }
        }
        if !out.is_empty() {
            let mut im = image::RgbImage::new(m.w as u32, m.h as u32);
            for (k, l) in base.iter().enumerate() {
                let fy = (l[0] + 16.0) / 116.0;
                let (fx, fz) = (fy + l[1] / 500.0, fy - l[2] / 200.0);
                let inv = |t: f32| if t > 0.2069 { t * t * t } else { (t - 16.0 / 116.0) / 7.787 };
                let (x, y, z) = (0.95047 * inv(fx), inv(fy), 1.08883 * inv(fz));
                let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
                let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
                let bl = 0.0557 * x - 0.2040 * y + 1.0570 * z;
                let enc = |c: f32| {
                    let c = c.clamp(0.0, 1.0);
                    ((if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }) * 255.0) as u8
                };
                im.put_pixel((k % m.w) as u32, (m.h - 1 - k / m.w) as u32, image::Rgb([enc(r), enc(g), enc(bl)]));
            }
            im.save(&out).unwrap();
        }
    }
}
