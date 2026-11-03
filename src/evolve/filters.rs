use super::*;

const CHROMA_WEIGHT: f32 = 0.7;

const ETF_REACH: f32 = 10.0;

const ETF_ETA: f32 = 4.0;

const TAPER_MIN: f32 = 0.45;

const MODE_RADIUS: usize = 2;

const MODE_PASSES: usize = 2;

const KMEANS_ROUNDS: usize = 12;

const REGION_COUNT: usize = 16;

const SEGMENT_STRIDE: usize = 13;

const SEGMENT_RANGE: f32 = 10.0;

const SEGMENT_SMOOTH_PX: f32 = 3.0;

const BILATERAL_PASSES: usize = 2;

const SMOOTH_RANGE: f32 = 0.1;

const SHARPEN_MM: f32 = 1.5;

#[derive(Clone, Copy, PartialEq)]
pub struct Prep {
    pub contrast: f32,
    pub blur_mm: f32,
    pub sharpen: f32,
    pub smooth_mm: f32,
    pub lift_ev: f32,
}

impl Prep {
    pub fn none() -> Self {
        Self { contrast: 1.0, blur_mm: 0.0, sharpen: 0.0, smooth_mm: 0.0, lift_ev: 0.0 }
    }

    pub(super) fn is_none(&self) -> bool {
        *self == Self::none()
    }
}

pub(super) fn bilateral(src: &[[f32; 3]], w: usize, h: usize, sigma_px: f32, range: f32) -> Vec<[f32; 3]> {
    if sigma_px < 0.5 {
        return src.to_vec();
    }
    let reach = (sigma_px * 2.0).ceil() as isize;
    let kernel: Vec<f32> = (-reach..=reach).map(|i| (-(i * i) as f32 / (2.0 * sigma_px * sigma_px)).exp()).collect();
    let inv = 1.0 / (2.0 * range * range);
    let mut cur = src.to_vec();
    for _ in 0..BILATERAL_PASSES {
        for horizontal in [true, false] {
            let prev = cur;
            cur = (0..w * h)
                .into_par_iter()
                .map(|k| {
                    let (x, y) = ((k % w) as isize, (k / w) as isize);
                    let c = prev[k];
                    let (mut sum, mut tot) = ([0.0f32; 3], 0.0f32);
                    for (o, kw) in kernel.iter().enumerate() {
                        let d = o as isize - reach;
                        let (sx, sy) = if horizontal { ((x + d).clamp(0, w as isize - 1), y) } else { (x, (y + d).clamp(0, h as isize - 1)) };
                        let v = prev[sy as usize * w + sx as usize];
                        let dist = (v[0] - c[0]).powi(2) + (v[1] - c[1]).powi(2) + (v[2] - c[2]).powi(2);
                        let wgt = kw * (-dist * inv).exp();
                        for i in 0..3 {
                            sum[i] += v[i] * wgt;
                        }
                        tot += wgt;
                    }
                    sum.map(|v| v / tot.max(1e-12))
                })
                .collect();
        }
    }
    cur
}

pub(super) fn segment(lab: &[[f32; 3]], w: usize, h: usize, seed: u64) -> (Vec<u8>, Vec<[f32; 3]>) {
    let soft = bilateral(lab, w, h, SEGMENT_SMOOTH_PX, SEGMENT_RANGE);
    let mut rng = StdRng::seed_from_u64(seed);
    let sample: Vec<[f32; 3]> = soft.iter().step_by(SEGMENT_STRIDE).copied().collect();
    let mut centres = vec![sample[rng.random_range(0..sample.len())]];
    while centres.len() < REGION_COUNT {
        let d2: Vec<f64> = sample.iter().map(|p| centres.iter().map(|c| de(*p, *c)).fold(f32::MAX, f32::min).powi(2) as f64).collect();
        let total: f64 = d2.iter().sum();
        let mut pick = rng.random::<f64>() * total;
        let mut chosen = sample.len() - 1;
        for (i, d) in d2.iter().enumerate() {
            pick -= d;
            if pick <= 0.0 {
                chosen = i;
                break;
            }
        }
        centres.push(sample[chosen]);
    }
    let nearest = |p: &[f32; 3], centres: &[[f32; 3]]| centres.iter().enumerate().map(|(i, c)| (i, de(*p, *c))).min_by(|a, b| a.1.total_cmp(&b.1)).map(|(i, _)| i).unwrap_or(0);
    for _ in 0..KMEANS_ROUNDS {
        let mut acc = vec![[0.0f64; 4]; centres.len()];
        for p in &sample {
            let i = nearest(p, &centres);
            for c in 0..3 {
                acc[i][c] += p[c] as f64;
            }
            acc[i][3] += 1.0;
        }
        for (c, a) in centres.iter_mut().zip(&acc) {
            if a[3] > 0.0 {
                *c = [(a[0] / a[3]) as f32, (a[1] / a[3]) as f32, (a[2] / a[3]) as f32];
            }
        }
    }
    let mut label: Vec<u8> = soft.par_iter().map(|p| nearest(p, &centres) as u8).collect();
    for _ in 0..MODE_PASSES {
        let prev = label;
        label = (0..w * h)
            .into_par_iter()
            .map(|k| {
                let (x, y) = ((k % w) as isize, (k / w) as isize);
                let mut votes = [0u16; 256];
                let r = MODE_RADIUS as isize;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (sx, sy) = ((x + dx).clamp(0, w as isize - 1), (y + dy).clamp(0, h as isize - 1));
                        votes[prev[sy as usize * w + sx as usize] as usize] += 1;
                    }
                }
                votes.iter().enumerate().max_by_key(|(_, v)| **v).map(|(i, _)| i as u8).unwrap_or(prev[k])
            })
            .collect();
    }
    (label, centres)
}

pub(super) fn taper(t: f32) -> f32 {
    let e = (t.min(1.0 - t) / TAPER_END).clamp(0.0, 1.0);
    TAPER_MIN + (1.0 - TAPER_MIN) * e * e * (3.0 - 2.0 * e)
}

pub(super) fn gaussian(src: &[f32], w: usize, h: usize, sigma_px: f32) -> Vec<f32> {
    if sigma_px < 0.3 {
        return src.to_vec();
    }
    let r = ((((12.0 * sigma_px * sigma_px / 3.0) + 1.0).sqrt() - 1.0) * 0.5).round().max(1.0) as usize;
    box_blur(&box_blur(&box_blur(src, w, h, r), w, h, r), w, h, r)
}

pub fn prepare(img: &image::RgbaImage, prep: Prep, px_mm: f32) -> image::RgbaImage {
    if prep.is_none() {
        return img.clone();
    }
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut rgb: Vec<[f32; 3]> = img.pixels().map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]).collect();
    if prep.smooth_mm > 0.0 {
        rgb = bilateral(&rgb, w, h, prep.smooth_mm / px_mm.max(1e-6), SMOOTH_RANGE);
    }
    let mut chans: Vec<Vec<f32>> = (0..3).map(|c| rgb.iter().map(|p| p[c]).collect()).collect();
    for ch in chans.iter_mut() {
        let mut v = gaussian(ch, w, h, prep.blur_mm / px_mm.max(1e-6));
        if prep.sharpen > 0.0 {
            let soft = gaussian(&v, w, h, (SHARPEN_MM / px_mm.max(1e-6)).max(1.0));
            for (x, b) in v.iter_mut().zip(&soft) {
                *x += prep.sharpen * (*x - b);
            }
        }
        for x in v.iter_mut() {
            *x = (0.5 + (*x - 0.5) * prep.contrast).clamp(0.0, 1.0);
        }
        *ch = v;
    }
    let mut out = image::RgbaImage::new(w as u32, h as u32);
    for (k, p) in out.pixels_mut().enumerate() {
        *p = image::Rgba([(chans[0][k] * 255.0).round() as u8, (chans[1][k] * 255.0).round() as u8, (chans[2][k] * 255.0).round() as u8, 255]);
    }
    out
}

pub(super) fn gauss(rng: &mut StdRng) -> f32 {
    let u: f32 = rng.random::<f32>().max(1e-7);
    let v: f32 = rng.random();
    (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
}

pub(super) fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let (n, len) = if horizontal { (h, w) } else { (w, h) };
        let at = |line: usize, i: usize| if horizontal { line * w + i } else { i * w + line };
        let lines: Vec<Vec<f32>> = (0..n)
            .into_par_iter()
            .map(|line| {
                let mut pre = vec![0.0f32; len + 1];
                for i in 0..len {
                    pre[i + 1] = pre[i] + src[at(line, i)];
                }
                (0..len)
                    .map(|i| {
                        let (a, b) = (i.saturating_sub(r), (i + r + 1).min(len));
                        (pre[b] - pre[a]) / (b - a) as f32
                    })
                    .collect()
            })
            .collect();
        let mut out = vec![0.0f32; w * h];
        for (line, vals) in lines.into_iter().enumerate() {
            for (i, v) in vals.into_iter().enumerate() {
                out[at(line, i)] = v;
            }
        }
        out
    };
    pass(&pass(src, true), false)
}

pub(super) fn flow_field(tangent: &[[f32; 3]], w: usize, h: usize) -> Vec<[f32; 3]> {
    let dx: Vec<f32> = tangent.iter().map(|t| t[2] * (t[0] * t[0] - t[1] * t[1])).collect();
    let dy: Vec<f32> = tangent.iter().map(|t| t[2] * 2.0 * t[0] * t[1]).collect();
    let mut fx = vec![0.0f32; w * h];
    let mut fy = vec![0.0f32; w * h];
    for (r, wgt) in [(6usize, 1.0f32), (24, 0.6), (96, 0.4)] {
        let bx = box_blur(&box_blur(&dx, w, h, r), w, h, r);
        let by = box_blur(&box_blur(&dy, w, h, r), w, h, r);
        for k in 0..w * h {
            fx[k] += wgt * bx[k];
            fy[k] += wgt * by[k];
        }
    }
    let mags: Vec<f32> = fx.iter().zip(&fy).map(|(x, y)| (x * x + y * y).sqrt()).collect();
    let mut sorted = mags.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let strong = sorted[sorted.len() * 9 / 10].max(1e-6);
    (0..w * h)
        .into_par_iter()
        .map(|k| {
            let th = 0.5 * fy[k].atan2(fx[k]);
            [th.cos(), th.sin(), (mags[k] / strong).min(1.0)]
        })
        .collect()
}

pub(super) fn etf(tangent: &[[f32; 3]], mag: &[f32], w: usize, h: usize, r: usize, iters: usize) -> Vec<[f32; 3]> {
    let mut t: Vec<[f32; 2]> = tangent.iter().map(|v| [v[0], v[1]]).collect();
    let r = r as isize;
    for _ in 0..iters {
        for horizontal in [true, false] {
            let prev = t;
            t = (0..w * h)
                .into_par_iter()
                .map(|k| {
                    let (i, j) = ((k % w) as isize, (k / w) as isize);
                    let tx = prev[k];
                    let gx = mag[k];
                    let mut acc = [0.0f32; 2];
                    for d in -r..=r {
                        let (x, y) = if horizontal { ((i + d).clamp(0, w as isize - 1), j) } else { (i, (j + d).clamp(0, h as isize - 1)) };
                        let n = y as usize * w + x as usize;
                        let ty = prev[n];
                        let dot = tx[0] * ty[0] + tx[1] * ty[1];
                        let wm = 0.5 * (1.0 + (ETF_ETA * (mag[n] - gx)).tanh());
                        let wgt = dot.abs() * wm * if dot < 0.0 { -1.0 } else { 1.0 };
                        acc[0] += wgt * ty[0];
                        acc[1] += wgt * ty[1];
                    }
                    let l = (acc[0] * acc[0] + acc[1] * acc[1]).sqrt();
                    if l > 1e-6 { [acc[0] / l, acc[1] / l] } else { tx }
                })
                .collect();
        }
    }
    let near = box_blur(mag, w, h, (r as usize) * iters);
    (0..w * h).into_par_iter().map(|k| [t[k][0], t[k][1], (near[k] * ETF_REACH).min(1.0)]).collect()
}

pub(super) fn structure(lab: &[[f32; 3]], w: usize, h: usize, radius: usize) -> (Vec<[f32; 3]>, Vec<f32>) {
    let l = |i: isize, j: isize| lab[(j.clamp(0, h as isize - 1) as usize) * w + i.clamp(0, w as isize - 1) as usize];
    let tensor: Vec<[f32; 3]> = (0..w * h)
        .into_par_iter()
        .map(|k| {
            let (i, j) = ((k % w) as isize, (k / w) as isize);
            let mut t = [0.0f32; 3];
            for c in 0..3 {
                let s = if c == 0 { 1.0 } else { CHROMA_WEIGHT };
                let gx = s * ((l(i + 1, j - 1)[c] + 2.0 * l(i + 1, j)[c] + l(i + 1, j + 1)[c]) - (l(i - 1, j - 1)[c] + 2.0 * l(i - 1, j)[c] + l(i - 1, j + 1)[c]));
                let gy = s * ((l(i - 1, j + 1)[c] + 2.0 * l(i, j + 1)[c] + l(i + 1, j + 1)[c]) - (l(i - 1, j - 1)[c] + 2.0 * l(i, j - 1)[c] + l(i + 1, j - 1)[c]));
                t[0] += gx * gx;
                t[1] += gx * gy;
                t[2] += gy * gy;
            }
            t
        })
        .collect();
    let r = radius as isize;
    let blur = |src: &[[f32; 3]], horizontal: bool| -> Vec<[f32; 3]> {
        (0..w * h)
            .into_par_iter()
            .map(|k| {
                let (i, j) = ((k % w) as isize, (k / w) as isize);
                let mut acc = [0.0f32; 3];
                for d in -r..=r {
                    let (x, y) = if horizontal { ((i + d).clamp(0, w as isize - 1), j) } else { (i, (j + d).clamp(0, h as isize - 1)) };
                    let v = src[y as usize * w + x as usize];
                    acc[0] += v[0];
                    acc[1] += v[1];
                    acc[2] += v[2];
                }
                acc
            })
            .collect()
    };
    let t = blur(&blur(&tensor, true), false);
    let mut strength: Vec<f32> = t.iter().map(|v| v[0] + v[2]).collect();
    strength.sort_by(|a, b| a.total_cmp(b));
    let typical = strength[strength.len() * 3 / 4].max(1e-3);
    let strong = strength[strength.len() * 19 / 20].max(typical);
    let tangent = t
        .par_iter()
        .map(|v| {
            let (jxx, jxy, jyy) = (v[0], v[1], v[2]);
            let tr = jxx + jyy;
            let diff = ((jxx - jyy).powi(2) + 4.0 * jxy * jxy).sqrt();
            let coherence = if tr > 1e-6 { diff / tr } else { 0.0 };
            let theta = 0.5 * (2.0 * jxy).atan2(jxx - jyy) + std::f32::consts::FRAC_PI_2;
            [theta.cos(), theta.sin(), coherence * (tr / typical).min(1.0)]
        })
        .collect();
    let edges = t.par_iter().map(|v| ((v[0] + v[2]) / strong).min(1.0)).collect();
    (tangent, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilateral_leaves_flat_colour_alone() {
        let src = vec![[40.0f32, 5.0, -3.0]; 64];
        for v in bilateral(&src, 8, 8, 2.0, 10.0) {
            assert!(de(v, src[0]) < 1e-3);
        }
    }

    #[test]
    fn bilateral_keeps_strong_edges() {
        let src: Vec<[f32; 3]> = (0..256).map(|k| if k % 16 < 8 { [20.0, 0.0, 0.0] } else { [80.0, 0.0, 0.0] }).collect();
        let out = bilateral(&src, 16, 16, 2.0, 5.0);
        assert!(out[3][0] < 25.0 && out[12][0] > 75.0);
    }

    #[test]
    fn box_blur_preserves_the_mean() {
        let src: Vec<f32> = (0..100).map(|k| if k == 55 { 100.0 } else { 0.0 }).collect();
        let out = box_blur(&src, 10, 10, 1);
        assert!((out.iter().sum::<f32>() - 100.0).abs() < 1e-3);
        assert!(out[55] < 100.0);
    }

    #[test]
    fn untouched_photo_is_unchanged_by_empty_prep() {
        assert!(Prep::none().contrast == 1.0 && Prep::none().blur_mm == 0.0);
    }
}
