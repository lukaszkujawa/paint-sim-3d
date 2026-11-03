use super::*;

const CHROMATIC: [usize; 10] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

pub(super) struct ColourModel {
    pub(super) set: color::PigmentSet,
    pub(super) cmf: Vec<f32>,
}

impl ColourModel {
    pub(super) fn load() -> Self {
        Self { set: color::load_pigments(), cmf: color::cmf_table() }
    }

    pub(super) fn lab(&self, m: &[f32; 16]) -> [f64; 3] {
        let l = color::mix_lab(&self.set, &self.cmf, m);
        [l[0] as f64, l[1] as f64, l[2] as f64]
    }
}

pub(super) fn delta_e(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

pub(super) fn colour_name(lab: [f64; 3]) -> String {
    let (l, a, b) = (lab[0], lab[1], lab[2]);
    let chroma = (a * a + b * b).sqrt();
    if chroma < 9.0 {
        return match l {
            x if x > 88.0 => "white",
            x if x > 68.0 => "light grey",
            x if x > 45.0 => "grey",
            x if x > 25.0 => "charcoal",
            _ => "black",
        }
        .to_string();
    }
    let hue = b.atan2(a).to_degrees().rem_euclid(360.0);
    let word = match hue {
        h if h < 15.0 => "crimson",
        h if h < 45.0 => if l < 45.0 { "brown" } else { "red" },
        h if h < 65.0 => if l < 50.0 { "umber" } else { "orange" },
        h if h < 80.0 => if l < 62.0 { "ochre" } else { "yellow" },
        h if h < 105.0 => if l < 65.0 { "olive" } else { "yellow" },
        h if h < 160.0 => "green",
        h if h < 215.0 => "teal",
        h if h < 285.0 => "blue",
        h if h < 330.0 => "violet",
        _ => "magenta",
    };
    let tone = if l > 80.0 { "pale " } else if l < 32.0 { "deep " } else { "" };
    format!("{}{}", tone, word)
}

pub(super) fn random_parts(rng: &mut StdRng) -> Vec<(usize, f32)> {
    random_parts_from(rng, &CHROMATIC)
}

pub(super) fn random_parts_from(rng: &mut StdRng, bases: &[usize]) -> Vec<(usize, f32)> {
    let base = *pick(rng, bases);
    let mut parts = vec![(base, 1.0f32)];
    if rng.random::<f64>() < 0.45 {
        let other = *pick(rng, &CHROMATIC);
        if other != base {
            parts.push((other, rng.random_range(0.1..0.6) as f32));
        }
    }
    parts
}

pub(super) fn tone_to(cm: &ColourModel, parts: &[(usize, f32)], target_l: f64, darkener: usize) -> [f32; 16] {
    let base = mix(parts, 0.0, 0.0);
    let l0 = cm.lab(&base)[0];
    let toward = if target_l > l0 { 0 } else { darkener };
    let blend_at = |t: f32| {
        let mut m = [0.0f32; 16];
        let total: f32 = base[..12].iter().sum::<f32>().max(1e-6);
        for i in 0..12 {
            m[i] = base[i] / total * (1.0 - t);
        }
        m[toward] += t;
        m
    };
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..22 {
        let mid = 0.5 * (lo + hi);
        let l = cm.lab(&blend_at(mid))[0];
        let reached = if toward == 0 { l >= target_l } else { l <= target_l };
        if reached { hi = mid; } else { lo = mid; }
    }
    blend_at(hi)
}

pub(super) fn named_paint(cm: &ColourModel, rng: &mut StdRng, m: [f32; 16], thinner: f32, yield_pa: f32, mu: f64) -> Paint {
    let mut m = m;
    m[12] = (thinner + 0.05 * gauss(rng) as f32).clamp(0.0, 0.75);
    m[13] = yield_pa;
    Paint { name: colour_name(cm.lab(&m)), mix: m, mu: lognormal(rng, mu, 0.2) }
}

pub(super) fn cm() -> &'static ColourModel {
    static CM: std::sync::OnceLock<ColourModel> = std::sync::OnceLock::new();
    CM.get_or_init(ColourModel::load)
}

pub(super) fn lch(l: f64, c: f64, h_deg: f64) -> [f64; 3] {
    let h = h_deg.to_radians();
    [l, c * h.cos(), c * h.sin()]
}

pub(super) fn hue_chroma(lab: [f64; 3]) -> (f64, f64) {
    (lab[2].atan2(lab[1]).to_degrees().rem_euclid(360.0), (lab[1] * lab[1] + lab[2] * lab[2]).sqrt())
}

pub(super) fn srgb_to_lab(rgb: [f32; 3]) -> [f64; 3] {
    let lin = |c: f32| {
        let c = c as f64;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let (r, g, b) = (lin(rgb[0]), lin(rgb[1]), lin(rgb[2]));
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.9505;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.089;
    let f = |t: f64| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

pub(super) fn build_mix(i: usize, j: usize, r: f64, white: f64, black: f64) -> [f32; 16] {
    let mut m = [0.0f32; 16];
    m[i] += (1.0 - r) as f32;
    m[j] += r as f32;
    m[0] += white as f32;
    m[1] += black as f32;
    let s: f32 = m[..12].iter().sum::<f32>().max(1e-6);
    for v in m[..12].iter_mut() {
        *v /= s;
    }
    m
}

pub(super) fn best_mix(target: [f64; 3], lab_of: &dyn Fn(&[f32; 16]) -> [f64; 3]) -> [f32; 16] {
    let n = color::NPIG;
    let mut best = (f64::MAX, 0usize, 0usize, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        for j in i..n {
            let ratios: &[f64] = if i == j { &[0.0] } else { &[0.2, 0.4, 0.6, 0.8] };
            for &r in ratios {
                for &w in &[0.0, 0.15, 0.4, 1.0, 2.5] {
                    for &b in &[0.0, 0.05, 0.15, 0.4] {
                        let e = delta_e(lab_of(&build_mix(i, j, r, w, b)), target);
                        if e < best.0 {
                            best = (e, i, j, r, w, b);
                        }
                    }
                }
            }
        }
    }
    let (mut e, i, j, mut r, mut w, mut b) = best;
    let mut step = 1.0;
    for _ in 0..5 {
        let mut improved = true;
        while improved {
            improved = false;
            let tries = [
                ((r + 0.1 * step).min(1.0), w, b),
                ((r - 0.1 * step).max(0.0), w, b),
                (r, w * (1.0 + 0.4 * step) + 0.05 * step, b),
                (r, (w * (1.0 - 0.3 * step) - 0.03 * step).max(0.0), b),
                (r, w, b * (1.0 + 0.4 * step) + 0.02 * step),
                (r, w, (b * (1.0 - 0.3 * step) - 0.01 * step).max(0.0)),
            ];
            for (r2, w2, b2) in tries {
                let e2 = delta_e(lab_of(&build_mix(i, j, r2, w2, b2)), target);
                if e2 < e - 1e-3 {
                    e = e2;
                    r = r2;
                    w = w2;
                    b = b2;
                    improved = true;
                }
            }
        }
        step *= 0.5;
    }
    build_mix(i, j, r, w, b)
}

pub fn match_rgb(rgb: [f32; 3]) -> [f32; 16] {
    let c = cm();
    best_mix(srgb_to_lab(rgb), &|m| srgb_to_lab(color::mix_rgb(&c.set, &c.cmf, m)))
}

pub(super) fn tone_mix(cm: &ColourModel, base: &[f32; 16], target_l: f64) -> [f32; 16] {
    let l0 = cm.lab(base)[0];
    let toward = if target_l > l0 { 0 } else if target_l < 40.0 { 1 } else { 9 };
    let total: f32 = base[..12].iter().sum::<f32>().max(1e-6);
    let blend_at = |t: f32| {
        let mut m = *base;
        for i in 0..12 {
            m[i] = base[i] / total * (1.0 - t);
        }
        m[toward] += t;
        m
    };
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..22 {
        let mid = 0.5 * (lo + hi);
        let l = cm.lab(&blend_at(mid))[0];
        let reached = if toward == 0 { l >= target_l } else { l <= target_l };
        if reached { hi = mid; } else { lo = mid; }
    }
    blend_at(hi)
}
