use super::*;

const GAMUT_BIN: f32 = 1.5;

const GAMUT_SLACK: f32 = 3.0;

pub(super) fn decode(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub(super) fn lab(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = rgb.map(|c| c.max(0.0));
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    let f = |t: f32| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

pub(super) fn de(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

pub(super) fn tonemap_inverse(y: f32) -> f32 {
    let (a, b, c, d, e) = (2.51f32, 0.03f32, 2.43f32, 0.59f32, 0.14f32);
    let y = y.clamp(0.0, 0.995);
    let qa = a - y * c;
    let qb = b - y * d;
    (-qb + (qb * qb + 4.0 * qa * y * e).sqrt()) / (2.0 * qa)
}

pub(super) fn tonemap(x: f32) -> f32 {
    ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
}

pub(crate) fn encode(c: f32) -> f32 {
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

pub(super) fn view_lab(rgb: [f32; 3], gain: [f32; 3]) -> [f32; 3] {
    lab([tonemap(gain[0] * rgb[0]), tonemap(gain[1] * rgb[1]), tonemap(gain[2] * rgb[2])])
}

pub(super) fn gamut_map(target: Vec<[f32; 3]>, palette: &[Swatch]) -> Vec<[f32; 3]> {
    let bin = |v: f32| (v / GAMUT_BIN).round() as i32;
    let keys: std::collections::HashSet<[i32; 3]> = target.iter().map(|t| t.map(bin)).collect();
    let keys: Vec<[i32; 3]> = keys.into_iter().collect();
    let found: std::collections::HashMap<[i32; 3], [f32; 3]> = keys
        .par_iter()
        .map(|k| {
            let want = k.map(|v| v as f32 * GAMUT_BIN);
            let near = palette.iter().map(|s| s.lab).min_by(|a, b| de(*a, want).total_cmp(&de(*b, want))).unwrap_or(want);
            (*k, near)
        })
        .collect();
    target
        .into_par_iter()
        .map(|t| {
            let near = found[&t.map(bin)];
            let d = de(t, near);
            if d <= GAMUT_SLACK {
                return t;
            }
            let f = GAMUT_SLACK / d;
            [near[0] + f * (t[0] - near[0]), near[1] + f * (t[1] - near[1]), near[2] + f * (t[2] - near[2])]
        })
        .collect()
}

pub(super) struct Swatch {
    pub(super) mix: [f32; 16],
    pub(super) rgb: [f32; 3],
    pub(super) lab: [f32; 3],
}

pub(super) fn swatches(set: &PigmentSet, cmf: &[f32]) -> Vec<Swatch> {
    let mut mixes: Vec<[f32; 16]> = Vec::new();
    let n = color::NPIG.min(set.pigments.len());
    let mut bases: Vec<[f32; 16]> = Vec::new();
    for i in 0..n {
        let mut m = [0.0f32; 16];
        m[i] = 1.0;
        bases.push(m);
        for j in i + 1..n {
            for (a, b) in [(3.0, 1.0), (1.0, 1.0), (1.0, 3.0)] {
                let mut m = [0.0f32; 16];
                m[i] = a;
                m[j] = b;
                bases.push(m);
            }
        }
    }
    for b in &bases {
        let total: f32 = b.iter().sum();
        for white in [0.0, 0.05, 0.15, 0.5, 1.5, 4.0, 10.0] {
            let mut m = *b;
            m[WHITE] += white * total;
            mixes.push(m);
        }
        if b[BLACK] == 0.0 {
            for black in [0.08, 0.25, 0.7] {
                let mut m = *b;
                m[BLACK] += black * total;
                mixes.push(m);
            }
        }
    }
    mixes
        .into_par_iter()
        .map(|m| {
            let total: f32 = m.iter().sum();
            let mut mix = [0.0f32; 16];
            for i in 0..n {
                mix[i] = m[i] / total;
            }
            let srgb = color::mix_rgb(set, cmf, &mix);
            Swatch { mix, rgb: srgb.map(decode), lab: lab(srgb.map(decode)) }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_endpoints() {
        let black = lab([0.0; 3]);
        let white = lab([1.0; 3]);
        assert!(black[0].abs() < 0.5);
        assert!((white[0] - 100.0).abs() < 0.5);
        assert!(white[1].abs() < 1.0 && white[2].abs() < 1.0);
    }

    #[test]
    fn colour_distance_is_a_metric() {
        let (a, b) = ([10.0, 2.0, 3.0], [20.0, 2.0, 3.0]);
        assert_eq!(de(a, a), 0.0);
        assert_eq!(de(a, b), de(b, a));
        assert!((de(a, b) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn tone_map_inverts() {
        for x in [0.01f32, 0.1, 0.5, 1.0, 3.0] {
            assert!((tonemap_inverse(tonemap(x)) - x).abs() < 0.02 * x.max(0.1), "{x}");
        }
    }

    #[test]
    fn srgb_round_trips() {
        for v in [0.0f32, 0.002, 0.04, 0.2, 0.5, 0.9, 1.0] {
            assert!((decode(encode(v)) - v).abs() < 1e-4, "{v}");
        }
    }

    #[test]
    fn gamut_map_keeps_reachable_and_pulls_in_the_rest() {
        let swatch = Swatch { mix: [0.0; 16], rgb: [0.5; 3], lab: [50.0, 0.0, 0.0] };
        let out = gamut_map(vec![[50.0, 0.0, 0.0], [90.0, 0.0, 0.0]], std::slice::from_ref(&swatch));
        assert_eq!(out[0], [50.0, 0.0, 0.0]);
        assert!(de(out[1], swatch.lab) <= GAMUT_SLACK + 0.01);
    }
}
