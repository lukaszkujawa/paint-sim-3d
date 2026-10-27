use serde::{Deserialize, Serialize};

pub const NBAND: usize = 16;
pub const NPIG: usize = 12;
pub const PIG_STRIDE: usize = 40;

#[derive(Clone, Serialize, Deserialize)]
pub struct Pigment {
    pub name: String,
    pub reflectance: Vec<f32>,
    pub scatter: f32,
    pub viscosity: f32,
    pub yield_stress: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PigmentSet {
    pub pigments: Vec<Pigment>,
}

pub fn band_wavelengths() -> [f32; NBAND] {
    let mut w = [0.0; NBAND];
    for (b, x) in w.iter_mut().enumerate() {
        *x = 410.0 + 20.0 * b as f32;
    }
    w
}

const CMF: [[f32; 4]; 32] = [
    [400.0, 0.0143, 0.0004, 0.0679],
    [410.0, 0.0435, 0.0012, 0.2074],
    [420.0, 0.1344, 0.0040, 0.6456],
    [430.0, 0.2839, 0.0116, 1.3856],
    [440.0, 0.3483, 0.0230, 1.7471],
    [450.0, 0.3362, 0.0380, 1.7721],
    [460.0, 0.2908, 0.0600, 1.6692],
    [470.0, 0.1954, 0.0910, 1.2876],
    [480.0, 0.0956, 0.1390, 0.8130],
    [490.0, 0.0320, 0.2080, 0.4652],
    [500.0, 0.0049, 0.3230, 0.2720],
    [510.0, 0.0093, 0.5030, 0.1582],
    [520.0, 0.0633, 0.7100, 0.0782],
    [530.0, 0.1655, 0.8620, 0.0422],
    [540.0, 0.2904, 0.9540, 0.0203],
    [550.0, 0.4334, 0.9950, 0.0087],
    [560.0, 0.5945, 0.9950, 0.0039],
    [570.0, 0.7621, 0.9520, 0.0021],
    [580.0, 0.9163, 0.8700, 0.0017],
    [590.0, 1.0263, 0.7570, 0.0011],
    [600.0, 1.0622, 0.6310, 0.0008],
    [610.0, 1.0026, 0.5030, 0.0003],
    [620.0, 0.8544, 0.3810, 0.0002],
    [630.0, 0.6424, 0.2650, 0.0000],
    [640.0, 0.4479, 0.1750, 0.0],
    [650.0, 0.2835, 0.1070, 0.0],
    [660.0, 0.1649, 0.0610, 0.0],
    [670.0, 0.0874, 0.0320, 0.0],
    [680.0, 0.0468, 0.0170, 0.0],
    [690.0, 0.0227, 0.0082, 0.0],
    [700.0, 0.0114, 0.0041, 0.0],
    [710.0, 0.0058, 0.0021, 0.0],
];

const D65: [f32; 32] = [
    82.8, 91.5, 93.4, 86.7, 104.9, 117.0, 117.8, 114.9, 115.9, 108.8, 109.4, 107.8, 104.8, 107.7, 104.4, 104.0,
    100.0, 96.3, 95.8, 88.7, 90.0, 89.6, 87.7, 83.3, 83.7, 80.0, 80.2, 82.3, 78.3, 69.7, 71.6, 74.3,
];

pub fn cmf_table() -> Vec<f32> {
    let mut t = vec![0.0f32; NBAND * 3];
    let mut ysum = 0.0;
    for b in 0..NBAND {
        let mut acc = [0.0f32; 3];
        for s in 0..2 {
            let k = 2 * b + s;
            let w = D65[k];
            acc[0] += CMF[k][1] * w;
            acc[1] += CMF[k][2] * w;
            acc[2] += CMF[k][3] * w;
        }
        t[b * 3] = acc[0];
        t[b * 3 + 1] = acc[1];
        t[b * 3 + 2] = acc[2];
        ysum += acc[1];
    }
    for v in t.iter_mut() {
        *v /= ysum;
    }
    t
}

fn interp(samples: &[(f32, f32)], lam: f32) -> f32 {
    if lam <= samples[0].0 {
        return samples[0].1;
    }
    for w in samples.windows(2) {
        if lam >= w[0].0 && lam <= w[1].0 {
            let t = (lam - w[0].0) / (w[1].0 - w[0].0);
            return w[0].1 + t * (w[1].1 - w[0].1);
        }
    }
    samples[samples.len() - 1].1
}

fn curve(samples: &[(f32, f32)]) -> Vec<f32> {
    band_wavelengths().iter().map(|&l| interp(samples, l).clamp(0.01, 0.98)).collect()
}

pub fn default_pigments() -> PigmentSet {
    let mk = |name: &str, s: f32, mu: f32, ty: f32, pts: &[(f32, f32)]| Pigment {
        name: name.to_string(),
        reflectance: curve(pts),
        scatter: s,
        viscosity: mu,
        yield_stress: ty,
    };
    PigmentSet {
        pigments: vec![
            mk("Titanium White", 3.0e4, 40.0, 40.0, &[(400.0, 0.86), (450.0, 0.92), (550.0, 0.94), (700.0, 0.93)]),
            mk("Ivory Black", 6.0e3, 30.0, 30.0, &[(400.0, 0.045), (550.0, 0.045), (700.0, 0.055)]),
            mk("Cadmium Yellow", 2.2e4, 35.0, 35.0, &[(400.0, 0.05), (460.0, 0.06), (490.0, 0.15), (510.0, 0.45), (530.0, 0.75), (560.0, 0.86), (700.0, 0.88)]),
            mk("Cadmium Red", 2.0e4, 35.0, 35.0, &[(400.0, 0.05), (560.0, 0.06), (590.0, 0.20), (610.0, 0.55), (630.0, 0.78), (700.0, 0.82)]),
            mk("Alizarin Crimson", 9.0e2, 25.0, 15.0, &[(400.0, 0.12), (430.0, 0.14), (470.0, 0.07), (540.0, 0.04), (580.0, 0.06), (610.0, 0.22), (640.0, 0.48), (700.0, 0.60)]),
            mk("Ultramarine Blue", 4.0e3, 30.0, 20.0, &[(400.0, 0.25), (440.0, 0.36), (470.0, 0.32), (520.0, 0.12), (580.0, 0.05), (700.0, 0.06)]),
            mk("Phthalo Blue", 1.0e3, 28.0, 15.0, &[(400.0, 0.12), (440.0, 0.22), (470.0, 0.26), (510.0, 0.14), (560.0, 0.04), (700.0, 0.03)]),
            mk("Phthalo Green", 1.0e3, 28.0, 15.0, &[(400.0, 0.04), (470.0, 0.09), (500.0, 0.27), (520.0, 0.30), (550.0, 0.17), (600.0, 0.04), (700.0, 0.03)]),
            mk("Yellow Ochre", 1.5e4, 38.0, 40.0, &[(400.0, 0.07), (470.0, 0.10), (520.0, 0.22), (570.0, 0.42), (620.0, 0.52), (700.0, 0.56)]),
            mk("Burnt Umber", 4.0e3, 38.0, 40.0, &[(400.0, 0.04), (500.0, 0.05), (600.0, 0.10), (700.0, 0.17)]),
            mk("Viridian", 2.0e3, 30.0, 20.0, &[(400.0, 0.06), (460.0, 0.12), (500.0, 0.26), (530.0, 0.22), (580.0, 0.07), (700.0, 0.05)]),
            mk("Cadmium Orange", 2.0e4, 35.0, 35.0, &[(400.0, 0.05), (520.0, 0.06), (550.0, 0.15), (580.0, 0.50), (610.0, 0.80), (700.0, 0.85)]),
        ],
    }
}

pub fn load_pigments() -> PigmentSet {
    let path = std::path::Path::new("assets/pigments.ron");
    if let Ok(txt) = std::fs::read_to_string(path) {
        if let Ok(set) = ron::from_str::<PigmentSet>(&txt) {
            if set.pigments.len() == NPIG {
                return set;
            }
        }
    }
    let set = default_pigments();
    if let Ok(txt) = ron::ser::to_string_pretty(&set, ron::ser::PrettyConfig::default()) {
        let _ = std::fs::create_dir_all("assets");
        let _ = std::fs::write(path, txt);
    }
    set
}

pub fn pigment_table(set: &PigmentSet) -> Vec<f32> {
    let mut t = vec![0.0f32; NPIG * PIG_STRIDE];
    for (i, p) in set.pigments.iter().enumerate().take(NPIG) {
        for b in 0..NBAND {
            let r = p.reflectance[b].clamp(0.01, 0.98);
            let ks = (1.0 - r) * (1.0 - r) / (2.0 * r);
            let s = p.scatter;
            t[i * PIG_STRIDE + b] = ks * s;
            t[i * PIG_STRIDE + NBAND + b] = s;
        }
        t[i * PIG_STRIDE + 32] = p.viscosity;
        t[i * PIG_STRIDE + 33] = p.yield_stress;
    }
    t
}

pub fn masstone_rgb(set: &PigmentSet, cmf: &[f32], i: usize) -> [f32; 3] {
    let p = &set.pigments[i];
    let mut xyz = [0.0f32; 3];
    for b in 0..NBAND {
        let r = p.reflectance[b];
        xyz[0] += r * cmf[b * 3];
        xyz[1] += r * cmf[b * 3 + 1];
        xyz[2] += r * cmf[b * 3 + 2];
    }
    xyz_to_srgb(xyz)
}

pub fn xyz_to_srgb(xyz: [f32; 3]) -> [f32; 3] {
    let r = 3.2406 * xyz[0] - 1.5372 * xyz[1] - 0.4986 * xyz[2];
    let g = -0.9689 * xyz[0] + 1.8758 * xyz[1] + 0.0415 * xyz[2];
    let b = 0.0557 * xyz[0] - 0.2040 * xyz[1] + 1.0570 * xyz[2];
    let enc = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    };
    [enc(r), enc(g), enc(b)]
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaintMix {
    pub frac: [f32; 16],
}

impl PaintMix {
    pub fn pure(i: usize) -> Self {
        let mut frac = [0.0; 16];
        frac[i] = 1.0;
        frac[13] = 0.0;
        Self { frac }
    }

    pub fn packed(&self) -> [u32; 8] {
        let mut o = [0u32; 8];
        for k in 0..8 {
            let a = half::f16::from_f32(self.frac[2 * k]).to_bits() as u32;
            let b = half::f16::from_f32(self.frac[2 * k + 1]).to_bits() as u32;
            o[k] = a | (b << 16);
        }
        o
    }

    pub fn normalized(&self) -> Self {
        let mut s = 0.0;
        for i in 0..13 {
            s += self.frac[i];
        }
        let mut o = *self;
        if s > 0.0 {
            for i in 0..13 {
                o.frac[i] /= s;
            }
        }
        o
    }
}
