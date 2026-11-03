use super::*;

const SCHEMES: [(&str, &[f64], f64); 5] = [
    ("complementary", &[180.0, 0.0, 210.0, 150.0], 1.0),
    ("split complementary", &[150.0, 210.0, 0.0, 180.0], 1.2),
    ("triadic", &[120.0, 240.0, 0.0, 180.0], 1.0),
    ("analogous", &[40.0, -40.0, 180.0, 75.0], 0.6),
    ("tetradic", &[90.0, 180.0, 270.0, 0.0], 0.6),
];

const DEEP_PIGMENTS: [usize; 7] = [4, 5, 6, 7, 10, 4, 9];

const MIN_HUE_GAP: f64 = 28.0;

const MIN_LINE_SPREAD: f64 = 26.0;

const MIN_GROUND_CONTRAST: f64 = 45.0;

const CLASSIC_SHARE: f64 = 0.2;

#[derive(Clone)]
pub(super) struct Paint {
    pub(super) name: String,
    pub(super) mix: [f32; 16],
    pub(super) mu: f64,
}

#[derive(Clone)]
pub struct Palette {
    pub(super) name: String,
    pub(super) scheme: String,
    pub(super) ground: Vec<Paint>,
    pub(super) ground_dark: Option<Paint>,
    pub(super) main: Paint,
    pub(super) second: Paint,
    pub(super) accents: Vec<Paint>,
}

pub(super) fn mix(parts: &[(usize, f32)], thinner: f32, yield_pa: f32) -> [f32; 16] {
    let mut m = [0.0f32; 16];
    for (i, a) in parts {
        m[*i] += *a;
    }
    m[12] = thinner;
    m[13] = yield_pa;
    m
}

pub(super) fn paint(rng: &mut StdRng, name: &str, parts: &[(usize, f32)], thinner: f32, yield_pa: f32, mu: f64) -> Paint {
    let mut jittered: Vec<(usize, f32)> = Vec::new();
    for (i, a) in parts {
        jittered.push((*i, a * lognormal(rng, 1.0, 0.22) as f32));
    }
    let th = (thinner + 0.06 * gauss(rng) as f32).clamp(0.0, 0.75);
    Paint { name: name.to_string(), mix: mix(&jittered, th, yield_pa), mu: lognormal(rng, mu, 0.2) }
}

pub(super) fn harmonic_palette(rng: &mut StdRng, cm: &ColourModel) -> Palette {
    let key = *weighted(rng, &[("dark", 1.2), ("mid", 1.3), ("light", 0.6)]);
    let target_l = match key {
        "dark" => rng.random_range(28.0..38.0),
        "mid" => rng.random_range(38.0..56.0),
        _ => rng.random_range(64.0..80.0),
    };
    let darkener = if target_l < 42.0 || rng.random::<f64>() < 0.5 { 1 } else { 9 };
    let ground_parts = match key {
        "dark" => random_parts_from(rng, &DEEP_PIGMENTS),
        _ => random_parts(rng),
    };
    let g0 = tone_to(cm, &ground_parts, target_l, darkener);
    let gl = cm.lab(&g0);
    let (h0, c0) = hue_chroma(gl);
    let h0 = if c0 < 8.0 { rng.random_range(0.0..360.0) } else { h0 };
    let lab_of = |m: &[f32; 16]| cm.lab(m);
    let side = if rng.random::<f64>() < 0.5 { 1.0 } else { -1.0 };
    let companion = if rng.random::<f64>() < 0.6 {
        lch(gl[0] + rng.random_range(-6.0..8.0), (c0 * rng.random_range(0.8..1.2)).max(15.0), h0 + side * rng.random_range(25.0..50.0))
    } else {
        lch(gl[0] + rng.random_range(-10.0..10.0), c0 * rng.random_range(0.3..0.6), h0 + rng.random_range(-15.0..15.0))
    };
    let g1 = best_mix(companion, &lab_of);
    let (scheme, offsets, _) = *weighted(rng, &SCHEMES.iter().map(|s| ((s.0, s.1, s.2), s.2)).collect::<Vec<_>>());
    let wants_light = |k: usize, flip: bool| -> bool {
        if gl[0] < 45.0 {
            true
        } else if gl[0] > 62.0 {
            false
        } else {
            (k % 2 == 0) ^ flip
        }
    };
    let flip = rng.random::<f64>() < 0.5;
    let contrast = |lab: [f64; 3]| (lab[0] - gl[0]).abs() * 1.4 + delta_e(lab, gl);
    let mut fg: Vec<[f32; 16]> = Vec::new();
    let mut labs: Vec<[f64; 3]> = Vec::new();
    let neutral_main = rng.random::<f64>() < 0.6;
    let mut hue_slot = 0usize;
    for k in 0..4 {
        let light = wants_light(k, flip);
        let mut best: Option<([f32; 16], f64)> = None;
        for attempt in 0..10 {
            let push = (attempt % 5) as f64 * 7.0;
            if attempt == 5 && !(k == 0 && neutral_main) {
                hue_slot += 1;
            }
            let m = if k == 0 && neutral_main {
                let t = if light { lch((90.0 - push * 0.5).max(70.0), rng.random_range(2.0..8.0), h0) } else { lch(rng.random_range(16.0..24.0), rng.random_range(0.0..6.0), h0 + 180.0) };
                best_mix(t, &lab_of)
            } else {
                let h = h0 + offsets[hue_slot % offsets.len()] + rng.random_range(-10.0..10.0);
                let l = if light { rng.random_range(62.0..88.0) + push * 0.4 } else { rng.random_range(18.0..42.0) - push * 0.5 };
                best_mix(lch(l.clamp(12.0, 94.0), rng.random_range(40.0..72.0), h), &lab_of)
            };
            let lab = cm.lab(&m);
            let spread = labs.iter().map(|o| delta_e(lab, *o)).fold(200.0f64, f64::min);
            let (hk, ck) = hue_chroma(lab);
            let hue_gap = labs
                .iter()
                .filter(|o| ck > 15.0 && hue_chroma(**o).1 > 15.0)
                .map(|o| {
                    let d = (hue_chroma(*o).0 - hk).rem_euclid(360.0);
                    d.min(360.0 - d)
                })
                .fold(180.0f64, f64::min);
            let name = colour_name(lab);
            let fresh = !labs.iter().any(|o| colour_name(*o) == name) && name != colour_name(gl);
            let score = contrast(lab).min(MIN_GROUND_CONTRAST * 1.5) + spread.min(MIN_LINE_SPREAD * 1.5) + hue_gap.min(MIN_HUE_GAP) + if fresh { 30.0 } else { 0.0 };
            if best.as_ref().map(|b| score > b.1).unwrap_or(true) {
                best = Some((m, score));
            }
            if contrast(lab) >= MIN_GROUND_CONTRAST && spread >= MIN_LINE_SPREAD && hue_gap >= MIN_HUE_GAP && fresh {
                break;
            }
        }
        if !(k == 0 && neutral_main) {
            hue_slot += 1;
        }
        let m = best.unwrap().0;
        labs.push(cm.lab(&m));
        fg.push(m);
    }
    let body = [(0.2, 0.0, 10.0), (0.15, 5.0, 10.0), (0.22, 0.0, 7.5), (0.22, 0.0, 7.5)];
    let mut fgp: Vec<Paint> = fg.iter().zip(body.iter()).map(|(m, &(th, y, mu))| named_paint(cm, rng, *m, th, y, mu)).collect();
    let accents = fgp.split_off(2);
    let mut pal = Palette {
        name: String::new(),
        scheme: format!("{} on {}", scheme, key),
        ground: vec![named_paint(cm, rng, g0, 0.45, 0.0, 4.0), named_paint(cm, rng, g1, 0.45, 0.0, 4.0)],
        ground_dark: None,
        main: fgp[0].clone(),
        second: fgp[1].clone(),
        accents,
    };
    pal.rebuild(cm);
    pal
}

impl Palette {
    pub const SLOTS: usize = 6;

    pub fn random(seed: u64) -> Palette {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15);
        palette(&mut rng)
    }

    pub(super) fn slot(&self, s: usize) -> &Paint {
        match s {
            0 | 1 => &self.ground[s],
            2 => &self.main,
            3 => &self.second,
            _ => &self.accents[s - 4],
        }
    }

    pub(super) fn slot_mut(&mut self, s: usize) -> &mut Paint {
        match s {
            0 | 1 => &mut self.ground[s],
            2 => &mut self.main,
            3 => &mut self.second,
            _ => &mut self.accents[s - 4],
        }
    }

    pub fn slot_mix(&self, s: usize) -> [f32; 16] {
        self.slot(s).mix
    }

    pub fn slot_name(&self, s: usize) -> &str {
        &self.slot(s).name
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set_slot(&mut self, s: usize, pigments: &[f32; 16]) {
        let total: f32 = pigments[..12].iter().sum::<f32>().max(1e-6);
        let p = self.slot_mut(s);
        for i in 0..12 {
            p.mix[i] = pigments[i] / total;
        }
        self.scheme = "custom".to_string();
        self.rebuild(cm());
    }

    pub(super) fn rebuild(&mut self, cm: &ColourModel) {
        while self.ground.len() < 2 {
            let g = self.ground[0].clone();
            self.ground.push(g);
        }
        while self.accents.len() < 2 {
            let w = Paint { name: String::new(), mix: mix(&[(0, 1.0)], 0.22, 0.0), mu: 7.5 };
            self.accents.push(w);
        }
        self.ground.truncate(2);
        self.accents.truncate(2);
        let l0 = cm.lab(&self.ground[0].mix)[0];
        let mut light = self.ground[0].clone();
        light.mix = tone_mix(cm, &self.ground[0].mix, (l0 + 12.0).min(95.0));
        let mut dark = self.ground[0].clone();
        dark.mix = tone_mix(cm, &self.ground[0].mix, (l0 - 15.0).max(8.0));
        let mut seen: Vec<(String, f64)> = Vec::new();
        for s in 0..Self::SLOTS {
            let lab = cm.lab(&self.slot(s).mix);
            let base = colour_name(lab);
            let mut nm = base.clone();
            if let Some((_, l_other)) = seen.iter().find(|(n, _)| *n == base) {
                let stripped = base.trim_start_matches("pale ").trim_start_matches("deep ");
                nm = if stripped == "white" { "off-white".to_string() } else { format!("{} {}", if lab[0] > *l_other { "light" } else { "dark" }, stripped) };
            }
            let mut k = 2;
            let first = nm.clone();
            while seen.iter().any(|(n, _)| *n == nm) {
                nm = format!("{} {}", first, k);
                k += 1;
            }
            seen.push((nm.clone(), lab[0]));
            self.slot_mut(s).name = nm;
        }
        light.name = format!("light {}", self.ground[0].name);
        dark.name = format!("dark {}", self.ground[0].name);
        self.ground.push(light);
        self.ground_dark = Some(dark);
        let fg: Vec<&str> = (2..Self::SLOTS).map(|s| self.slot_name(s)).collect();
        self.name = format!("{}: {} and {} ground, {}", self.scheme, self.ground[0].name, self.ground[1].name, fg.join(", "));
    }
}

pub(super) fn palette(rng: &mut StdRng) -> Palette {
    if rng.random::<f64>() >= CLASSIC_SHARE {
        return harmonic_palette(rng, cm());
    }
    let mut pal = classic_palette(rng);
    pal.scheme = pal.name.clone();
    pal.rebuild(cm());
    pal
}

pub(super) fn classic_palette(rng: &mut StdRng) -> Palette {
    let family = rng.random_range(0..6);
    match family {
        0 => Palette {
            name: "autumn rhythm".to_string(),
            ground_dark: None,
            scheme: String::new(),
            ground: vec![
                paint(rng, "umber", &[(9, 0.6), (8, 0.3), (1, 0.1)], 0.45, 0.0, 4.0),
                paint(rng, "white", &[(0, 0.9), (8, 0.1)], 0.4, 0.0, 4.0),
                paint(rng, "teal", &[(10, 0.5), (6, 0.2), (0, 0.3)], 0.45, 0.0, 4.0),
            ],
            main: paint(rng, "black", &[(1, 1.0)], 0.2, 0.0, 10.0),
            second: paint(rng, "white", &[(0, 1.0)], 0.15, 5.0, 10.0),
            accents: vec![
                paint(rng, "umber", &[(9, 0.8), (1, 0.2)], 0.2, 0.0, 8.0),
                paint(rng, "ochre", &[(8, 0.9), (2, 0.1)], 0.2, 0.0, 7.0),
                paint(rng, "silver", &[(0, 0.75), (1, 0.25)], 0.35, 0.0, 1.5),
                paint(rng, "teal", &[(10, 0.5), (6, 0.3), (0, 0.2)], 0.2, 0.0, 8.0),
            ],
        },
        1 => Palette {
            name: "blue poles".to_string(),
            ground_dark: None,
            scheme: String::new(),
            ground: vec![
                paint(rng, "teal", &[(10, 0.4), (6, 0.2), (0, 0.3), (9, 0.1)], 0.5, 0.0, 3.0),
                paint(rng, "pale yellow", &[(2, 0.25), (0, 0.55), (8, 0.2)], 0.5, 0.0, 3.0),
                paint(rng, "white", &[(0, 0.95), (5, 0.05)], 0.45, 0.0, 3.0),
            ],
            main: paint(rng, "deep blue", &[(5, 0.7), (1, 0.3)], 0.2, 0.0, 10.0),
            second: paint(rng, "white", &[(0, 1.0)], 0.15, 5.0, 10.0),
            accents: vec![
                paint(rng, "orange", &[(11, 0.85), (3, 0.15)], 0.25, 0.0, 7.0),
                paint(rng, "yellow", &[(2, 0.75), (11, 0.1), (8, 0.15)], 0.3, 0.0, 6.0),
                paint(rng, "silver", &[(0, 0.75), (1, 0.25)], 0.35, 0.0, 1.5),
                paint(rng, "crimson", &[(4, 0.8), (3, 0.2)], 0.25, 0.0, 7.0),
            ],
        },
        2 => Palette {
            name: "convergence".to_string(),
            ground_dark: None,
            scheme: String::new(),
            ground: vec![
                paint(rng, "black", &[(1, 0.9), (9, 0.1)], 0.4, 0.0, 4.0),
                paint(rng, "umber", &[(9, 0.8), (1, 0.2)], 0.4, 0.0, 4.0),
            ],
            main: paint(rng, "white", &[(0, 1.0)], 0.15, 5.0, 10.0),
            second: paint(rng, "yellow", &[(2, 0.85), (0, 0.15)], 0.25, 0.0, 8.0),
            accents: vec![
                paint(rng, "red", &[(3, 0.8), (11, 0.2)], 0.2, 0.0, 8.0),
                paint(rng, "ultramarine", &[(5, 0.85), (0, 0.15)], 0.2, 0.0, 8.0),
                paint(rng, "orange", &[(11, 0.85), (2, 0.15)], 0.25, 0.0, 7.0),
                paint(rng, "white", &[(0, 1.0)], 0.3, 0.0, 6.0),
            ],
        },
        3 => Palette {
            name: "earth".to_string(),
            ground_dark: None,
            scheme: String::new(),
            ground: vec![
                paint(rng, "ochre", &[(8, 0.8), (0, 0.15), (9, 0.05)], 0.45, 0.0, 4.0),
                paint(rng, "umber", &[(9, 0.7), (1, 0.1), (8, 0.2)], 0.45, 0.0, 4.0),
            ],
            main: paint(rng, "umber black", &[(9, 0.55), (1, 0.45)], 0.2, 0.0, 10.0),
            second: paint(rng, "white", &[(0, 0.95), (8, 0.05)], 0.15, 5.0, 10.0),
            accents: vec![
                paint(rng, "red", &[(3, 0.7), (4, 0.3)], 0.2, 0.0, 8.0),
                paint(rng, "ochre", &[(8, 0.9), (2, 0.1)], 0.2, 0.0, 7.0),
                paint(rng, "orange", &[(11, 0.8), (8, 0.2)], 0.25, 0.0, 7.0),
                paint(rng, "black", &[(1, 1.0)], 0.2, 0.0, 10.0),
            ],
        },
        4 => Palette {
            name: "monochrome".to_string(),
            ground_dark: None,
            scheme: String::new(),
            ground: vec![
                paint(rng, "grey", &[(0, 0.7), (1, 0.3)], 0.45, 0.0, 4.0),
                paint(rng, "white", &[(0, 1.0)], 0.4, 0.0, 4.0),
            ],
            main: paint(rng, "black", &[(1, 1.0)], 0.2, 0.0, 10.0),
            second: paint(rng, "white", &[(0, 1.0)], 0.15, 5.0, 10.0),
            accents: vec![
                paint(rng, "silver", &[(0, 0.75), (1, 0.25)], 0.35, 0.0, 1.5),
                paint(rng, "charcoal", &[(1, 0.85), (0, 0.15)], 0.25, 0.0, 8.0),
                paint(rng, "white", &[(0, 1.0)], 0.3, 0.0, 6.0),
            ],
        },
        _ => Palette {
            name: "lavender mist".to_string(),
            ground_dark: None,
            scheme: String::new(),
            ground: vec![
                paint(rng, "white", &[(0, 1.0)], 0.45, 0.0, 3.0),
                paint(rng, "pink", &[(0, 0.85), (4, 0.15)], 0.5, 0.0, 3.0),
                paint(rng, "lavender", &[(0, 0.8), (5, 0.12), (4, 0.08)], 0.5, 0.0, 3.0),
            ],
            main: paint(rng, "silver grey", &[(0, 0.6), (1, 0.4)], 0.2, 0.0, 10.0),
            second: paint(rng, "black", &[(1, 1.0)], 0.2, 0.0, 10.0),
            accents: vec![
                paint(rng, "crimson", &[(4, 0.85), (0, 0.15)], 0.25, 0.0, 7.0),
                paint(rng, "ochre", &[(8, 0.85), (0, 0.15)], 0.25, 0.0, 7.0),
                paint(rng, "teal", &[(10, 0.5), (6, 0.2), (0, 0.3)], 0.25, 0.0, 7.0),
                paint(rng, "white", &[(0, 1.0)], 0.3, 0.0, 6.0),
            ],
        },
    }
}
