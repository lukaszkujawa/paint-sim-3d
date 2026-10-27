use crate::tools::ToolKind;
use glam::{DVec2, DVec3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct Command {
    pub kind: ToolKind,
    pub pos: DVec3,
    pub vel: DVec3,
    pub flowing: bool,
    pub height: f64,
    pub mu: f64,
    pub sigma: f64,
    pub mix: [f32; 16],
    pub stick_diameter: f64,
    pub bottle_flow: f64,
    pub bottle_radius: f64,
    pub bucket_flow: f64,
    pub bucket_radius: f64,
    pub pour_rate: f64,
    pub pour_radius: f64,
    pub knife_width: f64,
    pub knife_height: f64,
    pub tilt: Option<(f64, f64)>,
    pub reload: bool,
    pub cut: bool,
    pub cure_rate: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Style {
    Flood,
    Drip,
    Pour,
    Fling,
    Dots,
    Lasso,
    Snap,
    Dry,
    Wash,
    Scrape,
    Tilt,
}

#[derive(Clone)]
struct Paint {
    name: String,
    mix: [f32; 16],
    mu: f64,
}

struct Palette {
    name: &'static str,
    ground: Vec<Paint>,
    main: Paint,
    second: Paint,
    accents: Vec<Paint>,
}

struct Layer {
    name: String,
    mix: [f32; 16],
    mu: f64,
    style: Style,
    duration: f64,
    pace: f64,
    loop_chance: f64,
    height_median: f64,
    stick_diameter: f64,
    flow: f64,
    pour_radius: f64,
    knife_width: f64,
    tilt: (f64, f64),
}

struct Gesture {
    t_start: f64,
    t_end: f64,
    speed: f64,
    height: f64,
    turn_sigma: f64,
    flow_until: f64,
    hop_at: f64,
    centre: DVec2,
    radius: f64,
    phase0: f64,
    lissajous: bool,
    drift: DVec2,
    wobble: f64,
}

pub struct Painter {
    rng: StdRng,
    w: f64,
    h: f64,
    t: f64,
    layer: usize,
    layer_t: f64,
    layers: Vec<Layer>,
    gesture: Option<Gesture>,
    pos: DVec2,
    heading: f64,
    ang_vel: f64,
    speed: f64,
    speed_noise: f64,
    height_noise: f64,
    cut_pending: bool,
    reload_pending: bool,
    pub palette: String,
    pub ground: String,
    pub seed: u64,
    pub done: bool,
}

const RHO: f64 = 1000.0;
const G: f64 = 9.81;

fn mix(parts: &[(usize, f32)], thinner: f32, yield_pa: f32) -> [f32; 16] {
    let mut m = [0.0f32; 16];
    for (i, a) in parts {
        m[*i] += *a;
    }
    m[12] = thinner;
    m[13] = yield_pa;
    m
}

pub fn transition_speed(q: f64, h: f64, nu: f64) -> f64 {
    let hs = h * (G / (nu * nu)).powf(1.0 / 3.0);
    let qs = q * (G / nu.powi(5)).powf(1.0 / 3.0);
    let us = 3.04 * qs.sqrt() * hs.powf(0.75);
    us * (G * nu).powf(1.0 / 3.0)
}

pub fn stick_flow(d: f64, nu: f64) -> f64 {
    0.016 * d * d * (G * nu * 0.5).powf(0.25)
}

fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-12);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn lognormal(rng: &mut StdRng, median: f64, sigma: f64) -> f64 {
    median * (sigma * gauss(rng)).exp()
}

fn pick<'a, T>(rng: &mut StdRng, items: &'a [T]) -> &'a T {
    &items[rng.random_range(0..items.len())]
}

fn weighted<'a, T>(rng: &mut StdRng, items: &'a [(T, f64)]) -> &'a T {
    let total: f64 = items.iter().map(|(_, w)| w).sum();
    let mut u = rng.random::<f64>() * total;
    for (item, w) in items {
        if u < *w {
            return item;
        }
        u -= w;
    }
    &items[items.len() - 1].0
}

fn paint(rng: &mut StdRng, name: &str, parts: &[(usize, f32)], thinner: f32, yield_pa: f32, mu: f64) -> Paint {
    let mut jittered: Vec<(usize, f32)> = Vec::new();
    for (i, a) in parts {
        jittered.push((*i, a * lognormal(rng, 1.0, 0.22) as f32));
    }
    let th = (thinner + 0.06 * gauss(rng) as f32).clamp(0.0, 0.75);
    Paint { name: name.to_string(), mix: mix(&jittered, th, yield_pa), mu: lognormal(rng, mu, 0.2) }
}

fn palette(rng: &mut StdRng) -> Palette {
    let family = rng.random_range(0..6);
    match family {
        0 => Palette {
            name: "autumn rhythm",
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
            name: "blue poles",
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
            name: "convergence",
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
            name: "earth",
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
            name: "monochrome",
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
            name: "lavender mist",
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

fn thinned(p: &Paint, thinner: f32, yield_pa: f32, mu: f64) -> Paint {
    let mut m = p.mix;
    m[12] = thinner;
    m[13] = yield_pa;
    Paint { name: p.name.clone(), mix: m, mu }
}

fn layer(name: String, p: &Paint, style: Style, duration: f64) -> Layer {
    let (pace, loop_chance, height_median, stick_diameter, flow) = match style {
        Style::Flood => (0.15, 0.0, 0.12, 0.01, 4.0e-5),
        Style::Dry | Style::Tilt => (0.0, 0.0, 0.2, 0.01, 0.0),
        Style::Pour => (1.5, 0.12, 0.08, 0.008, 2.5e-6),
        Style::Lasso => (1.4, 0.0, 0.22, 0.014, 3.0e-6),
        Style::Drip => (1.7, 0.1, 0.14, 0.008, 2.0e-6),
        Style::Snap => (4.0, 0.0, 0.3, 0.012, 2.0e-6),
        Style::Fling => (2.8, 0.0, 0.4, 0.01, 2.0e-6),
        Style::Dots => (0.0, 0.0, 0.09, 0.011, 4.0e-7),
        Style::Wash => (0.35, 0.0, 0.1, 0.01, 1.2e-5),
        Style::Scrape => (0.18, 0.0, 0.0, 0.01, 0.0),
    };
    Layer { name, mix: p.mix, mu: p.mu, style, duration, pace, loop_chance, height_median, stick_diameter, flow, pour_radius: 0.02, knife_width: 0.08, tilt: (0.0, 0.0) }
}

fn plan(rng: &mut StdRng, pal: &Palette) -> (Vec<Layer>, String) {
    let mut layers: Vec<Layer> = Vec::new();
    let dry = |rng: &mut StdRng, layers: &mut Vec<Layer>, chance: f64| {
        if rng.random::<f64>() < chance {
            let d = rng.random_range(3.5..6.0);
            layers.push(layer("drying".to_string(), &Paint { name: String::new(), mix: mix(&[(1, 1.0)], 0.2, 0.0), mu: 10.0 }, Style::Dry, d));
        }
    };
    let grounds = [("raw canvas", 1.0), ("floods", 2.0), ("washes", 2.0), ("scraped", 1.5), ("tilt run", 1.3)];
    let ground = *weighted(rng, &grounds);
    let mut gcols: Vec<Paint> = pal.ground.clone();
    for i in (1..gcols.len()).rev() {
        let j = rng.random_range(0..=i);
        gcols.swap(i, j);
    }
    match ground {
        "floods" => {
            let n = rng.random_range(1..=gcols.len().min(3));
            for p in gcols.iter().take(n) {
                let d = rng.random_range(7.0..14.0);
                layers.push(layer(format!("{} flood", p.name), p, Style::Flood, d));
            }
        }
        "washes" => {
            let n = rng.random_range(2..=gcols.len().min(3));
            for p in gcols.iter().take(n) {
                let d = rng.random_range(8.0..14.0);
                let th = (p.mix[12] + 0.1).min(0.65);
                let mut l = layer(format!("{} sweep", p.name), &thinned(p, th, 0.0, 2.5), Style::Wash, d);
                l.pour_radius = lognormal(rng, 0.02, 0.3).clamp(0.01, 0.035);
                layers.push(l);
            }
        }
        "scraped" => {
            let rounds = rng.random_range(1..=2);
            for k in 0..rounds {
                let p = &gcols[k % gcols.len()];
                let d = rng.random_range(5.0..9.0);
                layers.push(layer(format!("{} pools", p.name), &thinned(p, 0.25, 0.0, 6.0), Style::Flood, d));
                let mut l = layer(format!("{} scraped", p.name), p, Style::Scrape, rng.random_range(10.0..16.0));
                l.knife_width = lognormal(rng, 0.08, 0.3).clamp(0.04, 0.14);
                layers.push(l);
            }
        }
        "tilt run" => {
            let n = rng.random_range(1..=gcols.len().min(2));
            for p in gcols.iter().take(n) {
                let d = rng.random_range(6.0..9.0);
                layers.push(layer(format!("{} pools", p.name), &thinned(p, 0.35, 0.0, 4.0), Style::Flood, d));
            }
            let mut l = layer("tilt run".to_string(), &gcols[0], Style::Tilt, rng.random_range(6.0..10.0));
            l.tilt = (rng.random_range(25.0..40.0), rng.random_range(-180.0..180.0));
            layers.push(l);
            if rng.random::<f64>() < 0.5 {
                let mut l = layer("tilt back".to_string(), &gcols[0], Style::Tilt, rng.random_range(4.0..7.0));
                l.tilt = (rng.random_range(15.0..30.0), layers.last().unwrap().tilt.1 + 180.0);
                layers.push(l);
            }
        }
        _ => {}
    }
    if ground != "raw canvas" {
        dry(rng, &mut layers, 0.85);
    }
    let skel_style = *weighted(rng, &[(Style::Pour, 1.0), (Style::Drip, 1.0)]);
    let mut skel = layer(format!("{} skeleton", pal.main.name), &pal.main, skel_style, rng.random_range(30.0..50.0));
    skel.loop_chance = rng.random_range(0.08..0.22);
    layers.push(skel);
    if rng.random::<f64>() < 0.6 {
        layers.push(layer(format!("{} lasso", pal.main.name), &thinned(&pal.main, 0.12, 0.0, 12.0), Style::Lasso, rng.random_range(12.0..20.0)));
    }
    dry(rng, &mut layers, 0.5);
    let n_mid = rng.random_range(3..=6);
    let mut last_name = String::new();
    for _ in 0..n_mid {
        let which = *weighted(rng, &[("second", 0.4), ("accent", 0.45), ("main", 0.15)]);
        let p = match which {
            "second" => pal.second.clone(),
            "main" => pal.main.clone(),
            _ => pick(rng, &pal.accents).clone(),
        };
        if p.name == last_name && rng.random::<f64>() < 0.7 {
            continue;
        }
        last_name = p.name.clone();
        let style = *weighted(rng, &[(Style::Drip, 3.0), (Style::Pour, 2.5), (Style::Lasso, 1.2), (Style::Snap, 1.2), (Style::Fling, 1.0), (Style::Dots, 0.6)]);
        let (name, pp, dur) = match style {
            Style::Snap => (format!("{} snaps", p.name), thinned(&p, 0.45, 0.0, 1.5), rng.random_range(10.0..16.0)),
            Style::Fling => (format!("{} flings", p.name), thinned(&p, 0.5, 0.0, 1.0), rng.random_range(8.0..14.0)),
            Style::Lasso => (format!("{} lasso", p.name), thinned(&p, 0.12, 0.0, 12.0), rng.random_range(10.0..18.0)),
            Style::Dots => (format!("{} dots", p.name), p.clone(), rng.random_range(8.0..13.0)),
            Style::Pour => (format!("{} pours", p.name), p.clone(), rng.random_range(18.0..32.0)),
            _ => (format!("{} lines", p.name), p.clone(), rng.random_range(16.0..28.0)),
        };
        layers.push(layer(name, &pp, style, dur));
        if matches!(style, Style::Pour | Style::Drip | Style::Lasso) {
            dry(rng, &mut layers, 0.3);
        }
    }
    let n_acc = rng.random_range(1..=3);
    let mut used: Vec<String> = Vec::new();
    for _ in 0..n_acc {
        let mut p = pick(rng, &pal.accents).clone();
        for _ in 0..4 {
            if !used.contains(&p.name) {
                break;
            }
            p = pick(rng, &pal.accents).clone();
        }
        used.push(p.name.clone());
        let style = *weighted(rng, &[(Style::Dots, 1.0), (Style::Fling, 1.3), (Style::Snap, 1.0)]);
        let (name, pp) = match style {
            Style::Dots => (format!("{} drops", p.name), p.clone()),
            Style::Fling => (format!("{} spatter", p.name), thinned(&p, 0.55, 0.0, 0.6)),
            _ => (format!("{} snaps", p.name), thinned(&p, 0.45, 0.0, 1.2)),
        };
        layers.push(layer(name, &pp, style, rng.random_range(7.0..13.0)));
    }
    if rng.random::<f64>() < 0.7 {
        let p = if rng.random::<f64>() < 0.6 { pal.main.clone() } else { pal.second.clone() };
        let mut l = layer(format!("{} tracery", p.name), &thinned(&p, p.mix[12], 0.0, 6.0), Style::Drip, rng.random_range(18.0..30.0));
        l.stick_diameter = rng.random_range(0.005..0.0065);
        l.pace = 2.1;
        l.loop_chance = 0.06;
        layers.push(l);
        if rng.random::<f64>() < 0.5 {
            let q = if p.name == pal.main.name { pal.second.clone() } else { pal.main.clone() };
            let mut l = layer(format!("{} tracery", q.name), &thinned(&q, q.mix[12], 0.0, 6.0), Style::Drip, rng.random_range(14.0..24.0));
            l.stick_diameter = rng.random_range(0.005..0.0065);
            l.pace = 2.1;
            l.loop_chance = 0.06;
            layers.push(l);
        }
    }
    if rng.random::<f64>() < 0.4 {
        layers.push(layer(format!("{} drops", pal.main.name), &pal.main, Style::Dots, rng.random_range(6.0..10.0)));
    }
    (layers, ground.to_string())
}

impl Painter {
    pub fn new(seed: u64, w: f64, h: f64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let pal = palette(&mut rng);
        let (layers, ground) = plan(&mut rng, &pal);
        let total: f64 = layers.iter().map(|l| l.duration).sum();
        let names: Vec<String> = layers.iter().map(|l| format!("{} {:.0}s", l.name, l.duration)).collect();
        log::warn!("pollock seed {} palette {} ground {} total {:.0} s: {}", seed, pal.name, ground, total, names.join(", "));
        let pos = DVec2::new(w * rng.random_range(0.2..0.8), h * rng.random_range(0.2..0.8));
        Self {
            rng,
            w,
            h,
            t: 0.0,
            layer: 0,
            layer_t: 0.0,
            layers,
            gesture: None,
            pos,
            heading: 0.0,
            ang_vel: 0.0,
            speed: 0.0,
            speed_noise: 0.0,
            height_noise: 0.0,
            cut_pending: false,
            reload_pending: false,
            palette: pal.name.to_string(),
            ground,
            seed,
            done: false,
        }
    }

    pub fn skip_to(&mut self, name: &str) {
        if let Some(i) = self.layers.iter().position(|l| l.name == name) {
            self.layer = i;
            self.layer_t = 0.0;
            self.gesture = None;
        }
    }

    pub fn status(&self) -> String {
        if self.done {
            return format!("{} on {}: finished", self.palette, self.ground);
        }
        let l = &self.layers[self.layer];
        format!("{} on {}  layer {}/{} {}  {:.0}/{:.0} s", self.palette, self.ground, self.layer + 1, self.layers.len(), l.name, self.layer_t, l.duration)
    }

    fn gauss(&mut self) -> f64 {
        gauss(&mut self.rng)
    }

    fn lognormal(&mut self, median: f64, sigma: f64) -> f64 {
        lognormal(&mut self.rng, median, sigma)
    }

    fn layer_nu(&self, li: usize) -> f64 {
        let l = &self.layers[li];
        let thin = (l.mix[12] as f64).clamp(0.0, 0.95);
        l.mu * (1.0 - thin).powi(3) / RHO
    }

    fn layer_flow(&self, li: usize) -> f64 {
        let l = &self.layers[li];
        match l.style {
            Style::Pour | Style::Flood | Style::Lasso | Style::Wash => l.flow,
            _ => stick_flow(l.stick_diameter, self.layer_nu(li)),
        }
    }

    fn teleport(&mut self) {
        self.pos = DVec2::new(self.w * self.rng.random_range(0.1..0.9), self.h * self.rng.random_range(0.1..0.9));
        self.cut_pending = true;
    }

    fn run_to_border(&self, from: DVec2, dir: DVec2, margin: f64) -> f64 {
        let tx = if dir.x > 1e-9 { (self.w - margin - from.x) / dir.x } else if dir.x < -1e-9 { (margin - from.x) / dir.x } else { f64::INFINITY };
        let ty = if dir.y > 1e-9 { (self.h - margin - from.y) / dir.y } else if dir.y < -1e-9 { (margin - from.y) / dir.y } else { f64::INFINITY };
        tx.min(ty).max(0.0)
    }

    fn new_gesture(&mut self) {
        let li = self.layer;
        let style = self.layers[li].style;
        let pace = self.layers[li].pace;
        let hm = self.layers[li].height_median;
        let loop_chance = self.layers[li].loop_chance;
        let nu = self.layer_nu(li);
        let q = self.layer_flow(li);
        let base = Gesture { t_start: self.t, t_end: self.t, speed: 0.0, height: hm, turn_sigma: 0.0, flow_until: self.t, hop_at: 0.0, centre: self.pos, radius: 0.0, phase0: 0.0, lissajous: false, drift: DVec2::ZERO, wobble: 0.0 };
        let g = match style {
            Style::Drip | Style::Pour => {
                let dur = if style == Style::Drip { self.rng.random_range(1.5..5.0) } else { self.rng.random_range(1.5..4.0) };
                let height = self.lognormal(hm, 0.3).clamp(0.04, 0.3);
                let ustar = transition_speed(q, height, nu);
                let is_loop = self.rng.random::<f64>() < loop_chance;
                let factor = if is_loop { self.lognormal(0.45, 0.3).clamp(0.15, 0.75) } else { self.lognormal(pace, 0.22).max(1.15) };
                let speed = (ustar * factor).clamp(0.06, 2.2);
                let r_turn = self.lognormal(0.18, 0.5).clamp(0.05, 0.6);
                Gesture { t_end: self.t + dur, speed, height, turn_sigma: speed / r_turn, flow_until: self.t + dur, ..base }
            }
            Style::Lasso => {
                let speed = self.lognormal(pace, 0.2).clamp(0.8, 4.0);
                let height = self.lognormal(hm, 0.25).clamp(0.12, 0.6);
                let radius = self.lognormal(0.14, 0.4).clamp(0.06, 0.3);
                let turns = self.rng.random_range(1.2..2.6);
                let dur = turns * std::f64::consts::TAU * radius / speed;
                let lissajous = self.rng.random::<f64>() < 0.4;
                let m = radius + 0.05;
                let centre = DVec2::new(self.pos.x.clamp(m, self.w - m), self.pos.y.clamp(m, self.h - m));
                let phase0 = self.rng.random_range(0.0..6.28);
                let da: f64 = self.rng.random_range(0.0..6.28);
                let ds = self.lognormal(0.25, 0.4).clamp(0.05, 0.6);
                let drift = DVec2::new(da.cos(), da.sin()) * ds;
                let wobble = self.rng.random_range(0.0..6.28);
                self.cut_pending = true;
                Gesture { t_end: self.t + dur, speed, height, flow_until: self.t + dur, centre, radius, phase0, lissajous, drift, wobble, ..base }
            }
            Style::Dry | Style::Tilt => {
                let d = self.layers[li].duration;
                self.cut_pending = true;
                Gesture { t_end: self.t + d, flow_until: self.t, ..base }
            }
            Style::Snap => {
                let speed = self.lognormal(pace, 0.2).clamp(2.5, 6.0);
                let height = self.lognormal(hm, 0.25).clamp(0.2, 0.5);
                self.heading = self.rng.random_range(0.0..6.28);
                self.cut_pending = true;
                Gesture { t_end: self.t + 1.0, speed, height, flow_until: self.t + 0.32, ..base }
            }
            Style::Fling => {
                let dur = 0.8;
                let speed = self.lognormal(pace, 0.25).clamp(1.5, 4.0);
                let height = self.lognormal(hm, 0.3).clamp(0.2, 0.6);
                self.heading = self.rng.random_range(0.0..6.28);
                Gesture { t_end: self.t + dur, speed, height, flow_until: self.t + 0.3, ..base }
            }
            Style::Dots => {
                let hold = self.rng.random_range(0.15..0.7);
                let height = self.lognormal(hm, 0.3).clamp(0.04, 0.15);
                Gesture { t_end: self.t + hold + 0.5, height, flow_until: self.t + hold, hop_at: self.t + hold, ..base }
            }
            Style::Flood => {
                let hold = self.rng.random_range(0.8..1.6);
                let height = self.lognormal(hm, 0.3).clamp(0.06, 0.2);
                self.pos = DVec2::new(self.w * self.rng.random_range(0.15..0.85), self.h * self.rng.random_range(0.15..0.85));
                self.cut_pending = true;
                Gesture { t_end: self.t + hold + 0.6, speed: pace, height, flow_until: self.t + hold, hop_at: self.t + hold, ..base }
            }
            Style::Wash => {
                let dur = self.rng.random_range(2.5..6.0);
                let speed = self.lognormal(pace, 0.25).clamp(0.15, 0.7);
                let r_turn = self.lognormal(0.3, 0.4).clamp(0.1, 0.8);
                let radius = self.lognormal(self.layers[li].pour_radius, 0.3).clamp(0.008, 0.04);
                Gesture { t_end: self.t + dur, speed, height: hm, turn_sigma: speed / r_turn, flow_until: self.t + dur, radius, ..base }
            }
            Style::Scrape => {
                let speed = self.lognormal(pace, 0.25).clamp(0.08, 0.35);
                let m = 0.06;
                let start = DVec2::new(self.rng.random_range(m..self.w - m), self.rng.random_range(m..self.h - m));
                let a: f64 = self.rng.random_range(0.0..6.28);
                let dir = DVec2::new(a.cos(), a.sin());
                let reach = self.run_to_border(start, dir, m);
                let len = self.lognormal(0.45, 0.4).clamp(0.15, 0.9).min(reach);
                let dur = len / speed + 0.4;
                self.pos = start;
                self.heading = a;
                self.cut_pending = true;
                Gesture { t_end: self.t + dur, speed, height: 0.0, flow_until: self.t + len / speed, centre: start, drift: dir * speed, ..base }
            }
        };
        if !matches!(style, Style::Fling | Style::Snap | Style::Scrape) {
            self.heading = self.heading + self.gauss() * 0.8;
        }
        self.gesture = Some(g);
    }

    fn steer(&mut self, dt: f64, turn_sigma: f64) {
        let margin = 0.08;
        let c = DVec2::new(self.w * 0.5, self.h * 0.5);
        let to_c = c - self.pos;
        let out = ((margin - self.pos.x).max(0.0) + (margin - self.pos.y).max(0.0) + (self.pos.x - (self.w - margin)).max(0.0) + (self.pos.y - (self.h - margin)).max(0.0)) / margin;
        let target = to_c.y.atan2(to_c.x);
        let mut d = target - self.heading;
        while d > std::f64::consts::PI {
            d -= std::f64::consts::TAU;
        }
        while d < -std::f64::consts::PI {
            d += std::f64::consts::TAU;
        }
        self.heading += d * (out * 6.0 * dt).min(1.0);
        let tau = 0.35;
        let g = self.gauss();
        self.ang_vel += -self.ang_vel * dt / tau + turn_sigma * (2.0 * dt / tau).sqrt() * g;
        self.heading += self.ang_vel * dt;
    }

    fn noise(&mut self, dt: f64) {
        let g1 = self.gauss();
        let g2 = self.gauss();
        let ts = 0.45;
        let th = 0.9;
        self.speed_noise += -self.speed_noise * dt / ts + 0.22 * (2.0 * dt / ts).sqrt() * g1;
        self.height_noise += -self.height_noise * dt / th + 0.15 * (2.0 * dt / th).sqrt() * g2;
    }

    pub fn step(&mut self, dt: f64) -> Option<Command> {
        if self.done {
            return None;
        }
        self.t += dt;
        self.layer_t += dt;
        if self.layer_t >= self.layers[self.layer].duration {
            self.layer += 1;
            self.layer_t = 0.0;
            self.gesture = None;
            self.cut_pending = true;
            self.reload_pending = true;
            if self.layer >= self.layers.len() {
                self.done = true;
                return None;
            }
        }
        let mut reload = false;
        if self.gesture.as_ref().map(|g| self.t >= g.t_end).unwrap_or(true) {
            let style = self.layers[self.layer].style;
            if self.gesture.is_some() && style != Style::Scrape && self.rng.random::<f64>() < 0.35 {
                self.teleport();
            }
            self.new_gesture();
            reload = true;
        }
        self.noise(dt);
        let li = self.layer;
        let layer = &self.layers[li];
        let (mix, mu, sd, flow, pour_radius, knife_width, tilt) = (layer.mix, layer.mu, layer.stick_diameter, layer.flow, layer.pour_radius, layer.knife_width, layer.tilt);
        let style = layer.style;
        let g = self.gesture.as_ref().unwrap();
        let (gt_start, gt_end, gspeed, gheight, turn_sigma, flow_until, hop_at) = (g.t_start, g.t_end, g.speed, g.height, g.turn_sigma, g.flow_until, g.hop_at);
        let (centre, radius, phase0, lissajous, drift, wobble) = (g.centre, g.radius, g.phase0, g.lissajous, g.drift, g.wobble);
        let mut flowing = self.t < flow_until;
        let mut forced: Option<DVec2> = None;
        match style {
            Style::Lasso => {
                let om = gspeed / radius;
                let tl = self.t - gt_start;
                let ph = phase0 + om * tl;
                let rr = radius * (1.0 + 0.22 * (0.37 * ph + wobble).sin());
                let c = centre + drift * tl;
                let p = if lissajous {
                    c + DVec2::new(rr * ph.cos(), 0.6 * rr * (2.0 * ph).sin())
                } else {
                    c + DVec2::new(rr * ph.cos(), rr * ph.sin())
                };
                forced = Some(p);
            }
            Style::Dry | Style::Tilt => {
                self.speed = 0.0;
                flowing = false;
            }
            Style::Snap => {
                let tl = self.t - gt_start;
                let s = if tl < 0.25 {
                    let u = tl / 0.25;
                    u * u * (3.0 - 2.0 * u)
                } else if tl < 0.3 {
                    1.0 - (tl - 0.25) / 0.05
                } else {
                    0.0
                };
                self.speed = gspeed * s;
            }
            Style::Drip | Style::Pour => {
                self.steer(dt, turn_sigma);
                let kappa = self.ang_vel.abs() / gspeed.max(0.05);
                let cf = (3.0 / kappa.max(0.3)).powf(1.0 / 3.0).clamp(0.4, 1.0);
                let target = gspeed * self.speed_noise.exp() * cf;
                self.speed += (target - self.speed) * (8.0 * dt).min(1.0);
            }
            Style::Wash => {
                self.steer(dt, turn_sigma);
                self.speed = gspeed * (0.5 * self.speed_noise).exp();
            }
            Style::Fling => {
                let u = ((gt_end - self.t) / 0.8).clamp(0.0, 1.0);
                let ramp = 1.0 - u;
                let s = ramp * ramp * (3.0 - 2.0 * ramp);
                self.speed = gspeed * s.max(0.05);
                flowing = flowing && self.t > gt_end - 0.75;
            }
            Style::Dots => {
                if self.t >= hop_at {
                    if self.t - dt < hop_at {
                        self.cut_pending = true;
                    }
                    self.speed = 0.9;
                    self.steer(dt, 4.0);
                    flowing = false;
                } else {
                    self.speed = 0.02;
                }
            }
            Style::Flood => {
                if self.t >= hop_at {
                    self.speed = 0.0;
                    flowing = false;
                } else {
                    self.steer(dt, 1.5);
                    self.speed = gspeed;
                }
            }
            Style::Scrape => {
                let tl = (self.t - gt_start).min((flow_until - gt_start).max(0.0));
                forced = Some(centre + drift * tl);
            }
        }
        let dir = DVec2::new(self.heading.cos(), self.heading.sin());
        let prev = self.pos;
        match forced {
            Some(p) => {
                self.pos = p;
            }
            None => {
                self.pos += dir * self.speed * dt;
            }
        }
        self.pos.x = self.pos.x.clamp(0.02, self.w - 0.02);
        self.pos.y = self.pos.y.clamp(0.02, self.h - 0.02);
        let v = (self.pos - prev) / dt.max(1e-6);
        let hmod = if style == Style::Drip || style == Style::Pour { self.height_noise.exp() } else { 1.0 };
        let height = if style == Style::Scrape { 0.0 } else { (gheight * hmod).clamp(0.03, 0.6) };
        let kind = match style {
            Style::Pour | Style::Lasso => ToolKind::Bottle,
            Style::Flood => ToolKind::Bucket,
            Style::Wash => ToolKind::Pour,
            Style::Scrape => ToolKind::Knife,
            _ => ToolKind::Stick,
        };
        if matches!(style, Style::Lasso | Style::Snap | Style::Scrape) {
            self.speed = v.length();
        }
        let cut = self.cut_pending;
        self.cut_pending = false;
        if self.reload_pending {
            reload = true;
            self.reload_pending = false;
        }
        let cure_rate = if style == Style::Dry { 0.6 } else { 0.0 };
        let tilt_cmd = if style == Style::Tilt { Some(tilt) } else { None };
        Some(Command {
            kind,
            pos: DVec3::new(self.pos.x, self.pos.y, height),
            vel: DVec3::new(v.x, v.y, 0.0),
            flowing,
            height,
            mu,
            sigma: 0.04,
            mix,
            stick_diameter: sd,
            bottle_flow: flow,
            bottle_radius: 0.0025,
            bucket_flow: flow,
            bucket_radius: 0.008,
            pour_rate: flow,
            pour_radius: if style == Style::Wash { radius } else { pour_radius },
            knife_width,
            knife_height: 0.0004,
            tilt: tilt_cmd,
            reload,
            cut,
            cure_rate,
        })
    }
}
