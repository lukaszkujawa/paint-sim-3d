use super::*;

const GROUND_MARGIN_FLOATING: f64 = 0.04;

const GROUND_MARGIN_BORDER: f64 = 0.05;

const GROUND_MARGIN_BLEED: f64 = -0.03;

#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Style {
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
    Scrub,
    Tilt,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum GroundMethod {
    Auto,
    Brush,
    Bucket,
}

pub(super) struct Layer {
    pub(super) name: String,
    pub(super) mix: [f32; 16],
    pub(super) mu: f64,
    pub(super) style: Style,
    pub(super) duration: f64,
    pub(super) pace: f64,
    pub(super) loop_chance: f64,
    pub(super) height_median: f64,
    pub(super) stick_diameter: f64,
    pub(super) flow: f64,
    pub(super) pour_radius: f64,
    pub(super) knife_width: f64,
    pub(super) brush_width: f64,
    pub(super) shades: Vec<[f32; 16]>,
    pub(super) tilt: (f64, f64),
    pub(super) margin: f64,
    pub(super) edge_bias: bool,
    pub(super) strokes: usize,
}

pub(super) fn thinned(p: &Paint, thinner: f32, yield_pa: f32, mu: f64) -> Paint {
    let mut m = p.mix;
    m[12] = thinner;
    m[13] = yield_pa;
    Paint { name: p.name.clone(), mix: m, mu }
}

pub(super) fn layer(name: String, p: &Paint, style: Style, duration: f64) -> Layer {
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
        Style::Scrub => (0.2, 0.0, 0.62, 0.01, 0.0),
    };
    Layer { name, mix: p.mix, mu: p.mu, style, duration, pace, loop_chance, height_median, stick_diameter, flow, pour_radius: 0.02, knife_width: 0.08, brush_width: 0.09, shades: Vec::new(), tilt: (0.0, 0.0), margin: GROUND_MARGIN_FLOATING, edge_bias: false, strokes: 0 }
}

pub(super) fn shade(rng: &mut StdRng, base: &[f32; 16], partner: &[f32; 16], light: f32, dark: f32, cross: f32) -> [f32; 16] {
    let mut m = [0.0f32; 16];
    let total: f32 = base[..12].iter().sum::<f32>().max(1e-6);
    for i in 0..12 {
        m[i] = base[i] / total * lognormal(rng, 1.0, 0.15) as f32 + partner[i] / partner[..12].iter().sum::<f32>().max(1e-6) * cross;
    }
    m[0] += light;
    m[1] += 0.4 * dark;
    m[9] += 0.6 * dark;
    m[12] = base[12];
    m[13] = base[13];
    m
}

pub(super) fn shades(rng: &mut StdRng, base: &Paint, partner: &Paint, pass: usize) -> Vec<[f32; 16]> {
    let n = rng.random_range(3..=4);
    let mut out = Vec::new();
    for _ in 0..n {
        let cross = rng.random_range(0.0..0.25);
        let (light, dark) = match pass {
            0 => (rng.random_range(0.0..0.1), rng.random_range(0.0..0.08)),
            1 => (rng.random_range(0.12..0.35), rng.random_range(0.0..0.02)),
            _ => (rng.random_range(0.0..0.04), rng.random_range(0.15..0.4)),
        };
        out.push(shade(rng, &base.mix, &partner.mix, light, dark, cross));
    }
    out
}

pub(super) fn plan(rng: &mut StdRng, pal: &Palette) -> (Vec<Layer>, String) {
    let (mut layers, ground) = ground_plan(rng, pal, GroundMethod::Auto, 0);
    layers.extend(figure(rng, pal));
    (layers, ground)
}

pub(super) fn ground_plan(rng: &mut StdRng, pal: &Palette, method: GroundMethod, strokes: usize) -> (Vec<Layer>, String) {
    let mut layers: Vec<Layer> = Vec::new();
    let dry = |rng: &mut StdRng, layers: &mut Vec<Layer>, chance: f64| {
        if rng.random::<f64>() < chance {
            layers.push(dry_layer(rng));
        }
    };
    let grounds = [("brushed", 6.0), ("raw canvas", 0.5), ("floods", 1.0), ("washes", 1.0), ("scraped", 0.8), ("tilt run", 0.8)];
    let ground = match method {
        GroundMethod::Auto => *weighted(rng, &grounds),
        GroundMethod::Brush => "brushed",
        GroundMethod::Bucket => "floods",
    };
    let mut gcols: Vec<Paint> = pal.ground.clone();
    if gcols.len() > 1 && rng.random::<f64>() < 0.3 {
        gcols.swap(0, 1);
    }
    let layout = *weighted(rng, &[("bleed", 0.55), ("border", 0.25), ("floating", 0.2)]);
    let margin = match layout {
        "bleed" => GROUND_MARGIN_BLEED,
        "border" => GROUND_MARGIN_BORDER,
        _ => GROUND_MARGIN_FLOATING,
    };
    match ground {
        "brushed" => {
            let base = gcols[0].clone();
            let partner = gcols[1 % gcols.len()].clone();
            let passes = rng.random_range(2..=3);
            for k in 0..passes {
                let (what, dur) = match k {
                    0 => ("ground", rng.random_range(110.0..150.0)),
                    1 => ("lights", rng.random_range(35.0..55.0)),
                    _ => ("darks", rng.random_range(18.0..30.0)),
                };
                let mut l = layer(format!("{} brushed {}", base.name, what), &thinned(&base, 0.12, 20.0, 18.0), Style::Scrub, dur);
                l.shades = shades(rng, &base, &partner, k);
                l.brush_width = rng.random_range(0.08..0.12);
                l.margin = margin;
                layers.push(l);
            }
            let edge_chance = match layout {
                "bleed" => 0.8,
                "border" => 0.45,
                _ => 0.25,
            };
            if rng.random::<f64>() < edge_chance {
                let deep = pal.ground_dark.clone().unwrap_or_else(|| base.clone());
                let mut dark = blend(rng, &deep, &base);
                dark.name = format!("{} edges", base.name);
                let mut l = layer(dark.name.clone(), &thinned(&dark, 0.12, 20.0, 18.0), Style::Scrub, rng.random_range(18.0..28.0));
                l.shades = shades(rng, &dark, &base, 0);
                l.brush_width = rng.random_range(0.07..0.1);
                l.margin = margin;
                l.edge_bias = true;
                layers.push(l);
            }
            if method == GroundMethod::Auto && rng.random::<f64>() < 0.35 {
                let p = &gcols[gcols.len() - 1];
                let d = rng.random_range(6.0..10.0);
                layers.push(layer(format!("{} flood", p.name), p, Style::Flood, d));
            }
            if strokes > 0 {
                layers.truncate(strokes);
                let total: f64 = layers.iter().map(|l| l.duration).sum();
                let mut left = strokes;
                for k in (1..layers.len()).rev() {
                    let n = ((strokes as f64 * layers[k].duration / total).round() as usize).clamp(1, left - k);
                    layers[k].strokes = n;
                    left -= n;
                }
                layers[0].strokes = left;
            }
        }
        "floods" if strokes > 0 => {
            let chunk = strokes.div_ceil(6).max(1);
            let mut k = 0;
            while k < strokes {
                let n = chunk.min(strokes - k);
                let p = &gcols[(k / chunk) % gcols.len()];
                let mut l = layer(format!("{} flood", p.name), p, Style::Flood, 1.0);
                l.strokes = n;
                layers.push(l);
                k += n;
            }
        }
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
    (layers, format!("{} {}", ground, layout))
}

pub(super) fn dry_layer(rng: &mut StdRng) -> Layer {
    let d = rng.random_range(3.5..6.0);
    layer("drying".to_string(), &Paint { name: String::new(), mix: mix(&[(1, 1.0)], 0.2, 0.0), mu: 10.0 }, Style::Dry, d)
}

pub(super) fn blend(rng: &mut StdRng, a: &Paint, b: &Paint) -> Paint {
    let t = rng.random_range(0.25..0.75) as f32;
    let sa: f32 = a.mix[..12].iter().sum::<f32>().max(1e-6);
    let sb: f32 = b.mix[..12].iter().sum::<f32>().max(1e-6);
    let mut m = [0.0f32; 16];
    for i in 0..12 {
        m[i] = (1.0 - t) * a.mix[i] / sa + t * b.mix[i] / sb;
    }
    m[12] = (1.0 - t) * a.mix[12] + t * b.mix[12];
    m[13] = (1.0 - t) * a.mix[13] + t * b.mix[13];
    let mu = (a.mu.ln() * (1.0 - t as f64) + b.mu.ln() * t as f64).exp();
    Paint { name: format!("{}-{}", a.name, b.name), mix: m, mu }
}

pub(super) fn palette_colours(pal: &Palette) -> Vec<Paint> {
    let mut all = vec![pal.main.clone(), pal.second.clone()];
    for p in pal.accents.iter().chain(pal.ground.iter()) {
        if !all.iter().any(|q| q.name == p.name) {
            all.push(p.clone());
        }
    }
    all
}

pub(super) fn line_layer(rng: &mut StdRng, p: &Paint, styles: &[(Style, f64)], fine: f64) -> Layer {
    let style = *weighted(rng, styles);
    let (name, pp, dur) = match style {
        Style::Snap => (format!("{} snaps", p.name), thinned(p, 0.45, 0.0, 1.5), rng.random_range(10.0..16.0)),
        Style::Fling => (format!("{} flings", p.name), thinned(p, 0.5, 0.0, 1.0), rng.random_range(8.0..14.0)),
        Style::Lasso => (format!("{} lasso", p.name), thinned(p, 0.12, 0.0, 12.0), rng.random_range(10.0..18.0)),
        Style::Dots => (format!("{} dots", p.name), p.clone(), rng.random_range(8.0..13.0)),
        Style::Pour => (format!("{} pours", p.name), p.clone(), rng.random_range(18.0..32.0)),
        _ => (format!("{} lines", p.name), p.clone(), rng.random_range(16.0..28.0)),
    };
    let mut l = layer(name, &pp, style, dur);
    if matches!(style, Style::Drip | Style::Lasso | Style::Snap | Style::Fling) {
        l.stick_diameter = (l.stick_diameter * lognormal(rng, 1.0 - 0.25 * fine, 0.18)).clamp(0.0045, 0.016);
    }
    if style == Style::Pour {
        l.flow *= lognormal(rng, 1.0 - 0.35 * fine, 0.25);
    }
    l.pace *= lognormal(rng, 1.0 + 0.15 * fine, 0.1);
    l
}

pub(super) fn continuation(rng: &mut StdRng, pal: &Palette, round: usize) -> Vec<Layer> {
    let mut layers: Vec<Layer> = Vec::new();
    if rng.random::<f64>() < 0.75 {
        layers.push(dry_layer(rng));
    }
    let colours = palette_colours(pal);
    let mut theme = pick(rng, &colours).clone();
    if rng.random::<f64>() < 0.35 {
        let other = pick(rng, &colours).clone();
        if other.name != theme.name {
            theme = blend(rng, &theme, &other);
        }
    }
    let fine = (round as f64 / 4.0).min(1.0);
    let n = rng.random_range(4..=7);
    let mut last_name = String::new();
    for _ in 0..n {
        let which = *weighted(rng, &[("theme", 0.35), ("main", 0.2), ("second", 0.15), ("accent", 0.2), ("blend", 0.1)]);
        let p = match which {
            "theme" => theme.clone(),
            "main" => pal.main.clone(),
            "second" => pal.second.clone(),
            "blend" => {
                let a = pick(rng, &colours).clone();
                let b = pick(rng, &colours).clone();
                if a.name == b.name { a } else { blend(rng, &a, &b) }
            }
            _ => pick(rng, &pal.accents).clone(),
        };
        if p.name == last_name && rng.random::<f64>() < 0.7 {
            continue;
        }
        last_name = p.name.clone();
        let l = line_layer(rng, &p, &[(Style::Drip, 3.5), (Style::Pour, 2.0), (Style::Lasso, 1.4), (Style::Snap, 1.0), (Style::Fling, 0.8), (Style::Dots, 0.5)], fine);
        let style = l.style;
        layers.push(l);
        if matches!(style, Style::Pour | Style::Lasso) && rng.random::<f64>() < 0.25 {
            layers.push(dry_layer(rng));
        }
    }
    if rng.random::<f64>() < 0.6 {
        let p = if rng.random::<f64>() < 0.5 { pal.main.clone() } else { theme.clone() };
        let mut l = layer(format!("{} tracery", p.name), &thinned(&p, p.mix[12], 0.0, 6.0), Style::Drip, rng.random_range(16.0..26.0));
        l.stick_diameter = rng.random_range(0.0045..0.006);
        l.pace = 2.1;
        l.loop_chance = 0.06;
        layers.push(l);
    }
    if rng.random::<f64>() < 0.5 {
        let p = pick(rng, &pal.accents).clone();
        let style = *weighted(rng, &[(Style::Dots, 1.0), (Style::Fling, 1.0), (Style::Snap, 0.8)]);
        let (name, pp) = match style {
            Style::Dots => (format!("{} drops", p.name), p.clone()),
            Style::Fling => (format!("{} spatter", p.name), thinned(&p, 0.55, 0.0, 0.6)),
            _ => (format!("{} snaps", p.name), thinned(&p, 0.45, 0.0, 1.2)),
        };
        layers.push(layer(name, &pp, style, rng.random_range(7.0..12.0)));
    }
    for l in layers.iter_mut() {
        if l.style != Style::Dry {
            l.name = format!("r{} {}", round + 1, l.name);
        }
    }
    layers
}

pub(super) fn figure(rng: &mut StdRng, pal: &Palette) -> Vec<Layer> {
    let mut layers: Vec<Layer> = Vec::new();
    let dry = |rng: &mut StdRng, layers: &mut Vec<Layer>, chance: f64| {
        if rng.random::<f64>() < chance {
            layers.push(dry_layer(rng));
        }
    };
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
    layers
}
