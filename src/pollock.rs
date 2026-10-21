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
}

struct Layer {
    name: &'static str,
    mix: [f32; 16],
    mu: f64,
    style: Style,
    duration: f64,
    pace: f64,
    loop_chance: f64,
    height_median: f64,
    stick_diameter: f64,
    flow: f64,
}

struct Gesture {
    style: Style,
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

impl Painter {
    pub fn new(seed: u64, w: f64, h: f64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let black = mix(&[(1, 1.0)], 0.2, 0.0);
        let umber = mix(&[(9, 0.8), (1, 0.2)], 0.2, 0.0);
        let ochre = mix(&[(8, 0.9), (2, 0.1)], 0.2, 0.0);
        let white = mix(&[(0, 1.0)], 0.15, 5.0);
        let silver = mix(&[(0, 0.75), (1, 0.25)], 0.35, 0.0);
        let teal = mix(&[(10, 0.5), (6, 0.3), (0, 0.2)], 0.2, 0.0);
        let red = mix(&[(3, 0.8), (11, 0.2)], 0.2, 0.0);
        let wash_umber = mix(&[(9, 0.6), (8, 0.3), (1, 0.1)], 0.45, 0.0);
        let wash_white = mix(&[(0, 0.9), (8, 0.1)], 0.4, 0.0);
        let wash_teal = mix(&[(10, 0.5), (6, 0.2), (0, 0.3)], 0.45, 0.0);
        let yellow = mix(&[(2, 0.9), (11, 0.1)], 0.3, 0.0);
        let black_thin = mix(&[(1, 1.0)], 0.12, 0.0);
        let white_thin = mix(&[(0, 1.0)], 0.12, 0.0);
        let black_snap = mix(&[(1, 1.0)], 0.45, 0.0);
        let white_snap = mix(&[(0, 1.0)], 0.45, 0.0);
        let red_spatter = mix(&[(3, 0.85), (11, 0.15)], 0.55, 0.0);
        let black_spatter = mix(&[(1, 1.0)], 0.55, 0.0);
        let mut layers = vec![
            Layer { name: "umber flood", mix: wash_umber, mu: 4.0, style: Style::Flood, duration: 14.0, pace: 0.15, loop_chance: 0.0, height_median: 0.12, stick_diameter: 0.01, flow: 4.0e-5 },
            Layer { name: "white flood", mix: wash_white, mu: 4.0, style: Style::Flood, duration: 12.0, pace: 0.15, loop_chance: 0.0, height_median: 0.12, stick_diameter: 0.01, flow: 4.0e-5 },
            Layer { name: "teal flood", mix: wash_teal, mu: 4.0, style: Style::Flood, duration: 8.0, pace: 0.15, loop_chance: 0.0, height_median: 0.12, stick_diameter: 0.01, flow: 3.0e-5 },
            Layer { name: "drying", mix: black, mu: 10.0, style: Style::Dry, duration: 5.0, pace: 0.0, loop_chance: 0.0, height_median: 0.2, stick_diameter: 0.01, flow: 0.0 },
            Layer { name: "black skeleton", mix: black, mu: 10.0, style: Style::Pour, duration: 45.0, pace: 1.5, loop_chance: 0.12, height_median: 0.08, stick_diameter: 0.008, flow: 3.0e-6 },
            Layer { name: "black lasso", mix: black_thin, mu: 12.0, style: Style::Lasso, duration: 18.0, pace: 1.4, loop_chance: 0.0, height_median: 0.22, stick_diameter: 0.014, flow: 3.0e-6 },
            Layer { name: "drying", mix: black, mu: 10.0, style: Style::Dry, duration: 4.0, pace: 0.0, loop_chance: 0.0, height_median: 0.2, stick_diameter: 0.01, flow: 0.0 },
            Layer { name: "umber lines", mix: umber, mu: 8.0, style: Style::Drip, duration: 25.0, pace: 1.6, loop_chance: 0.1, height_median: 0.13, stick_diameter: 0.009, flow: 2.0e-6 },
            Layer { name: "white snaps", mix: white_snap, mu: 1.5, style: Style::Snap, duration: 14.0, pace: 4.0, loop_chance: 0.0, height_median: 0.3, stick_diameter: 0.012, flow: 2.0e-6 },
            Layer { name: "white pours", mix: white, mu: 10.0, style: Style::Pour, duration: 30.0, pace: 1.4, loop_chance: 0.15, height_median: 0.07, stick_diameter: 0.008, flow: 2.5e-6 },
            Layer { name: "white lines", mix: white, mu: 8.0, style: Style::Drip, duration: 25.0, pace: 1.8, loop_chance: 0.08, height_median: 0.16, stick_diameter: 0.008, flow: 2.0e-6 },
            Layer { name: "silver flings", mix: silver, mu: 1.5, style: Style::Fling, duration: 15.0, pace: 2.5, loop_chance: 0.0, height_median: 0.35, stick_diameter: 0.01, flow: 2.0e-6 },
            Layer { name: "ochre", mix: ochre, mu: 7.0, style: Style::Drip, duration: 20.0, pace: 1.6, loop_chance: 0.1, height_median: 0.14, stick_diameter: 0.008, flow: 2.0e-6 },
            Layer { name: "red spatter", mix: red_spatter, mu: 0.6, style: Style::Fling, duration: 10.0, pace: 3.2, loop_chance: 0.0, height_median: 0.4, stick_diameter: 0.01, flow: 2.0e-6 },
            Layer { name: "teal accents", mix: teal, mu: 8.0, style: Style::Pour, duration: 15.0, pace: 1.7, loop_chance: 0.1, height_median: 0.1, stick_diameter: 0.006, flow: 1.5e-6 },
            Layer { name: "drying", mix: black, mu: 10.0, style: Style::Dry, duration: 4.0, pace: 0.0, loop_chance: 0.0, height_median: 0.2, stick_diameter: 0.01, flow: 0.0 },
            Layer { name: "white lasso", mix: white_thin, mu: 12.0, style: Style::Lasso, duration: 14.0, pace: 1.4, loop_chance: 0.0, height_median: 0.22, stick_diameter: 0.014, flow: 3.0e-6 },
            Layer { name: "black loops", mix: black, mu: 10.0, style: Style::Pour, duration: 25.0, pace: 1.2, loop_chance: 0.4, height_median: 0.12, stick_diameter: 0.008, flow: 2.5e-6 },
            Layer { name: "black snaps", mix: black_snap, mu: 1.2, style: Style::Snap, duration: 15.0, pace: 4.2, loop_chance: 0.0, height_median: 0.3, stick_diameter: 0.012, flow: 2.0e-6 },
            Layer { name: "drying", mix: black, mu: 10.0, style: Style::Dry, duration: 4.0, pace: 0.0, loop_chance: 0.0, height_median: 0.2, stick_diameter: 0.01, flow: 0.0 },
            Layer { name: "red dots", mix: red, mu: 8.0, style: Style::Dots, duration: 12.0, pace: 0.0, loop_chance: 0.0, height_median: 0.08, stick_diameter: 0.01, flow: 4.0e-7 },
            Layer { name: "yellow flecks", mix: yellow, mu: 1.5, style: Style::Fling, duration: 10.0, pace: 2.8, loop_chance: 0.0, height_median: 0.4, stick_diameter: 0.008, flow: 2.0e-6 },
            Layer { name: "white tracery", mix: white, mu: 6.0, style: Style::Drip, duration: 25.0, pace: 2.2, loop_chance: 0.05, height_median: 0.18, stick_diameter: 0.005, flow: 1.0e-6 },
            Layer { name: "black tracery", mix: black, mu: 6.0, style: Style::Drip, duration: 30.0, pace: 2.0, loop_chance: 0.08, height_median: 0.18, stick_diameter: 0.006, flow: 1.0e-6 },
            Layer { name: "black spatter", mix: black_spatter, mu: 0.6, style: Style::Fling, duration: 8.0, pace: 3.4, loop_chance: 0.0, height_median: 0.45, stick_diameter: 0.01, flow: 2.0e-6 },
            Layer { name: "black drops", mix: black, mu: 8.0, style: Style::Dots, duration: 10.0, pace: 0.0, loop_chance: 0.0, height_median: 0.1, stick_diameter: 0.012, flow: 4.0e-7 },
        ];
        if rng.random::<f64>() < 0.5 {
            let a = layers.iter().position(|l| l.name == "umber lines").unwrap();
            let b = layers.iter().position(|l| l.name == "ochre").unwrap();
            layers.swap(a, b);
        }
        if rng.random::<f64>() < 0.5 {
            layers.swap(0, 1);
        }
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
            return "finished".to_string();
        }
        let l = &self.layers[self.layer];
        format!("layer {}/{} {}  {:.0}/{:.0} s", self.layer + 1, self.layers.len(), l.name, self.layer_t, l.duration)
    }

    fn gauss(&mut self) -> f64 {
        let u1: f64 = self.rng.random::<f64>().max(1e-12);
        let u2: f64 = self.rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn lognormal(&mut self, median: f64, sigma: f64) -> f64 {
        median * (sigma * self.gauss()).exp()
    }

    fn layer_nu(&self, li: usize) -> f64 {
        let l = &self.layers[li];
        let thin = (l.mix[12] as f64).clamp(0.0, 0.95);
        l.mu * (1.0 - thin).powi(3) / RHO
    }

    fn layer_flow(&self, li: usize) -> f64 {
        let l = &self.layers[li];
        match l.style {
            Style::Pour | Style::Flood | Style::Lasso => l.flow,
            _ => stick_flow(l.stick_diameter, self.layer_nu(li)),
        }
    }

    fn teleport(&mut self) {
        self.pos = DVec2::new(self.w * self.rng.random_range(0.1..0.9), self.h * self.rng.random_range(0.1..0.9));
        self.cut_pending = true;
    }

    fn new_gesture(&mut self) {
        let li = self.layer;
        let style = self.layers[li].style;
        let pace = self.layers[li].pace;
        let hm = self.layers[li].height_median;
        let loop_chance = self.layers[li].loop_chance;
        let nu = self.layer_nu(li);
        let q = self.layer_flow(li);
        let base = Gesture { style, t_start: self.t, t_end: self.t, speed: 0.0, height: hm, turn_sigma: 0.0, flow_until: self.t, hop_at: 0.0, centre: self.pos, radius: 0.0, phase0: 0.0, lissajous: false, drift: DVec2::ZERO, wobble: 0.0 };
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
            Style::Dry => {
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
        };
        if style != Style::Fling && style != Style::Snap {
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
            if self.layer >= self.layers.len() {
                self.done = true;
                return None;
            }
        }
        let mut reload = false;
        if self.gesture.as_ref().map(|g| self.t >= g.t_end).unwrap_or(true) {
            if self.gesture.is_some() && self.rng.random::<f64>() < 0.35 {
                self.teleport();
            }
            self.new_gesture();
            reload = true;
        }
        self.noise(dt);
        let li = self.layer;
        let layer = &self.layers[li];
        let (mix, mu, sd, flow) = (layer.mix, layer.mu, layer.stick_diameter, layer.flow);
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
            Style::Dry => {
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
        let height = (gheight * hmod).clamp(0.03, 0.6);
        let kind = match style {
            Style::Pour | Style::Lasso => ToolKind::Bottle,
            Style::Flood => ToolKind::Bucket,
            _ => ToolKind::Stick,
        };
        if style == Style::Lasso || style == Style::Snap {
            self.speed = v.length();
        }
        let cut = self.cut_pending;
        self.cut_pending = false;
        let cure_rate = if style == Style::Dry { 0.6 } else { 0.0 };
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
            reload,
            cut,
            cure_rate,
        })
    }
}
