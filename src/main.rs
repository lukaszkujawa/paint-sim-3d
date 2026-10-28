mod brush;
mod color;
mod pollock;
mod film;
mod gizmo;
mod gpu;
mod panels;
mod render;
mod rod;
mod roller;
mod tools;

use film::{Deposit, Film, FilmSettings, KnifeParams, Method, Scrap};
use glam::Vec3;
use render::{Beauty, Camera, Refine, Renderer, TubeRenderer};
use tools::{Airborne, CoarseSurface, Tool, ToolKind};
use glam::DVec3;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const KNIFE_TURN_MIN: f64 = 2.0e-4;
const KNIFE_TURN_DIST: f64 = 0.04;
const KNIFE_CAPACITY_PER_WIDTH: f64 = 5.0e-5;
const KNIFE_HOLD_PER_WIDTH: f64 = 1.2e-5;
const KNIFE_SPILL_PER_M: f64 = 2.5;
const KNIFE_END_RIDGE: f64 = 0.8;
const KNIFE_BANDS: usize = 8;
const KNIFE_POSE_HISTORY: usize = 8;
const KNIFE_PENDING_DECAY: f64 = 0.5;
const KNIFE_SMEAR: f64 = 0.35;
const KNIFE_BAND_MIX_PER_M: f64 = 1.2;
const PEN_STALE_S: f32 = 0.3;
const PEN_FLOW_MIN: f64 = 0.1;
const PEN_FLOW_MAX: f64 = 1.5;
const PEN_WIDTH_MIN: f32 = 0.35;
const PEN_KNIFE_CLOSE: f64 = 0.9;
const PEN_BRUSH_PRESS: f64 = 0.4;

struct Ui {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

struct App {
    window: Arc<Window>,
    gpu: gpu::Gpu,
    film: Film,
    renderer: Renderer,
    tubes: TubeRenderer,
    brush: brush::Brush,
    roller: roller::Roller,
    brush_scraped: f64,
    brush_released: f64,
    prev_kind: ToolKind,
    painter: Option<pollock::Painter>,
    beauty: Beauty,
    refine: Refine,
    last_beauty: String,
    painter_speed: f32,
    sim_dt: f64,
    painter_cure_saved: Option<f32>,
    painter_tilt_saved: Option<(f32, f32)>,
    painter_last_pour: Option<[f32; 2]>,
    painter_active: bool,
    painter_rounds_left: u32,
    tool: Tool,
    air: Airborne,
    surface: CoarseSurface,
    prev_tool_pos: Option<DVec3>,
    knife_dir: [f64; 2],
    knife_bands: [KnifeBand; KNIFE_BANDS],
    knife_pending: [f64; KNIFE_BANDS],
    knife_poses: std::collections::VecDeque<(u32, DVec3, DVec3)>,
    knife_prev: Option<DVec3>,
    knife_track: Option<DVec3>,
    pending_deposits: Vec<Deposit>,
    scrap_total: f64,
    view_rect: [f32; 4],
    block_rects: Vec<egui::Rect>,
    menu_open: bool,
    show_inspector: bool,
    show_help: bool,
    view_fitted: bool,
    pollock_pal: pollock::Palette,
    pen_down: bool,
    pen_last: Instant,
    pen_raw: f32,
    pen_seen: bool,
    pen_pressure: bool,
    pen_curve: f32,
    pal_picks: [[f32; 3]; 6],
    prof: String,
    auto_quality: bool,
    frame_budget_ms: f32,
    brush_sink: f64,
    last_cpu_ms: f32,
    ctrl_down: bool,
    cam: Camera,
    ui: Ui,
    last: Instant,
    cursor: [f32; 2],
    mouse_down: [bool; 3],
    last_pour: Option<Vec3>,
    paint: color::PaintMix,
    pour_rate: f32,
    pour_radius: f32,
    fps: f32,
    sim_time: f32,
    paused: bool,
    frame_index: u64,
    auto: Option<AutoScript>,
    shot_request: Option<String>,
    stats: String,
}

#[derive(Clone, Copy)]
struct KnifeBand {
    vol: f64,
    comp: [f32; 16],
}

impl KnifeBand {
    fn blend(&mut self, vol: f64, comp: &[f32; 16]) {
        let total = self.vol + vol;
        if total <= 0.0 {
            return;
        }
        for i in 0..16 {
            self.comp[i] = ((self.comp[i] as f64 * self.vol + comp[i] as f64 * vol) / total) as f32;
        }
        self.vol = total;
    }
}

struct AutoScript {
    frames: u64,
    shot: String,
    beauty: String,
    view: String,
    tilt: f32,
    cam: String,
    script: String,
    debug: bool,
    cure: f32,
    pace: bool,
    sigma: f32,
    angle: f32,
    yield_pa: f32,
    mu: f64,
    pig: i32,
    layer: String,
    seed: u64,
    dry: u32,
    export: String,
    stop: u64,
    rounds: u32,
    snap_save: String,
    snap_load: String,
    camat: String,
    vres: [u32; 2],
    spp: u32,
    light: Vec<f32>,
    aces: bool,
}

struct Pending {
    window: Option<Arc<Window>>,
    app: Option<App>,
}

impl App {
    fn new(window: Arc<Window>) -> Self {
        let gpu = gpu::Gpu::new(window.clone());
        let settings = FilmSettings::default();
        let w = std::env::args().collect::<Vec<_>>().windows(2).find(|p| p[0] == "--res").and_then(|p| p[1].parse().ok()).unwrap_or(3072u32);
        let h = ((w as f32) * settings.height_m / settings.width_m) as u32;
        let auto = parse_auto();
        let mut settings = settings;
        if let Some(a) = &auto { settings.tilt_deg = a.tilt; settings.tilt_dir_deg = -90.0; settings.debug_view = if a.debug { 1 } else { 0 }; if a.cure >= 0.0 { settings.cure_rate = a.cure; } if a.sigma >= 0.0 { settings.sigma = a.sigma; } if a.angle >= 0.0 { settings.contact_angle_deg = a.angle; } }
        let film = Film::new(&gpu, w, h, settings);
        let renderer = Renderer::new(&gpu, &film);
        let tubes = TubeRenderer::new(&gpu, &renderer.cam_buf);
        let beauty = Beauty::new(&gpu, &renderer);
        let refine = Refine::new(&gpu, &film, &renderer.layout, &renderer.cam_buf);
        let surface = CoarseSurface::new(film.pool_dims[0], film.pool_dims[1], 4.0 * film.dx as f64);
        let size = window.inner_size();
        let cam = Camera {
            target: Vec3::new(film.settings.width_m * 0.5, film.settings.height_m * 0.5, 0.0),
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 1.52,
            dist: 1.05,
            fov: 45f32.to_radians(),
            aspect: size.width as f32 / size.height.max(1) as f32,
        };
        let mut cam = cam;
        if let Some(a) = &auto {
            match a.cam.as_str() {
                "top" => { cam.pitch = 1.55; cam.dist = 0.9; cam.yaw = -1.5708; }
                "close" => { cam.pitch = 1.2; cam.dist = 0.3; cam.yaw = -1.5708; }
                "blob" => { cam.pitch = 1.0; cam.dist = 0.12; cam.yaw = -1.0; }
                "side" => { cam.pitch = 0.35; cam.dist = 0.6; cam.yaw = -0.9; }
                "macro" => { cam.pitch = 0.9; cam.dist = 0.07; cam.yaw = -1.2; }
                _ => {}
            }
            let v: Vec<f32> = a.camat.split(',').filter_map(|t| t.parse().ok()).collect();
            if v.len() == 5 {
                cam.target = Vec3::new(v[0], v[1], 0.0);
                cam.dist = v[2];
                cam.pitch = v[3];
                cam.yaw = v[4];
            }
        }
        let ctx = egui::Context::default();
        panels::apply_style(&ctx);
        let state = egui_winit::State::new(ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, None);
        let eg_renderer = egui_wgpu::Renderer::new(&gpu.device, gpu.config.format, egui_wgpu::RendererOptions::default());
        let mut paint = color::PaintMix::pure(1);
        paint.frac[12] = 0.3;
        let mut app = Self {
            window,
            gpu,
            film,
            renderer,
            tubes,
            tool: Tool::default(),
            air: Airborne::new(),
            surface,
            brush: if std::env::var("PAINT_BRUSH_FLAT").is_ok() { brush::Brush::build(brush::BrushShape::Flat, 160, 0.035, 0.025, 0.00015) } else { brush::Brush::new(160, 0.035, 0.008, 0.00015) },
            brush_scraped: 0.0,
            roller: roller::Roller::new(),
            brush_released: 0.0,
            prev_kind: ToolKind::Stick,
            painter: None,
            beauty,
            refine,
            last_beauty: String::new(),
            painter_speed: 2.0,
            sim_dt: 1.0 / 60.0,
            painter_cure_saved: None,
            painter_tilt_saved: None,
            painter_last_pour: None,
            painter_active: false,
            painter_rounds_left: 0,
            prev_tool_pos: None,
            knife_dir: [1.0, 0.0],
            knife_bands: [KnifeBand { vol: 0.0, comp: [0.0; 16] }; KNIFE_BANDS],
            knife_pending: [0.0; KNIFE_BANDS],
            knife_poses: std::collections::VecDeque::new(),
            knife_prev: None,
            knife_track: None,
            pending_deposits: Vec::new(),
            scrap_total: 0.0,
            view_rect: [0.0, 0.0, size.width as f32, size.height.max(1) as f32],
            block_rects: Vec::new(),
            menu_open: false,
            show_inspector: true,
            show_help: false,
            view_fitted: false,
            pollock_pal: pollock::Palette::random(rand::random::<u64>()),
            pen_down: false,
            pen_last: Instant::now(),
            pen_raw: 0.0,
            pen_seen: false,
            pen_pressure: true,
            pen_curve: 1.0,
            pal_picks: [[0.5; 3]; 6],
            prof: String::new(),
            auto_quality: true,
            frame_budget_ms: 14.0,
            brush_sink: 0.88,
            last_cpu_ms: 0.0,
            ctrl_down: false,
            cam,
            ui: Ui { ctx, state, renderer: eg_renderer },
            last: Instant::now(),
            cursor: [0.0, 0.0],
            mouse_down: [false; 3],
            last_pour: None,
            paint,
            pour_rate: 2.0e-6,
            pour_radius: 0.004,
            fps: 0.0,
            sim_time: 0.0,
            paused: false,
            frame_index: 0,
            auto,
            shot_request: None,
            stats: String::new(),
        };
        app.sync_pal_picks();
        let snap = app.auto.as_ref().map(|a| a.snap_load.clone()).unwrap_or_default();
        if !snap.is_empty() {
            app.load_snapshot(&snap);
        }
        if app.auto.as_ref().map(|a| a.aces).unwrap_or(false) {
            app.renderer.agx = false;
        }
        let light = app.auto.as_ref().map(|a| a.light.clone()).unwrap_or_default();
        if light.len() >= 3 {
            app.renderer.key_elevation_deg = light[0];
            app.renderer.key_azimuth_deg = light[1];
            app.renderer.key_power = light[2];
            if light.len() >= 4 {
                app.renderer.fill = light[3];
            }
            if light.len() >= 5 {
                app.renderer.exposure = light[4];
            }
        }
        app
    }

    fn scripted_deposits(&mut self, deposits: &mut Vec<Deposit>) {
        let f = self.frame_index;
        let dt = 1.0 / 60.0;
        let w = self.film.settings.width_m;
        let hgt = self.film.settings.height_m;
        if (10..130).contains(&f) {
            let t0 = (f - 10) as f32 / 120.0;
            let t1 = (f - 9) as f32 / 120.0;
            let curve = |t: f32| [w * (0.15 + 0.7 * t), hgt * (0.3 + 0.35 * (t * 6.0).sin() * 0.5 + 0.2 * t)];
            let mut paint = color::PaintMix::pure(1);
            paint.frac[12] = 0.3;
            paint.frac[13] = 0.0;
            deposits.push(Deposit::capsule(curve(t0), curve(t1), 0.0012, 1.5e-7 * dt, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour));
        }
        if (40..100).contains(&f) {
            let t0 = (f - 40) as f32 / 60.0;
            let t1 = (f - 39) as f32 / 60.0;
            let line = |t: f32| [w * (0.2 + 0.6 * t), hgt * (0.7 - 0.3 * t)];
            let paint = color::PaintMix::pure(3);
            deposits.push(Deposit::capsule(line(t0), line(t1), 0.0025, 1.2e-6 * dt, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour));
        }
        if (60..75).contains(&f) {
            let mut paint = color::PaintMix::pure(2);
            paint.frac[0] = 0.3;
            paint.frac[12] = 0.3;
            paint.frac[13] = 0.0;
            deposits.push(Deposit::capsule([w * 0.5, hgt * 0.5], [w * 0.5, hgt * 0.5], 0.02, 6.0e-6 * dt * 4.0, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour));
        }
        if (150..170).contains(&f) {
            let mut paint = color::PaintMix::pure(5);
            paint.frac[12] = 0.1;
            deposits.push(Deposit::capsule([w * 0.5, hgt * 0.5], [w * 0.52, hgt * 0.51], 0.008, 4.0e-6 * dt, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour));
        }
    }

    fn rebuild_brush(&mut self, shape: brush::BrushShape, n: usize, len: f64, size: f64) {
        let old = &self.brush;
        let (st, fr, rl, radius) = (old.stiffness, old.friction, old.release, old.radius);
        let load = old.total_load();
        let cap = old.total_capacity();
        let mut comp = [0.0f32; 16];
        if load > 0.0 {
            for b in &old.bristles {
                for i in 0..16 {
                    comp[i] += b.comp[i] * (b.load / load) as f32;
                }
            }
        }
        let press = (old.len - self.tool.height).max(0.0);
        let mut b = brush::Brush::build(shape, n, len, size, radius);
        b.stiffness = st;
        b.friction = fr;
        b.release = rl;
        if load > 0.0 && cap > 0.0 {
            b.dip(&comp);
            let f = (load / cap).min(1.0);
            for br in &mut b.bristles {
                br.load *= f;
            }
        }
        self.brush = b;
        if self.tool.kind == ToolKind::Brush {
            self.tool.height = (len - press).clamp(0.002, len - 0.003);
        }
    }

    fn apply_painter(&mut self) -> bool {
        let Some(p) = self.painter.as_mut() else { return false };
        if self.paused {
            return true;
        }
        match p.step(self.sim_dt) {
            Some(c) => {
                self.painter_active = true;
                if self.tool.kind != c.kind {
                    self.tool.reload();
                }
                self.tool.kind = c.kind;
                self.prev_kind = c.kind;
                if c.kind == ToolKind::Brush && c.brush_width > 0.0 {
                    let b = &self.brush;
                    if b.shape != brush::BrushShape::Flat || (b.width - c.brush_width).abs() > 1e-4 || b.bristles.len() != c.brush_count {
                        self.rebuild_brush(brush::BrushShape::Flat, c.brush_count, 0.05, c.brush_width);
                    }
                    self.brush.capacity = c.brush_capacity;
                    self.brush.release = c.brush_release;
                }
                self.tool.pos = c.pos;
                self.tool.vel = c.vel;
                self.tool.flowing = c.flowing;
                self.tool.height = if c.kind == ToolKind::Brush { self.brush.len * c.height } else { c.height };
                self.tool.mu = c.mu;
                self.tool.sigma = c.sigma;
                self.tool.stick_diameter = c.stick_diameter;
                self.tool.bottle_flow = c.bottle_flow;
                self.tool.bottle_radius = c.bottle_radius;
                self.tool.bucket_flow = c.bucket_flow;
                self.tool.bucket_radius = c.bucket_radius;
                self.tool.knife_width = c.knife_width;
                self.tool.knife_height = c.knife_height;
                self.paint.frac = c.mix;
                if c.reload {
                    if c.kind == ToolKind::Brush {
                        let mut f = c.mix;
                        f[14] = (c.mu as f32).ln() + 10.0;
                        if c.brush_width > 0.0 {
                            self.brush.dip_uneven(&f, self.frame_index as u64);
                        } else {
                            self.brush.dip(&f);
                        }
                    } else {
                        self.tool.reload();
                    }
                }
                if c.cut {
                    self.air.detach();
                }
                if c.kind == ToolKind::Pour && c.flowing {
                    let p = [c.pos.x as f32, c.pos.y as f32];
                    let p0 = self.painter_last_pour.unwrap_or(p);
                    let vol = (c.pour_rate * self.sim_dt) as f32;
                    let pig = self.paint.normalized().packed();
                    self.pending_deposits.push(Deposit::stroke(p0, p, c.pour_radius as f32, vol, pig, self.film.dx).laid_by(Method::Pour));
                    self.painter_last_pour = Some(p);
                } else {
                    self.painter_last_pour = None;
                }
                if c.cure_rate > 0.0 {
                    if self.painter_cure_saved.is_none() {
                        self.painter_cure_saved = Some(self.film.settings.cure_rate);
                    }
                    self.film.settings.cure_rate = c.cure_rate as f32;
                } else if let Some(v) = self.painter_cure_saved.take() {
                    self.film.settings.cure_rate = v;
                }
                match c.tilt {
                    Some((deg, dir)) => {
                        if self.painter_tilt_saved.is_none() {
                            self.painter_tilt_saved = Some((self.film.settings.tilt_deg, self.film.settings.tilt_dir_deg));
                        }
                        self.film.settings.tilt_deg = deg as f32;
                        self.film.settings.tilt_dir_deg = dir as f32;
                    }
                    None => {
                        if let Some((deg, dir)) = self.painter_tilt_saved.take() {
                            self.film.settings.tilt_deg = deg;
                            self.film.settings.tilt_dir_deg = dir;
                        }
                    }
                }
                true
            }
            None => {
                if self.painter_active {
                    self.painter_active = false;
                    self.tool.flowing = false;
                    self.painter_last_pour = None;
                    if let Some(v) = self.painter_cure_saved.take() {
                        self.film.settings.cure_rate = v;
                    }
                    if let Some((deg, dir)) = self.painter_tilt_saved.take() {
                        self.film.settings.tilt_deg = deg;
                        self.film.settings.tilt_dir_deg = dir;
                    }
                }
                false
            }
        }
    }

    fn scripted_tool(&mut self) -> bool {
        if self.apply_painter() {
            return true;
        }
        let Some(a) = &self.auto else { return false };
        if a.script == "pollock" {
            if self.painter.is_none() {
                let mut p = pollock::Painter::new(a.seed, self.film.settings.width_m as f64, self.film.settings.height_m as f64);
                if !a.layer.is_empty() {
                    p.skip_to(&a.layer);
                }
                self.painter = Some(p);
                self.painter_rounds_left = a.rounds;
            }
            if self.painter_rounds_left > 0 && self.painter.as_ref().map(|p| p.done).unwrap_or(false) {
                self.painter_rounds_left -= 1;
                if let Some(p) = self.painter.as_mut() {
                    p.continue_painting();
                }
            }
            return self.apply_painter();
        }
        if a.script == "brush4" {
            let f = self.frame_index;
            let w = self.film.settings.width_m as f64;
            let hgt = self.film.settings.height_m as f64;
            let cx = w * 0.5;
            let cy = hgt * 0.5;
            if f < 200 {
                self.tool.kind = ToolKind::Bucket;
                self.prev_kind = ToolKind::Bucket;
                self.tool.height = 0.15;
                self.tool.pos = DVec3::new(cx, cy, 0.15);
                self.tool.vel = DVec3::ZERO;
                self.tool.flowing = f > 10 && f < 130;
            } else {
                self.tool.kind = ToolKind::Brush;
                if self.prev_kind != ToolKind::Brush { self.tool.height = self.brush.len * 0.75; }
                self.prev_kind = ToolKind::Brush;
                let t = (f as f64 - 220.0) / 60.0;
                self.tool.flowing = f >= 220 && t < 3.0;
                self.tool.pos = DVec3::new(cx - 0.15 + 0.1 * t.max(0.0), cy, self.tool.height);
            }
            return true;
        }
        if a.script == "brush3" {
            let f = self.frame_index;
            let w = self.film.settings.width_m as f64;
            let hgt = self.film.settings.height_m as f64;
            let cx = w * 0.5;
            let cy = hgt * 0.5;
            if f < 240 {
                self.tool.kind = ToolKind::Bottle;
                self.prev_kind = ToolKind::Bottle;
                self.tool.height = 0.15;
                self.tool.mu = 8.0;
                self.tool.sigma = 0.0585;
                self.paint.frac[12] = 0.3;
                self.paint.frac[13] = 35.0;
                self.tool.pos = DVec3::new(cx, cy, 0.15);
                self.tool.vel = DVec3::ZERO;
                self.tool.flowing = f > 10 && f < 200;
            } else {
                self.tool.kind = ToolKind::Brush;
                if self.prev_kind != ToolKind::Brush { self.tool.height = self.brush.len * 0.6; }
                self.prev_kind = ToolKind::Brush;
                self.tool.mu = 20.0;
                if f == 260 {
                    let mut p = color::PaintMix::pure(0);
                    p.frac[13] = 40.0;
                    p.frac[14] = (20f32).ln() + 10.0;
                    self.brush.dip(&p.frac);
                }
                let t = (f as f64 - 270.0) / 60.0;
                self.tool.flowing = f >= 270 && t < 2.0;
                self.tool.pos = DVec3::new(cx - 0.08 + 0.08 * t.max(0.0), cy, self.tool.height);
            }
            return true;
        }
        if a.script == "brush2" {
            let f = self.frame_index;
            let w = self.film.settings.width_m as f64;
            let hgt = self.film.settings.height_m as f64;
            self.tool.kind = ToolKind::Brush;
            self.prev_kind = ToolKind::Brush;
            self.tool.height = 0.021;
            self.tool.mu = 20.0;
            if f == 10 {
                let mut p = color::PaintMix::pure(1);
                p.frac[13] = 40.0;
                p.frac[14] = (20f32).ln() + 10.0;
                self.brush.dip(&p.frac);
            }
            let t = (f as f64 - 20.0) / 60.0;
            self.tool.flowing = f >= 20 && t < 2.0;
            self.tool.pos = DVec3::new(w * 0.5 - 0.08 + 0.08 * t.max(0.0), hgt * 0.5, 0.021);
            return true;
        }
        if a.script == "brush" {
            let f = self.frame_index;
            let w = self.film.settings.width_m as f64;
            let hgt = self.film.settings.height_m as f64;
            self.tool.kind = ToolKind::Brush;
            self.prev_kind = ToolKind::Brush;
            self.tool.height = 0.021;
            self.tool.mu = 20.0;
            let lnmu = (20f32).ln() + 10.0;
            if f < 40 {
                let mut paint = color::PaintMix::pure(1);
                paint.frac[12] = 0.0;
                paint.frac[13] = 40.0;
                paint.frac[14] = lnmu;
                let y0 = (hgt * 0.5 - 0.05 + f as f64 * 0.0015) as f32;
                let d = Deposit::stroke([(w * 0.5 - 0.05) as f32, y0], [(w * 0.5 + 0.05) as f32, y0], 0.006, 2.4e-7, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour);
                self.pending_deposits.push(d);
                self.tool.flowing = false;
                self.tool.pos = DVec3::new(w * 0.5, hgt * 0.5 - 0.1, 0.021);
            } else {
                if f == 60 {
                    let mut p = color::PaintMix::pure(0);
                    p.frac[13] = 40.0;
                    p.frac[14] = lnmu;
                    self.brush.dip(&p.frac);
                }
                let t = (f as f64 - 60.0) / 60.0;
                self.tool.flowing = f >= 60 && t < 2.0;
                let lane = if t < 2.0 { 0.0 } else { 0.02 };
                let tt = if t < 2.0 { t.max(0.0) } else { (t - 2.3).max(0.0) };
                self.tool.pos = DVec3::new(w * 0.5 + lane * 2.0, hgt * 0.5 - 0.09 + 0.08 * tt, 0.021);
                self.tool.flowing = self.tool.flowing || (t >= 2.3 && t < 4.3);
            }
            return true;
        }
        if a.script == "layers" {
            let f = self.frame_index;
            let w = self.film.settings.width_m as f64;
            let hgt = self.film.settings.height_m as f64;
            let dt = 1.0 / 60.0;
            self.tool.kind = ToolKind::Pour;
            self.tool.flowing = false;
            if f < 60 {
                let t0 = f as f64 / 60.0;
                let mut paint = color::PaintMix::pure(2);
                paint.frac[12] = 0.1;
                self.knife_test_deposit(t0, w, hgt, dt, paint);
            }
            if (200..260).contains(&f) {
                let t0 = (f - 200) as f64 / 60.0;
                let t1 = (f - 199) as f64 / 60.0;
                let line = |t: f64| [(w * 0.5 - 0.06 + 0.12 * t) as f32, (hgt * 0.5 - 0.03 + 0.06 * t) as f32];
                let mut paint = color::PaintMix::pure(5);
                paint.frac[12] = 0.2;
                let d = Deposit::stroke(line(t0), line(t1), 0.003, (1.5e-6 * dt) as f32, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour);
                self.pending_deposits.push(d);
            }
            return true;
        }
        if a.script == "tiltoil" {
            let f = self.frame_index;
            let w = self.film.settings.width_m as f64;
            let hgt = self.film.settings.height_m as f64;
            self.tool.kind = ToolKind::Bucket;
            self.tool.mu = if a.mu > 0.0 { a.mu } else { 40.0 };
            self.paint = color::PaintMix::pure(if a.pig >= 0 { a.pig as usize } else { 3 });
            self.paint.frac[12] = 0.0;
            self.paint.frac[13] = if a.yield_pa >= 0.0 { a.yield_pa } else { 250.0 };
            let t = f as f64 / 60.0;
            let span = 3.0 * a.seed.max(1) as f64;
            let tilt_at = (60.0 * span) as u64 + 60;
            if a.seed > 0 {
                self.tool.pos = DVec3::new(w * (0.25 + 0.5 * (t / span).min(1.0)), hgt * (0.6 + 0.08 * (t * 7.5 / span).sin()), 0.15);
                self.tool.flowing = f > 5 && (f as f64) < 5.0 + 60.0 * span;
            } else {
                self.tool.pos = DVec3::new(w * 0.5, hgt * 0.6, 0.15);
                self.tool.flowing = f > 5 && f < 90;
            }
            self.tool.vel = DVec3::ZERO;
            if f < tilt_at {
                self.film.settings.tilt_deg = 0.0;
            }
            if f == tilt_at {
                self.film.settings.tilt_deg = if a.tilt > 0.0 { a.tilt } else { 90.0 };
                self.film.settings.tilt_dir_deg = -90.0;
            }
            return true;
        }
        if a.script == "mixing" {
            let f = self.frame_index;
            let dt = 1.0 / 60.0;
            let lnmu = (4f32).ln() + 10.0;
            let paint = |parts: &[(usize, f32)]| {
                let mut m = color::PaintMix { frac: [0.0; 16] };
                for (i, v) in parts {
                    m.frac[*i] = *v;
                }
                m.frac[12] = 0.25;
                m.frac[14] = lnmu;
                m.normalized().packed()
            };
            let dx = self.film.dx;
            let bar = |x: f64, y0: f64, y1: f64, pig: [u32; 8]| Deposit::stroke([x as f32, y0 as f32], [x as f32, y1 as f32], 0.012, 3.5e-6, pig, dx).laid_by(Method::Pour);
            if f == 0 {
                let pairs: [(usize, usize, f32); 8] = [(2, 5, 0.5), (2, 6, 0.5), (3, 2, 0.5), (4, 5, 0.5), (3, 7, 0.5), (0, 6, 0.85), (1, 2, 0.2), (0, 8, 0.5)];
                for (k, (a_pig, b_pig, wa)) in pairs.iter().enumerate() {
                    let x = (0.1 + 0.143 * k as f64) as f32;
                    for (row, pig) in [paint(&[(*a_pig, 1.0)]), paint(&[(*a_pig, *wa), (*b_pig, 1.0 - *wa)]), paint(&[(*b_pig, 1.0)])].iter().enumerate() {
                        let y = (0.72 - 0.08 * row as f64) as f32;
                        self.pending_deposits.push(Deposit::capsule([x, y], [x, y], 0.026, 1.0e-6, *pig, dx).laid_by(Method::Pour));
                    }
                }
                self.pending_deposits.push(Deposit::stroke([0.05, 0.25], [0.27, 0.25], 0.015, 2.5e-6, paint(&[(2, 1.0)]), dx).laid_by(Method::Pour));
                for x0 in [0.43, 0.73, 1.02] {
                    self.pending_deposits.push(bar(x0, 0.12, 0.38, paint(&[(2, 1.0)])));
                    self.pending_deposits.push(bar(x0 + 0.06, 0.12, 0.38, paint(&[(5, 1.0)])));
                }
            }
            if f == 20 {
                self.pending_deposits.push(Deposit::stroke([0.16, 0.13], [0.16, 0.37], 0.015, 2.5e-6, paint(&[(5, 1.0)]), dx).laid_by(Method::Pour));
            }
            self.tool.flowing = false;
            if (60..380).contains(&f) {
                self.tool.kind = ToolKind::Knife;
                let pass = (f - 60) / 80;
                let t = ((f - 60) % 80) as f64 / 80.0;
                let ys = [0.2, 0.3, 0.25, 0.25];
                let right = pass % 2 == 0;
                let x = if right { 0.36 + 0.3 * t } else { 0.66 - 0.3 * t };
                self.knife_dir = if right { [1.0, 0.0] } else { [-1.0, 0.0] };
                self.tool.pos = DVec3::new(x, ys[pass as usize], 0.0);
                self.tool.flowing = t > 0.04 && t < 0.96;
            } else if (400..760).contains(&f) {
                self.tool.kind = ToolKind::Brush;
                self.prev_kind = ToolKind::Brush;
                self.tool.height = 0.024;
                let t = (f - 400) as f64 * dt;
                self.tool.pos = DVec3::new(0.76 + 0.09 * (std::f64::consts::TAU * t / 0.8).sin(), 0.13 + 0.04 * t, 0.024);
                self.tool.flowing = true;
            } else if (780..1080).contains(&f) {
                self.tool.kind = ToolKind::Roller;
                let t = (f - 780) as f64 * dt;
                self.tool.pos = DVec3::new(1.05 + 0.1 * (std::f64::consts::TAU * t / 1.5).sin(), 0.22 + 0.012 * t, 0.0);
                self.tool.flowing = true;
            } else {
                self.tool.kind = ToolKind::Pour;
            }
            return true;
        }
        if a.script == "roller" {
            let f = self.frame_index;
            let w = self.film.settings.width_m as f64;
            let hgt = self.film.settings.height_m as f64;
            let dt = 1.0 / 60.0;
            if f < 60 {
                self.tool.kind = ToolKind::Pour;
                self.tool.flowing = false;
                for (i, pig) in [2usize, 3, 5, 7, 11].iter().enumerate() {
                    let x = (w * 0.5 - 0.04 + 0.02 * i as f64) as f32;
                    let t0 = f as f64 / 60.0;
                    let t1 = (f + 1) as f64 / 60.0;
                    let y = |t: f64| (hgt * 0.5 + 0.1 + 0.15 * t) as f32;
                    let mut paint = color::PaintMix::pure(*pig);
                    paint.frac[12] = 0.25;
                    self.pending_deposits.push(Deposit::stroke([x, y(t0)], [x, y(t1)], 0.006, 2.5e-8, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour));
                }
                return true;
            }
            self.tool.kind = ToolKind::Roller;
            let t = (f as f64 - 120.0).max(0.0) * dt;
            let p = DVec3::new(w * 0.5, hgt * 0.5 + 0.28 - 0.15 * t, 0.0);
            self.tool.pos = p;
            self.tool.flowing = f > 121 && p.y > hgt * 0.5 - 0.32;
            return true;
        }
        if a.script == "knife" {
            let f = self.frame_index;
            let w = self.film.settings.width_m as f64;
            let hgt = self.film.settings.height_m as f64;
            let dt = 1.0 / 60.0;
            if f < 90 {
                self.tool.kind = ToolKind::Pour;
                let t0 = f as f64 / 90.0;
                let mut paint = color::PaintMix::pure(3);
                paint.frac[12] = 0.2;
                if a.yield_pa >= 0.0 {
                    paint.frac[13] = a.yield_pa;
                    paint.frac[12] = 0.0;
                }
                self.knife_test_deposit(t0, w, hgt, dt, paint);
                self.tool.flowing = false;
                return true;
            }
            self.tool.kind = ToolKind::Knife;
            let t = (f - 90) as f64 * dt;
            let speed = 0.12;
            let path = |t: f64| DVec3::new(w * 0.5 - 0.1 + speed * t, hgt * 0.5, 0.0);
            let p = path(t);
            self.tool.vel = (path(t + dt) - p) / dt;
            self.tool.pos = p;
            self.tool.flowing = f > 100 && f < 175;
            return true;
        }
        if a.script == "lines" {
            let f = self.frame_index;
            let w = self.film.settings.width_m as f64;
            let hgt = self.film.settings.height_m as f64;
            let dt = 1.0 / 60.0;
            self.tool.kind = ToolKind::Bottle;
            self.tool.height = 0.1;
            self.tool.mu = if a.mu > 0.0 { a.mu } else { 10.0 };
            self.tool.bottle_flow = 1.8e-6;
            self.tool.bottle_radius = 0.0025;
            self.paint.frac = [0.0; 16];
            self.paint.frac[1] = 1.0;
            self.paint.frac[12] = 0.2;
            let speeds = [0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0];
            let len = w - 0.3;
            let pause = 0.4;
            let t = f as f64 * dt;
            let mut t0 = 0.5;
            let mut cur = None;
            for (i, s) in speeds.iter().enumerate() {
                let dur = len / s;
                if t >= t0 && t < t0 + dur + pause {
                    cur = Some((i, *s, t - t0, dur));
                    break;
                }
                t0 += dur + pause;
            }
            let was_flowing = self.tool.flowing;
            match cur {
                Some((i, s, tl, dur)) => {
                    let y = hgt * (0.08 + 0.14 * i as f64);
                    let on = tl < dur;
                    let x = 0.15 + s * tl.min(dur);
                    self.tool.pos = DVec3::new(x, y, 0.1);
                    self.tool.vel = DVec3::new(if on { s } else { 0.0 }, 0.0, 0.0);
                    self.tool.flowing = on;
                }
                None => {
                    self.tool.flowing = false;
                }
            }
            if was_flowing && !self.tool.flowing {
                self.air.detach();
            }
            return true;
        }
        if a.script == "flick" {
            let f = self.frame_index;
            let w = self.film.settings.width_m as f64;
            let hgt = self.film.settings.height_m as f64;
            let dt = 1.0 / 60.0;
            self.tool.kind = ToolKind::Stick;
            self.tool.height = 0.3;
            self.tool.mu = 0.3;
            self.paint.frac[12] = 0.7;
            self.paint.frac[13] = 0.0;
            let t = f as f64 * dt;
            let swing = |t: f64| {
                let u = (t / 0.35).min(1.0);
                let s = u * u * (3.0 - 2.0 * u);
                DVec3::new(w * 0.3 + 0.25 * s, hgt * 0.5, 0.3)
            };
            let p = swing(t);
            self.tool.vel = (swing(t + dt) - p) / dt;
            self.tool.pos = p;
            self.tool.flowing = f > 2 && t < 0.4;
            return true;
        }
        if a.script != "jet" && a.script != "coil" && a.script != "bucket" {
            return false;
        }
        let f = self.frame_index;
        let w = self.film.settings.width_m as f64;
        let hgt = self.film.settings.height_m as f64;
        let dt = 1.0 / 60.0;
        let (kind, height, speed) = match a.script.as_str() {
            "jet" => (ToolKind::Stick, 0.15, 0.6),
            "coil" => (ToolKind::Bottle, 0.12, 0.02),
            _ => (ToolKind::Bucket, 0.25, 0.15),
        };
        self.tool.kind = kind;
        self.tool.height = height;
        let t = f as f64 * dt;
        let amp = (0.45 * w).min(speed * 0.6);
        let om = speed / amp.max(1e-3);
        let path = |t: f64| DVec3::new(w * 0.5 + amp * (om * t).sin(), hgt * 0.5 + 0.3 * hgt * (2.0 * om * t).sin(), height);
        let p = path(t);
        self.tool.vel = (path(t + dt) - p) / dt;
        self.tool.pos = p;
        self.tool.flowing = f > 5 && f + 1 < a.frames;
        true
    }

    fn knife_test_deposit(&mut self, t0: f64, w: f64, hgt: f64, dt: f64, paint: color::PaintMix) {
        let c = [w * 0.5 + 0.02 * (t0 * 12.0).sin(), hgt * 0.5 + 0.02 * (t0 * 12.0).cos()];
        let d = Deposit::capsule([c[0] as f32, c[1] as f32], [c[0] as f32, c[1] as f32], 0.012, (8.0e-6 * dt) as f32, paint.normalized().packed(), self.film.dx).laid_by(Method::Pour);
        self.pending_deposits.push(d);
    }

    fn beauty_render(&mut self, ortho: bool) {
        let (w, h) = if ortho { (self.film.w * 2, self.film.h * 2) } else { (6144u32, 4096u32) };
        let t0 = Instant::now();
        self.refine.run(&self.gpu, &self.film);
        let img = self.beauty.render(&self.gpu, &self.renderer, &self.cam, &self.film, ortho, w, h, 64, Some(&self.refine));
        let path = format!("beauty_{}_{}.png", if ortho { "top" } else { "view" }, self.frame_index);
        let _ = img.save(&path);
        self.last_beauty = format!("{} ({}x{}, {:.1} s)", path, w, h, t0.elapsed().as_secs_f32());
        log::warn!("saved {}", self.last_beauty);
    }

    fn save_screenshot(&mut self, path: &str, tex: &wgpu::Texture) {
        let w = self.gpu.config.width;
        let h = self.gpu.config.height;
        let bpr = ((w * 4 + 255) / 256) * 256;
        let buf = self.gpu.readback("shot", (bpr * h) as u64);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("shot") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range().expect("map");
        let mut img = image::RgbaImage::new(w, h);
        let bgra = matches!(self.gpu.config.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
        for y in 0..h {
            for x in 0..w {
                let o = (y * bpr + x * 4) as usize;
                let (r, g, b) = if bgra { (data[o + 2], data[o + 1], data[o]) } else { (data[o], data[o + 1], data[o + 2]) };
                img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
            }
        }
        drop(data);
        buf.unmap();
        let [vx, vy, vw, vh] = self.view_rect.map(|v| v.max(0.0) as u32);
        let img = if vw > 0 && vh > 0 && (vw < w || vh < h) { image::imageops::crop_imm(&img, vx, vy, vw.min(w - vx.min(w)), vh.min(h - vy.min(h))).to_image() } else { img };
        let _ = img.save(path);
        log::warn!("saved {}", path);
    }

    fn read_texture_f16(&self, tex: &wgpu::Texture) -> Vec<f32> {
        let w = self.film.w;
        let h = self.film.h;
        let bpr = ((w * 8 + 255) / 256) * 256;
        let buf = self.gpu.readback("export", (bpr * h) as u64);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("export") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range().expect("map");
        let mut out = vec![0.0f32; (w * h * 4) as usize];
        for y in 0..h {
            let row = &data[(y * bpr) as usize..(y * bpr + w * 8) as usize];
            let halves: &[u16] = bytemuck::cast_slice(row);
            for (i, v) in halves.iter().enumerate() {
                out[(y * w * 4) as usize + i] = half::f16::from_bits(*v).to_f32();
            }
        }
        drop(data);
        buf.unmap();
        out
    }

    fn read_texture_raw(&self, tex: &wgpu::Texture) -> Vec<u8> {
        let w = self.film.w;
        let h = self.film.h;
        let bpr = ((w * 8 + 255) / 256) * 256;
        let buf = self.gpu.readback("snap_tex", (bpr * h) as u64);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("snap") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range().expect("map");
        let mut out = Vec::with_capacity((w * h * 8) as usize);
        for y in 0..h {
            out.extend_from_slice(&data[(y * bpr) as usize..(y * bpr + w * 8) as usize]);
        }
        drop(data);
        buf.unmap();
        out
    }

    fn read_buffer_raw(&self, src: &wgpu::Buffer) -> Vec<u8> {
        let size = src.size();
        let buf = self.gpu.readback("snap_buf", size);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("snap") });
        enc.copy_buffer_to_buffer(src, 0, &buf, 0, size);
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let out = buf.slice(..).get_mapped_range().expect("map").to_vec();
        buf.unmap();
        out
    }

    fn save_snapshot(&mut self, path: &str) {
        self.compute_stats();
        let parts = [
            self.read_buffer_raw(&self.film.h_buf),
            self.read_buffer_raw(&self.film.m_buf),
            self.read_buffer_raw(&self.film.tile_flags),
            self.read_texture_raw(&self.film.color_tex),
            self.read_texture_raw(&self.film.detail_tex),
        ];
        let mut out = Vec::new();
        out.extend_from_slice(bytemuck::bytes_of(&[self.film.w, self.film.h]));
        out.extend_from_slice(bytemuck::bytes_of(&[self.film.h_max, self.film.s_max]));
        for p in &parts {
            out.extend_from_slice(bytemuck::bytes_of(&(p.len() as u64)));
            out.extend_from_slice(p);
        }
        let _ = std::fs::write(path, out);
        log::warn!("saved snapshot {}", path);
    }

    fn load_snapshot(&mut self, path: &str) {
        let Ok(data) = std::fs::read(path) else {
            log::warn!("cannot read snapshot {}", path);
            return;
        };
        let dims: [u32; 2] = bytemuck::pod_read_unaligned(&data[0..8]);
        if dims != [self.film.w, self.film.h] {
            log::warn!("snapshot is {}x{}, film is {}x{}", dims[0], dims[1], self.film.w, self.film.h);
            return;
        }
        let maxes: [f32; 2] = bytemuck::pod_read_unaligned(&data[8..16]);
        let mut off = 16usize;
        let mut parts = Vec::new();
        for _ in 0..5 {
            let len: u64 = bytemuck::pod_read_unaligned(&data[off..off + 8]);
            off += 8;
            parts.push(&data[off..off + len as usize]);
            off += len as usize;
        }
        let q = &self.gpu.queue;
        q.write_buffer(&self.film.h_buf, 0, parts[0]);
        q.write_buffer(&self.film.m_buf, 0, parts[1]);
        q.write_buffer(&self.film.tile_flags, 0, parts[2]);
        let w = self.film.w;
        let h = self.film.h;
        for (tex, bytes) in [(&self.film.color_tex, parts[3]), (&self.film.detail_tex, parts[4])] {
            q.write_texture(
                wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                bytes,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
        self.film.h_max = maxes[0];
        self.film.s_max = maxes[1];
        self.paused = true;
        log::warn!("loaded snapshot {}", path);
    }

    fn export_blender(&mut self, dir: &str) {
        let _ = std::fs::create_dir_all(dir);
        let w = self.film.w;
        let h = self.film.h;
        let n = (w * h) as u64;
        let buf = self.gpu.readback("exp_h", n * 4);
        let bufm = self.gpu.readback("exp_m", n * 4);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("export") });
        enc.copy_buffer_to_buffer(&self.film.h_buf, 0, &buf, 0, n * 4);
        enc.copy_buffer_to_buffer(&self.film.m_buf, 0, &bufm, 0, n * 4);
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        bufm.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let hv: Vec<f32> = bytemuck::cast_slice(&buf.slice(..).get_mapped_range().expect("map")).to_vec();
        let mv: Vec<f32> = bytemuck::cast_slice(&bufm.slice(..).get_mapped_range().expect("map")).to_vec();
        buf.unmap();
        bufm.unmap();
        let color = self.read_texture_f16(&self.film.color_tex);
        let detail = self.read_texture_f16(&self.film.detail_tex);
        let mut height = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::new(w, h);
        let mut albedo = image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::new(w, h);
        let mut rough = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::new(w, h);
        let mut mask = image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::new(w, h);
        let q = |v: f32| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        let mut hmax = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                let c = (y * w + x) as usize;
                let total = hv[c].max(0.0) + mv[c].max(0.0);
                hmax = hmax.max(total);
                height.put_pixel(x, h - 1 - y, image::Luma([(total * 1e6).clamp(0.0, 65535.0) as u16]));
                let col = &color[c * 4..c * 4 + 4];
                albedo.put_pixel(x, h - 1 - y, image::Rgb([q(col[0]), q(col[1]), q(col[2])]));
                rough.put_pixel(x, h - 1 - y, image::Luma([q(col[3])]));
                let d = &detail[c * 4..c * 4 + 4];
                let method = d[0].floor();
                let angle = (d[0] - method).clamp(0.0, 1.0);
                let brushed = if (method - 5.0).abs() < 0.5 || (method - 6.0).abs() < 0.5 { 1.0 } else { 0.0 };
                let wet = (hv[c] / 0.5e-3).clamp(0.0, 1.0);
                mask.put_pixel(x, h - 1 - y, image::Rgb([q(wet), q(angle), q(brushed)]));
            }
        }
        let jobs: Vec<(String, Vec<u16>, image::ColorType)> = vec![
            (format!("{}/height.png", dir), height.into_raw(), image::ColorType::L16),
            (format!("{}/albedo.png", dir), albedo.into_raw(), image::ColorType::Rgb16),
            (format!("{}/roughness.png", dir), rough.into_raw(), image::ColorType::L16),
            (format!("{}/mask.png", dir), mask.into_raw(), image::ColorType::Rgb16),
        ];
        use rayon::prelude::*;
        jobs.par_iter().for_each(|(path, data, ct)| {
            let be: &[u8] = bytemuck::cast_slice(data);
            if let Ok(f) = std::fs::File::create(path) {
                let wr = std::io::BufWriter::new(f);
                let enc = image::codecs::png::PngEncoder::new_with_quality(wr, image::codecs::png::CompressionType::Fast, image::codecs::png::FilterType::NoFilter);
                let _ = image::ImageEncoder::write_image(enc, be, w, h, (*ct).into());
            }
        });
        let meta = format!(
            "{{\n  \"cells\": [{}, {}],\n  \"size_m\": [{}, {}],\n  \"height_scale_m\": 0.065535,\n  \"height_max_m\": {:.6},\n  \"frame\": {}\n}}\n",
            w, h, self.film.settings.width_m, self.film.settings.height_m, hmax, self.frame_index
        );
        let _ = std::fs::write(format!("{}/canvas.json", dir), meta);
        let _ = std::fs::write(format!("{}/render.py", dir), include_str!("../blender/render.py"));
        self.last_beauty = format!("exported {} ({}x{})", dir, w, h);
        log::warn!("exported Blender set to {}", dir);
    }

    fn compute_stats(&mut self) {
        let n = (self.film.w * self.film.h) as u64;
        let buf = self.gpu.readback("stats", n * 4);
        let bufm = self.gpu.readback("statsm", n * 4);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stats") });
        enc.copy_buffer_to_buffer(&self.film.h_buf, 0, &buf, 0, n * 4);
        enc.copy_buffer_to_buffer(&self.film.m_buf, 0, &bufm, 0, n * 4);
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        bufm.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let hv: Vec<f32> = bytemuck::cast_slice(&buf.slice(..).get_mapped_range().expect("map")).to_vec();
        let mv: Vec<f32> = bytemuck::cast_slice(&bufm.slice(..).get_mapped_range().expect("map")).to_vec();
        buf.unmap();
        bufm.unmap();
        if let Ok(prefix) = std::env::var("PAINT_DUMP") {
            let dims = [self.film.w, self.film.h];
            let _ = std::fs::write(format!("{}_dims.bin", prefix), bytemuck::cast_slice::<u32, u8>(&dims));
            let _ = std::fs::write(format!("{}_h.bin", prefix), bytemuck::cast_slice::<f32, u8>(&hv));
            let _ = std::fs::write(format!("{}_m.bin", prefix), bytemuck::cast_slice::<f32, u8>(&mv));
            let bd = self.gpu.readback("delta_r", n * 4);
            let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dump") });
            enc.copy_buffer_to_buffer(&self.film.delta, 0, &bd, 0, n * 4);
            self.gpu.queue.submit(std::iter::once(enc.finish()));
            bd.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            let dv: Vec<u32> = bytemuck::cast_slice(&bd.slice(..).get_mapped_range().expect("map")).to_vec();
            bd.unmap();
            let _ = std::fs::write(format!("{}_d.bin", prefix), bytemuck::cast_slice::<u32, u8>(&dv));
        }
        let dx = self.film.dx as f64;
        let mut sum = 0.0f64;
        let mut summ = 0.0f64;
        let mut hmax = 0.0f32;
        let mut smax = 0.0f32;
        let mut wet = 0u64;
        let mut nan = 0u64;
        for (a, b) in hv.iter().zip(mv.iter()) {
            if !a.is_finite() { nan += 1; continue; }
            sum += *a as f64;
            summ += *b as f64;
            if *a > 0.0 { wet += 1; }
            hmax = hmax.max(*a);
            smax = smax.max(*a + *b);
        }
        self.film.s_max = smax;
        let vol = sum * dx * dx;
        let volm = summ * dx * dx - self.film.weave_volume;
        {
            let (imax, _) = hv.iter().enumerate().fold((0usize, 0.0f32), |a, (i, v)| if *v > a.1 { (i, *v) } else { a });
            let w = self.film.w as usize;
            let row = imax / w;
            let col = imax % w;
            let mut prof = String::new();
            let mut k = 0usize;
            while col + k < w && k <= 1024 {
                prof.push_str(&format!(" {:.0}", hv[row * w + col + k] * 1e6));
                k += 32;
            }
            log::warn!("profile um every {:.1} mm from max:{}", 32.0 * self.film.dx * 1e3, prof);
            let mut fine = String::new();
            let mut k = 0usize;
            while col + k < w && k <= 240 && (k < 2 || hv[row * w + col + k - 1] > 0.0) {
                fine.push_str(&format!(" {:.0}", hv[row * w + col + k] * 1e6));
                k += 1;
            }
            log::warn!("fine profile um per cell from max:{}", fine);
            let mut nb = String::new();
            for dj in -3i64..=3 {
                nb.push_str(" |");
                for di in -3i64..=3 {
                    let ii = col as i64 + di;
                    let jj = row as i64 + dj;
                    if ii >= 0 && jj >= 0 && (ii as usize) < w && (jj as usize) < self.film.h as usize {
                        nb.push_str(&format!(" {:.0}", hv[jj as usize * w + ii as usize] * 1e6));
                    }
                }
            }
            log::warn!("7x7 um around max:{}", nb);
            let mut art = String::new();
            for dj in -60i64..=60 {
                art.push_str("
");
                for di in -80i64..=80 {
                    let ii = col as i64 + di * 2;
                    let jj = row as i64 + dj * 2;
                    let mut ch = ' ';
                    if ii >= 0 && jj >= 0 && (ii as usize) < w && (jj as usize) < self.film.h as usize {
                        let v = hv[jj as usize * w + ii as usize];
                        ch = if v > 2e-3 { '#' } else if v > 1e-3 { '%' } else if v > 3e-4 { '+' } else if v > 0.0 { '.' } else { ' ' };
                    }
                    art.push(ch);
                }
            }
            log::warn!("planform around max (rows 0.39 mm, cols 0.39 mm):{}", art);
            let nn = (self.film.w * self.film.h) as u64;
            let br = self.gpu.readback("rheo_r", nn * 8);
            let bs = self.gpu.readback("sh_r", nn * 16);
            let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stats3") });
            enc.copy_buffer_to_buffer(&self.film.rheo, 0, &br, 0, nn * 8);
            enc.copy_buffer_to_buffer(&self.film.strata_h, 0, &bs, 0, nn * 16);
            self.gpu.queue.submit(std::iter::once(enc.finish()));
            br.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            bs.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            let rv: Vec<[f32; 2]> = bytemuck::cast_slice(&br.slice(..).get_mapped_range().expect("map")).to_vec();
            let sv: Vec<[f32; 4]> = bytemuck::cast_slice(&bs.slice(..).get_mapped_range().expect("map")).to_vec();
            br.unmap();
            bs.unmap();
            let mut pr = String::new();
            let mut k = 0usize;
            while col + k < w && k <= 512 {
                let c = row * w + col + k;
                let st = sv[c];
                pr.push_str(&format!(" [mu {:.2} ty {:.1} st {:.0}/{:.0}/{:.0}/{:.0}]", rv[c][0], rv[c][1], st[0] * 1e6, st[1] * 1e6, st[2] * 1e6, st[3] * 1e6));
                k += 32;
            }
            log::warn!("rheo/strata along profile:{}", pr);
            let mut low_ty = 0usize;
            let mut no_strata = 0usize;
            let mut wetc = 0usize;
            for c in 0..(nn as usize) {
                if hv[c] > 0.0 {
                    wetc += 1;
                    if rv[c][1] < 1.0 { low_ty += 1; }
                    let st = sv[c];
                    if st[0] + st[1] + st[2] + st[3] <= 0.0 { no_strata += 1; }
                }
            }
            log::warn!("wet {} low_ty {} no_strata {}", wetc, low_ty, no_strata);
        }
        let cr = self.film.cg_report;
        log::warn!("implicit: {} steps, residual {:.2e}, drift {:.2e} of {:.3e}", self.film.implicit_steps, if cr[0] > 0.0 { (cr[3] / cr[0]).sqrt() } else { 0.0 }, cr[1], cr[2]);
        self.stats = format!(
            "wet {:.4} mL  cured {:.4} mL  deposited {:.4} mL  brush held {:.4} mL  err {:.2}%  hmax {:.3} mm  wet cells {}  active tiles {}  nan {}  fps {:.0}  substeps {}",
            vol * 1e6, volm * 1e6, self.film.total_deposited * 1e6, self.brush.total_load() * 1e6,
            if self.film.total_deposited > 0.0 { (vol + volm + self.brush_scraped - self.brush_released - self.film.total_deposited) / self.film.total_deposited * 100.0 } else { 0.0 },
            hmax * 1e3, wet, self.film.active_tiles, nan, self.fps, self.film.substeps_last
        );
        log::warn!("{}", self.stats);
    }

    fn cursor_over_panel(&self) -> bool {
        let [x, y, w, h] = self.view_rect;
        let (cx, cy) = (self.cursor[0], self.cursor[1]);
        if cx < x || cy < y || cx >= x + w || cy >= y + h || self.menu_open {
            return true;
        }
        let ppp = self.ui.ctx.pixels_per_point().max(0.01);
        let p = egui::pos2(cx / ppp, cy / ppp);
        self.block_rects.iter().any(|r| r.expand(4.0).contains(p))
    }

    fn load_knife(&mut self) {
        let each = self.tool.knife_width * KNIFE_CAPACITY_PER_WIDTH / KNIFE_BANDS as f64;
        for b in &mut self.knife_bands {
            b.vol = each;
            b.comp = self.paint.frac;
        }
    }

    fn knife_load(&self) -> f64 {
        self.knife_bands.iter().map(|b| b.vol).sum()
    }

    fn knife_mix(&self) -> [f32; 16] {
        let mut m = KnifeBand { vol: 0.0, comp: [0.0; 16] };
        for b in &self.knife_bands {
            m.blend(b.vol, &b.comp);
        }
        m.comp
    }

    fn wipe_knife(&mut self) {
        for b in &mut self.knife_bands {
            b.vol = 0.0;
        }
    }

    fn knife_capacity(&self) -> f64 {
        self.tool.knife_width * KNIFE_CAPACITY_PER_WIDTH
    }

    fn pen(&self) -> Option<f64> {
        if self.pen_down && self.pen_pressure {
            Some((self.pen_raw.clamp(0.0, 1.0) as f64).powf(self.pen_curve as f64))
        } else {
            None
        }
    }

    fn ndc(&self) -> [f32; 2] {
        let [x, y, w, h] = self.view_rect;
        [(self.cursor[0] - x) / w.max(1.0) * 2.0 - 1.0, 1.0 - (self.cursor[1] - y) / h.max(1.0) * 2.0]
    }

    fn frame(&mut self) {
        let size = self.window.inner_size();
        let surface_tex = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                self.gpu.resize(size.width, size.height);
                return;
            }
        };
        if self.auto.as_ref().map(|a| a.pace).unwrap_or(false) {
            let el = self.last.elapsed();
            if el < std::time::Duration::from_micros(16600) {
                std::thread::sleep(std::time::Duration::from_micros(16600) - el);
            }
        }
        let now = Instant::now();
        let real_dt = (now - self.last).as_secs_f32().min(0.1);
        self.last = now;
        self.fps = 0.9 * self.fps + 0.1 / real_dt.max(1e-4);
        let frame_dt = (1.0 / 60.0) * if self.painter.as_ref().map(|p| !p.done).unwrap_or(false) { self.painter_speed } else { 1.0 };
        self.sim_dt = frame_dt as f64;
        if self.auto_quality && !self.paused {
            let ms = (real_dt * 1e3 - self.last_cpu_ms).max(0.0);
            let s = &mut self.film.settings;
            if ms > self.frame_budget_ms * 1.15 {
                s.max_substeps = (s.max_substeps as f32 * 0.85).floor().max(8.0) as u32;
            } else if ms < self.frame_budget_ms * 0.7 {
                s.max_substeps = (s.max_substeps + 2).min(128);
            }
        }

        let mut deposits: Vec<Deposit> = Vec::new();
        if self.auto.as_ref().map(|a| a.script.is_empty()).unwrap_or(false) {
            self.scripted_deposits(&mut deposits);
        }
        let want_pour = self.mouse_down[0] && !self.cursor_over_panel();
        let pen = self.pen();
        self.tool.flow_scale = pen.map(|p| PEN_FLOW_MIN + (PEN_FLOW_MAX - PEN_FLOW_MIN) * p).unwrap_or(1.0);
        let mut scraps: Vec<Scrap> = Vec::new();
        self.film.poll_scraps(&self.gpu, &mut scraps);
        if self.tool.kind == ToolKind::Pour {
            if want_pour {
                if let Some(p) = self.cam.plane_hit(self.ndc(), 0.0) {
                    let p0 = self.last_pour.unwrap_or(p);
                    let vol = self.pour_rate * frame_dt * self.tool.flow_scale as f32;
                    let radius = self.pour_radius * pen.map(|p| PEN_WIDTH_MIN + (1.0 - PEN_WIDTH_MIN) * p as f32).unwrap_or(1.0);
                    let pig = self.paint.normalized().packed();
                    deposits.push(Deposit::stroke([p0.x, p0.y], [p.x, p.y], radius, vol, pig, self.film.dx).laid_by(Method::Pour));
                    self.last_pour = Some(p);
                }
            } else {
                self.last_pour = None;
            }
        } else {
            self.last_pour = None;
        }
        if let Some(a) = &self.auto {
            if a.script == "hold" {
                let size = self.window.inner_size();
                let f = self.frame_index;
                let phase = (f / 420) % 3;
                let off = phase as f32 * 0.06;
                self.cursor = [size.width as f32 * (0.55 + off), size.height as f32 * 0.5];
                self.mouse_down[0] = f > 10 && (f % 420) < 360;
                self.tool.kind = ToolKind::Bottle;
                self.tool.height = 0.15;
                self.tool.mu = 8.0;
                self.tool.sigma = 0.0585;
                self.paint.frac[12] = 0.3;
                self.paint.frac[13] = 0.0;
                if a.yield_pa >= 0.0 { self.paint.frac[13] = a.yield_pa; self.paint.frac[12] = 0.0; }
                if a.pig >= 0 { let y = self.paint.frac[13]; let th = self.paint.frac[12]; self.paint = color::PaintMix::pure(a.pig as usize); self.paint.frac[13] = y; self.paint.frac[12] = th; }
                if a.mu > 0.0 { self.tool.mu = a.mu; }
            }
            if a.script == "pour" {
                let size = self.window.inner_size();
                let f = self.frame_index;
                let t = f as f32 / 60.0;
                let x = if f < 130 { 0.5 } else if f < 400 { 0.5 + 0.05 * (t - 130.0 / 60.0) } else { 0.5 + 0.05 * 4.5 + 0.3 * (t - 400.0 / 60.0) };
                let y = if f < 400 { 0.5 } else { 0.5 + 0.06 * ((t - 400.0 / 60.0) * 3.0).sin() };
                self.cursor = [size.width as f32 * x.min(0.95), size.height as f32 * y];
                self.mouse_down[0] = f > 10 && f < 560;
                self.tool.kind = ToolKind::Pour;
                self.tool.mu = if a.mu > 0.0 { a.mu } else { 4.0 };
                self.paint.frac[12] = 0.3;
                self.paint.frac[13] = if a.yield_pa >= 0.0 { a.yield_pa } else { 0.0 };
            }
            if a.script == "settle" {
                let size = self.window.inner_size();
                let f = self.frame_index;
                self.cursor = [size.width as f32 * 0.55, size.height as f32 * 0.5];
                self.mouse_down[0] = f > 10 && f < 300;
                self.tool.kind = ToolKind::Bottle;
                self.tool.height = 0.15;
                self.tool.mu = 8.0;
                self.tool.sigma = 0.0585;
                self.paint.frac[12] = 0.3;
                self.paint.frac[13] = 0.0;
                if a.yield_pa >= 0.0 { self.paint.frac[13] = a.yield_pa; self.paint.frac[12] = 0.0; }
                if a.mu > 0.0 { self.tool.mu = a.mu; }
            }
            if a.script == "stress2" {
                let size = self.window.inner_size();
                let f = self.frame_index;
                let t = f as f32 / 60.0;
                let seed = ((f as u64).wrapping_mul(2654435761) >> 7) as f32 / 4294967296.0 * 32.0;
                let jump = (f % 17 == 0) as u32 as f32;
                self.cursor = [
                    size.width as f32 * (0.55 + 0.3 * (t * 2.1 + seed).sin() + 0.2 * jump * (seed * 3.0).sin()),
                    size.height as f32 * (0.5 + 0.35 * (t * 3.7 + seed * 0.5).cos()),
                ];
                self.mouse_down[0] = f > 10 && (f / 70) % 3 != 2;
                self.tool.kind = if (f / 300) % 2 == 0 { ToolKind::Stick } else { ToolKind::Bottle };
                self.tool.height = 0.05 + 0.4 * ((t * 0.9).sin() * 0.5 + 0.5) as f64;
                self.tool.mu = 0.05;
                self.paint.frac[12] = 0.8;
                self.paint.frac[13] = 0.0;
                if f % 200 == 0 { self.tool.reload(); }
            }
            if a.script == "stress" {
                let size = self.window.inner_size();
                let f = self.frame_index;
                let t = f as f32 / 60.0;
                let jerk = ((t * 7.3).sin() * (t * 3.1).cos()).abs();
                self.cursor = [size.width as f32 * (0.55 + 0.3 * (t * 2.1).sin() * jerk), size.height as f32 * (0.5 + 0.35 * (t * 3.7).cos())];
                self.mouse_down[0] = f > 10 && (f / 90) % 4 != 3;
                self.tool.kind = ToolKind::Stick;
                self.tool.mu = 0.5;
                self.paint.frac[12] = 0.6;
                self.paint.frac[13] = 0.0;
                if f % 240 == 0 { self.tool.reload(); }
                if f % 500 == 250 { self.tool.kind = ToolKind::Bottle; }
            }
            if a.script == "mouse" {
                let size = self.window.inner_size();
                let f = self.frame_index as f32;
                let t = (f / 60.0).min(4.0);
                self.cursor = [size.width as f32 * (0.45 + 0.08 * (t * 1.5).sin()), size.height as f32 * (0.5 + 0.1 * (t * 2.0).cos())];
                self.mouse_down[0] = f > 10.0 && f < 230.0;
            }
        }
        let scripted = self.scripted_tool();
        if let Some(a) = &self.auto {
            if a.dry > 0 && self.frame_index == a.dry as u64 {
                self.film.dry_now(&self.gpu);
            }
            if a.stop > 0 && self.frame_index >= a.stop {
                let was = self.tool.flowing;
                self.tool.flowing = false;
                if was { self.air.detach(); }
            }
        }
        deposits.append(&mut self.pending_deposits);
        if !scripted {
            let pick_z = if matches!(self.tool.kind, ToolKind::Brush | ToolKind::Roller) { 0.0 } else { self.tool.height as f32 };
            if let Some(p) = self.cam.plane_hit(self.ndc(), pick_z).or_else(|| self.cam.plane_hit(self.ndc(), 0.0)) {
                let p = DVec3::new(p.x as f64, p.y as f64, self.tool.height);
                let v = match self.prev_tool_pos {
                    Some(q) => (p - q) / frame_dt as f64,
                    None => DVec3::ZERO,
                };
                self.tool.vel = self.tool.vel * 0.5 + v * 0.5;
                self.tool.pos = p;
                self.prev_tool_pos = Some(p);
            }
            self.tool.flowing = want_pour && self.tool.kind != ToolKind::Pour;
            if self.tool.kind == ToolKind::Knife {
                if let Some(p) = self.cam.plane_hit(self.ndc(), 0.0) {
                    let p = DVec3::new(p.x as f64, p.y as f64, 0.0);
                    let v = match self.prev_tool_pos { Some(q) => (p - q) / frame_dt as f64, None => DVec3::ZERO };
                    self.tool.vel = v;
                    self.tool.pos = p;
                    self.prev_tool_pos = Some(p);
                }
            }
        }
        self.paint.frac[14] = (self.tool.mu as f32).ln() + 10.0;
        self.film.settings.sigma = self.tool.sigma as f32;
        self.tool.paint = self.paint;
        let mut knife = None;
        if self.tool.kind == ToolKind::Knife {
            let here = DVec3::new(self.tool.pos.x, self.tool.pos.y, 0.0);
            let moved = self.knife_track.map(|q| here - q).unwrap_or(DVec3::ZERO);
            let dist = moved.length();
            if dist > KNIFE_TURN_MIN {
                let k = (dist / KNIFE_TURN_DIST).min(1.0);
                let flip = if moved.x * self.knife_dir[0] + moved.y * self.knife_dir[1] < 0.0 { -1.0 } else { 1.0 };
                let nx = self.knife_dir[0] + (flip * moved.x / dist - self.knife_dir[0]) * k;
                let ny = self.knife_dir[1] + (flip * moved.y / dist - self.knife_dir[1]) * k;
                let l = (nx * nx + ny * ny).sqrt().max(1e-9);
                self.knife_dir = [flip * nx / l, flip * ny / l];
            }
            self.knife_track = Some(here);
            let d = DVec3::new(self.knife_dir[0], self.knife_dir[1], 0.0);
            let blade_dir = DVec3::new(-d.y, d.x, 0.0);
            let hw = self.tool.knife_width * 0.5;
            let p = DVec3::new(self.tool.pos.x, self.tool.pos.y, 0.0);
            let gap = self.tool.knife_height * pen.map(|p| 1.0 - PEN_KNIFE_CLOSE * p).unwrap_or(1.0);
            let hold = self.tool.knife_width * KNIFE_HOLD_PER_WIDTH;
            let band_w = self.tool.knife_width / KNIFE_BANDS as f64;
            if self.tool.flowing && !self.paused {
                let step = self.knife_prev.map(|q| p - q).unwrap_or(DVec3::ZERO);
                let ahead = step.length();
                let across = if ahead > 0.0 { (step / ahead).dot(d) } else { 0.0 };
                if ahead > 1e-6 && across > 0.2 {
                    let mdir = step / ahead;
                    let a = p - blade_dir * hw;
                    let b = p + blade_dir * hw;
                    let need = gap.max(1e-6) * band_w * (ahead + self.film.dx as f64);
                    let mut pig = [[0u32; 8]; KNIFE_BANDS];
                    let mut fill = [0.0f32; KNIFE_BANDS];
                    let mut smear = [0.0f32; KNIFE_BANDS];
                    for (i, band) in self.knife_bands.iter().enumerate() {
                        let ratio = ((band.vol - self.knife_pending[i]).max(0.0) / need).min(1.0);
                        fill[i] = ratio as f32;
                        smear[i] = (KNIFE_SMEAR * ratio) as f32;
                        pig[i] = color::PaintMix { frac: band.comp }.normalized().packed();
                        self.knife_pending[i] += need * (1.5 * ratio).min(1.0) * ratio.powf(0.6);
                    }
                    self.knife_poses.push_back((self.frame_index as u32, p, blade_dir));
                    while self.knife_poses.len() > KNIFE_POSE_HISTORY {
                        self.knife_poses.pop_front();
                    }
                    knife = Some(KnifeParams {
                        a: [a.x as f32, a.y as f32],
                        b: [b.x as f32, b.y as f32],
                        dir: [mdir.x as f32, mdir.y as f32],
                        sweep: ahead as f32,
                        height: gap as f32,
                        half_width: hw as f32,
                        active: 1,
                        frame: self.frame_index as u32,
                        pad1: 0,
                        bands: KNIFE_BANDS as u32,
                        pad2: 0,
                        pad3: 0,
                        pad4: 0,
                        pig,
                        fill,
                        smear,
                    });
                    let keep = hold / KNIFE_BANDS as f64;
                    for i in 0..KNIFE_BANDS {
                        let bow = (self.knife_bands[i].vol - self.knife_pending[i] - keep).max(0.0);
                        if bow <= 0.0 {
                            continue;
                        }
                        self.knife_bands[i].vol -= bow;
                        let r = (0.75 * bow / band_w).sqrt().max(self.film.dx as f64);
                        let q = p + mdir * (r + self.film.dx as f64);
                        let a = q + blade_dir * (-hw + (i as f64 + 0.15) * band_w);
                        let b = q + blade_dir * (-hw + (i as f64 + 0.85) * band_w);
                        let pig = color::PaintMix { frac: self.knife_bands[i].comp }.normalized().packed();
                        deposits.push(Deposit::stroke([a.x as f32, a.y as f32], [b.x as f32, b.y as f32], r as f32, bow as f32, pig, self.film.dx).laid_by(Method::Knife).recycled());
                    }
                }
                if let Some(q) = self.knife_prev {
                    let dist = (p - q).length();
                    let bow = self.knife_load() - hold;
                    if bow > 0.0 && dist > 0.0 {
                        let spill = bow * (dist * KNIFE_SPILL_PER_M).min(1.0);
                        for (side, bi) in [(-1.0, 0usize), (1.0, KNIFE_BANDS - 1)] {
                            let take = (0.5 * spill).min(self.knife_bands[bi].vol);
                            if take <= 0.0 {
                                continue;
                            }
                            self.knife_bands[bi].vol -= take;
                            let pig = color::PaintMix { frac: self.knife_bands[bi].comp }.normalized().packed();
                            let r = (0.75 * take / dist).sqrt().max(self.film.dx as f64) as f32;
                            let e0 = q + blade_dir * (side * (hw + r as f64 * 0.6));
                            let e1 = p + blade_dir * (side * (hw + r as f64 * 0.6));
                            deposits.push(Deposit::stroke([e0.x as f32, e0.y as f32], [e1.x as f32, e1.y as f32], r, take as f32, pig, self.film.dx).laid_by(Method::Knife).recycled());
                        }
                    }
                    let mix = (dist * KNIFE_BAND_MIX_PER_M).min(0.3);
                    if mix > 0.0 {
                        let old = self.knife_bands;
                        for i in 0..KNIFE_BANDS {
                            let neighbours = [i.checked_sub(1), Some(i + 1).filter(|&n| n < KNIFE_BANDS)];
                            let given = neighbours.iter().flatten().count() as f64 * 0.5 * mix * old[i].vol;
                            let mut b = KnifeBand { vol: old[i].vol - given, comp: old[i].comp };
                            for j in neighbours.iter().flatten() {
                                b.blend(0.5 * mix * old[*j].vol, &old[*j].comp);
                            }
                            self.knife_bands[i] = b;
                        }
                    }
                }
                self.knife_prev = Some(p);
            } else {
                if let Some(last) = self.knife_prev {
                    let keep = hold / KNIFE_BANDS as f64;
                    for i in 0..KNIFE_BANDS {
                        let vol = (self.knife_bands[i].vol - keep).max(0.0) * KNIFE_END_RIDGE;
                        if vol <= 0.0 {
                            continue;
                        }
                        self.knife_bands[i].vol -= vol;
                        let r = (0.75 * vol / band_w).sqrt().max(self.film.dx as f64);
                        let q = last + d * r;
                        let a = q + blade_dir * (-hw + (i as f64 + 0.15) * band_w);
                        let b = q + blade_dir * (-hw + (i as f64 + 0.85) * band_w);
                        let pig = color::PaintMix { frac: self.knife_bands[i].comp }.normalized().packed();
                        deposits.push(Deposit::stroke([a.x as f32, a.y as f32], [b.x as f32, b.y as f32], r as f32, vol as f32, pig, self.film.dx).laid_by(Method::Knife).recycled());
                    }
                }
                self.knife_prev = None;
            }
        } else {
            self.knife_prev = None;
            self.knife_track = None;
        }
        self.scrap_total += scraps.iter().map(|s| s.vol.max(0.0) as f64).sum::<f64>();
        for p in &mut self.knife_pending {
            *p *= KNIFE_PENDING_DECAY;
        }
        if !scraps.is_empty() {
            let width = self.tool.knife_width.max(1e-6);
            let fallback = (DVec3::new(self.tool.pos.x, self.tool.pos.y, 0.0), DVec3::new(-self.knife_dir[1], self.knife_dir[0], 0.0));
            for s in &scraps {
                let (centre, blade) = self.knife_poses.iter().rev().find(|q| q.0 == s.frame).map(|q| (q.1, q.2)).unwrap_or(fallback);
                let i = (s.cell % self.film.w) as f64 + 0.5;
                let j = (s.cell / self.film.w) as f64 + 0.5;
                let cell = DVec3::new(i * self.film.dx as f64, j * self.film.dx as f64, 0.0);
                let along = (cell - centre).dot(blade) / width + 0.5;
                let bi = ((along * KNIFE_BANDS as f64).floor().max(0.0) as usize).min(KNIFE_BANDS - 1);
                let vol = s.vol as f64;
                if vol < 0.0 {
                    self.knife_bands[bi].vol = (self.knife_bands[bi].vol + vol).max(0.0);
                    self.knife_pending[bi] = (self.knife_pending[bi] + vol).max(0.0);
                    continue;
                }
                let mut comp = [0.0f32; 16];
                for k in 0..8 {
                    comp[2 * k] = half::f16::from_bits((s.pig[k] & 0xffff) as u16).to_f32();
                    comp[2 * k + 1] = half::f16::from_bits((s.pig[k] >> 16) as u16).to_f32();
                }
                self.knife_bands[bi].blend(vol, &comp);
            }
        }
        let mut contacts = Vec::new();
        if self.tool.kind == ToolKind::Brush {
            if self.prev_kind != ToolKind::Brush {
                self.tool.height = self.brush.len * 0.75;
            }
            self.tool.height = self.tool.height.clamp(0.002, self.brush.len - 0.003);
            let mut res = Vec::new();
            self.film.poll_reservoir(&self.gpu, &mut res);
            self.brush_scraped += self.brush.absorb(&res);
            if !self.paused {
                let lift = if self.tool.flowing { 0.0 } else { 0.04 };
                let h = match pen {
                    Some(p) => (self.brush.len * (1.0 - PEN_BRUSH_PRESS * p)).clamp(0.002, self.brush.len - 0.0005),
                    None => self.tool.height,
                };
                let handle = DVec3::new(self.tool.pos.x, self.tool.pos.y, h + lift);
                let surf = &self.surface;
                let brush = &mut self.brush;
                let sink = self.brush_sink;
                let (c, rel) = brush.update(handle, frame_dt as f64, &|x, y| surf.brush_floor(x, y, sink), self.film.dx as f64);
                contacts = c;
                for r in rel {
                    let pig = color::PaintMix { frac: r.comp }.normalized().packed();
                    let d = Deposit::stroke(r.p0, r.p1, r.r, r.vol as f32, pig, self.film.dx).laid_by(Method::Brush).recycled();
                    self.brush_released += r.vol * d.inside_fraction(self.film.settings.width_m, self.film.settings.height_m) as f64;
                    deposits.push(d);
                }
            }
        }
        if self.tool.kind == ToolKind::Roller {
            let mut res = Vec::new();
            self.film.poll_reservoir(&self.gpu, &mut res);
            self.roller.absorb(&res);
            if !self.paused {
                let pressure = pen.unwrap_or(self.roller.pressure);
                let at = glam::DVec2::new(self.tool.pos.x, self.tool.pos.y);
                let (c, d) = self.roller.update(at, self.tool.flowing, pressure, self.film.dx);
                contacts = c;
                deposits.extend(d);
            }
        } else {
            self.roller.lift();
        }
        self.film.upload_contacts(&self.gpu, &contacts);
        self.prev_kind = self.tool.kind;
        let t_air = Instant::now();
        if !self.paused {
            let g = self.film.gravity_vec();
            let gravity = DVec3::new(g[0] as f64, g[1] as f64, -(g[2] as f64));
            self.surface.decay_recent();
            let bounds = [self.film.settings.width_m as f64, self.film.settings.height_m as f64];
            self.air.update(&mut self.tool, frame_dt as f64, &mut self.surface, gravity, self.film.dx, &mut deposits, bounds);
        }
        let ms_air = t_air.elapsed().as_secs_f32() * 1e3;
        let air_detail = format!("(rods {:.1} land {:.1} drops {:.1} caps {} steps {})", self.air.prof[0], self.air.prof[1], self.air.prof[2], self.surface.caps.len(), self.air.rod_steps);
        let t_mesh = Instant::now();
        let colors: Vec<[f32; 3]> = (0..color::NPIG).map(|i| color::masstone_rgb(&self.film.pigments, &self.film.cmf_table, i)).collect();
        let color_of = |pig: &[u32; 8]| -> [f32; 3] {
            let mut c = [0.0f32; 3];
            let mut w = 0.0;
            for k in 0..6 {
                let a = half::f16::from_bits((pig[k] & 0xffff) as u16).to_f32();
                let b = half::f16::from_bits((pig[k] >> 16) as u16).to_f32();
                for (f, i) in [(a, 2 * k), (b, 2 * k + 1)] {
                    if i < color::NPIG && f > 0.0 {
                        let lin = |x: f32| x.powf(2.2);
                        c[0] += f * lin(colors[i][0]);
                        c[1] += f * lin(colors[i][1]);
                        c[2] += f * lin(colors[i][2]);
                        w += f;
                    }
                }
            }
            if w > 0.0 { [c[0] / w, c[1] / w, c[2] / w] } else { [0.5, 0.5, 0.5] }
        };
        let (mut tv, mut ti) = tools::tube_mesh(&self.air.rods, &self.air.drops, &color_of);
        let solid = ti.len();
        if self.tool.kind == ToolKind::Brush {
            let r = self.brush.ferrule_r;
            let ferrule = [rod::Drop { x: self.brush.handle, v: DVec3::ZERO, vol: 4.0 / 3.0 * std::f64::consts::PI * r * r * r, pig: [0; 8], fluid: self.tool.fluid() }];
            let (fv, fi) = tools::tube_mesh(&[], &ferrule, &color_of);
            let base = tv.len() as u32;
            tv.extend(fv);
            ti.extend(fi.iter().map(|i| i + base));
        }
        if self.auto.is_none() && (self.painter_active || !self.cursor_over_panel()) {
            let ground = if self.painter_active { Some(DVec3::new(self.tool.pos.x, self.tool.pos.y, 0.0)) } else { self.cam.plane_hit(self.ndc(), 0.0).map(|g| DVec3::new(g.x as f64, g.y as f64, 0.0)) };
            let gx = gizmo::GizmoCtx {
                ground,
                knife_dir: glam::DVec2::new(self.knife_dir[0], self.knife_dir[1]),
                paint: color_of(&self.paint.normalized().packed()),
                px: self.cam.dist as f64 * 0.0012,
                knife_fill: (self.knife_load() / self.knife_capacity().max(1e-12)).clamp(0.0, 1.0) as f32,
                knife_paint: color_of(&color::PaintMix { frac: self.knife_mix() }.normalized().packed()),
                knife_gap: self.tool.knife_height * self.pen().map(|p| 1.0 - PEN_KNIFE_CLOSE * p).unwrap_or(1.0),
                pour_radius: self.pour_radius as f64 * self.pen().map(|p| PEN_WIDTH_MIN as f64 + (1.0 - PEN_WIDTH_MIN as f64) * p).unwrap_or(1.0),
            };
            gizmo::tool_gizmo(&mut tv, &mut ti, &self.tool, &gx);
        }
        if self.tool.kind == ToolKind::Brush {
            let pc = |comp: &[f32; 16]| color_of(&color::PaintMix { frac: *comp }.normalized().packed());
            self.brush.mesh(&mut tv, &mut ti, &pc, (self.cam.dist as f64 * 0.0012).max(1.5e-4));
        }
        if self.tool.kind == ToolKind::Roller {
            let pc = |comp: &[f32; 16]| color_of(&color::PaintMix { frac: *comp }.normalized().packed());
            self.roller.mesh(&mut tv, &mut ti, self.tool.pos, &pc);
        }
        self.tubes.upload(&self.gpu, &tv, &ti, solid);
        let ms_mesh = t_mesh.elapsed().as_secs_f32() * 1e3;
        let nverts: usize = self.air.rods.iter().map(|r| r.n()).sum();

        let raw = self.ui.state.take_egui_input(&self.window);
        self.ui.ctx.begin_pass(raw);
        if self.auto.is_none() {
            self.ui_panel();
            if !self.view_fitted && self.view_rect[2] > 1.0 {
                self.view_fitted = true;
                self.set_top_view();
            }
        } else {
            self.view_rect = [0.0, 0.0, size.width as f32, size.height.max(1) as f32];
        }
        let mut out = self.ui.ctx.end_pass();
        let mut textures_delta = std::mem::take(&mut out.textures_delta);
        self.ui.state.handle_platform_output(&self.window, std::mem::take(&mut out.platform_output));
        let prims = self.ui.ctx.tessellate(std::mem::take(&mut out.shapes), out.pixels_per_point);

        let vr = self.view_rect;
        let vr = [vr[0].clamp(0.0, size.width as f32 - 1.0), vr[1].clamp(0.0, size.height as f32 - 1.0), vr[2].max(1.0), vr[3].max(1.0)];
        let vr = [vr[0], vr[1], vr[2].min(size.width as f32 - vr[0]), vr[3].min(size.height as f32 - vr[1])];
        self.cam.aspect = vr[2] / vr[3];
        self.renderer.update(&self.gpu, &self.cam, &self.film, vr[3] as u32);

        let view = surface_tex.texture.create_view(&Default::default());
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        let mut stats_slot = None;
        let mut scrap_slot = None;
        let mut res_slot = None;
        let t_enc = Instant::now();
        if !self.paused {
            self.film.poll_stats(&self.gpu, &mut self.surface.data, &mut self.surface.sub);
            self.film.upload_deposits(&self.gpu, &deposits);
            self.film.set_knife(&self.gpu, knife);
            let substeps = self.film.choose_substeps(frame_dt);
            self.film.update_params(&self.gpu, frame_dt, substeps);
            self.film.encode_frame(&mut enc, substeps);
            stats_slot = self.film.request_stats(&self.gpu, &mut enc);
            scrap_slot = self.film.request_scraps(&mut enc);
            if matches!(self.tool.kind, ToolKind::Brush | ToolKind::Roller) {
                res_slot = self.film.request_reservoir(&mut enc);
            }
            self.sim_time += frame_dt;
        }
        let ms_enc = t_enc.elapsed().as_secs_f32() * 1e3;
        self.prof = format!("air {:.1} ms {}  mesh {:.1} ms  encode {:.1} ms  frame {:.1} ms  rod verts {}  deposits {}", ms_air, air_detail, ms_mesh, ms_enc, real_dt * 1e3, nverts, deposits.len());
        self.last_cpu_ms = ms_air + ms_mesh + ms_enc;
        if self.auto.as_ref().map(|a| a.script.starts_with("stress") || a.script == "pollock").unwrap_or(false) && (self.frame_index % 300 == 0 || (real_dt > 0.05 && self.frame_index % 20 == 0)) {
            let vmax = self.air.rods.iter().flat_map(|r| r.v.iter()).map(|v| v.length()).fold(0.0f64, f64::max);
            log::warn!("vmax {:.2} m/s", vmax);
            log::warn!("f{} {:.0} fps sub {} rods {} drops {} {}", self.frame_index, self.fps, self.film.substeps_last, self.air.rods.len(), self.air.drops.len(), self.prof);
        }

        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [size.width, size.height], pixels_per_point: out.pixels_per_point };
        for (id, deltas) in &textures_delta.set {
            for delta in deltas.iter() {
                self.ui.renderer.update_texture(&self.gpu.device, &self.gpu.queue, *id, delta);
            }
        }
        let cmds = self.ui.renderer.update_buffers(&self.gpu.device, &self.gpu.queue, &mut enc, &prims, &screen);
        {
            let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            pass.set_viewport(vr[0], vr[1], vr[2], vr[3], 0.0, 1.0);
            self.renderer.draw(&mut pass);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tubes"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.tubes.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(vr[0], vr[1], vr[2], vr[3], 0.0, 1.0);
            self.tubes.draw(&mut pass);
        }
        {
            let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            self.ui.renderer.render(&mut pass, &prims, &screen);
        }
        for id in &textures_delta.free {
            self.ui.renderer.free_texture(id);
        }
        textures_delta.clear();
        self.gpu.queue.submit(cmds.into_iter().chain(std::iter::once(enc.finish())));
        if let Some(k) = stats_slot {
            self.film.map_stats(k);
        }
        if let Some(k) = scrap_slot {
            self.film.map_scraps(k);
        }
        if let Some(k) = res_slot {
            self.film.map_reservoir(k);
        }
        self.frame_index += 1;
        let mut exit = false;
        let export_dir = self.auto.as_ref().filter(|a| self.frame_index == a.frames).map(|a| a.export.clone()).unwrap_or_default();
        if !export_dir.is_empty() {
            self.export_blender(&export_dir);
        }
        if let Some(auto) = &self.auto {
            if self.frame_index == auto.frames {
                self.shot_request = Some(auto.shot.clone());
                if !auto.snap_save.is_empty() {
                    let sp = auto.snap_save.clone();
                    self.save_snapshot(&sp);
                }
                let auto = self.auto.as_ref().unwrap();
                let spp = auto.spp;
                let vres = auto.vres;
                if !auto.beauty.is_empty() {
                    let bp = auto.beauty.clone();
                    self.refine.run(&self.gpu, &self.film);
                    let t0 = Instant::now();
                    let img = self.beauty.render(&self.gpu, &self.renderer, &self.cam, &self.film, true, self.film.w * 2, self.film.h * 2, spp, Some(&self.refine));
                    log::warn!("beauty took {:.1} s", t0.elapsed().as_secs_f32());
                    let _ = img.save(&bp);
                    log::warn!("saved beauty {}", bp);
                }
                if !auto.view.is_empty() {
                    let vp = auto.view.clone();
                    self.refine.run(&self.gpu, &self.film);
                    let t0 = Instant::now();
                    let img = self.beauty.render(&self.gpu, &self.renderer, &self.cam, &self.film, false, vres[0], vres[1], spp, Some(&self.refine));
                    log::warn!("view took {:.1} s", t0.elapsed().as_secs_f32());
                    let _ = img.save(&vp);
                    log::warn!("saved view {}", vp);
                }
                exit = true;
            }
        }
        if let Some(path) = self.shot_request.take() {
            for r in &self.air.rods {
                let zmin = r.x.iter().map(|x| x.z).fold(1.0f64, f64::min);
                let rmax = (0..r.n().saturating_sub(1)).map(|e| r.edge_radius(e)).fold(0.0f64, f64::max);
                let fz = r.frozen.iter().filter(|f| **f).count();
                let lens: Vec<String> = (0..r.n().saturating_sub(1)).map(|e| format!("{:.1}/{:.2}/{:.2}", (r.x[e + 1] - r.x[e]).length() * 1e3, r.edge_radius(e) * 1e3, r.vol[e] * 1e6)).collect();
                log::warn!("rod {} n {} nozzle {} vol {:.3} mL frozen {} zmin {:.4} rmax {:.2} mm edges(L mm/r mm/V uL) {:?}", r.id, r.n(), r.nozzle.is_some(), r.total_volume() * 1e6, fz, zmin, rmax * 1e3, lens);
            }
            self.save_screenshot(&path, &surface_tex.texture);
            self.compute_stats();
        }
        self.gpu.queue.present(surface_tex);
        if exit {
            std::process::exit(0);
        }
    }
}

impl ApplicationHandler for Pending {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("paint-sim").with_inner_size(winit::dpi::LogicalSize::new(1600.0, 1000.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("window"));
        self.window = Some(window.clone());
        self.app = Some(App::new(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(app) = self.app.as_mut() else { return };
        let _resp = app.ui.state.on_window_event(&app.window, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => { app.gpu.resize(s.width, s.height); app.tubes.resize(&app.gpu); }
            WindowEvent::RedrawRequested => {
                app.frame();
                app.window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                if app.pen_down {
                    if app.pen_last.elapsed().as_secs_f32() < PEN_STALE_S {
                        return;
                    }
                    app.pen_down = false;
                    app.pen_raw = 0.0;
                    app.mouse_down[0] = false;
                }
                let prev = app.cursor;
                app.cursor = [position.x as f32, position.y as f32];
                let dx = app.cursor[0] - prev[0];
                let dy = app.cursor[1] - prev[1];
                if app.mouse_down[1] {
                    app.cam.yaw -= dx * 0.005;
                    app.cam.pitch = (app.cam.pitch + dy * 0.005).clamp(0.1, 1.55);
                }
                if app.mouse_down[2] {
                    let eye = app.cam.eye();
                    let fwd = (app.cam.target - eye).normalize();
                    let right = fwd.cross(Vec3::Z).normalize();
                    let up = right.cross(fwd);
                    let k = app.cam.dist * 0.0012;
                    app.cam.target -= right * dx * k;
                    app.cam.target += up * dy * k;
                }
            }
            WindowEvent::Touch(t) => {
                app.pen_last = Instant::now();
                app.cursor = [t.location.x as f32, t.location.y as f32];
                if let Some(f) = t.force {
                    app.pen_raw = f.normalized() as f32;
                    app.pen_seen = true;
                }
                match t.phase {
                    TouchPhase::Started => {
                        if t.force.is_none() {
                            app.pen_raw = 1.0;
                        }
                        if !app.cursor_over_panel() {
                            app.pen_down = true;
                            app.mouse_down[0] = true;
                        }
                    }
                    TouchPhase::Ended | TouchPhase::Cancelled => {
                        if app.pen_down {
                            app.mouse_down[0] = false;
                        }
                        app.pen_down = false;
                        app.pen_raw = 0.0;
                    }
                    TouchPhase::Moved => {}
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let i = match button {
                    MouseButton::Left => 0,
                    MouseButton::Right => 1,
                    MouseButton::Middle => 2,
                    _ => return,
                };
                if state == ElementState::Pressed {
                    if app.cursor_over_panel() {
                        return;
                    }
                    app.mouse_down[i] = true;
                } else {
                    app.mouse_down[i] = false;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if app.cursor_over_panel() {
                    return;
                }
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };
                if app.mouse_down[1] || app.ctrl_down || app.tool.kind == ToolKind::Pour {
                    app.cam.dist = (app.cam.dist * (1.0 - d * 0.1)).clamp(0.05, 6.0);
                } else if app.tool.kind == ToolKind::Roller {
                    app.roller.pressure = (app.roller.pressure + d as f64 * 0.05).clamp(0.0, 1.0);
                } else {
                    app.tool.height = (app.tool.height * (1.0 + d as f64 * 0.1)).clamp(0.002, 1.0);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if matches!(event.physical_key, PhysicalKey::Code(KeyCode::ControlLeft) | PhysicalKey::Code(KeyCode::ControlRight)) {
                    app.ctrl_down = event.state == ElementState::Pressed;
                }
                if event.state == ElementState::Pressed && !app.ui.ctx.egui_wants_keyboard_input() {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => app.paused = !app.paused,
                        PhysicalKey::Code(KeyCode::Tab) => app.show_inspector = !app.show_inspector,
                        PhysicalKey::Code(KeyCode::KeyT) => app.set_top_view(),
                        PhysicalKey::Code(KeyCode::KeyB) => app.beauty_render(true),
                        PhysicalKey::Code(KeyCode::Digit1) => app.tool.kind = ToolKind::Stick,
                        PhysicalKey::Code(KeyCode::Digit2) => app.tool.kind = ToolKind::Bottle,
                        PhysicalKey::Code(KeyCode::Digit3) => app.tool.kind = ToolKind::Bucket,
                        PhysicalKey::Code(KeyCode::Digit4) => app.tool.kind = ToolKind::Pour,
                        PhysicalKey::Code(KeyCode::Digit5) => app.tool.kind = ToolKind::Knife,
                        PhysicalKey::Code(KeyCode::Digit6) => app.tool.kind = ToolKind::Brush,
                        PhysicalKey::Code(KeyCode::Digit7) => app.tool.kind = ToolKind::Roller,
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            match app.tool.kind {
                                ToolKind::Brush => app.brush.dip(&app.paint.frac),
                                ToolKind::Roller => app.roller.dip(&app.paint.frac),
                                ToolKind::Knife => app.load_knife(),
                                _ => app.tool.reload(),
                            }
                        }
                        PhysicalKey::Code(KeyCode::KeyD) => app.film.dry_now(&app.gpu),
                        PhysicalKey::Code(KeyCode::Equal) => app.cam.dist = (app.cam.dist * 0.8).max(0.05),
                        PhysicalKey::Code(KeyCode::Minus) => app.cam.dist = (app.cam.dist * 1.25).min(6.0),
                        PhysicalKey::Code(KeyCode::F12) => app.shot_request = Some(format!("shot_{}.png", app.frame_index)),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn parse_auto() -> Option<AutoScript> {
    let args: Vec<String> = std::env::args().collect();
    let mut frames = None;
    let mut shot = "auto.png".to_string();
    let mut beauty = String::new();
    let mut view = String::new();
    let mut tilt = 0.0f32;
    let mut cam = String::new();
    let mut script = String::new();
    let mut debug = false;
    let mut cure = -1.0f32;
    let mut pace = false;
    let mut sigma = -1.0f32;
    let mut yield_pa = -1.0f32;
    let mut pig = -1i32;
    let mut layer = String::new();
    let mut seed: u64 = 7;
    let mut dry: u32 = 0;
    let mut export = String::new();
    let mut mu = -1.0f64;
    let mut angle = -1.0f32;
    let mut stop = 0u64;
    let mut rounds = 0u32;
    let mut snap_save = String::new();
    let mut snap_load = String::new();
    let mut camat = String::new();
    let mut vres = [3072u32, 2048u32];
    let mut spp = 64u32;
    let mut light = Vec::new();
    let mut aces = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--auto" => { frames = Some(args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(200)); i += 1; }
            "--shot" => { shot = args.get(i + 1).cloned().unwrap_or(shot); i += 1; }
            "--beauty" => { beauty = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--view" => { view = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--tilt" => { tilt = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.0); i += 1; }
            "--cam" => { cam = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--script" => { script = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--debug" => { debug = true; }
            "--pace" => { pace = true; }
            "--sigma" => { sigma = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(-1.0); i += 1; }
            "--pig" => { pig = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(-1); i += 1; }
            "--layer" => { layer = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--seed" => { seed = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(7); i += 1; }
            "--dry" => { dry = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0); i += 1; }
            "--export" => { export = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--yield" => { yield_pa = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(-1.0); i += 1; }
            "--mu" => { mu = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(-1.0); i += 1; }
            "--angle" => { angle = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(-1.0); i += 1; }
            "--cure" => { cure = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.0); i += 1; }
            "--stop" => { stop = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0); i += 1; }
            "--aces" => { aces = true; }
            "--light" => { light = args.get(i + 1).map(|s| s.split(',').filter_map(|t| t.parse().ok()).collect()).unwrap_or_default(); i += 1; }
            "--rounds" => { rounds = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0); i += 1; }
            "--snap-save" => { snap_save = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--snap-load" => { snap_load = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--camat" => { camat = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--spp" => { spp = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(64); i += 1; }
            "--vres" => {
                let v: Vec<u32> = args.get(i + 1).map(|s| s.split('x').filter_map(|t| t.parse().ok()).collect()).unwrap_or_default();
                if v.len() == 2 { vres = [v[0], v[1]]; }
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    frames.map(|frames| AutoScript { frames, shot, beauty, view, tilt, cam, script, debug, cure, pace, sigma, angle, yield_pa, mu, pig, layer, seed, dry, export, stop, rounds, snap_save, snap_load, camat, vres, spp, light, aces })
}

fn pollock_stats(seeds: u64) {
    let settings = FilmSettings::default();
    let dt = 1.0 / 60.0;
    let mut ratios = Vec::new();
    let mut rings = vec![0.0f32; 8];
    for seed in 0..seeds {
        let mut p = pollock::Painter::new(seed, settings.width_m as f64, settings.height_m as f64);
        while !p.done {
            p.step(dt);
        }
        let (edge, inner) = p.ink_balance();
        ratios.push(edge / inner.max(1e-6));
        for (k, v) in p.ink_rings().iter().enumerate() {
            rings[k] += v / seeds as f32;
        }
    }
    let mean = ratios.iter().sum::<f32>() / ratios.len().max(1) as f32;
    let min = ratios.iter().cloned().fold(f32::MAX, f32::min);
    let max = ratios.iter().cloned().fold(0.0f32, f32::max);
    println!("edge/interior ink over {} seeds: mean {:.2} min {:.2} max {:.2}", seeds, mean, min, max);
    let profile: Vec<String> = rings.iter().map(|v| format!("{:.2}", v)).collect();
    println!("ink density by ring, edge to centre: {}", profile.join(" "));
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args: Vec<String> = std::env::args().collect();
    if let Some(n) = args.windows(2).find(|p| p[0] == "--pollock-stats").and_then(|p| p[1].parse::<u64>().ok()) {
        pollock_stats(n);
        return;
    }
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut pending = Pending { window: None, app: None };
    event_loop.run_app(&mut pending).expect("run");
}
