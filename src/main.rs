mod brush;
mod color;
mod pollock;
mod film;
mod gpu;
mod render;
mod rod;
mod tools;

use film::{Deposit, Film, FilmSettings, KnifeParams, Method, Scrap};
use glam::Vec3;
use render::{Beauty, Camera, Refine, Renderer, TubeRenderer};
use tools::{Airborne, CoarseSurface, Tool, ToolKind};
use glam::DVec3;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

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
    tool: Tool,
    air: Airborne,
    surface: CoarseSurface,
    prev_tool_pos: Option<DVec3>,
    knife_dir: [f64; 2],
    pending_deposits: Vec<Deposit>,
    scrap_total: f64,
    ui_rect: Option<egui::Rect>,
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
        }
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, None);
        let eg_renderer = egui_wgpu::Renderer::new(&gpu.device, gpu.config.format, egui_wgpu::RendererOptions::default());
        let mut paint = color::PaintMix::pure(1);
        paint.frac[12] = 0.3;
        Self {
            window,
            gpu,
            film,
            renderer,
            tubes,
            tool: Tool::default(),
            air: Airborne::new(),
            surface,
            brush: brush::Brush::new(160, 0.035, 0.008, 0.00015),
            brush_scraped: 0.0,
            brush_released: 0.0,
            prev_kind: ToolKind::Stick,
            painter: None,
            beauty,
            refine,
            last_beauty: String::new(),
            painter_speed: 2.0,
            sim_dt: 1.0 / 60.0,
            painter_cure_saved: None,
            prev_tool_pos: None,
            knife_dir: [1.0, 0.0],
            pending_deposits: Vec::new(),
            scrap_total: 0.0,
            ui_rect: None,
            prof: String::new(),
            auto_quality: true,
            frame_budget_ms: 14.0,
            brush_sink: 0.7,
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
        }
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

    fn apply_painter(&mut self) -> bool {
        let Some(p) = self.painter.as_mut() else { return false };
        if self.paused {
            return true;
        }
        match p.step(self.sim_dt) {
            Some(c) => {
                if self.tool.kind != c.kind {
                    self.tool.reload();
                }
                self.tool.kind = c.kind;
                self.prev_kind = c.kind;
                self.tool.pos = c.pos;
                self.tool.vel = c.vel;
                self.tool.flowing = c.flowing;
                self.tool.height = c.height;
                self.tool.mu = c.mu;
                self.tool.sigma = c.sigma;
                self.tool.stick_diameter = c.stick_diameter;
                self.tool.bottle_flow = c.bottle_flow;
                self.tool.bottle_radius = c.bottle_radius;
                self.tool.bucket_flow = c.bucket_flow;
                self.tool.bucket_radius = c.bucket_radius;
                self.paint.frac = c.mix;
                if c.reload {
                    self.tool.reload();
                }
                if c.cut {
                    self.air.detach();
                }
                if c.cure_rate > 0.0 {
                    if self.painter_cure_saved.is_none() {
                        self.painter_cure_saved = Some(self.film.settings.cure_rate);
                    }
                    self.film.settings.cure_rate = c.cure_rate as f32;
                } else if let Some(v) = self.painter_cure_saved.take() {
                    self.film.settings.cure_rate = v;
                }
                true
            }
            None => {
                self.tool.flowing = false;
                if let Some(v) = self.painter_cure_saved.take() {
                    self.film.settings.cure_rate = v;
                }
                true
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
                let mut p = pollock::Painter::new(7, self.film.settings.width_m as f64, self.film.settings.height_m as f64);
                if !a.layer.is_empty() {
                    p.skip_to(&a.layer);
                }
                self.painter = Some(p);
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
        let img = self.beauty.render(&self.gpu, &self.renderer, &self.cam, &self.film, ortho, w, h, 36, Some(&self.refine));
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
        let _ = img.save(path);
        log::warn!("saved {}", path);
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
        self.stats = format!(
            "wet {:.4} mL  cured {:.4} mL  deposited {:.4} mL  brush held {:.4} mL  err {:.2}%  hmax {:.3} mm  wet cells {}  active tiles {}  nan {}  fps {:.0}  substeps {}",
            vol * 1e6, volm * 1e6, self.film.total_deposited * 1e6, self.brush.total_load() * 1e6,
            if self.film.total_deposited > 0.0 { (vol + volm + self.brush_scraped - self.brush_released - self.film.total_deposited) / self.film.total_deposited * 100.0 } else { 0.0 },
            hmax * 1e3, wet, self.film.active_tiles, nan, self.fps, self.film.substeps_last
        );
        log::warn!("{}", self.stats);
    }

    fn cursor_over_panel(&self) -> bool {
        let Some(rect) = self.ui_rect else { return false };
        let ppp = self.ui.ctx.pixels_per_point().max(0.01);
        let p = egui::pos2(self.cursor[0] / ppp, self.cursor[1] / ppp);
        rect.expand(4.0).contains(p)
    }

    fn ndc(&self) -> [f32; 2] {
        let s = self.window.inner_size();
        [self.cursor[0] / s.width as f32 * 2.0 - 1.0, 1.0 - self.cursor[1] / s.height as f32 * 2.0]
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
        let mut scraps: Vec<Scrap> = Vec::new();
        self.film.poll_scraps(&self.gpu, &mut scraps);
        if self.tool.kind == ToolKind::Pour {
            if want_pour {
                if let Some(p) = self.cam.plane_hit(self.ndc(), 0.0) {
                    let p0 = self.last_pour.unwrap_or(p);
                    let vol = self.pour_rate * frame_dt;
                    let pig = self.paint.normalized().packed();
                    deposits.push(Deposit::stroke([p0.x, p0.y], [p.x, p.y], self.pour_radius, vol, pig, self.film.dx).laid_by(Method::Pour));
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
                self.paint.frac[13] = 35.0;
                if a.yield_pa >= 0.0 { self.paint.frac[13] = a.yield_pa; self.paint.frac[12] = 0.0; }
                if a.pig >= 0 { let y = self.paint.frac[13]; let th = self.paint.frac[12]; self.paint = color::PaintMix::pure(a.pig as usize); self.paint.frac[13] = y; self.paint.frac[12] = th; }
                if a.mu > 0.0 { self.tool.mu = a.mu; }
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
                self.paint.frac[13] = 35.0;
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
        deposits.append(&mut self.pending_deposits);
        if !scripted {
            let pick_z = if self.tool.kind == ToolKind::Brush { 0.0 } else { self.tool.height as f32 };
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
        if self.tool.kind == ToolKind::Knife && self.tool.flowing {
            let dir = DVec3::new(self.tool.vel.x, self.tool.vel.y, 0.0);
            let speed = dir.length();
            if speed > 1e-4 {
                let d = dir / speed;
                let blade_dir = DVec3::new(-d.y, d.x, 0.0);
                let hw = self.tool.knife_width * 0.5;
                let p = DVec3::new(self.tool.pos.x, self.tool.pos.y, 0.0);
                let a = p - blade_dir * hw;
                let b = p + blade_dir * hw;
                knife = Some(KnifeParams {
                    a: [a.x as f32, a.y as f32],
                    b: [b.x as f32, b.y as f32],
                    dir: [d.x as f32, d.y as f32],
                    sweep: (speed * frame_dt as f64) as f32,
                    height: self.tool.knife_height as f32,
                    half_width: hw as f32,
                    active: 1,
                    pad0: 0,
                    pad1: 0,
                });
                self.knife_dir = [d.x, d.y];
            }
        }
        self.scrap_total += scraps.iter().map(|s| s.vol as f64).sum::<f64>();
        for s in &scraps {
            let i = (s.cell % self.film.w) as f64 + 0.5;
            let j = (s.cell / self.film.w) as f64 + 0.5;
            let cell = DVec3::new(i * self.film.dx as f64, j * self.film.dx as f64, 0.0);
            let d = DVec3::new(self.knife_dir[0], self.knife_dir[1], 0.0);
            let blade = DVec3::new(-d.y, d.x, 0.0);
            let center = DVec3::new(self.tool.pos.x, self.tool.pos.y, 0.0);
            let along = (cell - center).dot(blade);
            let ahead = self.film.dx as f64 * 1.2;
            let q = center + blade * along + d * ahead;
            let p0 = [q.x as f32, q.y as f32];
            deposits.push(Deposit::capsule(p0, p0, 0.8 * self.film.dx, s.vol, s.pig, self.film.dx).laid_by(Method::Knife).recycled());
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
                let handle = DVec3::new(self.tool.pos.x, self.tool.pos.y, self.tool.height + lift);
                let surf = &self.surface;
                let brush = &mut self.brush;
                let sink = self.brush_sink;
                let (c, rel) = brush.update(handle, frame_dt as f64, &|x, y| surf.brush_floor(x, y, sink), self.film.dx as f64);
                contacts = c;
                for r in rel {
                    let pig = color::PaintMix { frac: r.comp }.normalized().packed();
                    deposits.push(Deposit::stroke(r.p0, r.p1, r.r, r.vol as f32, pig, self.film.dx).laid_by(Method::Brush).recycled());
                    self.brush_released += r.vol;
                }
            }
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
        let mut marker = self.air.drops.clone();
        if self.tool.kind != ToolKind::Pour {
            let r = match self.tool.kind { ToolKind::Knife => 0.002, ToolKind::Brush => self.brush.ferrule_r, _ => 0.004 };
            let mpos = if self.tool.kind == ToolKind::Brush { self.brush.handle } else { self.tool.pos };
            marker.push(rod::Drop { x: mpos, v: DVec3::ZERO, vol: 4.0 / 3.0 * std::f64::consts::PI * r * r * r, pig: [0; 8], fluid: self.tool.fluid() });
        }
        let (mut tv, mut ti) = tools::tube_mesh(&self.air.rods, &marker, &color_of);
        if self.tool.kind == ToolKind::Brush {
            let pc = |comp: &[f32; 16]| color_of(&color::PaintMix { frac: *comp }.normalized().packed());
            self.brush.mesh(&mut tv, &mut ti, &pc, (self.cam.dist as f64 * 0.0012).max(1.5e-4));
        }
        self.tubes.upload(&self.gpu, &tv, &ti);
        let ms_mesh = t_mesh.elapsed().as_secs_f32() * 1e3;
        let nverts: usize = self.air.rods.iter().map(|r| r.n()).sum();

        let raw = self.ui.state.take_egui_input(&self.window);
        self.ui.ctx.begin_pass(raw);
        self.ui_panel();
        let mut out = self.ui.ctx.end_pass();
        let mut textures_delta = std::mem::take(&mut out.textures_delta);
        self.ui.state.handle_platform_output(&self.window, std::mem::take(&mut out.platform_output));
        let prims = self.ui.ctx.tessellate(std::mem::take(&mut out.shapes), out.pixels_per_point);

        self.cam.aspect = size.width as f32 / size.height.max(1) as f32;
        self.renderer.update(&self.gpu, &self.cam, &self.film);

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
            if self.tool.kind == ToolKind::Brush {
                res_slot = self.film.request_reservoir(&mut enc);
            }
            self.sim_time += frame_dt;
        }
        let ms_enc = t_enc.elapsed().as_secs_f32() * 1e3;
        self.prof = format!("air {:.1} ms  mesh {:.1} ms  encode {:.1} ms  frame {:.1} ms  rod verts {}  deposits {}", ms_air, ms_mesh, ms_enc, real_dt * 1e3, nverts, deposits.len());
        self.last_cpu_ms = ms_air + ms_mesh + ms_enc;
        if self.auto.as_ref().map(|a| a.script.starts_with("stress")).unwrap_or(false) && (self.frame_index % 60 == 0 || real_dt > 0.05) {
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
            self.renderer.draw(&mut pass);
            self.ui.renderer.render(&mut pass, &prims, &screen);
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
            self.tubes.draw(&mut pass);
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
        if let Some(auto) = &self.auto {
            if self.frame_index == auto.frames {
                self.shot_request = Some(auto.shot.clone());
                if !auto.beauty.is_empty() {
                    let bp = auto.beauty.clone();
                    self.refine.run(&self.gpu, &self.film);
                    let img = self.beauty.render(&self.gpu, &self.renderer, &self.cam, &self.film, true, self.film.w * 2, self.film.h * 2, 36, Some(&self.refine));
                    let _ = img.save(&bp);
                    log::warn!("saved beauty {}", bp);
                }
                if !auto.view.is_empty() {
                    let vp = auto.view.clone();
                    self.refine.run(&self.gpu, &self.film);
                    let img = self.beauty.render(&self.gpu, &self.renderer, &self.cam, &self.film, false, 3072, 2048, 36, Some(&self.refine));
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

    fn ui_panel(&mut self) {
        let ctx = self.ui.ctx.clone();
        let resp = egui::Window::new("Paint").default_width(300.0).show(&ctx, |ui| {
            ui.label(format!("{:.0} fps   substeps {}   t={:.1}s   hmax {:.2} mm   mu_min {:.2}", self.fps, self.film.substeps_last, self.sim_time, self.film.h_max * 1e3, self.film.mu_min));
            ui.label(format!("deposited {:.3} mL", self.film.total_deposited * 1e6));
            if !self.stats.is_empty() { ui.label(&self.stats); }
            if !self.last_beauty.is_empty() { ui.label(format!("beauty: {}", self.last_beauty)); }
            ui.label(&self.prof);
            if ui.button("stats").clicked() { self.compute_stats(); }
            ui.horizontal(|ui| {
                if ui.button("top view").clicked() {
                    self.cam.yaw = -std::f32::consts::FRAC_PI_2;
                    self.cam.pitch = 1.52;
                    self.cam.dist = 1.05;
                    self.cam.target = Vec3::new(self.film.settings.width_m * 0.5, self.film.settings.height_m * 0.5, 0.0);
                }
                if ui.button("angled view").clicked() {
                    self.cam.yaw = -1.2;
                    self.cam.pitch = 0.9;
                    self.cam.dist = 1.3;
                    self.cam.target = Vec3::new(self.film.settings.width_m * 0.5, self.film.settings.height_m * 0.5, 0.0);
                }
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.paused, "pause");
                ui.checkbox(&mut self.film.settings.raw_canvas, "raw canvas (on clear)");
                if ui.button("clear canvas").clicked() {
                    self.film.clear(&self.gpu);
                    self.air = Airborne::new();
                    self.brush_scraped = 0.0;
                    self.brush_released = 0.0;
                    self.surface = CoarseSurface::new(self.film.pool_dims[0], self.film.pool_dims[1], 4.0 * self.film.dx as f64);
                    self.sim_time = 0.0;
                }
                if ui.button("beauty render (top)").clicked() {
                    self.beauty_render(true);
                }
                if ui.button("beauty render (view)").clicked() {
                    self.beauty_render(false);
                }
                if ui.button("screenshot (F12)").clicked() {
                    self.shot_request = Some(format!("shot_{}.png", self.frame_index));
                }
            });
            ui.label(format!("airborne {:.3} mL  landed {:.3} mL  emitted {:.3} mL  rods {}  drops {}", self.air.airborne_volume() * 1e6, self.air.landed_volume * 1e6, self.air.emitted_volume * 1e6, self.air.rods.len(), self.air.drops.len()));
            if self.tool.kind == ToolKind::Stick && self.tool.stick_load <= 1e-9 && !self.tool.infinite {
                ui.colored_label(egui::Color32::YELLOW, "stick is empty: press R or click dip stick");
            }
            ui.label("Hold the left mouse button over the canvas to pour. The sphere is the tool tip; wheel changes its height.");
            ui.separator();
            ui.horizontal(|ui| {
                let running = self.painter.as_ref().map(|p| !p.done).unwrap_or(false);
                if !running {
                    if ui.button("Pollock: paint (experimental)").clicked() {
                        let seed = rand::random::<u64>() % 100000;
                        self.painter = Some(pollock::Painter::new(seed, self.film.settings.width_m as f64, self.film.settings.height_m as f64));
                    }
                } else if ui.button("Pollock: stop").clicked() {
                    self.painter = None;
                    self.tool.flowing = false;
                }
                if let Some(p) = &self.painter {
                    ui.label(format!("seed {}  {}", p.seed, p.status()));
                }
            });
            ui.add(egui::Slider::new(&mut self.painter_speed, 1.0..=4.0).text("painter time scale (x real time)"));
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tool.kind, ToolKind::Stick, "1 stick");
                ui.selectable_value(&mut self.tool.kind, ToolKind::Bottle, "2 bottle");
                ui.selectable_value(&mut self.tool.kind, ToolKind::Bucket, "3 bucket");
                ui.selectable_value(&mut self.tool.kind, ToolKind::Pour, "4 pour");
                ui.selectable_value(&mut self.tool.kind, ToolKind::Knife, "5 knife");
                ui.selectable_value(&mut self.tool.kind, ToolKind::Brush, "6 brush");
            });
            ui.add(egui::Slider::new(&mut self.tool.height, 0.002..=1.0).logarithmic(true).text("height m (wheel)"));
            ui.add(egui::Slider::new(&mut self.tool.mu, 0.05..=100.0).logarithmic(true).text("paint viscosity Pa s"));
            ui.add(egui::Slider::new(&mut self.tool.sigma, 0.02..=0.07).text("paint surface tension"));
            match self.tool.kind {
                ToolKind::Stick => {
                    ui.label(format!("stick load {:.2} mL  flow {:.3} mL/s  (R to dip)", self.tool.stick_load * 1e6, self.tool.stick_flow() * 1e6));
                    ui.add(egui::Slider::new(&mut self.tool.stick_diameter, 0.003..=0.03).text("stick diameter m"));
                    ui.horizontal(|ui| {
                        if ui.button("dip stick").clicked() { self.tool.reload(); }
                        if ui.checkbox(&mut self.tool.infinite, "infinite paint").changed() && self.tool.infinite { self.tool.reload(); }
                    });
                }
                ToolKind::Bottle => {
                    ui.add(egui::Slider::new(&mut self.tool.bottle_flow, 1e-7..=2e-5).logarithmic(true).text("bottle flow m3/s"));
                    ui.add(egui::Slider::new(&mut self.tool.bottle_radius, 0.0005..=0.006).logarithmic(true).text("nozzle radius m"));
                }
                ToolKind::Bucket => {
                    ui.add(egui::Slider::new(&mut self.tool.bucket_flow, 5e-6..=3e-4).logarithmic(true).text("bucket flow m3/s"));
                    ui.add(egui::Slider::new(&mut self.tool.bucket_radius, 0.003..=0.03).logarithmic(true).text("stream radius m"));
                }
                ToolKind::Pour => {}
                ToolKind::Knife => {
                    ui.add(egui::Slider::new(&mut self.tool.knife_width, 0.005..=0.15).text("blade width m"));
                    ui.add(egui::Slider::new(&mut self.tool.knife_height, 0.0..=0.003).text("blade gap m"));
                }
                ToolKind::Brush => {
                    let press = ((self.brush.len - self.tool.height) * 1e3).max(0.0);
                    ui.label(format!("brush load {:.3} / {:.3} mL   bristles touching {}   press {:.1} mm (wheel)", self.brush.total_load() * 1e6, self.brush.total_capacity() * 1e6, self.brush.touching, press));
                    ui.label("Hold the left button to press the brush onto the canvas; release to lift it. R dips it in the current mix.");
                    ui.horizontal(|ui| {
                        if ui.button("dip brush").clicked() { self.brush.dip(&self.paint.frac); }
                        if ui.button("clean brush").clicked() { self.brush.clean(); }
                    });
                    ui.add(egui::Slider::new(&mut self.brush.stiffness, 0.05..=0.9).text("bristle stiffness"));
                    ui.add(egui::Slider::new(&mut self.brush.friction, 0.0..=0.95).text("bristle friction"));
                    ui.add(egui::Slider::new(&mut self.brush.release, 0.1..=10.0).logarithmic(true).text("paint release 1/s"));
                    ui.add(egui::Slider::new(&mut self.brush_sink, 0.0..=1.0).text("sink into wet paint"));
                    let mut n = self.brush.bristles.len();
                    if ui.add(egui::Slider::new(&mut n, 16..=400).text("bristle count")).changed() {
                        let b = brush::Brush::new(n, self.brush.len, self.brush.ferrule_r, self.brush.radius);
                        let (st, fr, rl) = (self.brush.stiffness, self.brush.friction, self.brush.release);
                        self.brush = b;
                        self.brush.stiffness = st;
                        self.brush.friction = fr;
                        self.brush.release = rl;
                    }
                    let mut fr = self.brush.ferrule_r * 1e3;
                    if ui.add(egui::Slider::new(&mut fr, 1.0..=25.0).text("ferrule radius mm")).changed() {
                        let n = self.brush.bristles.len();
                        let (st, frc, rl) = (self.brush.stiffness, self.brush.friction, self.brush.release);
                        self.brush = brush::Brush::new(n, self.brush.len, fr * 1e-3, self.brush.radius);
                        self.brush.stiffness = st;
                        self.brush.friction = frc;
                        self.brush.release = rl;
                    }
                }
            }
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("enamel (Pollock)").clicked() {
                    self.paint.frac[12] = 0.3;
                    self.paint.frac[13] = 0.0;
                    self.tool.mu = 4.0;
                }
                if ui.button("thin enamel").clicked() {
                    self.paint.frac[12] = 0.7;
                    self.paint.frac[13] = 0.0;
                    self.tool.mu = 2.0;
                }
                if ui.button("tube oil").clicked() {
                    self.paint.frac[12] = 0.0;
                    self.paint.frac[13] = 250.0;
                    self.tool.mu = 40.0;
                }
            });
            egui::CollapsingHeader::new("Paint mix").default_open(true).show(ui, |ui| {
                let names: Vec<String> = self.film.pigments.pigments.iter().map(|p| p.name.clone()).collect();
                for (i, name) in names.iter().enumerate() {
                    let rgb = color::masstone_rgb(&self.film.pigments, &self.film.cmf_table, i);
                    ui.horizontal(|ui| {
                        let (rect, resp) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::click());
                        ui.painter().rect_filled(rect, 2.0, egui::Color32::from_rgb((rgb[0] * 255.0) as u8, (rgb[1] * 255.0) as u8, (rgb[2] * 255.0) as u8));
                        if resp.clicked() {
                            for k in 0..12 { self.paint.frac[k] = 0.0; }
                            self.paint.frac[i] = 1.0;
                        }
                        ui.add(egui::Slider::new(&mut self.paint.frac[i], 0.0..=1.0).text(name));
                    });
                }
                ui.add(egui::Slider::new(&mut self.paint.frac[12], 0.0..=0.9).text("thinner"));
                ui.add(egui::Slider::new(&mut self.paint.frac[13], 0.0..=600.0).text("yield stress Pa (impasto body)"));
                ui.add(egui::Slider::new(&mut self.film.settings.contact_angle_deg, 0.0..=90.0).text("wetting angle deg"));
                ui.add(egui::Slider::new(&mut self.pour_rate, 1e-7..=5e-5).logarithmic(true).text("pour rate m3/s"));
                ui.add(egui::Slider::new(&mut self.pour_radius, 0.0005..=0.03).logarithmic(true).text("pour radius m"));
            });
            egui::CollapsingHeader::new("Canvas").default_open(true).show(ui, |ui| {
                let s = &mut self.film.settings;
                ui.add(egui::Slider::new(&mut s.tilt_deg, 0.0..=90.0).text("tilt deg"));
                ui.add(egui::Slider::new(&mut s.tilt_dir_deg, -180.0..=180.0).text("tilt dir"));
                ui.add(egui::Slider::new(&mut s.cure_rate, 0.0..=1.0).logarithmic(true).text("cure rate 1/s"));
                ui.add(egui::Slider::new(&mut self.renderer.light_intensity, 0.0..=6.0).text("light"));
            });
            egui::CollapsingHeader::new("Physics / quality").show(ui, |ui| {
                let s = &mut self.film.settings;
                ui.add(egui::Slider::new(&mut s.m_cap, 1e-11..=1e-7).logarithmic(true).text("capillary mobility cap"));
                ui.add(egui::Slider::new(&mut s.m_cap_hydro, 1e-10..=1e-6).logarithmic(true).text("hydrostatic mobility cap"));
                ui.add(egui::Slider::new(&mut s.h_min, 0.0..=5e-5).text("min film m"));
                ui.add(egui::Slider::new(&mut s.min_substeps, 1..=32).text("min substeps"));
                ui.add(egui::Slider::new(&mut s.max_substeps, 1..=128).text("max substeps"));
                ui.checkbox(&mut self.auto_quality, "auto quality (adapt substeps to frame budget)");
                ui.add(egui::Slider::new(&mut self.frame_budget_ms, 5.0..=40.0).text("frame budget ms"));
                ui.add(egui::Slider::new(&mut s.debug_view, 0..=1).text("debug view"));
            });
            ui.label("LMB: apply tool   RMB drag: orbit   MMB drag: pan   wheel: tool height   Ctrl+wheel or +/-: zoom   R: dip stick   1-5: tools");
        });
        self.ui_rect = resp.map(|r| r.response.rect);
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
                } else {
                    app.tool.height = (app.tool.height * (1.0 + d as f64 * 0.1)).clamp(0.002, 1.0);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if matches!(event.physical_key, PhysicalKey::Code(KeyCode::ControlLeft) | PhysicalKey::Code(KeyCode::ControlRight)) {
                    app.ctrl_down = event.state == ElementState::Pressed;
                }
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => app.paused = !app.paused,
                        PhysicalKey::Code(KeyCode::Digit1) => app.tool.kind = ToolKind::Stick,
                        PhysicalKey::Code(KeyCode::Digit2) => app.tool.kind = ToolKind::Bottle,
                        PhysicalKey::Code(KeyCode::Digit3) => app.tool.kind = ToolKind::Bucket,
                        PhysicalKey::Code(KeyCode::Digit4) => app.tool.kind = ToolKind::Pour,
                        PhysicalKey::Code(KeyCode::Digit5) => app.tool.kind = ToolKind::Knife,
                        PhysicalKey::Code(KeyCode::Digit6) => app.tool.kind = ToolKind::Brush,
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            if app.tool.kind == ToolKind::Brush { app.brush.dip(&app.paint.frac); } else { app.tool.reload(); }
                        }
                        PhysicalKey::Code(KeyCode::Equal) => app.cam.dist = (app.cam.dist * 0.8).max(0.05),
                        PhysicalKey::Code(KeyCode::Minus) => app.cam.dist = (app.cam.dist * 1.25).min(6.0),
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
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
    let mut mu = -1.0f64;
    let mut angle = -1.0f32;
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
            "--yield" => { yield_pa = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(-1.0); i += 1; }
            "--mu" => { mu = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(-1.0); i += 1; }
            "--angle" => { angle = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(-1.0); i += 1; }
            "--cure" => { cure = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.0); i += 1; }
            _ => {}
        }
        i += 1;
    }
    frames.map(|frames| AutoScript { frames, shot, beauty, view, tilt, cam, script, debug, cure, pace, sigma, angle, yield_pa, mu, pig, layer })
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut pending = Pending { window: None, app: None };
    event_loop.run_app(&mut pending).expect("run");
}
