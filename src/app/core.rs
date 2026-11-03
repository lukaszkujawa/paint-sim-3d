use crate::film::{Deposit, Film, FilmSettings};
use crate::render::{Beauty, Camera, Refine, Renderer, TubeRenderer};
use crate::tools::{Airborne, CoarseSurface, Tool, ToolKind};
use crate::{brush, color, evolve, gpu, panels, pollock, roller, session, sponge};
use crate::app::*;
use crate::script::*;
use glam::{DVec3, Vec3};
use std::sync::Arc;
use std::time::Instant;
use winit::window::Window;

pub(crate) struct Ui {
    pub(crate) ctx: egui::Context,
    pub(crate) state: egui_winit::State,
    pub(crate) renderer: egui_wgpu::Renderer,
}

pub(crate) struct App {
    pub(crate) window: Arc<Window>,
    pub(crate) gpu: gpu::Gpu,
    pub(crate) film: Film,
    pub(crate) renderer: Renderer,
    pub(crate) tubes: TubeRenderer,
    pub(crate) brush: brush::Brush,
    pub(crate) roller: roller::Roller,
    pub(crate) sponge: sponge::Sponge,
    pub(crate) tool_scraped: f64,
    pub(crate) tool_released: f64,
    pub(crate) prev_kind: ToolKind,
    pub(crate) painter: Option<pollock::Painter>,
    pub(crate) evo: evolve::EvolveState,
    pub(crate) beauty: Beauty,
    pub(crate) refine: Refine,
    pub(crate) last_beauty: String,
    pub(crate) painter_speed: f32,
    pub(crate) sim_dt: f64,
    pub(crate) painter_cure_saved: Option<f32>,
    pub(crate) painter_tilt_saved: Option<(f32, f32)>,
    pub(crate) painter_last_pour: Option<[f32; 2]>,
    pub(crate) painter_active: bool,
    pub(crate) painter_rounds_left: u32,
    pub(crate) tool: Tool,
    pub(crate) air: Airborne,
    pub(crate) surface: CoarseSurface,
    pub(crate) prev_tool_pos: Option<DVec3>,
    pub(crate) knife_dir: [f64; 2],
    pub(crate) knife_bands: [KnifeBand; KNIFE_BANDS],
    pub(crate) knife_pending: [f64; KNIFE_BANDS],
    pub(crate) knife_poses: std::collections::VecDeque<(u32, DVec3, DVec3)>,
    pub(crate) knife_prev: Option<DVec3>,
    pub(crate) knife_track: Option<DVec3>,
    pub(crate) pending_deposits: Vec<Deposit>,
    pub(crate) scrap_total: f64,
    pub(crate) view_rect: [f32; 4],
    pub(crate) block_rects: Vec<egui::Rect>,
    pub(crate) menu_open: bool,
    pub(crate) show_inspector: bool,
    pub(crate) show_help: bool,
    pub(crate) view_fitted: bool,
    pub(crate) pollock_pal: pollock::Palette,
    pub(crate) ground_method: pollock::GroundMethod,
    pub(crate) ground_strokes: [usize; 2],
    pub(crate) history: session::History,
    pub(crate) stroke_down: bool,
    pub(crate) show_gallery: bool,
    pub(crate) save_dialog: Option<String>,
    pub(crate) save_focus: bool,
    pub(crate) gallery: Vec<(session::Painting, Option<egui::TextureHandle>)>,
    pub(crate) pen_down: bool,
    pub(crate) pen_last: Instant,
    pub(crate) pen_raw: f32,
    pub(crate) pen_seen: bool,
    pub(crate) pen_pressure: bool,
    pub(crate) pen_cut: bool,
    pub(crate) pen_curve: f32,
    pub(crate) pal_picks: [[f32; 3]; 6],
    pub(crate) paint_pick: [f32; 3],
    pub(crate) paint_pick_mix: [f32; 12],
    pub(crate) paint_pick_ev: f32,
    pub(crate) paint_hex: String,
    pub(crate) eyedropper: bool,
    pub(crate) pick_hover: Option<[f32; 3]>,
    pub(crate) custom_mixes: Vec<[f32; 16]>,
    pub(crate) prof: String,
    pub(crate) auto_quality: bool,
    pub(crate) frame_budget_ms: f32,
    pub(crate) brush_sink: f64,
    pub(crate) brush_pressure: f64,
    pub(crate) brush_raise: f64,
    pub(crate) brush_z: Option<f64>,
    pub(crate) brush_hold: f64,
    pub(crate) held_deposits: Vec<Deposit>,
    pub(crate) canvas_dialog: Option<CanvasChoice>,
    pub(crate) tool_heights: Vec<(ToolKind, f64)>,
    pub(crate) height_kind: ToolKind,
    pub(crate) last_cpu_ms: f32,
    pub(crate) ctrl_down: bool,
    pub(crate) cam: Camera,
    pub(crate) ui: Ui,
    pub(crate) last: Instant,
    pub(crate) cursor: [f32; 2],
    pub(crate) mouse_down: [bool; 3],
    pub(crate) last_pour: Option<Vec3>,
    pub(crate) paint: color::PaintMix,
    pub(crate) pour_rate: f32,
    pub(crate) pour_radius: f32,
    pub(crate) rubber_size: f64,
    pub(crate) rubber_prev: Option<DVec3>,
    pub(crate) rubber_seg: Option<([f32; 2], [f32; 2], f32)>,
    pub(crate) fps: f32,
    pub(crate) sim_time: f32,
    pub(crate) paused: bool,
    pub(crate) frame_index: u64,
    pub(crate) auto: Option<AutoScript>,
    pub(crate) shot_request: Option<String>,
    pub(crate) stats: String,
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) struct CanvasChoice {
    pub res: u32,
    pub colour: Option<[f32; 3]>,
    pub weave: bool,
    pub custom: [f32; 3],
    pub first: bool,
}

impl App {
    pub(crate) fn new(window: Arc<Window>) -> Self {
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
            tool_scraped: 0.0,
            roller: roller::Roller::new(),
            sponge: sponge::Sponge::new(),
            tool_released: 0.0,
            prev_kind: ToolKind::Stick,
            painter: None,
            evo: evolve::EvolveState::new(),
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
            ground_method: pollock::GroundMethod::Auto,
            ground_strokes: [40, 12],
            history: session::History::new(),
            stroke_down: false,
            show_gallery: false,
            save_dialog: None,
            save_focus: false,
            gallery: Vec::new(),
            pen_down: false,
            pen_last: Instant::now(),
            pen_raw: 0.0,
            pen_seen: false,
            pen_pressure: true,
            pen_cut: false,
            pen_curve: 1.0,
            pal_picks: [[0.5; 3]; 6],
            paint_pick: [0.5; 3],
            paint_pick_mix: [0.0; 12],
            paint_pick_ev: 0.0,
            paint_hex: String::new(),
            eyedropper: false,
            pick_hover: None,
            custom_mixes: Vec::new(),
            prof: String::new(),
            auto_quality: std::env::var("PAINT_NO_AUTOQ").is_err(),
            frame_budget_ms: 14.0,
            brush_sink: 0.88,
            brush_pressure: 0.5,
            brush_raise: 0.0,
            brush_z: None,
            brush_hold: 0.0,
            held_deposits: Vec::new(),
            canvas_dialog: None,
            tool_heights: Vec::new(),
            height_kind: ToolKind::Stick,
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
            rubber_size: 0.03,
            rubber_prev: None,
            rubber_seg: None,
            fps: 0.0,
            sim_time: 0.0,
            paused: false,
            frame_index: 0,
            auto,
            shot_request: None,
            stats: String::new(),
        };
        app.sync_pal_picks();
        if app.auto.is_none() {
            app.tool.kind = ToolKind::Brush;
            app.prev_kind = ToolKind::Brush;
            app.height_kind = ToolKind::Brush;
        }
        if let Some(path) = std::env::args().skip(1).filter(|_| app.auto.is_none()).find(|a| !a.starts_with("--") && std::path::Path::new(a).is_file() && ["jpg", "jpeg", "png", "webp", "bmp"].iter().any(|e| a.to_lowercase().ends_with(e))) {
            app.load_photo(std::path::Path::new(&path));
        }
        let snap = app.auto.as_ref().map(|a| a.snap_load.clone()).unwrap_or_default();
        if !snap.is_empty() {
            app.load_snapshot(&snap);
        }
        if app.auto.as_ref().map(|a| a.aces).unwrap_or(false) {
            app.renderer.agx = false;
        }
        if app.auto.as_ref().map(|a| a.agx).unwrap_or(false) {
            app.renderer.agx = true;
        }
        if app.auto.is_none() {
            app.canvas_dialog = Some(CanvasChoice { res: app.film.w, colour: app.film.settings.canvas_rgb, weave: app.film.settings.raw_canvas, custom: [0.90, 0.85, 0.76], first: true });
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
}
