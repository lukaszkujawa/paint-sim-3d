mod widgets;
mod tools;
mod paint;
mod photo;
mod pollock_panel;
mod files;
use widgets::*;
use crate::{brush, color, evolve, pollock, session, App};
use crate::tools::ToolKind;
use egui::{Align, Color32, CornerRadius, FontId, Layout, RichText, Stroke, TextStyle, Vec2};
use glam::Vec3;

const TOOLBAR_W: f32 = 74.0;
const INSPECTOR_W: f32 = 320.0;
const ACCENT: Color32 = Color32::from_rgb(78, 132, 204);
const ACCENT_DIM: Color32 = Color32::from_rgb(46, 78, 120);
const BAR_BG: Color32 = Color32::from_rgb(22, 23, 25);
const STRIP_BG: Color32 = Color32::from_rgb(26, 27, 29);
const PANEL_BG: Color32 = Color32::from_rgb(30, 31, 34);
const WIDGET_BG: Color32 = Color32::from_rgb(42, 44, 48);
const WIDGET_HOVER: Color32 = Color32::from_rgb(52, 55, 60);
const WIDGET_ACTIVE: Color32 = Color32::from_rgb(60, 64, 70);
const BORDER: Color32 = Color32::from_rgb(45, 47, 51);
const TEXT: Color32 = Color32::from_rgb(208, 211, 216);
const OK: Color32 = Color32::from_rgb(104, 178, 128);
const WARN: Color32 = Color32::from_rgb(214, 172, 92);
const INFO: Color32 = Color32::from_rgb(150, 178, 214);
const TOOLS: [(ToolKind, &str, &str, &str); 9] = [
    (ToolKind::Brush, "Brush", "1", "Press a bristle brush onto the canvas"),
    (ToolKind::Roller, "Roller", "2", "Roll an even coat; picks up wet paint as it goes"),
    (ToolKind::Knife, "Knife", "3", "Drag a palette knife through the paint"),
    (ToolKind::Stick, "Stick", "4", "Drip paint from a loaded stick"),
    (ToolKind::Bottle, "Bottle", "5", "Squeeze a thin steady thread"),
    (ToolKind::Bucket, "Bucket", "6", "Pour a heavy stream"),
    (ToolKind::Pour, "Pour", "7", "Lay paint straight onto the canvas"),
    (ToolKind::Sponge, "Sponge", "8", "Dab or drag porous texture; a clean sponge lifts wet paint"),
    (ToolKind::Rubber, "Rubber", "9", "Scrape away paint and texture, wet or dry, down to bare canvas"),
];

pub fn apply_style(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    ctx.all_styles_mut(|s| {
        s.text_styles = [
            (TextStyle::Small, FontId::proportional(11.0)),
            (TextStyle::Body, FontId::proportional(13.0)),
            (TextStyle::Button, FontId::proportional(13.0)),
            (TextStyle::Heading, FontId::proportional(15.0)),
            (TextStyle::Monospace, FontId::monospace(12.0)),
        ]
        .into();
        s.spacing.item_spacing = Vec2::new(6.0, 5.0);
        s.spacing.button_padding = Vec2::new(8.0, 3.0);
        s.spacing.interact_size.y = 20.0;
        s.spacing.window_margin = egui::Margin::same(10);
        s.spacing.menu_margin = egui::Margin::same(6);
        let v = &mut s.visuals;
        v.panel_fill = PANEL_BG;
        v.window_fill = Color32::from_rgb(34, 35, 38);
        v.extreme_bg_color = BAR_BG;
        v.faint_bg_color = Color32::from_rgb(36, 37, 40);
        v.window_stroke = Stroke::new(1.0, BORDER);
        v.window_shadow = egui::epaint::Shadow { offset: [0, 4], blur: 16, spread: 0, color: Color32::from_black_alpha(110) };
        v.popup_shadow = egui::epaint::Shadow { offset: [0, 3], blur: 12, spread: 0, color: Color32::from_black_alpha(100) };
        v.override_text_color = None;
        v.hyperlink_color = ACCENT;
        v.window_corner_radius = CornerRadius::same(6);
        v.menu_corner_radius = CornerRadius::same(6);
        v.selection.bg_fill = ACCENT_DIM;
        v.selection.stroke = Stroke::new(1.0, Color32::from_rgb(196, 214, 238));
        v.slider_trailing_fill = true;
        for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open, &mut v.widgets.noninteractive] {
            w.corner_radius = CornerRadius::same(4);
        }
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
        v.widgets.noninteractive.bg_fill = PANEL_BG;
        v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
        v.widgets.inactive.weak_bg_fill = WIDGET_BG;
        v.widgets.inactive.bg_fill = WIDGET_BG;
        v.widgets.inactive.bg_stroke = Stroke::NONE;
        v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
        v.widgets.hovered.weak_bg_fill = WIDGET_HOVER;
        v.widgets.hovered.bg_fill = WIDGET_HOVER;
        v.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(70, 74, 80));
        v.widgets.hovered.fg_stroke = Stroke::new(1.5, Color32::from_rgb(236, 238, 241));
        v.widgets.active.weak_bg_fill = WIDGET_ACTIVE;
        v.widgets.active.bg_fill = WIDGET_ACTIVE;
        v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
        v.widgets.active.fg_stroke = Stroke::new(1.5, Color32::WHITE);
        v.widgets.open.weak_bg_fill = WIDGET_HOVER;
        v.widgets.open.bg_fill = WIDGET_HOVER;
        v.widgets.open.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    });
}

impl App {
    pub(crate) fn tool_name(&self) -> &'static str {
        TOOLS.iter().find(|t| t.0 == self.tool.kind).map(|t| t.1).unwrap_or("")
    }

    pub(crate) fn set_top_view(&mut self) {
        let (w, h) = (self.film.settings.width_m, self.film.settings.height_m);
        let aspect = (self.view_rect[2] / self.view_rect[3].max(1.0)).max(0.1);
        let span = h.max(w / aspect) * 1.08;
        self.cam.yaw = -std::f32::consts::FRAC_PI_2;
        self.cam.pitch = 1.52;
        self.cam.dist = span / (2.0 * (0.5 * self.cam.fov).tan());
        self.cam.target = Vec3::new(w * 0.5, h * 0.5, 0.0);
    }

    pub(crate) fn set_angled_view(&mut self) {
        self.cam.yaw = -1.2;
        self.cam.pitch = 0.9;
        self.cam.dist = 1.3;
        self.cam.target = Vec3::new(self.film.settings.width_m * 0.5, self.film.settings.height_m * 0.5, 0.0);
    }

    pub(crate) fn clear_canvas(&mut self) {
        self.film.clear(&self.gpu);
        self.air = crate::tools::Airborne::new();
        self.tool_scraped = 0.0;
        self.tool_released = 0.0;
        self.surface = crate::tools::CoarseSurface::new(self.film.pool_dims[0], self.film.pool_dims[1], 4.0 * self.film.dx as f64);
        self.sim_time = 0.0;
    }

    pub(crate) fn reload_tool(&mut self) {
        match self.tool.kind {
            ToolKind::Brush => self.brush.dip(&self.paint.frac),
            ToolKind::Roller => self.roller.dip(&self.paint.frac),
            ToolKind::Sponge => self.sponge.dip(&self.paint.frac),
            ToolKind::Knife => self.load_knife(),
            _ => self.tool.reload(),
        }
    }

    pub(crate) fn ui_panel(&mut self) {
        if self.eyedropper {
            self.eyedropper_overlay();
        }
        let ctx = self.ui.ctx.clone();
        self.menu_open = false;
        self.block_rects.clear();
        let central = ctx.root_ui(|ui| {
            egui::Panel::top("menubar").frame(egui::Frame::new().fill(BAR_BG).inner_margin(egui::Margin::symmetric(8, 3)).stroke(Stroke::new(1.0, BORDER))).show(ui, |ui| self.menu_bar(ui));
            egui::Panel::bottom("status").frame(egui::Frame::new().fill(BAR_BG).inner_margin(egui::Margin::symmetric(10, 3)).stroke(Stroke::new(1.0, BORDER))).show(ui, |ui| self.status_bar(ui));
            egui::Panel::left("tools").exact_size(TOOLBAR_W).resizable(false).frame(egui::Frame::new().fill(STRIP_BG).inner_margin(egui::Margin::symmetric(6, 8)).stroke(Stroke::new(1.0, BORDER))).show(ui, |ui| self.tool_strip(ui));
            let mut open = self.show_inspector;
            egui::Panel::right("inspector").exact_size(INSPECTOR_W).resizable(false).frame(egui::Frame::new().fill(PANEL_BG).inner_margin(egui::Margin::symmetric(10, 8)).stroke(Stroke::new(1.0, BORDER))).show_collapsible(ui, &mut open, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.inspector(ui));
            });
            self.show_inspector = open;
            ui.available_rect_before_wrap()
        });
        self.file_windows(&ctx);
        if self.show_help {
            let mut open = true;
            if let Some(r) = egui::Window::new("Controls").open(&mut open).collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(&ctx, |ui| Self::help(ui)) {
                self.block_rects.push(r.response.rect);
            }
            self.show_help = open;
        }
        let ppp = ctx.pixels_per_point();
        self.view_rect = [central.min.x * ppp, central.min.y * ppp, central.width() * ppp, central.height() * ppp];
    }

    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.label(RichText::new("Canvaz").strong().color(Color32::WHITE));
            ui.add_space(8.0);
            let file = ui.menu_button("File", |ui| {
                if ui.button("New painting...").on_hover_text("Start a fresh canvas: choose its resolution and colour").clicked() {
                    self.canvas_dialog = Some(crate::CanvasChoice { res: self.film.w, colour: self.film.settings.canvas_rgb, weave: self.film.settings.raw_canvas, custom: self.film.settings.canvas_rgb.unwrap_or([0.90, 0.85, 0.76]), first: false });
                }
                if ui.button("Open...").clicked() {
                    self.open_gallery();
                }
                if ui.add(egui::Button::new("Save").shortcut_text("Ctrl+S")).clicked() {
                    self.save_current();
                }
                if ui.button("Save as...").clicked() {
                    self.save_dialog = Some(self.history.name.clone());
                }
                ui.separator();
                if ui.add(egui::Button::new("Save screenshot").shortcut_text("F12")).clicked() {
                    self.shot_request = Some(format!("shot_{}.png", self.frame_index));
                }
                if ui.button("Export for Blender").clicked() {
                    let dir = format!("blender_export_{}", self.frame_index);
                    self.export_blender(&dir);
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    self.autosave_tick(true);
                    std::process::exit(0);
                }
            });
            let edit = ui.menu_button("Edit", |ui| {
                if ui.add_enabled(self.can_undo(), egui::Button::new("Undo").shortcut_text("Ctrl+Z")).clicked() {
                    self.undo();
                }
                ui.label(RichText::new(format!("{} steps can be undone", self.history.levels.len())).small().weak());
            });
            let canvas = ui.menu_button("Canvas", |ui| {
                if ui.button("Clear canvas").clicked() {
                    self.begin_full();
                    self.clear_canvas();
                }
                if ui.add(egui::Button::new("Dry canvas now").shortcut_text("D")).clicked() {
                    self.begin_full();
                    self.film.dry_now(&self.gpu);
                }
                ui.separator();
                ui.checkbox(&mut self.film.settings.raw_canvas, "Raw canvas on clear");
                ui.checkbox(&mut self.paused, "Pause simulation   Space");
            });
            let view = ui.menu_button("View", |ui| {
                if ui.add(egui::Button::new("Top view").shortcut_text("T")).clicked() {
                    self.set_top_view();
                }
                if ui.button("Angled view").clicked() {
                    self.set_angled_view();
                }
                ui.separator();
                ui.checkbox(&mut self.show_inspector, "Inspector   Tab");
            });
            let render = ui.menu_button("Render", |ui| {
                if ui.add(egui::Button::new("Beauty render, top").shortcut_text("B")).on_hover_text("Photographic straight-down render at twice the simulation resolution").clicked() {
                    self.beauty_render(true);
                }
                if ui.button("Beauty render, current view").on_hover_text("6144 x 4096 render from the current camera").clicked() {
                    self.beauty_render(false);
                }
                ui.separator();
                ui.label(RichText::new("Settings").small().weak());
                ui.set_min_width(320.0);
                ui.set_max_width(320.0);
                self.render_settings(ui);
            });
            let help = ui.menu_button("Help", |ui| {
                if ui.button("Controls").clicked() {
                    self.show_help = true;
                }
            });
            self.menu_open = [file.inner, edit.inner, canvas.inner, view.inner, render.inner, help.inner].iter().any(|o| o.is_some());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let (label, tip) = if self.paused { ("Resume", "Resume the simulation (Space)") } else { ("Pause", "Pause the simulation (Space)") };
                if ui.add(egui::Button::new(label).min_size(Vec2::new(84.0, 0.0))).on_hover_text(tip).clicked() {
                    self.paused = !self.paused;
                }
                let inspector = if self.show_inspector { "Hide panel" } else { "Show panel" };
                if ui.button(inspector).on_hover_text("Toggle the inspector (Tab)").clicked() {
                    self.show_inspector = !self.show_inspector;
                }
            });
        });
    }

    fn tool_strip(&mut self, ui: &mut egui::Ui) {
        ui.vertical_centered_justified(|ui| {
            ui.spacing_mut().item_spacing.y = 4.0;
            for (kind, name, key, tip) in TOOLS {
                let sel = self.tool.kind == kind;
                let text = RichText::new(name).size(12.0).color(if sel { Color32::WHITE } else { ui.visuals().text_color() });
                let b = egui::Button::selectable(sel, text).min_size(Vec2::new(TOOLBAR_W - 12.0, 34.0));
                if ui.add(b).on_hover_text(format!("{}   ({})", tip, key)).clicked() {
                    self.tool.kind = kind;
                }
            }
            ui.add_space(10.0);
            let rgb = color::mix_rgb(&self.film.pigments, &self.film.cmf_table, &self.paint.frac);
            let (rect, resp) = ui.allocate_exact_size(Vec2::new(TOOLBAR_W - 12.0, 30.0), egui::Sense::click());
            ui.painter().rect_filled(rect, 4.0, swatch_color(self.shown(rgb)));
            ui.painter().rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::from_gray(90)), egui::StrokeKind::Inside);
            if resp.on_hover_text("Current paint. Click to reload the tool (R)").clicked() {
                self.reload_tool();
            }
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 14.0;
            let (dot, state) = if self.paused { (WARN, "Paused") } else { (OK, "Running") };
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                let (r, _) = ui.allocate_exact_size(Vec2::splat(8.0), egui::Sense::hover());
                ui.painter().circle_filled(r.center(), 3.5, dot);
                ui.label(RichText::new(state).color(dot).small());
            });
            let small = |s: String| RichText::new(s).small().weak();
            ui.label(small(format!("t {:.1} s", self.sim_time)));
            ui.label(small(format!("{:.0} fps", self.fps)));
            ui.label(small(format!("{} substeps", self.film.substeps_last)));
            ui.label(small(format!("{:.2} mL on canvas", self.film.total_deposited * 1e6)));
            if let Some(p) = &self.painter {
                if !p.done {
                    ui.label(RichText::new(format!("Painter: {}", p.status())).small().color(INFO));
                }
            }
            if let Some(ev) = self.evo.run.as_ref().filter(|_| self.evo.active) {
                ui.label(RichText::new(format!("Evolving: {} strokes kept, {:.0}% like the photo", ev.kept, ev.progress() * 100.0)).small().color(INFO));
            }
            if self.pen_seen {
                let p = if self.pen_down { self.pen_raw.clamp(0.0, 1.0).powf(self.pen_curve) } else { 0.0 };
                ui.label(RichText::new(format!("Pen {:.0} %", p * 100.0)).small().weak());
            }
            if self.tool.kind == ToolKind::Stick && self.tool.stick_load <= 1e-9 && !self.tool.infinite {
                ui.label(RichText::new("Stick is empty, press R").small().color(WARN));
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(small("Left: paint   Right: orbit   Middle: pan   Wheel: height   Ctrl+wheel: zoom".to_string()));
                if !self.last_beauty.is_empty() {
                    ui.label(RichText::new(self.last_beauty.clone()).small().color(OK));
                }
            });
        });
    }

    fn inspector(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new(self.tool_name()).heading().strong());
        ui.add_space(2.0);
        self.tool_settings(ui);
        ui.separator();
        section(ui, "Paint", true, |ui| self.paint_settings(ui));
        section(ui, "Paint from a photo", true, |ui| self.evolve_settings(ui));
        section(ui, "Pollock painter", true, |ui| self.painter_settings(ui));
        section(ui, "Canvas", false, |ui| {
            num(ui, "Tilt", &mut self.film.settings.tilt_deg, 0.0, 90.0, 1.0, "°", false);
            num(ui, "Tilt direction", &mut self.film.settings.tilt_dir_deg, -180.0, 180.0, 1.0, "°", false);
            num(ui, "Drying rate", &mut self.film.settings.cure_rate, 0.0, 1.0, 1.0, " /s", true);
            num(ui, "Wetting angle", &mut self.film.settings.contact_angle_deg, 0.0, 90.0, 1.0, "°", false);
            num(ui, "Hysteresis", &mut self.film.settings.hysteresis_deg, 0.0, 45.0, 1.0, "°", false);
            num(ui, "View light", &mut self.renderer.light_intensity, 0.0, 6.0, 1.0, "", false);
        });
        section(ui, "Pen tablet", false, |ui| {
            ui.checkbox(&mut self.pen_pressure, "Pressure controls the tool");
            num(ui, "Pressure curve", &mut self.pen_curve, 0.4, 2.5, 1.0, "", false);
            prop(ui, "Pressure", |ui| {
                let p = if self.pen_down { self.pen_raw.clamp(0.0, 1.0).powf(self.pen_curve) } else { 0.0 };
                ui.add(egui::ProgressBar::new(p).desired_height(14.0).text(RichText::new(if self.pen_seen { format!("{:.0} %", p * 100.0) } else { "no pen seen yet".to_string() }).small()));
            });
            ui.add(egui::Label::new(RichText::new("Brush: press depth. Bottle and bucket: flow. Pour: amount and width. Knife: blade gap closes as you press. Stick: contact only. Needs Windows Ink enabled in the Wacom settings; with Windows Ink off the pen acts as a plain mouse.").small().weak()).wrap());
        });
        section(ui, "Simulation", false, |ui| {
            let s = &mut self.film.settings;
            ui.checkbox(&mut self.auto_quality, "Adapt substeps to frame budget");
            num(ui, "Frame budget", &mut self.frame_budget_ms, 5.0, 40.0, 1.0, " ms", false);
            num(ui, "Min substeps", &mut s.min_substeps, 1.0, 32.0, 1.0, "", false);
            num(ui, "Max substeps", &mut s.max_substeps, 1.0, 128.0, 1.0, "", false);
            ui.checkbox(&mut s.capillary, "Explicit capillary levelling");
            num(ui, "Capillary cap", &mut s.m_cap, 1e-11, 1e-7, 1.0, "", true);
            num(ui, "Hydrostatic cap", &mut s.m_cap_hydro, 1e-10, 1e-6, 1.0, "", true);
            num(ui, "Min film", &mut s.h_min, 0.0, 5e-5, 1e6, " µm", false);
            num(ui, "Debug view", &mut s.debug_view, 0.0, 1.0, 1.0, "", false);
        });
        section(ui, "Diagnostics", false, |ui| {
            let cr = self.film.cg_report;
            let mono = |s: String| RichText::new(s).monospace().size(11.0);
            ui.label(mono(format!("max height {:.2} mm   mu min {:.2}", self.film.h_max * 1e3, self.film.mu_min)));
            ui.label(mono(format!("implicit {} steps  res {:.1e}", self.film.implicit_steps, if cr[0] > 0.0 { (cr[3] / cr[0]).sqrt() } else { 0.0 })));
            ui.label(mono(format!("airborne {:.3} mL  landed {:.3} mL", self.air.airborne_volume() * 1e6, self.air.landed_volume * 1e6)));
            ui.label(mono(format!("rods {}  drops {}", self.air.rods.len(), self.air.drops.len())));
            ui.add(egui::Label::new(mono(self.prof.clone())).wrap());
            if !self.stats.is_empty() {
                ui.add(egui::Label::new(mono(self.stats.clone())).wrap());
            }
            if ui.button("Compute stats").clicked() {
                self.compute_stats();
            }
        });
    }

    fn render_settings(&mut self, ui: &mut egui::Ui) {
        let r = &mut self.renderer;
        num(ui, "Exposure", &mut r.exposure, -4.0, 2.0, 1.0, " EV", false);
        num(ui, "Light elevation", &mut r.key_elevation_deg, 5.0, 90.0, 1.0, "°", false);
        num(ui, "Light direction", &mut r.key_azimuth_deg, -180.0, 180.0, 1.0, "°", false);
        num(ui, "Light power", &mut r.key_power, 0.0, 10.0, 1.0, "", false);
        num(ui, "Ambient fill", &mut r.fill, 0.0, 1.5, 1.0, "", false);
        num(ui, "Bloom", &mut r.bloom, 0.0, 1.0, 1.0, "", false);
        num(ui, "Vignette", &mut r.vignette, 0.0, 0.6, 1.0, "", false);
        num(ui, "Grain", &mut r.grain, 0.0, 0.05, 1.0, "", false);
        num(ui, "Sharpen", &mut r.sharpen, 0.0, 1.5, 1.0, "", false);
        num(ui, "Aperture", &mut r.aperture_mm, 0.0, 8.0, 1.0, " mm", false);
        prop(ui, "Tone mapping", |ui| {
            ui.selectable_value(&mut r.agx, false, "ACES");
            ui.selectable_value(&mut r.agx, true, "AgX");
        });
    }

    fn help(ui: &mut egui::Ui) {
        egui::Grid::new("controls").num_columns(2).spacing([24.0, 6.0]).striped(true).show(ui, |ui| {
            for (k, v) in [
                ("Left mouse", "Apply the tool"),
                ("Right drag", "Orbit the camera"),
                ("Middle drag", "Pan"),
                ("Wheel", "Tool height (brush: pressure)"),
                ("Ctrl + wheel,  + / -", "Zoom"),
                ("1 – 8", "Select tool"),
                ("R", "Reload stick, dip brush or load roller"),
                ("T", "Top view"),
                ("B", "Beauty render, top"),
                ("D", "Dry canvas"),
                ("Space", "Pause / resume"),
                ("Tab", "Show / hide inspector"),
                ("F12", "Save screenshot"),
                ("Ctrl+Z", "Undo"),
                ("Ctrl+S", "Save painting"),
            ] {
                ui.label(RichText::new(k).strong());
                ui.label(v);
                ui.end_row();
            }
        });
    }
}
