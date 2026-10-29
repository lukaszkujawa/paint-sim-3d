use crate::{brush, color, evolve, pollock, session, App};
use crate::tools::ToolKind;
use egui::{Align, Color32, CornerRadius, FontId, Layout, RichText, Stroke, TextStyle, Vec2};
use glam::Vec3;

const LABEL_W: f32 = 104.0;
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
const TEXT_MUTED: Color32 = Color32::from_rgb(138, 143, 151);
const OK: Color32 = Color32::from_rgb(104, 178, 128);
const WARN: Color32 = Color32::from_rgb(214, 172, 92);
const INFO: Color32 = Color32::from_rgb(150, 178, 214);

const TOOLS: [(ToolKind, &str, &str, &str); 8] = [
    (ToolKind::Stick, "Stick", "1", "Drip paint from a loaded stick"),
    (ToolKind::Bottle, "Bottle", "2", "Squeeze a thin steady thread"),
    (ToolKind::Bucket, "Bucket", "3", "Pour a heavy stream"),
    (ToolKind::Pour, "Pour", "4", "Lay paint straight onto the canvas"),
    (ToolKind::Knife, "Knife", "5", "Drag a palette knife through the paint"),
    (ToolKind::Brush, "Brush", "6", "Press a bristle brush onto the canvas"),
    (ToolKind::Roller, "Roller", "7", "Roll an even coat; picks up wet paint as it goes"),
    (ToolKind::Sponge, "Sponge", "8", "Dab or drag porous texture; a clean sponge lifts wet paint"),
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

fn prop<R>(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(Vec2::new(LABEL_W, ui.spacing().interact_size.y), Layout::left_to_right(Align::Center), |ui| {
            ui.set_min_width(LABEL_W);
            ui.add(egui::Label::new(RichText::new(label).color(TEXT_MUTED)).truncate());
        });
        ui.spacing_mut().slider_width = (ui.available_width() - 84.0).max(40.0);
        add(ui)
    })
    .inner
}

fn num<N: egui::emath::Numeric>(ui: &mut egui::Ui, label: &str, v: &mut N, lo: f64, hi: f64, scale: f64, suffix: &str, log: bool) -> bool {
    prop(ui, label, |ui| {
        let mut d = v.to_f64() * scale;
        let r = ui.add(egui::Slider::new(&mut d, lo * scale..=hi * scale).clamping(egui::SliderClamping::Edits).logarithmic(log).suffix(suffix).max_decimals(if N::INTEGRAL { 0 } else if hi * scale >= 100.0 { 1 } else { 2 }));
        if r.changed() {
            *v = N::from_f64(d / scale);
        }
        r.changed()
    })
}

fn section(ui: &mut egui::Ui, title: &str, open: bool, add: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(RichText::new(title).strong().size(13.0)).default_open(open).show(ui, |ui| {
        ui.add_space(2.0);
        add(ui);
        ui.add_space(4.0);
    });
}

fn swatch_color(rgb: [f32; 3]) -> Color32 {
    Color32::from_rgb((rgb[0] * 255.0) as u8, (rgb[1] * 255.0) as u8, (rgb[2] * 255.0) as u8)
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
        self.air = crate::Airborne::new();
        self.brush_scraped = 0.0;
        self.brush_released = 0.0;
        self.surface = crate::CoarseSurface::new(self.film.pool_dims[0], self.film.pool_dims[1], 4.0 * self.film.dx as f64);
        self.sim_time = 0.0;
    }

    fn reload_tool(&mut self) {
        match self.tool.kind {
            ToolKind::Brush => self.brush.dip(&self.paint.frac),
            ToolKind::Roller => self.roller.dip(&self.paint.frac),
            ToolKind::Sponge => self.sponge.dip(&self.paint.frac),
            ToolKind::Knife => self.load_knife(),
            _ => self.tool.reload(),
        }
    }

    pub(crate) fn ui_panel(&mut self) {
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
            ui.label(RichText::new("Paint Sim").strong().color(Color32::WHITE));
            ui.add_space(8.0);
            let file = ui.menu_button("File", |ui| {
                if ui.button("New painting").on_hover_text("Clear the canvas (undoable)").clicked() {
                    self.begin_full();
                    self.clear_canvas();
                    self.history.name.clear();
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
            ui.painter().rect_filled(rect, 4.0, swatch_color(rgb));
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
                ui.label(RichText::new(format!("Evolving: {} strokes kept, {:.0}% closer", ev.kept, ev.progress() * 100.0)).small().color(INFO));
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

    fn tool_settings(&mut self, ui: &mut egui::Ui) {
        if self.tool.kind == ToolKind::Roller {
            self.roller_settings(ui);
            return;
        }
        if self.tool.kind == ToolKind::Sponge {
            self.sponge_settings(ui);
            return;
        }
        num(ui, "Height", &mut self.tool.height, 0.002, 1.0, 100.0, " cm", true);
        match self.tool.kind {
            ToolKind::Stick => {
                num(ui, "Stick diameter", &mut self.tool.stick_diameter, 0.003, 0.03, 1e3, " mm", false);
                prop(ui, "Load", |ui| {
                    let frac = (self.tool.stick_load / self.tool.stick_capacity.max(1e-12)).clamp(0.0, 1.0) as f32;
                    ui.add(egui::ProgressBar::new(if self.tool.infinite { 1.0 } else { frac }).desired_height(14.0).text(RichText::new(format!("{:.2} mL", self.tool.stick_load * 1e6)).small()));
                });
                ui.horizontal(|ui| {
                    if ui.button("Dip stick").on_hover_text("R").clicked() {
                        self.tool.reload();
                    }
                    if ui.checkbox(&mut self.tool.infinite, "Never runs dry").changed() && self.tool.infinite {
                        self.tool.reload();
                    }
                });
            }
            ToolKind::Bottle => {
                num(ui, "Flow", &mut self.tool.bottle_flow, 1e-7, 2e-5, 1e6, " mL/s", true);
                num(ui, "Nozzle radius", &mut self.tool.bottle_radius, 0.0005, 0.006, 1e3, " mm", true);
            }
            ToolKind::Bucket => {
                num(ui, "Flow", &mut self.tool.bucket_flow, 5e-6, 3e-4, 1e6, " mL/s", true);
                num(ui, "Stream radius", &mut self.tool.bucket_radius, 0.003, 0.03, 1e3, " mm", true);
            }
            ToolKind::Pour => {
                num(ui, "Rate", &mut self.pour_rate, 1e-7, 5e-5, 1e6, " mL/s", true);
                num(ui, "Radius", &mut self.pour_radius, 0.0005, 0.03, 1e3, " mm", true);
            }
            ToolKind::Knife => {
                num(ui, "Blade width", &mut self.tool.knife_width, 0.005, 0.15, 1e3, " mm", false);
                num(ui, "Blade gap", &mut self.tool.knife_height, 0.0, 0.003, 1e3, " mm", false);
                prop(ui, "Load", |ui| {
                    let frac = (self.knife_load() / self.knife_capacity().max(1e-12)).clamp(0.0, 1.0) as f32;
                    ui.add(egui::ProgressBar::new(frac).desired_height(14.0).text(RichText::new(format!("{:.2} mL", self.knife_load() * 1e6)).small()));
                });
                ui.horizontal(|ui| {
                    if ui.button("Load knife").on_hover_text("R: pick up the current mix").clicked() {
                        self.load_knife();
                    }
                    if ui.button("Wipe knife").clicked() {
                        self.wipe_knife();
                    }
                });
                ui.add(egui::Label::new(RichText::new("Drag to level the paint under the blade to the gap. Excess is carried on the blade and spread into thin or bare spots further on; surplus squeezes out as ridges along both ends, and lifting the knife leaves an end ridge.").small().weak()).wrap());
            }
            ToolKind::Brush => self.brush_settings(ui),
            ToolKind::Roller | ToolKind::Sponge => {}
        }
    }

    fn sponge_settings(&mut self, ui: &mut egui::Ui) {
        let s = &mut self.sponge;
        num(ui, "Size", &mut s.size, 0.02, 0.15, 1e3, " mm", false);
        num(ui, "Density", &mut s.density, 0.15, 1.0, 100.0, " %", false);
        num(ui, "Pressure", &mut s.pressure, 0.0, 1.0, 100.0, " %", false);
        num(ui, "Flow", &mut s.transfer, 0.2, 3.0, 100.0, " %", false);
        prop(ui, "Load", |ui| {
            let frac = (s.load() / s.capacity()).clamp(0.0, 1.0) as f32;
            ui.add(egui::ProgressBar::new(frac).desired_height(14.0).text(RichText::new(format!("{:.2} mL", s.load() * 1e6)).small()));
        });
        ui.horizontal(|ui| {
            if ui.button("Load sponge").on_hover_text("R: soak it in the current mix").clicked() {
                self.sponge.dip(&self.paint.frac);
            }
            if ui.button("Add colour").on_hover_text("Dab the current mix into patches of the sponge for mottled multi-colour prints").clicked() {
                self.sponge.dab_colour(&self.paint.frac);
            }
            if ui.button("Clean").clicked() {
                self.sponge.clean();
            }
        });
        ui.add(egui::Label::new(RichText::new("Press to dab: the sponge squashes and gives more paint the longer and harder you press. Drag for streaky broken colour. Each new dab lands at a new angle. A dry sponge soaks up wet paint, a saturated one gives it. The wheel or pen sets the pressure.").small().weak()).wrap());
    }

    fn roller_settings(&mut self, ui: &mut egui::Ui) {
        let r = &mut self.roller;
        num(ui, "Width", &mut r.width, 0.04, 0.25, 1e3, " mm", false);
        num(ui, "Diameter", &mut r.radius, 0.01, 0.04, 2e3, " mm", false);
        num(ui, "Pressure", &mut r.pressure, 0.0, 1.0, 100.0, " %", false);
        num(ui, "Nap texture", &mut r.nap, 0.0, 1.0, 100.0, " %", false);
        prop(ui, "Load", |ui| {
            let frac = (r.load() / r.capacity()).clamp(0.0, 1.0) as f32;
            ui.add(egui::ProgressBar::new(frac).desired_height(14.0).text(RichText::new(format!("{:.1} mL  ·  film {:.0} µm", r.load() * 1e6, r.film(r.pressure) * 1e6)).small()));
        });
        ui.horizontal(|ui| {
            if ui.button("Load roller").on_hover_text("R").clicked() {
                self.roller.dip(&self.paint.frac);
            }
            if ui.button("Clean roller").clicked() {
                self.roller.clean();
            }
        });
        ui.add(egui::Label::new(RichText::new("Hold the left button to roll. The frame swivels like a caster so the roller turns to face the way you move. It lays a film that thins as the nap empties, leaves ridges at its ends when full, and lifts wet paint thicker than its gap. The wheel sets the pressure.").small().weak()).wrap());
    }

    fn brush_settings(&mut self, ui: &mut egui::Ui) {
        let mut shape = self.brush.shape;
        prop(ui, "Shape", |ui| {
            ui.selectable_value(&mut shape, brush::BrushShape::Round, "Round");
            ui.selectable_value(&mut shape, brush::BrushShape::Flat, "Flat");
        });
        let mut size = self.brush.size() * 1e3;
        let mut len = self.brush.len * 1e3;
        let mut n = self.brush.bristles.len();
        let c1 = num(ui, if shape == brush::BrushShape::Flat { "Width" } else { "Diameter" }, &mut size, 0.1, 150.0, 1.0, " mm", true);
        let c2 = num(ui, "Bristle length", &mut len, 10.0, 60.0, 1.0, " mm", false);
        let c3 = num(ui, "Bristles", &mut n, 16.0, 400.0, 1.0, "", false);
        if shape != self.brush.shape || c1 || c2 || c3 {
            self.rebuild_brush(shape, n, len * 1e-3, size * 1e-3);
        }
        num(ui, "Stiffness", &mut self.brush.stiffness, 0.05, 0.9, 1.0, "", false);
        num(ui, "Friction", &mut self.brush.friction, 0.0, 0.95, 1.0, "", false);
        num(ui, "Paint release", &mut self.brush.release, 0.1, 10.0, 1.0, "", true);
        num(ui, "Sink into paint", &mut self.brush_sink, 0.0, 1.0, 100.0, " %", false);
        prop(ui, "Load", |ui| {
            let frac = (self.brush.total_load() / self.brush.total_capacity().max(1e-12)).clamp(0.0, 1.0) as f32;
            ui.add(egui::ProgressBar::new(frac).desired_height(14.0).text(RichText::new(format!("{:.3} mL  ·  {} touching", self.brush.total_load() * 1e6, self.brush.touching)).small()));
        });
        ui.horizontal(|ui| {
            if ui.button("Dip brush").on_hover_text("R").clicked() {
                self.brush.dip(&self.paint.frac);
            }
            if ui.button("Clean brush").clicked() {
                self.brush.clean();
            }
            if ui.checkbox(&mut self.tool.infinite, "Never runs dry").on_hover_text("Keep the bristles topped up with the current paint").changed() && self.tool.infinite {
                self.brush.top_up(&self.paint.frac);
            }
        });
    }

    fn paint_settings(&mut self, ui: &mut egui::Ui) {
        let rgb = color::mix_rgb(&self.film.pigments, &self.film.cmf_table, &self.paint.frac);
        let names: Vec<String> = self.film.pigments.pigments.iter().map(|p| p.name.clone()).collect();
        let total: f32 = self.paint.frac[..color::NPIG].iter().sum::<f32>().max(1e-6);
        let mut parts: Vec<(f32, &str)> = (0..color::NPIG).filter(|&i| self.paint.frac[i] > 0.0).map(|i| (self.paint.frac[i] / total, names[i].as_str())).collect();
        parts.sort_by(|a, b| b.0.total_cmp(&a.0));
        let desc = parts.iter().take(3).map(|(f, n)| format!("{:.0}% {}", f * 100.0, n)).collect::<Vec<_>>().join(", ");
        ui.horizontal(|ui| {
            let (rect, resp) = ui.allocate_exact_size(Vec2::new(44.0, 30.0), egui::Sense::click());
            ui.painter().rect_filled(rect, 4.0, swatch_color(rgb));
            let stroke = if resp.hovered() { Stroke::new(1.5, Color32::from_gray(220)) } else { Stroke::new(1.0, Color32::from_gray(90)) };
            ui.painter().rect_stroke(rect, 4.0, stroke, egui::StrokeKind::Inside);
            let resp = resp.on_hover_text("Click to pick any colour");
            if resp.clicked() {
                self.paint_pick = rgb;
            }
            self.paint_picker(ui, &resp, rgb);
            ui.add(egui::Label::new(RichText::new(if desc.is_empty() { "No pigment".to_string() } else { desc }).small()).wrap());
        });
        ui.add_space(4.0);
        let cols = 6;
        let gap = 4.0;
        let cell = ((ui.available_width() - gap * (cols as f32 - 1.0)) / cols as f32).floor();
        egui::Grid::new("swatches").spacing([gap, gap]).show(ui, |ui| {
            for i in 0..color::NPIG.min(names.len()) {
                let c = swatch_color(color::masstone_rgb(&self.film.pigments, &self.film.cmf_table, i));
                let (rect, resp) = ui.allocate_exact_size(Vec2::new(cell, 22.0), egui::Sense::click());
                ui.painter().rect_filled(rect, 3.0, c);
                let used = self.paint.frac[i] > 0.0;
                let stroke = if used { Stroke::new(2.0, ACCENT) } else if resp.hovered() { Stroke::new(1.0, Color32::from_gray(200)) } else { Stroke::new(1.0, Color32::from_gray(70)) };
                ui.painter().rect_stroke(rect, 3.0, stroke, egui::StrokeKind::Inside);
                let resp = resp.on_hover_text(format!("{}\nClick: use pure   Shift+click: add to mix", names[i]));
                if resp.clicked() {
                    if ui.input(|inp| inp.modifiers.shift) {
                        self.paint.frac[i] += 0.25 * total;
                    } else {
                        for k in 0..color::NPIG {
                            self.paint.frac[k] = 0.0;
                        }
                        self.paint.frac[i] = 1.0;
                    }
                }
                if (i + 1) % cols == 0 {
                    ui.end_row();
                }
            }
        });
        if !self.custom_mixes.is_empty() {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = gap;
                ui.label(RichText::new("Custom").small().weak());
                let mut pick = None;
                for (k, m) in self.custom_mixes.iter().enumerate() {
                    let c = swatch_color(color::mix_rgb(&self.film.pigments, &self.film.cmf_table, m));
                    let (rect, resp) = ui.allocate_exact_size(Vec2::new(22.0, 22.0), egui::Sense::click());
                    ui.painter().rect_filled(rect, 3.0, c);
                    let same = (0..color::NPIG).all(|i| (m[i] - self.paint.frac[i] / total).abs() < 1e-4);
                    let stroke = if same { Stroke::new(2.0, ACCENT) } else if resp.hovered() { Stroke::new(1.0, Color32::from_gray(200)) } else { Stroke::new(1.0, Color32::from_gray(70)) };
                    ui.painter().rect_stroke(rect, 3.0, stroke, egui::StrokeKind::Inside);
                    if resp.on_hover_text("Click: use this colour").clicked() {
                        pick = Some(k);
                    }
                }
                if let Some(k) = pick {
                    let m = self.custom_mixes[k];
                    for i in 0..color::NPIG {
                        self.paint.frac[i] = m[i];
                    }
                }
            });
        }
        ui.add_space(4.0);
        prop(ui, "Preset", |ui| {
            if ui.button("Enamel").on_hover_text("Fluid house paint, Pollock's choice").clicked() {
                self.paint.frac[12] = 0.3;
                self.paint.frac[13] = 0.0;
                self.tool.mu = 4.0;
            }
            if ui.button("Thin").on_hover_text("Thinned enamel").clicked() {
                self.paint.frac[12] = 0.7;
                self.paint.frac[13] = 0.0;
                self.tool.mu = 2.0;
            }
            if ui.button("Tube oil").on_hover_text("Stiff impasto oil").clicked() {
                self.paint.frac[12] = 0.0;
                self.paint.frac[13] = 250.0;
                self.tool.mu = 40.0;
            }
        });
        num(ui, "Viscosity", &mut self.tool.mu, 0.05, 100.0, 1.0, " Pa·s", true);
        num(ui, "Thinner", &mut self.paint.frac[12], 0.0, 0.9, 100.0, " %", false);
        num(ui, "Yield stress", &mut self.paint.frac[13], 0.0, 600.0, 1.0, " Pa", false);
        num(ui, "Surface tension", &mut self.tool.sigma, 0.02, 0.07, 1e3, " mN/m", false);
        egui::CollapsingHeader::new(RichText::new("Pigment amounts").small()).id_salt("mix_sliders").show(ui, |ui| {
            for (i, name) in names.iter().enumerate().take(color::NPIG) {
                num(ui, name, &mut self.paint.frac[i], 0.0, 1.0, 1.0, "", false);
            }
        });
    }

    fn evolve_settings(&mut self, ui: &mut egui::Ui) {
        let running = self.evolving();
        ui.horizontal(|ui| {
            if ui.add_enabled(!running, egui::Button::new("Load photo...")).on_hover_text("Choose a picture to paint. You can also drop an image file onto the window").clicked() {
                self.pick_photo();
            }
            let name = self.evo.photo.as_ref().map(|p| p.name.clone()).unwrap_or_else(|| "No photo yet".to_string());
            ui.add(egui::Label::new(RichText::new(name).small().weak()).truncate());
        });
        let ctx = ui.ctx().clone();
        if let Some(photo) = self.evo.photo.as_mut() {
            let tex = evolve::preview(photo, &ctx);
            let w = ui.available_width() - 12.0;
            let size = tex.size_vec2();
            let focus = &mut self.evo.focus;
            let sense = if focus.shape == evolve::FocusShape::Off { egui::Sense::hover() } else { egui::Sense::click_and_drag() };
            let resp = ui.add(egui::Image::new(&tex).fit_to_exact_size(Vec2::new(w, w * size.y / size.x.max(1.0))).corner_radius(4.0).sense(sense));
            let r = resp.rect;
            let uv = |p: egui::Pos2| [((p.x - r.min.x) / r.width()).clamp(0.0, 1.0), ((p.y - r.min.y) / r.height()).clamp(0.0, 1.0)];
            if let Some(p) = resp.interact_pointer_pos() {
                if resp.drag_started() {
                    focus.a = uv(p);
                }
                focus.b = uv(p);
            }
            let at = |q: [f32; 2]| egui::pos2(r.min.x + q[0] * r.width(), r.min.y + q[1] * r.height());
            let painter = ui.painter_at(r);
            let edge = Stroke::new(1.5, Color32::WHITE);
            let ramp = Stroke::new(1.0, Color32::from_white_alpha(110));
            match focus.shape {
                evolve::FocusShape::Off => {}
                evolve::FocusShape::Circle => {
                    let c = at(focus.a);
                    let rad = at(focus.b).distance(c);
                    let f = 0.5 * focus.soft * rad;
                    painter.circle_stroke(c, rad + f, ramp);
                    painter.circle_stroke(c, (rad - f).max(0.0), ramp);
                    painter.circle_stroke(c, rad, edge);
                }
                evolve::FocusShape::Rect => {
                    let q = focus.rect();
                    let rr = egui::Rect::from_min_max(at([q[0], q[1]]), at([q[2], q[3]]));
                    let f = 0.25 * focus.soft * rr.width().min(rr.height());
                    painter.rect_stroke(rr.expand(f), f, ramp, egui::StrokeKind::Middle);
                    if rr.width() > 2.0 * f && rr.height() > 2.0 * f {
                        painter.rect_stroke(rr.shrink(f), 0.0, ramp, egui::StrokeKind::Middle);
                    }
                    painter.rect_stroke(rr, 0.0, edge, egui::StrokeKind::Middle);
                }
            }
        } else {
            ui.add(egui::Label::new(RichText::new("Drop an image onto the window or load one. The painter plans strokes in memory with a quick brush model, going after the areas that differ most from the photo and following its edges, then paints only the strokes that will stay visible with the real tool, keeping those that bring the canvas closer to the photo.").small().weak()).wrap());
        }
        ui.checkbox(&mut self.evo.dry_passes, "Let each pass dry before the next").on_hover_text("When strokes start helping much less, the painter dries the canvas and moves to a smaller tool, like painting in layers");
        prop(ui, "Tries per stroke", |ui| {
            ui.add(egui::Slider::new(&mut self.evo.candidates, 4..=128).logarithmic(true).clamping(egui::SliderClamping::Edits));
        });
        let mut shape = self.evo.focus.shape;
        prop(ui, "Focus", |ui| {
            ui.selectable_value(&mut shape, evolve::FocusShape::Off, "All").on_hover_text("Paint the whole canvas");
            ui.selectable_value(&mut shape, evolve::FocusShape::Circle, "Circle").on_hover_text("Drag on the photo from the centre outwards");
            ui.selectable_value(&mut shape, evolve::FocusShape::Rect, "Box").on_hover_text("Drag on the photo from one corner to the other");
        });
        if shape != self.evo.focus.shape {
            let f = &mut self.evo.focus;
            let q = f.rect();
            if q[2] - q[0] < 0.02 && q[3] - q[1] < 0.02 {
                (f.a, f.b) = match shape {
                    evolve::FocusShape::Rect => ([0.3, 0.3], [0.7, 0.7]),
                    _ => ([0.5, 0.5], [0.75, 0.5]),
                };
            } else if shape == evolve::FocusShape::Circle && f.shape == evolve::FocusShape::Rect {
                (f.a, f.b) = ([0.5 * (q[0] + q[2]), 0.5 * (q[1] + q[3])], [q[2], 0.5 * (q[1] + q[3])]);
            } else if shape == evolve::FocusShape::Rect && f.shape == evolve::FocusShape::Circle {
                let aspect = self.film.settings.width_m / self.film.settings.height_m;
                let rad = ((f.b[0] - f.a[0]).powi(2) + ((f.b[1] - f.a[1]) / aspect).powi(2)).sqrt();
                (f.a, f.b) = ([f.a[0] - rad, f.a[1] - rad * aspect], [f.a[0] + rad, f.a[1] + rad * aspect]);
            }
            f.shape = shape;
        }
        if shape != evolve::FocusShape::Off {
            num(ui, "Soft edge", &mut self.evo.focus.soft, 0.0, 2.0, 100.0, " %", false);
            num(ui, "Outside", &mut self.evo.focus.outside, 0.0, 1.0, 100.0, " %", false);
        }
        num(ui, "Follow edges", &mut self.evo.follow, 0.0, 1.0, 100.0, " %", false);
        num(ui, "Longest stroke", &mut self.evo.longest, 0.01, 1.2, 100.0, " cm", true);
        num(ui, "Speed", &mut self.painter_speed, 1.0, 8.0, 1.0, "×", false);
        let has_photo = self.evo.photo.is_some();
        let bw = ui.available_width();
        if running {
            if ui.add(egui::Button::new(RichText::new("Stop").strong()).min_size(Vec2::new(bw, 26.0))).on_hover_text("Esc").clicked() {
                self.stop_evolve();
            }
        } else {
            let paint = egui::Button::new(RichText::new("Paint").strong().color(Color32::WHITE)).fill(ACCENT).min_size(Vec2::new(bw, 26.0));
            if ui.add_enabled(has_photo && self.tool.kind != ToolKind::Knife, paint).on_hover_text("Paint the photo over whatever is on the canvas with the selected tool, paint and size until you stop it. Strokes vary in colour and between 40 and 100% of the tool size. Change the tool, size or paint and press Paint again to carry on; use File > New painting for a fresh canvas").clicked() {
                self.start_evolve();
            }
        }
        if let Some(ev) = &self.evo.run {
            let mono = |s: String| RichText::new(s).small();
            ui.add(egui::ProgressBar::new(ev.progress()).desired_height(14.0).text(RichText::new(format!("{:.0}% closer to the photo", ev.progress() * 100.0)).small()));
            ui.label(mono(format!("{} kept of {} painted", ev.kept, ev.painted)));
            ui.label(mono(format!("{} planned in memory, {} skipped as hidden", ev.tried, ev.pruned)));
            if running {
                ui.label(mono(format!("Pass {}, tool size up to {:.1} mm", ev.passes + 1, ev.size_hi * 1e3)));
            }
            if !ev.status.is_empty() {
                ui.add(egui::Label::new(RichText::new(ev.status.clone()).small().color(WARN)).wrap());
            }
        }
    }

    fn painter_settings(&mut self, ui: &mut egui::Ui) {
        let running = self.painter.as_ref().map(|p| !p.done).unwrap_or(false);
        let (w, h) = (self.film.settings.width_m as f64, self.film.settings.height_m as f64);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Palette").strong());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.button("Randomise").on_hover_text("New palette built on a colour harmony: complementary, split complementary, triadic, analogous or tetradic around the ground hue, with line colours that contrast with the ground").clicked() {
                    self.pollock_pal = pollock::Palette::random(rand::random::<u64>());
                    self.sync_pal_picks();
                }
            });
        });
        ui.add(egui::Label::new(RichText::new(self.pollock_pal.name()).small().weak()).wrap());
        ui.add_space(2.0);
        let rows: [(&str, std::ops::Range<usize>); 2] = [("Background", 0..2), ("Foreground", 2..6)];
        for (title, slots) in rows {
            prop(ui, title, |ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                for s in slots {
                    self.palette_swatch(ui, s);
                }
            });
        }
        prop(ui, "Background with", |ui| {
            for (m, label, tip) in [
                (pollock::GroundMethod::Auto, "Auto", "Let the painter choose: brushed, poured, washed, scraped or tilted"),
                (pollock::GroundMethod::Brush, "Brush", "Scrub the ground on with a wide flat brush"),
                (pollock::GroundMethod::Bucket, "Bucket", "Pour pools from a bucket that flow into each other"),
            ] {
                if ui.selectable_label(self.ground_method == m, label).on_hover_text(tip).clicked() {
                    self.ground_method = m;
                }
            }
        });
        if self.ground_method != pollock::GroundMethod::Auto {
            let (k, label, hi) = if self.ground_method == pollock::GroundMethod::Brush { (0, "Brush strokes", 200) } else { (1, "Bucket pours", 60) };
            prop(ui, label, |ui| {
                ui.add(egui::Slider::new(&mut self.ground_strokes[k], 1..=hi).clamping(egui::SliderClamping::Edits));
            });
        }
        ui.add_space(4.0);
        ui.add_enabled_ui(!running, |ui| {
            ui.horizontal(|ui| {
                let bw = ((ui.available_width() - 6.0) * 0.5).max(60.0);
                if ui.add(egui::Button::new("Paint background").min_size(Vec2::new(bw, 26.0))).on_hover_text("Lay the ground in the two background colours, then let it dry").clicked() {
                    let seed = rand::random::<u64>() % 100000;
                    self.begin_full();
                    let strokes = match self.ground_method {
                        pollock::GroundMethod::Auto => 0,
                        pollock::GroundMethod::Brush => self.ground_strokes[0],
                        pollock::GroundMethod::Bucket => self.ground_strokes[1],
                    };
                    self.painter = Some(pollock::Painter::background(seed, w, h, self.pollock_pal.clone(), self.ground_method, strokes));
                }
                if ui.add(egui::Button::new("Paint foreground").min_size(Vec2::new(bw, 26.0))).on_hover_text("Pour, drip and fling the four foreground colours on top of what is there. Press again for another round").clicked() {
                    self.begin_full();
                    match self.painter.as_mut() {
                        Some(p) => p.paint_foreground(self.pollock_pal.clone()),
                        None => {
                            let seed = rand::random::<u64>() % 100000;
                            self.painter = Some(pollock::Painter::foreground(seed, w, h, self.pollock_pal.clone()));
                        }
                    }
                }
            });
        });
        ui.horizontal(|ui| {
            let bw = ((ui.available_width() - 6.0) * 0.5).max(60.0);
            if running {
                if ui.add(egui::Button::new(RichText::new("Stop").strong()).min_size(Vec2::new(bw, 26.0))).clicked() {
                    if let Some(p) = self.painter.as_mut() {
                        p.stop();
                    }
                }
            } else {
                let full = egui::Button::new(RichText::new("Full painting").strong().color(Color32::WHITE)).fill(ACCENT).min_size(Vec2::new(bw, 26.0));
                if ui.add(full).on_hover_text("Background, foreground and two or three more rounds of lines with this palette").clicked() {
                    let seed = rand::random::<u64>() % 100000;
                    self.begin_full();
                    self.painter = Some(pollock::Painter::full(seed, w, h, self.pollock_pal.clone()));
                }
            }
        });
        if let Some(p) = &self.painter {
            ui.add(egui::Label::new(RichText::new(p.status()).small()).wrap()).on_hover_text(format!("Seed {}", p.seed));
        }
        num(ui, "Speed", &mut self.painter_speed, 1.0, 8.0, 1.0, "×", false);
    }

    pub(crate) fn sync_pal_picks(&mut self) {
        for s in 0..pollock::Palette::SLOTS {
            self.pal_picks[s] = color::mix_rgb(&self.film.pigments, &self.film.cmf_table, &self.pollock_pal.slot_mix(s));
        }
    }

    fn paint_picker(&mut self, ui: &mut egui::Ui, resp: &egui::Response, rgb: [f32; 3]) {
        let popup = egui::Popup::from_toggle_button_response(resp).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
            ui.set_width(240.0);
            ui.label(RichText::new("Pick a colour").strong());
            ui.spacing_mut().slider_width = 210.0;
            let mut c = swatch_color(self.paint_pick);
            if egui::color_picker::color_picker_color32(ui, &mut c, egui::color_picker::Alpha::Opaque) {
                self.paint_pick = [c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0];
                let mix = pollock::match_rgb(self.paint_pick);
                for i in 0..color::NPIG {
                    self.paint.frac[i] = mix[i];
                }
            }
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(Vec2::new(28.0, 20.0), egui::Sense::hover());
                ui.painter().rect_filled(r, 3.0, swatch_color(self.paint_pick));
                ui.label(RichText::new("picked").small().weak());
                let (r, _) = ui.allocate_exact_size(Vec2::new(28.0, 20.0), egui::Sense::hover());
                ui.painter().rect_filled(r, 3.0, swatch_color(rgb));
                ui.label(RichText::new("paint").small().weak());
            });
            ui.label(RichText::new("The colour is mixed from the twelve pigments, so very saturated colours come out as the closest real paint. Thinner and yield stress are kept.").small().weak());
            ui.separator();
            if ui.button("Save to custom colours").clicked() {
                let total: f32 = self.paint.frac[..color::NPIG].iter().sum::<f32>().max(1e-6);
                let mut m = [0.0f32; 16];
                for i in 0..color::NPIG {
                    m[i] = self.paint.frac[i] / total;
                }
                self.custom_mixes.retain(|o| (0..color::NPIG).any(|i| (o[i] - m[i]).abs() > 1e-4));
                self.custom_mixes.insert(0, m);
                self.custom_mixes.truncate(12);
            }
        });
        if let Some(p) = popup {
            self.block_rects.push(p.response.rect);
            self.menu_open = true;
        }
    }

    fn palette_swatch(&mut self, ui: &mut egui::Ui, s: usize) {
        let m = self.pollock_pal.slot_mix(s);
        let rgb = color::mix_rgb(&self.film.pigments, &self.film.cmf_table, &m);
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(36.0, 26.0), egui::Sense::click());
        ui.painter().rect_filled(rect, 4.0, swatch_color(rgb));
        let stroke = if resp.hovered() { Stroke::new(1.5, Color32::from_gray(220)) } else { Stroke::new(1.0, Color32::from_gray(80)) };
        ui.painter().rect_stroke(rect, 4.0, stroke, egui::StrokeKind::Inside);
        let resp = resp.on_hover_text(format!("{}\nClick to change", self.pollock_pal.slot_name(s)));
        let popup = egui::Popup::from_toggle_button_response(&resp).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
            ui.set_width(240.0);
            let role = if s < 2 { format!("Background {}", s + 1) } else { format!("Foreground {}", s - 1) };
            ui.label(RichText::new(format!("{}: {}", role, self.pollock_pal.slot_name(s))).strong());
            ui.spacing_mut().slider_width = 210.0;
            let mut c = swatch_color(self.pal_picks[s]);
            if egui::color_picker::color_picker_color32(ui, &mut c, egui::color_picker::Alpha::Opaque) {
                self.pal_picks[s] = [c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0];
                let mix = pollock::match_rgb(self.pal_picks[s]);
                self.pollock_pal.set_slot(s, &mix);
            }
            let names: Vec<&str> = self.film.pigments.pigments.iter().map(|p| p.name.as_str()).collect();
            let mut parts: Vec<(f32, &str)> = (0..color::NPIG).filter(|&i| m[i] > 0.005).map(|i| (m[i], names[i])).collect();
            parts.sort_by(|a, b| b.0.total_cmp(&a.0));
            let desc = parts.iter().take(4).map(|(f, n)| format!("{:.0}% {}", f * 100.0, n)).collect::<Vec<_>>().join(", ");
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(Vec2::new(28.0, 20.0), egui::Sense::hover());
                ui.painter().rect_filled(r, 3.0, swatch_color(rgb));
                ui.add(egui::Label::new(RichText::new(desc).small()).wrap());
            });
            ui.label(RichText::new("The picked colour is matched with the closest mix of the twelve pigments.").small().weak());
            ui.separator();
            if ui.button("Use current paint mix").clicked() {
                let frac = self.paint.frac;
                self.pollock_pal.set_slot(s, &frac);
                self.pal_picks[s] = color::mix_rgb(&self.film.pigments, &self.film.cmf_table, &self.pollock_pal.slot_mix(s));
            }
            if ui.button("Load into paint mix").on_hover_text("Copy this colour to the hand tools, keeping the paint body").clicked() {
                for i in 0..color::NPIG {
                    self.paint.frac[i] = m[i];
                }
            }
        });
        if let Some(p) = popup {
            self.block_rects.push(p.response.rect);
            self.menu_open = true;
        }
    }

    fn render_settings(&mut self, ui: &mut egui::Ui) {
        let r = &mut self.renderer;
        num(ui, "Exposure", &mut r.exposure, -2.0, 2.0, 1.0, " EV", false);
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

    pub(crate) fn save_current(&mut self) {
        if self.history.name.is_empty() {
            self.save_dialog = Some(String::new());
        } else {
            let name = self.history.name.clone();
            self.save_painting(&name);
        }
    }

    fn open_gallery(&mut self) {
        let ctx = self.ui.ctx.clone();
        self.gallery = session::list_paintings()
            .into_iter()
            .map(|p| {
                let tex = image::open(session::thumb_path(&p.path)).ok().map(|img| {
                    let rgba = img.to_rgba8();
                    let size = [rgba.width() as usize, rgba.height() as usize];
                    ctx.load_texture(p.name.clone(), egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw()), egui::TextureOptions::LINEAR)
                });
                (p, tex)
            })
            .collect();
        self.show_gallery = true;
    }

    fn file_windows(&mut self, ctx: &egui::Context) {
        if let Some(mut name) = self.save_dialog.take() {
            let mut keep = true;
            let mut save = false;
            if let Some(r) = egui::Window::new("Save painting").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
                ui.label("Name");
                let edit = ui.add(egui::TextEdit::singleline(&mut name).desired_width(260.0).hint_text("my painting"));
                if !self.save_focus {
                    edit.request_focus();
                    self.save_focus = true;
                }
                if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    save = true;
                }
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        save = true;
                    }
                    if ui.button("Cancel").clicked() {
                        keep = false;
                    }
                });
            }) {
                self.block_rects.push(r.response.rect);
            }
            let clean: String = name.trim().chars().filter(|c| c.is_alphanumeric() || " -_".contains(*c)).collect();
            if save && !clean.is_empty() && clean != session::AUTOSAVE_NAME {
                self.save_painting(&clean);
                keep = false;
            }
            if keep {
                self.save_dialog = Some(name);
            } else {
                self.save_focus = false;
            }
        }
        if self.show_gallery {
            let mut open = true;
            let mut chosen = None;
            if let Some(r) = egui::Window::new("Open painting").open(&mut open).collapsible(false).default_size([820.0, 560.0]).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
                if self.gallery.is_empty() {
                    ui.label(RichText::new(format!("No saved paintings yet. Save with Ctrl+S; they go to the {} folder.", session::PAINTINGS_DIR)).weak());
                }
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = Vec2::new(12.0, 12.0);
                        for (k, (p, tex)) in self.gallery.iter().enumerate() {
                            ui.vertical(|ui| {
                                let resp = match tex {
                                    Some(t) => ui.add(egui::Button::image(egui::Image::new(t).fit_to_exact_size(Vec2::new(240.0, 160.0)))),
                                    None => ui.add(egui::Button::new("no preview").min_size(Vec2::new(240.0, 160.0))),
                                };
                                let label = if p.name == session::AUTOSAVE_NAME { "Last session (autosave)".to_string() } else { p.name.clone() };
                                ui.label(RichText::new(label).small());
                                if resp.clicked() {
                                    chosen = Some(k);
                                }
                            });
                        }
                    });
                });
            }) {
                self.block_rects.push(r.response.rect);
            }
            if let Some(k) = chosen {
                let path = self.gallery[k].0.path.clone();
                self.show_gallery = false;
                self.open_painting(&path);
            } else {
                self.show_gallery = open;
            }
        }
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
