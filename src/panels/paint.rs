use super::*;

impl App {
    pub(super) fn paint_settings(&mut self, ui: &mut egui::Ui) {
        let rgb = color::mix_rgb(&self.film.pigments, &self.film.cmf_table, &self.paint.frac);
        let names: Vec<String> = self.film.pigments.pigments.iter().map(|p| p.name.clone()).collect();
        let total: f32 = self.paint.frac[..color::NPIG].iter().sum::<f32>().max(1e-6);
        let mut parts: Vec<(f32, &str)> = (0..color::NPIG).filter(|&i| self.paint.frac[i] > 0.0).map(|i| (self.paint.frac[i] / total, names[i].as_str())).collect();
        parts.sort_by(|a, b| b.0.total_cmp(&a.0));
        let desc = parts.iter().take(3).map(|(f, n)| format!("{:.0}% {}", f * 100.0, n)).collect::<Vec<_>>().join(", ");
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(Vec2::new(44.0, 30.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 4.0, swatch_color(self.shown(rgb)));
            ui.painter().rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::from_gray(90)), egui::StrokeKind::Inside);
            ui.add(egui::Label::new(RichText::new(if desc.is_empty() { "No pigment".to_string() } else { desc }).small()).wrap());
        });
        ui.add_space(4.0);
        let dropper = egui::Button::selectable(self.eyedropper, "Pick colour from canvas").min_size(Vec2::new(ui.available_width(), 24.0));
        if ui.add(dropper).on_hover_text("Click a spot on the canvas to mix its colour and load the tool (I). Esc cancels.").clicked() {
            self.eyedropper = !self.eyedropper;
        }
        ui.add_space(4.0);
        let cols = 6;
        let gap = 4.0;
        let cell = ((ui.available_width() - gap * (cols as f32 - 1.0)) / cols as f32).floor();
        egui::Grid::new("swatches").spacing([gap, gap]).show(ui, |ui| {
            for i in 0..color::NPIG.min(names.len()) {
                let c = swatch_color(self.shown(color::masstone_rgb(&self.film.pigments, &self.film.cmf_table, i)));
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
                    let c = swatch_color(self.shown(color::mix_rgb(&self.film.pigments, &self.film.cmf_table, m)));
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
        if self.paint.frac[..color::NPIG] != self.paint_pick_mix || self.renderer.exposure != self.paint_pick_ev {
            self.paint_pick = self.shown(rgb);
            self.paint_pick_ev = self.renderer.exposure;
            self.paint_pick_mix.copy_from_slice(&self.paint.frac[..color::NPIG]);
        }
        egui::CollapsingHeader::new(RichText::new("Custom colour").small()).id_salt("custom_colour").default_open(true).show(ui, |ui| self.custom_colour(ui, rgb));
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

    pub(crate) fn shown(&self, paint: [f32; 3]) -> [f32; 3] {
        color::paint_shown(paint, self.renderer.exposure)
    }

    pub(crate) fn use_colour(&mut self, rgb: [f32; 3]) {
        let mix = pollock::match_rgb(color::paint_for_shown(rgb, self.renderer.exposure));
        for i in 0..color::NPIG {
            self.paint.frac[i] = mix[i];
        }
        self.paint_pick = rgb;
        self.paint_pick_ev = self.renderer.exposure;
        self.paint_pick_mix.copy_from_slice(&self.paint.frac[..color::NPIG]);
        if !self.evolving() {
            self.reload_tool();
        }
    }

    pub(super) fn custom_colour(&mut self, ui: &mut egui::Ui, rgb: [f32; 3]) {
        ui.spacing_mut().slider_width = (ui.available_width() - 24.0).clamp(120.0, 240.0);
        let mut c = swatch_color(self.paint_pick);
        if egui::color_picker::color_picker_color32(ui, &mut c, egui::color_picker::Alpha::Opaque) {
            self.use_colour([c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0]);
        }
        ui.add_space(4.0);
        let mut submitted = None;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Hex").small());
            let edit = ui.add(egui::TextEdit::singleline(&mut self.paint_hex).desired_width(72.0).char_limit(7));
            if edit.changed() {
                submitted = parse_hex(&self.paint_hex);
            }
            if !edit.has_focus() {
                let c = swatch_color(self.paint_pick);
                self.paint_hex = format!("#{:02X}{:02X}{:02X}", c.r(), c.g(), c.b());
            }
            let (r, _) = ui.allocate_exact_size(Vec2::new(22.0, 18.0), egui::Sense::hover());
            ui.painter().rect_filled(r, 3.0, swatch_color(self.paint_pick));
            ui.label(RichText::new("wanted").small().weak());
            let (r, _) = ui.allocate_exact_size(Vec2::new(22.0, 18.0), egui::Sense::hover());
            ui.painter().rect_filled(r, 3.0, swatch_color(self.shown(rgb)));
            ui.label(RichText::new("paint").small().weak());
        });
        if let Some(rgb) = submitted {
            self.use_colour(rgb);
        }
        ui.add(egui::Label::new(RichText::new("Mixed from the twelve pigments, so very saturated colours come out as the closest real paint. Thinner and yield stress are kept.").small().weak()).wrap());
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
    }

    pub(super) fn eyedropper_overlay(&mut self) {
        let ctx = self.ui.ctx.clone();
        let ppp = ctx.pixels_per_point().max(0.01);
        self.pick_hover = self.canvas_colour_at_cursor();
        ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
        let Some(rgb) = self.pick_hover.map(|c| self.shown(c)) else { return };
        let at = egui::pos2(self.cursor[0] / ppp + 18.0, self.cursor[1] / ppp + 18.0);
        egui::Area::new(egui::Id::unique("eyedropper")).order(egui::Order::Tooltip).fixed_pos(at).interactable(false).show(&ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    let (r, _) = ui.allocate_exact_size(Vec2::new(28.0, 20.0), egui::Sense::hover());
                    ui.painter().rect_filled(r, 3.0, swatch_color(rgb));
                    let c = swatch_color(rgb);
                    ui.label(RichText::new(format!("#{:02X}{:02X}{:02X}", c.r(), c.g(), c.b())).small());
                });
            });
        });
    }
}
