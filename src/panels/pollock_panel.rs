use super::*;

impl App {
    pub(super) fn painter_settings(&mut self, ui: &mut egui::Ui) {
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

    pub(super) fn palette_swatch(&mut self, ui: &mut egui::Ui, s: usize) {
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
}
