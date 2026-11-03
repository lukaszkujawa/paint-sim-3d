use super::*;

impl App {
    pub(super) fn evolve_settings(&mut self, ui: &mut egui::Ui) {
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
            let tex = evolve::preview(photo, &ctx, self.evo.prep, self.film.settings.width_m * 1e3);
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
            ui.add(egui::Label::new(RichText::new("Drop an image onto the window or load one. The painter works in layers like a person would: strokes are planned against a copy of the photo softened to the current tool size, keeping its edges, and placed only where the canvas still differs. You decide when it is time for detail: lower the tool size and it carries on with finer strokes. Each stroke starts at the worst spot, follows the contour there and stops where its colour no longer fits. Strokes are planned in memory with a quick brush model and painted with the real tool, keeping those that bring the canvas closer to the photo.").small().weak()).wrap());
        }
        ui.checkbox(&mut self.evo.dry_passes, "Dry the canvas when I change the tool size").on_hover_text("Each change of tool size starts a new pass; the canvas is dried first, like painting in layers");
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
        let p = &mut self.evo.prep;
        let touched = num(ui, "Contrast", &mut p.contrast, 0.5, 2.0, 100.0, " %", false) | num(ui, "Blur", &mut p.blur_mm, 0.0, 40.0, 1.0, " mm", false) | num(ui, "Sharpen", &mut p.sharpen, 0.0, 3.0, 100.0, " %", false) | num(ui, "Smooth, keep edges", &mut p.smooth_mm, 0.0, 10.0, 1.0, " mm", false) | num_hint(ui, "Lift dark photo", "Brightens the photo before painting so a dark picture is not crushed against black paint; the painting comes out that much lighter than the photo", &mut p.lift_ev, 0.0, 3.0, 1.0, " EV", false);
        if touched {
            self.evo.prep_touched = Some(std::time::Instant::now());
        }
        num_hint(ui, "Shape following", "At 100% every stroke is built from the shapes in the photo: it starts where the canvas is most wrong, runs along the contour there with one colour and ends where that colour stops fitting. Lower values mix in random tries that only chase the error", &mut self.evo.intent, 0.0, 1.0, 100.0, " %", false);
        num_hint(ui, "Follow edges", "How tightly a stroke bends to follow the contour it is on: 0% keeps it straight, 100% turns with every curve", &mut self.evo.follow, 0.0, 1.0, 100.0, " %", false);
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
            if ui.add_enabled(has_photo, paint).on_hover_text("Paint the photo over whatever is on the canvas with the selected tool, paint and size until you stop it. Strokes vary in colour and between 40 and 100% of the tool size. Change the tool, size or paint and press Paint again to carry on; use File > New painting for a fresh canvas").clicked() {
                self.start_evolve();
            }
        }
        if let Some(ev) = &self.evo.run {
            let mono = |s: String| RichText::new(s).small();
            ui.add(egui::ProgressBar::new(ev.progress()).desired_height(14.0).text(RichText::new(format!("{:.0}% like the photo", ev.progress() * 100.0)).small()));
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
}
