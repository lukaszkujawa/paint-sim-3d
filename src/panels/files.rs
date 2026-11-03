use super::*;

impl App {
    pub(crate) fn save_current(&mut self) {
        if self.history.name.is_empty() {
            self.save_dialog = Some(String::new());
        } else {
            let name = self.history.name.clone();
            self.save_painting(&name);
        }
    }

    pub(super) fn open_gallery(&mut self) {
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

    pub(super) fn canvas_window(&mut self, ctx: &egui::Context) {
        let Some(mut c) = self.canvas_dialog.take() else { return };
        const SIZES: [(u32, &str); 5] = [(1536, "Draft"), (2048, "Quick"), (3072, "Standard"), (4096, "Fine"), (4992, "Ultra")];
        const COLOURS: [(&str, Option<[f32; 3]>); 8] = [
            ("Raw cotton", None),
            ("Primed white", Some([0.94, 0.93, 0.90])),
            ("Warm ivory", Some([0.93, 0.88, 0.78])),
            ("Grey", Some([0.56, 0.56, 0.55])),
            ("Ochre ground", Some([0.74, 0.56, 0.33])),
            ("Terracotta", Some([0.64, 0.37, 0.26])),
            ("Blue grey", Some([0.50, 0.56, 0.62])),
            ("Black", Some([0.08, 0.08, 0.09])),
        ];
        let width_mm = self.film.settings.width_m * 1e3;
        let mut start = false;
        let mut keep = true;
        let title = if c.first { "Welcome to Canvaz" } else { "New painting" };
        if let Some(r) = egui::Window::new(title).collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            ui.set_width(380.0);
            ui.label(RichText::new("Resolution").strong());
            let aspect = self.film.settings.height_m / self.film.settings.width_m;
            let room = self.gpu.device.limits().max_storage_buffer_binding_size as f64;
            for (res, name) in SIZES.into_iter().filter(|(r, _)| *r as f64 * (*r as f32 * aspect) as f64 * 128.0 <= room) {
                let cell = width_mm / res as f32;
                let note = match res {
                    r if r <= 2048 => "fast, loose work",
                    3072 => "balanced",
                    4096 => "finer marks, slower",
                    _ => "finest detail, needs a strong GPU",
                };
                ui.radio_value(&mut c.res, res, format!("{}  {}x{}  ·  {:.2} mm cells  ·  {}", name, res, (res as f32 * self.film.settings.height_m / self.film.settings.width_m) as u32, cell, note));
            }
            ui.add_space(8.0);
            ui.label(RichText::new("Canvas").strong());
            egui::Grid::new("canvas_colours").spacing([6.0, 6.0]).show(ui, |ui| {
                for (k, (name, col)) in COLOURS.iter().enumerate() {
                    let rgb = col.unwrap_or([0.86, 0.80, 0.70]);
                    let fill = Color32::from_rgb((rgb[0] * 255.0) as u8, (rgb[1] * 255.0) as u8, (rgb[2] * 255.0) as u8);
                    let selected = c.colour == *col;
                    let btn = egui::Button::new(RichText::new(*name).small().color(if rgb.iter().sum::<f32>() > 1.5 { Color32::BLACK } else { Color32::WHITE })).fill(fill).min_size(Vec2::new(86.0, 34.0)).stroke(if selected { Stroke::new(2.5, ACCENT) } else { Stroke::new(1.0, Color32::from_gray(90)) });
                    if ui.add(btn).clicked() {
                        c.colour = *col;
                        c.weave = col.is_none() || c.weave;
                    }
                    if k % 4 == 3 {
                        ui.end_row();
                    }
                }
            });
            ui.horizontal(|ui| {
                let custom = c.colour.is_some() && !COLOURS.iter().any(|(_, col)| *col == c.colour);
                if ui.radio(custom, "Custom colour").clicked() {
                    c.colour = Some(c.custom);
                }
                if ui.color_edit_button_rgb(&mut c.custom).changed() {
                    c.colour = Some(c.custom);
                }
            });
            ui.checkbox(&mut c.weave, "Canvas weave texture");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new(RichText::new("Start painting").strong().color(Color32::WHITE)).fill(ACCENT).min_size(Vec2::new(150.0, 28.0))).clicked() {
                    start = true;
                }
                if ui.button(if c.first { "Use defaults" } else { "Cancel" }).clicked() {
                    keep = false;
                }
            });
            if c.res > 3072 {
                ui.label(RichText::new("Larger canvases run slower while paint is wet. Saved paintings open at the resolution they were made with.").small().weak());
            }
        }) {
            self.block_rects.push(r.response.rect);
            self.menu_open = true;
        }
        if start {
            self.start_canvas(c);
        } else if keep {
            self.canvas_dialog = Some(c);
        }
    }

    pub(super) fn file_windows(&mut self, ctx: &egui::Context) {
        self.canvas_window(ctx);
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
}
