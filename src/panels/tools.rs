use super::*;

impl App {
    pub(super) fn tool_settings(&mut self, ui: &mut egui::Ui) {
        if self.tool.kind == ToolKind::Roller {
            self.roller_settings(ui);
            return;
        }
        if self.tool.kind == ToolKind::Sponge {
            self.sponge_settings(ui);
            return;
        }
        if self.tool.kind == ToolKind::Rubber {
            num(ui, "Size", &mut self.rubber_size, crate::RUBBER_MIN, crate::RUBBER_MAX, 1e3, " mm", true);
            ui.add(egui::Label::new(RichText::new("Drag to remove everything under the rubber, wet or dry, right down to the bare canvas and its weave. The wheel sets the size, and a pen clears a wider path the harder you press. Wet paint beside the cleared area can still flow back in.").small().weak()).wrap());
            return;
        }
        if self.tool.kind == ToolKind::Brush {
            self.brush_settings(ui);
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
            ToolKind::Brush | ToolKind::Roller | ToolKind::Sponge | ToolKind::Rubber => {}
        }
    }

    pub(super) fn sponge_settings(&mut self, ui: &mut egui::Ui) {
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

    pub(super) fn roller_settings(&mut self, ui: &mut egui::Ui) {
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

    pub(super) fn brush_settings(&mut self, ui: &mut egui::Ui) {
        let mut shape = self.brush.shape;
        prop(ui, "Shape", |ui| {
            ui.selectable_value(&mut shape, brush::BrushShape::Round, "Round");
            ui.selectable_value(&mut shape, brush::BrushShape::Flat, "Flat");
        });
        let mut size = self.brush.size() * 1e3;
        let mut len = self.brush.len * 1e3;
        let mut n = self.brush.requested;
        num_hint(ui, "Pressure", "How hard the brush presses. The mouse wheel changes it; a pen sets it directly.", &mut self.brush_pressure, 0.0, 1.0, 100.0, " %", false);
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
}
