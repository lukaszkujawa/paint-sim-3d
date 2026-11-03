use super::*;

const TEXT_MUTED: Color32 = Color32::from_rgb(138, 143, 151);

const LABEL_W: f32 = 104.0;

pub(super) fn prop<R>(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
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

pub(super) fn num<N: egui::emath::Numeric>(ui: &mut egui::Ui, label: &str, v: &mut N, lo: f64, hi: f64, scale: f64, suffix: &str, log: bool) -> bool {
    prop(ui, label, |ui| {
        let mut d = v.to_f64() * scale;
        let r = ui.add(egui::Slider::new(&mut d, lo * scale..=hi * scale).clamping(egui::SliderClamping::Edits).logarithmic(log).suffix(suffix).max_decimals(if N::INTEGRAL { 0 } else if hi * scale >= 100.0 { 1 } else { 2 }));
        if r.changed() {
            *v = N::from_f64(d / scale);
        }
        r.changed()
    })
}

pub(super) fn num_hint<N: egui::emath::Numeric>(ui: &mut egui::Ui, label: &str, hint: &str, v: &mut N, lo: f64, hi: f64, scale: f64, suffix: &str, log: bool) -> bool {
    prop(ui, label, |ui| {
        let mut d = v.to_f64() * scale;
        let r = ui.add(egui::Slider::new(&mut d, lo * scale..=hi * scale).clamping(egui::SliderClamping::Edits).logarithmic(log).suffix(suffix).max_decimals(if N::INTEGRAL { 0 } else if hi * scale >= 100.0 { 1 } else { 2 })).on_hover_text(hint);
        if r.changed() {
            *v = N::from_f64(d / scale);
        }
        r.changed()
    })
}

pub(super) fn section(ui: &mut egui::Ui, title: &str, open: bool, add: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(RichText::new(title).strong().size(13.0)).default_open(open).show(ui, |ui| {
        ui.add_space(2.0);
        add(ui);
        ui.add_space(4.0);
    });
}

pub(super) fn parse_hex(text: &str) -> Option<[f32; 3]> {
    let h = text.trim().trim_start_matches('#');
    if h.len() != 6 || !h.is_ascii() {
        return None;
    }
    let part = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).ok().map(|v| v as f32 / 255.0);
    Some([part(0)?, part(2)?, part(4)?])
}

pub(super) fn swatch_color(rgb: [f32; 3]) -> Color32 {
    Color32::from_rgb((rgb[0] * 255.0) as u8, (rgb[1] * 255.0) as u8, (rgb[2] * 255.0) as u8)
}

#[cfg(test)]
mod tests {
    use super::parse_hex;

    #[test]
    fn reads_hex_with_or_without_hash() {
        assert_eq!(parse_hex("#ff0000"), Some([1.0, 0.0, 0.0]));
        assert_eq!(parse_hex(" 00FF00 "), Some([0.0, 1.0, 0.0]));
    }

    #[test]
    fn rejects_bad_hex() {
        for bad in ["", "#fff", "#ff00", "#gg0000", "#ff00000", "\u{e9}\u{e9}\u{e9}"] {
            assert_eq!(parse_hex(bad), None, "{bad}");
        }
    }
}
