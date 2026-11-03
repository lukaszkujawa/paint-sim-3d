use crate::app::*;
use crate::script::*;
use crate::tools::ToolKind;
use crate::{brush, evolve};

impl App {
    pub(super) fn scene_evolve(&mut self, _a: &AutoScript) -> bool {
        let args: Vec<String> = std::env::args().collect();
        let arg = |k: &str| args.windows(2).find(|p| p[0] == k).map(|p| p[1].clone());
        if let Some(t) = arg("--tool") {
            self.tool.kind = match t.as_str() {
                "stick" => ToolKind::Stick,
                "bottle" => ToolKind::Bottle,
                "bucket" => ToolKind::Bucket,
                "brush" => ToolKind::Brush,
                "roller" => ToolKind::Roller,
                "sponge" => ToolKind::Sponge,
                "knife" => ToolKind::Knife,
                _ => ToolKind::Pour,
            };
        }
        if self.tool.kind == ToolKind::Brush {
            self.tool.height = self.brush.len * 0.75;
            self.prev_kind = ToolKind::Brush;
        }
        if args.iter().any(|a| a == "--flat") {
            let (n, len) = (self.brush.requested, self.brush.len);
            self.rebuild_brush(brush::BrushShape::Flat, n, len, 0.03);
        }
        if let Some(mm) = arg("--size").and_then(|v| v.parse::<f64>().ok()) {
            let kind = self.tool.kind;
            self.set_tool_size(kind, mm * 1e-3);
        }
        if let Some(path) = arg("--target") {
            self.load_photo(std::path::Path::new(&path));
        }
        if let Some(v) = arg("--canvas") {
            let c: Vec<f32> = v.split(',').filter_map(|t| t.parse().ok()).collect();
            let colour = (c.len() >= 3).then(|| [c[0], c[1], c[2]]);
            self.start_canvas(CanvasChoice { res: self.film.w, colour, weave: arg("--noweave").is_none(), custom: [0.9; 3], first: false });
        }
        if let Some(spec) = arg("--focus") {
            let (kind, nums) = spec.split_once(':').unwrap_or((spec.as_str(), ""));
            let v: Vec<f32> = nums.split(',').filter_map(|x| x.parse().ok()).collect();
            if v.len() >= 4 {
                let f = &mut self.evo.focus;
                f.shape = if kind == "circle" { evolve::FocusShape::Circle } else { evolve::FocusShape::Rect };
                (f.a, f.b) = ([v[0], v[1]], [v[2], v[3]]);
                f.soft = v.get(4).copied().unwrap_or(f.soft);
                f.outside = v.get(5).copied().unwrap_or(f.outside);
            }
        }
        if let Some(v) = arg("--thin").and_then(|v| v.parse::<f32>().ok()) {
            self.paint.frac[12] = v / 100.0;
        }
        if let Some(v) = arg("--contrast").and_then(|v| v.parse::<f32>().ok()) {
            self.evo.prep.contrast = v / 100.0;
        }
        if let Some(v) = arg("--blur").and_then(|v| v.parse::<f32>().ok()) {
            self.evo.prep.blur_mm = v;
        }
        if let Some(v) = arg("--smooth").and_then(|v| v.parse::<f32>().ok()) {
            self.evo.prep.smooth_mm = v;
        }
        if let Some(v) = arg("--lift").and_then(|v| v.parse::<f32>().ok()) {
            self.evo.prep.lift_ev = v;
        }
        if let Some(v) = arg("--sharpen").and_then(|v| v.parse::<f32>().ok()) {
            self.evo.prep.sharpen = v / 100.0;
        }
        if let Some(v) = arg("--intent").and_then(|v| v.parse::<f64>().ok()) {
            self.evo.intent = v / 100.0;
        }
        if let Some(v) = arg("--longest").and_then(|v| v.parse::<f64>().ok()) {
            self.evo.longest = v / 100.0;
        }
        if let Some(v) = arg("--speed").and_then(|v| v.parse().ok()) {
            self.painter_speed = v;
        }
        if let Some(n) = arg("--candidates").and_then(|v| v.parse().ok()) {
            self.evo.candidates = n;
        }
        self.paused = false;
        self.start_evolve();
        if let Some(ev) = self.evo.run.as_mut() {
            log::warn!("evolving {}x{} eval grid from {}", ev.w, ev.h, self.evo.photo.as_ref().map(|p| p.name.clone()).unwrap_or_default());
        }
        if let Some(a) = self.auto.as_mut() {
            a.script = "evolving".to_string();
        }
        self.evolve_tick()
    }
}
