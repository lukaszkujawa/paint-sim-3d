use super::*;

const LOAD_MAX: f64 = 10.0;

const LOAD_SIZE: f64 = 0.016;

const ROUND_LEAD: f64 = 0.06;

const POUR_FILM: f64 = 2.0e-4;

const THREAD_SETTLE_MAX: u32 = 180;

const SETTLE_FRAMES: u32 = 2;

const LIFT_FRAMES: u32 = 1;

const LOWER_FRAMES: u32 = 1;

const BRISTLE_MIN: f64 = 0.006;

const BRISTLE_PER_SIZE: f64 = 5.0;

const PREP_SETTLE: f32 = 0.35;

const KNIFE_STREAK_DARK: f32 = 0.05;

const KNIFE_STREAK_LIGHT: f32 = 0.25;

const KNIFE_LOAD_MARGIN: f64 = 0.8;

const PRESS_JITTER: f64 = 0.15;

const TRAVEL_PER_SIZE: f64 = 1.0;

const APPROACH_STEPS: f64 = 4.0;

const PRESS_MIN: f64 = 0.0005;

const PRESS_PER_SIZE: f64 = 1.0;

const HOVER: f64 = 0.004;

pub struct EvolveState {
    pub photo: Option<Photo>,
    pub focus: Focus,
    pub run: Option<Evolver>,
    pub(super) fitted: Option<(f32, f32)>,
    pub candidates: usize,
    pub follow: f64,
    pub intent: f64,
    pub prep: Prep,
    pub prep_touched: Option<std::time::Instant>,
    pub dry_passes: bool,
    pub longest: f64,
    pub active: bool,
    pub(super) photo_gen: u64,
    pub(super) saved_paint: Option<[f32; 16]>,
    pub(super) saved_height: Option<f64>,
    pub(super) saved_load: Option<f64>,
    pub(super) saved_len: Option<f64>,
    pub(super) user_sizes: Vec<(ToolKind, BrushShape, f64)>,
    pub(super) stroke_press: f64,
}

impl EvolveState {
    pub fn new() -> Self {
        Self { photo: None, focus: Focus { shape: FocusShape::Off, a: [0.5, 0.5], b: [0.75, 0.5], soft: 0.5, outside: 0.0 }, run: None, fitted: None, candidates: 32, follow: 0.7, intent: 0.7, prep: Prep::none(), prep_touched: None, dry_passes: true, longest: 0.25, active: false, photo_gen: 0, saved_paint: None, saved_height: None, saved_load: None, saved_len: None, user_sizes: Vec::new(), stroke_press: 0.0 }
    }
}

impl App {
    pub(crate) fn evolving(&self) -> bool {
        self.evo.active
    }

    pub(crate) fn tool_size(&self, kind: ToolKind) -> f64 {
        match kind {
            ToolKind::Pour => self.pour_radius as f64,
            ToolKind::Brush => self.brush.size(),
            ToolKind::Roller => self.roller.width,
            ToolKind::Sponge => self.sponge.size,
            ToolKind::Stick => self.tool.stick_diameter,
            ToolKind::Bottle => self.tool.bottle_radius,
            ToolKind::Bucket => self.tool.bucket_radius,
            ToolKind::Knife => self.tool.knife_width,
            ToolKind::Rubber => self.rubber_size,
        }
    }

    pub(crate) fn set_tool_size(&mut self, kind: ToolKind, size: f64) {
        match kind {
            ToolKind::Pour => self.pour_radius = size as f32,
            ToolKind::Brush => {
                if (self.brush.size() - size).abs() > 0.02 * size {
                    let (shape, n, len) = (self.brush.shape, self.brush.requested, self.brush.len);
                    self.rebuild_brush(shape, n, len, size);
                }
            }
            ToolKind::Roller => self.roller.width = size,
            ToolKind::Sponge => self.sponge.size = size,
            ToolKind::Stick => self.tool.stick_diameter = size,
            ToolKind::Bottle => self.tool.bottle_radius = size,
            ToolKind::Bucket => self.tool.bucket_radius = size,
            ToolKind::Knife => self.tool.knife_width = size,
            ToolKind::Rubber => self.rubber_size = size,
        }
    }

    pub(crate) fn load_photo(&mut self, path: &Path) {
        let aspect = self.film.settings.width_m / self.film.settings.height_m;
        match load_photo(path, aspect) {
            Ok(p) => {
                self.notice(&format!("Loaded {}", p.name));
                self.evo.photo = Some(p);
                self.evo.photo_gen += 1;
            }
            Err(e) => self.notice(&format!("Could not open that image: {}", e)),
        }
    }

    pub(crate) fn pick_photo(&mut self) {
        let file = rfd::FileDialog::new().set_title("Choose a photo to paint").add_filter("Images", &["jpg", "jpeg", "png", "webp", "bmp"]).pick_file();
        if let Some(path) = file {
            self.load_photo(&path);
        }
    }

    pub(crate) fn start_evolve(&mut self) {
        let Some(photo) = &self.evo.photo else { return };
        if let Some(p) = self.painter.as_mut() {
            p.stop();
        }
        let gen = self.evo.photo_gen;
        match self.evo.run.as_mut().filter(|ev| ev.photo_gen == gen) {
            Some(ev) => ev.resume(),
            None => {
                let mut ev = Evolver::new(&photo.image, self.film.w, self.film.h, self.film.dx, &self.film.pigments, &self.film.cmf_table, self.auto.as_ref().map(|a| a.seed as u64).unwrap_or_else(rand::random), self.evo.prep);
                ev.photo_gen = gen;
                self.evo.run = Some(ev);
            }
        }
        self.begin_full();
        self.tool.flowing = false;
        self.air.detach();
        self.evo.saved_paint = Some(self.paint.frac);
        self.evo.saved_height = Some(self.tool.height);
        self.evo.saved_len = Some(self.brush.len);
        self.evo.saved_load = Some(self.brush.capacity / self.brush.base_capacity.max(1e-30));
        self.evo.user_sizes.clear();
        self.evo.active = true;
    }

    pub(crate) fn stop_evolve(&mut self) {
        if !self.evo.active {
            return;
        }
        self.evo.active = false;
        self.brush_raise = 0.0;
        self.tool.flowing = false;
        self.tool.vel = DVec3::ZERO;
        self.air.detach();
        if let Some(f) = self.evo.saved_paint.take() {
            self.paint.frac = f;
        }
        let shape = self.brush.shape;
        for (kind, s, size) in std::mem::take(&mut self.evo.user_sizes) {
            if kind != ToolKind::Brush || s == shape {
                self.set_tool_size(kind, size);
            }
        }
        if let Some(len) = self.evo.saved_len.take() {
            if (self.brush.len - len).abs() > 1e-6 {
                let (shape, n, size) = (self.brush.shape, self.brush.requested, self.brush.size());
                self.rebuild_brush(shape, n, len, size);
            }
        }
        if let Some(k) = self.evo.saved_load.take() {
            self.brush.capacity = self.brush.base_capacity * k;
        }
        if let Some(h) = self.evo.saved_height.take() {
            self.tool.height = h;
        }
    }

    pub(crate) fn restore_exposure(&mut self) {
        self.renderer.exposure = crate::render::DEFAULT_EXPOSURE;
        self.evo.fitted = None;
        self.stop_evolve();
        self.evo.run = None;
    }

    pub(crate) fn evolve_tick(&mut self) -> bool {
        if !self.evo.active {
            return false;
        }
        let Some(mut ev) = self.evo.run.take() else { return false };
        if !self.paused {
            let phase = std::mem::replace(&mut ev.phase, Phase::Plan);
            ev.phase = self.evolve_phase(&mut ev, phase);
        }
        self.evo.run = Some(ev);
        true
    }

    pub(super) fn evolve_phase(&mut self, ev: &mut Evolver, phase: Phase) -> Phase {
        let z = self.tool.height;
        match phase {
            Phase::Wait(n) => {
                self.tool.flowing = false;
                if n == 0 {
                    Phase::Observe
                } else {
                    Phase::Wait(n - 1)
                }
            }
            Phase::Observe => {
                self.tool.flowing = false;
                let rgb = self.read_linear(ev, [0, 0, ev.w, ev.h]);
                if !ev.calibrated {
                    if let Some((user, fitted)) = self.evo.fitted.take() {
                        if (self.renderer.exposure - fitted).abs() < 1e-4 {
                            self.renderer.exposure = user;
                        }
                    }
                    let user = self.renderer.exposure;
                    let unit = self.display_gain(ev, &rgb).map(|g| g * (-user).exp2());
                    let fit = ev.range_fit(unit);
                    ev.fit_ev = fit;
                    if fit < 0.0 && fit < user {
                        self.renderer.exposure = fit;
                        self.evo.fitted = Some((user, fit));
                        self.notice(&format!("Exposure set to {:.1} EV so paint covers this photo's tones; change it under Render", fit));
                    }
                    let base = unit.map(|g| g * fit.exp2());
                    ev.base_gain = base;
                    let gain = base.map(|g| g * (-ev.prep.lift_ev).exp2());
                    log::info!("display gain {:.2} {:.2} {:.2}", gain[0], gain[1], gain[2]);
                    ev.calibrate(gain, [0.0; 3]);
                }
                let show = ev.base_gain;
                ev.observe(rgb.into_par_iter().map(|c| view_lab(c, show)).collect());
                Phase::Plan
            }
            Phase::Plan => {
                self.tool.flowing = false;
                self.tool.vel = DVec3::ZERO;
                let settled = self.evo.prep_touched.map(|t| t.elapsed().as_secs_f32() > PREP_SETTLE).unwrap_or(true);
                if ev.prep != self.evo.prep && settled {
                    ev.set_prep(self.evo.prep);
                    return Phase::Observe;
                }
                let (kind, shape) = (self.tool.kind, self.brush.shape);
                if let Some(size) = ev.set_tool(kind, shape, self.tool_size(kind)) {
                    self.evo.user_sizes.retain(|u| u.0 != kind || u.1 != shape);
                    self.evo.user_sizes.push((kind, shape, size));
                }
                if !ev.has_tool() {
                    ev.status = "This tool cannot paint from a photo; pick another tool to keep painting".to_string();
                    return Phase::Plan;
                }
                ev.status.clear();
                match ev.next(self.evo.candidates, self.evo.follow, self.evo.intent, self.evo.longest, self.evo.focus, &self.paint.frac) {
                    None if !ev.synced() => Phase::Observe,
                    None => {
                        if ev.finished {
                            ev.status = "Nothing found to gain at this size right now; still looking. Lower the tool size for detail, or change the tool or paint".to_string();
                        }
                        Phase::Plan
                    }
                    Some(mut plan) => {
                        if plan.dry && self.evo.dry_passes {
                            self.begin_full();
                            self.film.dry_now(&self.gpu);
                        }
                        self.prepare_tool(&plan);
                        ev.applied(self.tool_size(plan.kind));
                        self.begin_stroke();
                        let (_, dir) = plan.entry();
                        if plan.kind == ToolKind::Brush {
                            self.brush.lean = DVec3::new(dir.x, dir.y, 0.0);
                            if self.brush.shape == BrushShape::Round {
                                let shift = dir * (ROUND_LEAD * self.brush.len);
                                for q in plan.path.iter_mut() {
                                    *q -= shift;
                                }
                            }
                        }
                        let (from, _) = plan.entry();
                        self.tool.pos = DVec3::new(from.x, from.y, z);
                        Phase::Lower(plan, 0)
                    }
                }
            }
            Phase::Lower(plan, n) => {
                let (from, _) = plan.entry();
                self.tool.pos = DVec3::new(from.x, from.y, z);
                let thread = is_thread(plan.kind);
                self.tool.flowing = thread;
                let fall = if thread { ((2.0 * z / 9.81).sqrt() / self.sim_dt.max(1e-6)).ceil() as u32 } else { 0 };
                if n + 1 < LOWER_FRAMES.max(fall) {
                    Phase::Lower(plan, n + 1)
                } else if plan.lead > 0.0 {
                    Phase::Approach(plan, 0.0)
                } else {
                    Phase::Stroke(plan, 0.0)
                }
            }
            Phase::Approach(plan, s) => {
                let s1 = (s + (plan.lead / APPROACH_STEPS).min(plan.speed * self.sim_dt)).min(plan.lead);
                let (from, dir) = plan.entry();
                let p = from + dir * s1;
                self.tool.pos = DVec3::new(p.x, p.y, z);
                self.tool.vel = DVec3::new(dir.x, dir.y, 0.0) * plan.speed;
                self.tool.flowing = true;
                self.brush_raise = (self.brush.len - self.tool.height).max(0.0) + HOVER * (1.0 - s1 / plan.lead);
                if s1 >= plan.lead {
                    self.brush_raise = 0.0;
                    Phase::Stroke(plan, 0.0)
                } else {
                    Phase::Approach(plan, s1)
                }
            }
            Phase::Exit(plan, s) => {
                let s1 = (s + (plan.lead / APPROACH_STEPS).min(plan.speed * self.sim_dt)).min(plan.lead);
                let dir = plan.exit_dir();
                let p = *plan.path.last().unwrap() + dir * s1;
                self.tool.pos = DVec3::new(p.x, p.y, z);
                self.tool.vel = DVec3::new(dir.x, dir.y, 0.0) * plan.speed;
                self.tool.flowing = true;
                self.brush_raise = self.brush_hover() * (s1 / plan.lead);
                if s1 >= plan.lead {
                    Phase::Lift(plan, 0)
                } else {
                    Phase::Exit(plan, s1)
                }
            }
            Phase::Stroke(plan, s) => {
                let total = plan.length();
                let travel = if plan.kind == ToolKind::Brush { (plan.speed * self.sim_dt).min(TRAVEL_PER_SIZE * plan.size) } else { plan.speed * self.sim_dt };
                let s1 = (s + travel).min(total);
                let (p0, p1) = (plan.ahead(s), plan.ahead(s1));
                let z = if plan.kind == ToolKind::Brush {
                    self.tool.height = self.brush.len - self.evo.stroke_press * press_profile(s1, total, plan.size);
                    self.tool.height
                } else {
                    z
                };
                self.tool.pos = DVec3::new(p1.x, p1.y, z);
                self.tool.vel = DVec3::new(p1.x - p0.x, p1.y - p0.y, 0.0) / self.sim_dt.max(1e-6);
                self.tool.flowing = plan.kind != ToolKind::Pour;
                if plan.kind == ToolKind::Pour {
                    let r = plan.size;
                    let pig = color::PaintMix { frac: plan.mix }.normalized().packed();
                    let n = ((p1 - p0).length() / r).ceil().max(1.0) as usize;
                    for k in 0..n {
                        let a = p0 + (p1 - p0) * (k as f64 / n as f64);
                        let b = p0 + (p1 - p0) * ((k + 1) as f64 / n as f64);
                        let seg = (b - a).length().max(if s == 0.0 && k == 0 { r } else { 0.0 });
                        let vol = (seg * 2.0 * r * POUR_FILM) as f32;
                        self.pending_deposits.push(Deposit::stroke([a.x as f32, a.y as f32], [b.x as f32, b.y as f32], r as f32, vol, pig, self.film.dx).laid_by(Method::Pour));
                    }
                }
                if s1 < total {
                    Phase::Stroke(plan, s1)
                } else if plan.lead > 0.0 {
                    Phase::Exit(plan, 0.0)
                } else {
                    Phase::Lift(plan, 0)
                }
            }
            Phase::Lift(plan, n) => {
                self.tool.flowing = false;
                self.tool.vel = DVec3::ZERO;
                self.brush_raise = 0.0;
                if n == 0 && is_thread(plan.kind) {
                    self.air.detach();
                }
                if n + 1 >= LIFT_FRAMES {
                    Phase::Settle(plan, 0)
                } else {
                    Phase::Lift(plan, n + 1)
                }
            }
            Phase::Settle(plan, n) => {
                self.tool.flowing = false;
                let airborne = !self.air.rods.is_empty() || !self.air.drops.is_empty();
                if n + 1 < SETTLE_FRAMES || (airborne && n < THREAD_SETTLE_MAX) {
                    return Phase::Settle(plan, n + 1);
                }
                let region = ev.region(&plan);
                let after = self.read_eval(ev, region);
                if !ev.judge(&plan, region, &after) && self.rollback().is_err() {
                    ev.store(region, &after);
                }
                Phase::Plan
            }
        }
    }

    pub(super) fn brush_hover(&self) -> f64 {
        (self.brush.len - self.tool.height).max(0.0) + HOVER
    }

    pub(super) fn prepare_tool(&mut self, plan: &Plan) {
        let mut f = plan.mix;
        f[14] = (self.tool.mu as f32).ln() + 10.0;
        for i in 0..13 {
            self.paint.frac[i] = plan.mix[i];
        }
        match plan.kind {
            ToolKind::Pour => self.pour_radius = plan.size as f32,
            ToolKind::Brush => {
                let user_len = self.evo.saved_len.unwrap_or(self.brush.len);
                let len = (BRISTLE_PER_SIZE * plan.size).clamp(BRISTLE_MIN.min(user_len), user_len);
                if (self.brush.size() - plan.size).abs() > 0.02 * plan.size || (self.brush.len - len).abs() > 0.02 * len {
                    let (shape, n) = (self.brush.shape, self.brush.requested);
                    self.rebuild_brush(shape, n, len, plan.size);
                }
                let press = (PRESS_PER_SIZE * plan.size).clamp(PRESS_MIN, self.brush_press(self.brush_pressure).max(PRESS_MIN));
                let press = press * (1.0 + PRESS_JITTER * (2.0 * rand::random::<f64>() - 1.0)) + self.brush.touch_depth();
                self.evo.stroke_press = press;
                self.tool.height = self.brush.len - press * PRESS_START;
                self.brush.capacity = self.brush.base_capacity * (plan.size / LOAD_SIZE).clamp(1.0, LOAD_MAX);
                self.brush.clean();
                self.brush.dip(&f);
            }
            ToolKind::Roller => {
                self.roller.width = plan.size;
                self.roller.clean();
                self.roller.dip(&f);
            }
            ToolKind::Sponge => {
                self.sponge.size = plan.size;
                self.sponge.clean();
                self.sponge.dip(&f);
            }
            ToolKind::Stick => {
                self.tool.stick_diameter = plan.size;
                self.tool.reload();
            }
            ToolKind::Rubber => {}
            ToolKind::Bottle => self.tool.bottle_radius = plan.size,
            ToolKind::Bucket => self.tool.bucket_radius = plan.size,
            ToolKind::Knife => {
                self.tool.knife_width = plan.size;
                self.load_knife();
                let need = KNIFE_LOAD_MARGIN * self.tool.knife_height.max(1e-4) * plan.size * plan.length() + self.tool.knife_width * crate::KNIFE_HOLD_PER_WIDTH;
                let each = (need / self.knife_bands.len() as f64).min(self.knife_bands[0].vol);
                for b in self.knife_bands.iter_mut() {
                    b.vol = each;
                    let t = (2.0 * rand::random::<f32>() - 1.0).powi(3);
                    let total: f32 = b.comp[..12].iter().sum::<f32>().max(1e-6);
                    if t > 0.0 {
                        b.comp[WHITE] += KNIFE_STREAK_LIGHT * t * total;
                    } else {
                        b.comp[BLACK] += KNIFE_STREAK_DARK * -t * total;
                    }
                }
                let d = plan.entry().1;
                self.knife_dir = [d.x, d.y];
                self.knife_track = None;
                self.knife_prev = None;
            }
        }
    }

    pub(super) fn display_gain(&mut self, ev: &Evolver, albedo: &[[f32; 3]]) -> [f32; 3] {
        self.refine.run(&self.gpu, &self.film);
        let agx = std::mem::replace(&mut self.renderer.agx, false);
        let shown = self.beauty.render(&self.gpu, &self.renderer, &self.cam, &self.film, true, ev.w as u32, ev.h as u32, 1, Some(&self.refine));
        self.renderer.agx = agx;
        let mut ratios: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for j in (0..ev.h).step_by(3) {
            for i in (0..ev.w).step_by(3) {
                let p = shown.get_pixel(i as u32, (ev.h - 1 - j) as u32);
                let a = albedo[j * ev.w + i];
                for c in 0..3 {
                    let y = decode(p[c] as f32 / 255.0);
                    if y < 0.93 && a[c] > 0.02 {
                        ratios[c].push(tonemap_inverse(y) / a[c]);
                    }
                }
            }
        }
        let mut gain = [1.0f32; 3];
        for c in 0..3 {
            if !ratios[c].is_empty() {
                ratios[c].sort_by(|a, b| a.total_cmp(b));
                gain[c] = ratios[c][ratios[c].len() / 2].clamp(0.2, 20.0);
            }
        }
        gain
    }

    pub(crate) fn dump_observed(&self, path: &str) {
        let Some(ev) = self.evo.run.as_ref() else { return };
        let gain = ev.base_gain;
        let rgb = self.read_linear(ev, [0, 0, ev.w, ev.h]);
        let mut im = image::RgbImage::new(ev.w as u32, ev.h as u32);
        for (k, p) in rgb.iter().enumerate() {
            let px = [0, 1, 2].map(|c| (encode(tonemap(p[c] * gain[c])) * 255.0) as u8);
            im.put_pixel((k % ev.w) as u32, (ev.h - 1 - k / ev.w) as u32, image::Rgb(px));
        }
        let _ = im.save(path);
    }

    pub(super) fn read_eval(&self, ev: &Evolver, region: [usize; 4]) -> Vec<[f32; 3]> {
        let show = ev.base_gain;
        self.read_linear(ev, region).into_par_iter().map(|c| view_lab(c, show)).collect()
    }

    pub(super) fn read_linear(&self, ev: &Evolver, region: [usize; 4]) -> Vec<[f32; 3]> {
        let f = ev.factor as usize;
        let (rw, rh) = (region[2] - region[0], region[3] - region[1]);
        if rw == 0 || rh == 0 {
            return Vec::new();
        }
        let (cw, ch) = ((rw * f) as u32, (rh * f) as u32);
        let bpr = ((cw * 8 + 255) / 256) * 256;
        let buf = self.gpu.readback("evolve_read", (bpr * ch) as u64);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("evolve_read") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &self.film.color_tex, mip_level: 0, origin: wgpu::Origin3d { x: (region[0] * f) as u32, y: (region[1] * f) as u32, z: 0 }, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(ch) } },
            wgpu::Extent3d { width: cw, height: ch, depth_or_array_layers: 1 },
        );
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range().expect("map");
        let halves: &[u16] = bytemuck::cast_slice(&data);
        let row = (bpr / 2) as usize;
        let inv = 1.0 / (f * f) as f32;
        let out = (0..rw * rh)
            .into_par_iter()
            .map(|k| {
                let (i, j) = (k % rw, k / rw);
                let mut rgb = [0.0f32; 3];
                for y in j * f..(j + 1) * f {
                    for x in i * f..(i + 1) * f {
                        let o = y * row + x * 4;
                        for c in 0..3 {
                            rgb[c] += half::f16::from_bits(halves[o + c]).to_f32();
                        }
                    }
                }
                rgb.map(|v| v * inv)
            })
            .collect();
        drop(data);
        buf.unmap();
        out
    }
}
