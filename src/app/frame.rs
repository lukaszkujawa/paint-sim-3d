use crate::film::{Deposit, KnifeParams, Method, Scrap};
use crate::tools::{self, ToolKind};
use crate::{color, gizmo};
use crate::app::*;
use glam::DVec3;
use std::time::Instant;








const PEN_WIDTH_MIN: f32 = 0.35;

const PEN_FLOW_MAX: f64 = 1.5;

const PEN_FLOW_MIN: f64 = 0.1;

const MAX_STEP_SPEED: f32 = 2.0;












pub(crate) fn fit_viewport(rect: [f32; 4], size: [f32; 2]) -> [f32; 4] {
    let x = rect[0].clamp(0.0, size[0] - 1.0);
    let y = rect[1].clamp(0.0, size[1] - 1.0);
    [x, y, rect[2].max(1.0).min(size[0] - x), rect[3].max(1.0).min(size[1] - y)]
}

impl App {
    pub(crate) fn encode_sim(&mut self, enc: &mut wgpu::CommandEncoder, deposits: &[Deposit], knife: Option<KnifeParams>, frame_dt: f32) -> [Option<usize>; 3] {
        let mut slots = [None; 3];
        if self.paused {
            self.held_deposits.extend_from_slice(deposits);
            return slots;
        }
        self.film.poll_stats(&self.gpu, &mut self.surface.data, &mut self.surface.sub);
        if self.held_deposits.is_empty() {
            let sent = self.film.upload_deposits(&self.gpu, deposits);
            self.held_deposits.extend_from_slice(&deposits[sent..]);
        } else {
            let mut all = std::mem::take(&mut self.held_deposits);
            all.extend_from_slice(deposits);
            let sent = self.film.upload_deposits(&self.gpu, &all);
            self.held_deposits.extend_from_slice(&all[sent..]);
        }
        self.film.set_knife(&self.gpu, knife);
        let seg = self.rubber_seg.take();
        self.film.set_rubber(&self.gpu, seg);
        let substeps = self.film.choose_substeps(frame_dt);
        self.film.update_params(&self.gpu, frame_dt, substeps);
        self.film.flush_journal(&self.gpu);
        self.film.encode_frame(enc, substeps);
        slots[0] = self.film.request_stats(&self.gpu, enc);
        slots[1] = self.film.request_scraps(enc);
        let tag = match self.tool.kind {
            ToolKind::Brush => Some(0),
            ToolKind::Roller => Some(1),
            ToolKind::Sponge => Some(2),
            _ => None,
        };
        if let Some(t) = tag {
            slots[2] = self.film.request_reservoir(enc, t);
        }
        self.sim_time += frame_dt;
        slots
    }

    pub(crate) fn map_slots(&mut self, slots: [Option<usize>; 3]) {
        if let Some(k) = slots[0] {
            self.film.map_stats(k);
        }
        if let Some(k) = slots[1] {
            self.film.map_scraps(k);
        }
        if let Some(k) = slots[2] {
            self.film.map_reservoir(k);
        }
    }

    pub(crate) fn frame(&mut self) {
        let driven = self.painter.as_ref().map(|p| !p.done).unwrap_or(false) || self.evolving();
        let steps = if driven { (self.painter_speed / MAX_STEP_SPEED).ceil().max(1.0) as u32 } else { 1 };
        let scale = if driven { self.painter_speed / steps as f32 } else { 1.0 };
        let size = self.window.inner_size();
        let surface_tex = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                self.gpu.resize(size.width, size.height);
                return;
            }
        };
        for _ in 1..steps {
            self.step(scale, steps, None);
        }
        self.step(scale, steps, Some(surface_tex));
    }

    pub(crate) fn frame_timing(&mut self, present: bool, scale: f32, steps: u32) -> (f32, f32) {
        if present && self.auto.as_ref().map(|a| a.pace).unwrap_or(false) {
            let el = self.last.elapsed();
            if el < std::time::Duration::from_micros(16600) {
                std::thread::sleep(std::time::Duration::from_micros(16600) - el);
            }
        }
        let now = Instant::now();
        let real_dt = (now - self.last).as_secs_f32().min(0.1);
        if present {
            self.last = now;
            self.fps = 0.9 * self.fps + 0.1 / real_dt.max(1e-4);
        }
        let frame_dt = (1.0 / 60.0) * scale;
        self.sim_dt = frame_dt as f64;
        if present && self.auto_quality && !self.paused {
            let ms = match self.film.clock.as_ref().filter(|c| c.valid) {
                Some(c) => steps as f32 * c.sim_ms + c.render_ms,
                None => (real_dt * 1e3 / steps as f32 - self.last_cpu_ms).max(0.0),
            };
            let s = &mut self.film.settings;
            if ms > self.frame_budget_ms * 1.15 {
                s.max_substeps = (s.max_substeps as f32 * 0.85).floor().max(8.0) as u32;
            } else if ms < self.frame_budget_ms * 0.7 {
                s.max_substeps = (s.max_substeps + 2).min(128);
            }
        }

        (frame_dt, real_dt)
    }

    pub(crate) fn remember_tool_height(&mut self) {
        if self.tool.kind != self.height_kind {
            let from = self.height_kind;
            self.tool_heights.retain(|(k, _)| *k != from);
            self.tool_heights.push((from, self.tool.height));
            self.height_kind = self.tool.kind;
            self.prev_tool_pos = None;
            if self.auto.is_none() && !self.evolving() && !self.painter_active {
                if let Some(&(_, h)) = self.tool_heights.iter().find(|(k, _)| *k == self.tool.kind) {
                    self.tool.height = h;
                    if self.tool.kind == ToolKind::Brush {
                        self.prev_kind = ToolKind::Brush;
                    }
                }
            }
        }
    }

    pub(crate) fn pour_frame(&mut self, want_pour: bool, pen: Option<f64>, frame_dt: f32, deposits: &mut Vec<Deposit>) {
        if self.tool.kind == ToolKind::Pour {
            if want_pour && !self.paused {
                if let Some(p) = self.cam.plane_hit(self.ndc(), 0.0) {
                    let p0 = self.last_pour.unwrap_or(p);
                    let vol = self.pour_rate * frame_dt * self.tool.flow_scale as f32;
                    let radius = self.pour_radius * pen.map(|p| PEN_WIDTH_MIN + (1.0 - PEN_WIDTH_MIN) * p as f32).unwrap_or(1.0);
                    let pig = self.paint.normalized().packed();
                    deposits.push(Deposit::stroke([p0.x, p0.y], [p.x, p.y], radius, vol, pig, self.film.dx).laid_by(Method::Pour));
                    self.last_pour = Some(p);
                }
            } else {
                self.last_pour = None;
            }
        } else {
            self.last_pour = None;
        }
    }

    pub(crate) fn track_cursor(&mut self, scripted: bool, want_pour: bool, frame_dt: f32) {
        if !scripted {
            let pick_z = if matches!(self.tool.kind, ToolKind::Brush | ToolKind::Roller | ToolKind::Sponge | ToolKind::Rubber) { 0.0 } else { self.tool.height as f32 };
            let general = if self.tool.kind == ToolKind::Knife { None } else { self.cam.plane_hit(self.ndc(), pick_z).or_else(|| self.cam.plane_hit(self.ndc(), 0.0)) };
            if let Some(p) = general {
                let p = DVec3::new(p.x as f64, p.y as f64, self.tool.height);
                let v = match self.prev_tool_pos {
                    Some(q) => (p - q) / frame_dt as f64,
                    None => DVec3::ZERO,
                };
                self.tool.vel = self.tool.vel * 0.5 + v * 0.5;
                self.tool.pos = p;
                self.prev_tool_pos = Some(p);
            }
            self.tool.flowing = want_pour && self.tool.kind != ToolKind::Pour;
            if self.tool.kind == ToolKind::Knife {
                if let Some(p) = self.cam.plane_hit(self.ndc(), 0.0) {
                    let p = DVec3::new(p.x as f64, p.y as f64, 0.0);
                    let v = match self.prev_tool_pos { Some(q) => (p - q) / frame_dt as f64, None => DVec3::ZERO };
                    self.tool.vel = v;
                    self.tool.pos = p;
                    self.prev_tool_pos = Some(p);
                }
            }
        } else {
            self.prev_tool_pos = None;
        }
    }

    pub(crate) fn upload_meshes(&mut self) {
        let colors: Vec<[f32; 3]> = (0..color::NPIG).map(|i| color::masstone_rgb(&self.film.pigments, &self.film.cmf_table, i)).collect();
        let color_of = |pig: &[u32; 8]| -> [f32; 3] {
            let mut c = [0.0f32; 3];
            let mut w = 0.0;
            for k in 0..6 {
                let a = half::f16::from_bits((pig[k] & 0xffff) as u16).to_f32();
                let b = half::f16::from_bits((pig[k] >> 16) as u16).to_f32();
                for (f, i) in [(a, 2 * k), (b, 2 * k + 1)] {
                    if i < color::NPIG && f > 0.0 {
                        let lin = |x: f32| x.powf(2.2);
                        c[0] += f * lin(colors[i][0]);
                        c[1] += f * lin(colors[i][1]);
                        c[2] += f * lin(colors[i][2]);
                        w += f;
                    }
                }
            }
            if w > 0.0 { [c[0] / w, c[1] / w, c[2] / w] } else { [0.5, 0.5, 0.5] }
        };
        let (mut tv, mut ti) = tools::tube_mesh(&self.air.rods, &self.air.drops, &color_of);
        let solid = ti.len();
        if self.auto.is_none() && (self.painter_active || self.evolving() || !self.cursor_over_panel()) {
            let ground = if self.painter_active || self.evolving() { Some(DVec3::new(self.tool.pos.x, self.tool.pos.y, 0.0)) } else { self.cam.plane_hit(self.ndc(), 0.0).map(|g| DVec3::new(g.x as f64, g.y as f64, 0.0)) };
            let gx = gizmo::GizmoCtx {
                ground,
                knife_dir: glam::DVec2::new(self.knife_dir[0], self.knife_dir[1]),
                paint: color_of(&self.paint.normalized().packed()),
                px: self.cam.dist as f64 * 0.0012,
                knife_fill: (self.knife_load() / self.knife_capacity().max(1e-12)).clamp(0.0, 1.0) as f32,
                knife_paint: color_of(&color::PaintMix { frac: self.knife_mix() }.normalized().packed()),
                sponge_radius: 0.5 * self.sponge.size,
                rubber_radius: 0.5 * self.rubber_size,
                knife_gap: self.tool.knife_height * self.pen().map(|p| 1.0 - PEN_KNIFE_CLOSE * p).unwrap_or(1.0),
                pour_radius: self.pour_radius as f64 * self.pen().map(|p| PEN_WIDTH_MIN as f64 + (1.0 - PEN_WIDTH_MIN as f64) * p).unwrap_or(1.0),
            };
            gizmo::tool_gizmo(&mut tv, &mut ti, &self.tool, &gx);
        }
        if self.tool.kind == ToolKind::Brush {
            let pc = |comp: &[f32; 16]| color_of(&color::PaintMix { frac: *comp }.normalized().packed());
            self.brush.mesh(&mut tv, &mut ti, &pc, (self.cam.dist as f64 * 0.0012).max(1.5e-4));
        }
        if self.tool.kind == ToolKind::Roller {
            let pc = |comp: &[f32; 16]| color_of(&color::PaintMix { frac: *comp }.normalized().packed());
            self.roller.mesh(&mut tv, &mut ti, self.tool.pos, &pc);
        }
        if self.tool.kind == ToolKind::Sponge {
            let pc = |comp: &[f32; 16]| color_of(&color::PaintMix { frac: *comp }.normalized().packed());
            self.sponge.mesh(&mut tv, &mut ti, self.tool.pos, &pc);
        }
        self.tubes.upload(&self.gpu, &tv, &ti, solid);
    }

    fn draw_passes(&mut self, enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, vr: [f32; 4], prims: &[egui::ClippedPrimitive], screen: &egui_wgpu::ScreenDescriptor) {
        {
            let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: match self.film.prof.as_mut() {
                    Some(p) => p.render_writes("render"),
                    None => self.film.clock.as_ref().and_then(|c| c.render(true)),
                },
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            pass.set_viewport(vr[0], vr[1], vr[2], vr[3], 0.0, 1.0);
            self.renderer.draw(&mut pass);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tubes"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.tubes.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: self.film.prof.as_mut().and_then(|p| p.render_writes("tubes")),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(vr[0], vr[1], vr[2], vr[3], 0.0, 1.0);
            self.tubes.draw(&mut pass);
        }
        {
            let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: self.film.clock.as_ref().and_then(|c| c.render(false)),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            self.ui.renderer.render(&mut pass, prims, screen);
        }
    }

    fn air_frame(&mut self, frame_dt: f32, deposits: &mut Vec<Deposit>) {
        if !self.paused {
            let g = self.film.gravity_vec();
            let gravity = DVec3::new(g[0] as f64, g[1] as f64, -(g[2] as f64));
            self.surface.decay_recent();
            let bounds = [self.film.settings.width_m as f64, self.film.settings.height_m as f64];
            self.air.update(&mut self.tool, frame_dt as f64, &mut self.surface, gravity, self.film.dx, deposits, bounds);
        }
    }

    fn build_ui(&mut self, size: winit::dpi::PhysicalSize<u32>) -> egui::FullOutput {
        let raw = self.ui.state.take_egui_input(&self.window);
        self.ui.ctx.begin_pass(raw);
        if self.auto.is_none() || self.auto.as_ref().map(|a| a.show_ui).unwrap_or(false) {
            self.ui_panel();
            if !self.view_fitted && self.view_rect[2] > 1.0 {
                self.view_fitted = true;
                self.set_top_view();
            }
        } else {
            self.view_rect = [0.0, 0.0, size.width as f32, size.height.max(1) as f32];
        }
        self.ui.ctx.end_pass()
    }

    fn log_rods(&self) {
        for r in &self.air.rods {
            let zmin = r.x.iter().map(|x| x.z).fold(1.0f64, f64::min);
            let rmax = (0..r.n().saturating_sub(1)).map(|e| r.edge_radius(e)).fold(0.0f64, f64::max);
            let fz = r.frozen.iter().filter(|f| **f).count();
            let lens: Vec<String> = (0..r.n().saturating_sub(1)).map(|e| format!("{:.1}/{:.2}/{:.2}", (r.x[e + 1] - r.x[e]).length() * 1e3, r.edge_radius(e) * 1e3, r.vol[e] * 1e6)).collect();
            log::warn!("rod {} n {} nozzle {} vol {:.3} mL frozen {} zmin {:.4} rmax {:.2} mm edges(L mm/r mm/V uL) {:?}", r.id, r.n(), r.nozzle.is_some(), r.total_volume() * 1e6, fz, zmin, rmax * 1e3, lens);
        }
    }

    pub(crate) fn step(&mut self, scale: f32, steps: u32, surface_tex: Option<wgpu::SurfaceTexture>) {
        let size = self.window.inner_size();
        let size = winit::dpi::PhysicalSize::new(size.width.max(1), size.height.max(1));
        let present = surface_tex.is_some();
        let (frame_dt, real_dt) = self.frame_timing(present, scale, steps);
        let t_tool = Instant::now();
        let mut deposits: Vec<Deposit> = Vec::new();
        if self.auto.as_ref().map(|a| a.script.is_empty()).unwrap_or(false) {
            self.scripted_deposits(&mut deposits);
        }
        let want_pour = self.mouse_down[0] && !self.cursor_over_panel() && !self.evolving();
        if want_pour && !self.stroke_down {
            self.begin_stroke();
        }
        self.stroke_down = want_pour;
        self.remember_tool_height();
        let pen = self.pen();
        self.tool.flow_scale = pen.map(|p| PEN_FLOW_MIN + (PEN_FLOW_MAX - PEN_FLOW_MIN) * p).unwrap_or(1.0);
        let mut scraps: Vec<Scrap> = Vec::new();
        self.film.poll_scraps(&self.gpu, &mut scraps);
        self.pour_frame(want_pour, pen, frame_dt, &mut deposits);
        self.scripted_input(size);
        let scripted = self.scripted_tool();
        if let Some(a) = &self.auto {
            if a.dry > 0 && self.frame_index == a.dry as u64 {
                self.film.dry_now(&self.gpu);
            }
            if a.stop > 0 && self.frame_index >= a.stop {
                let was = self.tool.flowing;
                self.tool.flowing = false;
                if was { self.air.detach(); }
            }
        }
        deposits.append(&mut self.pending_deposits);
        self.track_cursor(scripted, want_pour, frame_dt);
        self.paint.frac[14] = (self.tool.mu as f32).ln() + 10.0;
        self.film.settings.sigma = self.tool.sigma as f32;
        self.tool.paint = self.paint;
        self.rubber_frame(pen);
        let knife = self.knife_frame(pen, frame_dt, &mut deposits);
        self.absorb_scraps(&scraps);
        self.absorb_reservoir();
        let mut contacts = Vec::new();
        self.brush_frame(pen, frame_dt, &mut deposits, &mut contacts);
        self.roller_and_sponge_frame(pen, frame_dt, &mut deposits, &mut contacts);
        self.film.upload_contacts(&self.gpu, &contacts);
        self.prev_kind = self.tool.kind;
        let ms_tool = t_tool.elapsed().as_secs_f32() * 1e3;
        let t_air = Instant::now();
        self.air_frame(frame_dt, &mut deposits);
        let ms_air = t_air.elapsed().as_secs_f32() * 1e3;
        let air_detail = format!("(rods {:.1} land {:.1} drops {:.1} caps {} steps {})", self.air.prof[0], self.air.prof[1], self.air.prof[2], self.surface.caps.len(), self.air.rod_steps);
        let t_mesh = Instant::now();
        self.upload_meshes();
        let ms_mesh = t_mesh.elapsed().as_secs_f32() * 1e3;
        let nverts: usize = self.air.rods.iter().map(|r| r.n()).sum();

        let Some(surface_tex) = surface_tex else {
            let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("step") });
            let slots = self.encode_sim(&mut enc, &deposits, knife, frame_dt);
            match self.film.prof.as_mut() {
                Some(p) => p.finish(&self.gpu, enc),
                None => {
                    self.gpu.queue.submit(std::iter::once(enc.finish()));
                }
            }
            self.map_slots(slots);
            self.frame_index += 1;
            return;
        };
        let t_ui = Instant::now();
        let mut out = self.build_ui(size);
        let mut textures_delta = std::mem::take(&mut out.textures_delta);
        self.ui.state.handle_platform_output(&self.window, std::mem::take(&mut out.platform_output));
        let prims = self.ui.ctx.tessellate(std::mem::take(&mut out.shapes), out.pixels_per_point);
        let ms_ui = t_ui.elapsed().as_secs_f32() * 1e3;

        let vr = fit_viewport(self.view_rect, [size.width as f32, size.height as f32]);
        self.cam.aspect = vr[2] / vr[3];
        self.renderer.update(&self.gpu, &self.cam, &self.film, vr[3] as u32);

        let view = surface_tex.texture.create_view(&Default::default());
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        let t_enc = Instant::now();
        if let Some(c) = self.film.clock.as_mut() {
            c.poll();
            c.begin();
        }
        let slots = self.encode_sim(&mut enc, &deposits, knife, frame_dt);
        let ms_enc = t_enc.elapsed().as_secs_f32() * 1e3;
        let (gpu_sim, gpu_render) = self.film.clock.as_ref().map(|c| (c.sim_ms, c.render_ms)).unwrap_or((0.0, 0.0));
        self.prof = format!("tool {:.1} ms  air {:.1} ms {}  mesh {:.1} ms  ui {:.1} ms  encode {:.1} ms  gpu sim {:.1} ms  render {:.1} ms  frame {:.1} ms  rod verts {}  deposits {}", ms_tool, ms_air, air_detail, ms_mesh, ms_ui, ms_enc, gpu_sim, gpu_render, real_dt * 1e3, nverts, deposits.len());
        self.last_cpu_ms = ms_tool + ms_air + ms_mesh + ms_ui + ms_enc;
        if self.auto.as_ref().map(|a| a.script.starts_with("stress") || a.script == "pollock").unwrap_or(false) && (self.frame_index % 300 == 0 || (real_dt > 0.05 && self.frame_index % 20 == 0)) {
            let vmax = self.air.rods.iter().flat_map(|r| r.v.iter()).map(|v| v.length()).fold(0.0f64, f64::max);
            log::warn!("vmax {:.2} m/s", vmax);
            log::warn!("f{} {:.0} fps sub {} rods {} drops {} {}", self.frame_index, self.fps, self.film.substeps_last, self.air.rods.len(), self.air.drops.len(), self.prof);
        }

        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [size.width, size.height], pixels_per_point: out.pixels_per_point };
        for (id, deltas) in &textures_delta.set {
            for delta in deltas.iter() {
                self.ui.renderer.update_texture(&self.gpu.device, &self.gpu.queue, *id, delta);
            }
        }
        let cmds = self.ui.renderer.update_buffers(&self.gpu.device, &self.gpu.queue, &mut enc, &prims, &screen);
        self.draw_passes(&mut enc, &view, vr, &prims, &screen);
        for id in &textures_delta.free {
            self.ui.renderer.free_texture(id);
        }
        textures_delta.clear();
        if let Some(c) = self.film.clock.as_ref() {
            c.resolve(&mut enc);
        }
        match self.film.prof.as_mut() {
            Some(p) => {
                self.gpu.queue.submit(cmds);
                p.finish(&self.gpu, enc);
            }
            None => {
                self.gpu.queue.submit(cmds.into_iter().chain(std::iter::once(enc.finish())));
            }
        }
        if let Some(c) = self.film.clock.as_mut() {
            c.map();
        }
        self.map_slots(slots);
        self.frame_index += 1;
        self.autosave_tick(false);
        let exit = self.finish_auto();
        if let Some(path) = self.shot_request.take() {
            self.log_rods();
            self.save_screenshot(&path, &surface_tex.texture);
            self.compute_stats();
        }
        self.gpu.queue.present(surface_tex);
        if exit {
            std::process::exit(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fit_viewport;

    #[test]
    fn viewport_stays_inside_the_window() {
        assert_eq!(fit_viewport([100.0, 40.0, 500.0, 300.0], [800.0, 600.0]), [100.0, 40.0, 500.0, 300.0]);
        assert_eq!(fit_viewport([700.0, 500.0, 500.0, 300.0], [800.0, 600.0]), [700.0, 500.0, 100.0, 100.0]);
        assert_eq!(fit_viewport([-5.0, -5.0, 0.0, 0.0], [800.0, 600.0]), [0.0, 0.0, 1.0, 1.0]);
    }
}
