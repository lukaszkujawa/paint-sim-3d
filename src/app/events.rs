use crate::tools::ToolKind;
use crate::app::*;
use glam::Vec3;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const PEN_STALE_S: f32 = 0.3;

pub(crate) struct Pending {
    pub(crate) window: Option<Arc<Window>>,
    pub(crate) app: Option<App>,
}

impl ApplicationHandler for Pending {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("Canvaz").with_window_icon(app_icon()).with_inner_size(winit::dpi::LogicalSize::new(1600.0, 1000.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("window"));
        self.window = Some(window.clone());
        self.app = Some(App::new(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(app) = self.app.as_mut() else { return };
        let _resp = app.ui.state.on_window_event(&app.window, &event);
        match event {
            WindowEvent::CloseRequested => {
                app.autosave_tick(true);
                event_loop.exit();
            }
            WindowEvent::DroppedFile(path) => app.load_photo(&path),
            WindowEvent::Resized(s) => { app.gpu.resize(s.width, s.height); app.tubes.resize(&app.gpu); }
            WindowEvent::RedrawRequested => {
                app.frame();
                app.window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                if app.pen_down {
                    if app.pen_last.elapsed().as_secs_f32() < PEN_STALE_S {
                        return;
                    }
                    app.pen_down = false;
                    app.pen_raw = 0.0;
                    app.mouse_down[0] = false;
                    app.pen_cut = true;
                }
                let prev = app.cursor;
                app.cursor = [position.x as f32, position.y as f32];
                let dx = app.cursor[0] - prev[0];
                let dy = app.cursor[1] - prev[1];
                if app.mouse_down[1] {
                    app.cam.yaw -= dx * 0.005;
                    app.cam.pitch = (app.cam.pitch + dy * 0.005).clamp(0.1, 1.55);
                }
                if app.mouse_down[2] {
                    let eye = app.cam.eye();
                    let fwd = (app.cam.target - eye).normalize();
                    let right = fwd.cross(Vec3::Z).normalize();
                    let up = right.cross(fwd);
                    let k = app.cam.dist * 0.0012;
                    app.cam.target -= right * dx * k;
                    app.cam.target += up * dy * k;
                }
            }
            WindowEvent::Touch(t) => {
                app.pen_last = Instant::now();
                app.cursor = [t.location.x as f32, t.location.y as f32];
                if let Some(f) = t.force {
                    app.pen_raw = f.normalized() as f32;
                    app.pen_seen = true;
                }
                match t.phase {
                    TouchPhase::Started => {
                        if t.force.is_none() {
                            app.pen_raw = 1.0;
                        }
                        if app.eyedropper {
                            app.eyedrop_click();
                        } else if !app.cursor_over_panel() {
                            app.pen_down = true;
                            app.mouse_down[0] = true;
                        }
                    }
                    TouchPhase::Ended | TouchPhase::Cancelled => {
                        if app.pen_down {
                            app.mouse_down[0] = false;
                        }
                        app.pen_down = false;
                        app.pen_raw = 0.0;
                        app.pen_cut = false;
                    }
                    TouchPhase::Moved => {
                        if app.pen_cut && !app.pen_down && !app.cursor_over_panel() {
                            app.pen_down = true;
                            app.mouse_down[0] = true;
                            app.pen_cut = false;
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let i = match button {
                    MouseButton::Left => 0,
                    MouseButton::Right => 1,
                    MouseButton::Middle => 2,
                    _ => return,
                };
                if state == ElementState::Pressed {
                    if app.cursor_over_panel() {
                        return;
                    }
                    if app.eyedropper && i == 0 {
                        app.eyedrop_click();
                        return;
                    }
                    app.mouse_down[i] = true;
                } else {
                    app.mouse_down[i] = false;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if app.cursor_over_panel() {
                    return;
                }
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };
                if app.mouse_down[1] || app.ctrl_down || app.tool.kind == ToolKind::Pour {
                    app.cam.dist = (app.cam.dist * (1.0 - d * 0.1)).clamp(0.05, 6.0);
                } else if app.tool.kind == ToolKind::Roller {
                    app.roller.pressure = (app.roller.pressure + d as f64 * 0.05).clamp(0.0, 1.0);
                } else if app.tool.kind == ToolKind::Brush {
                    app.brush_pressure = (app.brush_pressure + d as f64 * 0.05).clamp(0.0, 1.0);
                } else if app.tool.kind == ToolKind::Rubber {
                    app.rubber_size = (app.rubber_size * (1.0 + d as f64 * 0.1)).clamp(RUBBER_MIN, RUBBER_MAX);
                } else if app.tool.kind == ToolKind::Sponge {
                    app.sponge.pressure = (app.sponge.pressure + d as f64 * 0.05).clamp(0.0, 1.0);
                } else {
                    app.tool.height = (app.tool.height * (1.0 + d as f64 * 0.1)).clamp(0.002, 1.0);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if matches!(event.physical_key, PhysicalKey::Code(KeyCode::ControlLeft) | PhysicalKey::Code(KeyCode::ControlRight)) {
                    app.ctrl_down = event.state == ElementState::Pressed;
                }
                if event.state == ElementState::Pressed && !app.ui.ctx.egui_wants_keyboard_input() {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => app.paused = !app.paused,
                        PhysicalKey::Code(KeyCode::Escape) if app.eyedropper => app.eyedropper = false,
                        PhysicalKey::Code(KeyCode::Escape) => app.stop_evolve(),
                        PhysicalKey::Code(KeyCode::KeyI) => app.eyedropper = !app.eyedropper,
                        PhysicalKey::Code(KeyCode::Tab) => app.show_inspector = !app.show_inspector,
                        PhysicalKey::Code(KeyCode::KeyT) => app.set_top_view(),
                        PhysicalKey::Code(KeyCode::KeyB) => app.beauty_render(true),
                        PhysicalKey::Code(KeyCode::Digit1) => app.tool.kind = ToolKind::Brush,
                        PhysicalKey::Code(KeyCode::Digit2) => app.tool.kind = ToolKind::Roller,
                        PhysicalKey::Code(KeyCode::Digit3) => app.tool.kind = ToolKind::Knife,
                        PhysicalKey::Code(KeyCode::Digit4) => app.tool.kind = ToolKind::Stick,
                        PhysicalKey::Code(KeyCode::Digit5) => app.tool.kind = ToolKind::Bottle,
                        PhysicalKey::Code(KeyCode::Digit6) => app.tool.kind = ToolKind::Bucket,
                        PhysicalKey::Code(KeyCode::Digit7) => app.tool.kind = ToolKind::Pour,
                        PhysicalKey::Code(KeyCode::Digit8) => app.tool.kind = ToolKind::Sponge,
                        PhysicalKey::Code(KeyCode::Digit9) => app.tool.kind = ToolKind::Rubber,
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            match app.tool.kind {
                                ToolKind::Brush => app.brush.dip(&app.paint.frac),
                                ToolKind::Roller => app.roller.dip(&app.paint.frac),
                                ToolKind::Sponge => app.sponge.dip(&app.paint.frac),
                                ToolKind::Knife => app.load_knife(),
                                _ => app.tool.reload(),
                            }
                        }
                        PhysicalKey::Code(KeyCode::KeyD) => {
                            app.begin_full();
                            app.film.dry_now(&app.gpu);
                        }
                        PhysicalKey::Code(KeyCode::KeyZ) if app.ctrl_down => app.undo(),
                        PhysicalKey::Code(KeyCode::KeyS) if app.ctrl_down => app.save_current(),
                        PhysicalKey::Code(KeyCode::Equal) => app.cam.dist = (app.cam.dist * 0.8).max(0.05),
                        PhysicalKey::Code(KeyCode::Minus) => app.cam.dist = (app.cam.dist * 1.25).min(6.0),
                        PhysicalKey::Code(KeyCode::F12) => app.shot_request = Some(format!("shot_{}.png", app.frame_index)),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

pub(crate) fn app_icon() -> Option<winit::window::Icon> {
    let img = image::load_from_memory(include_bytes!("../../assets/icon.png")).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    winit::window::Icon::from_rgba(img.into_raw(), w, h).ok()
}
