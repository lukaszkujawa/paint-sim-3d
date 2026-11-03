mod app;
mod brush;
mod color;
mod evolve;
mod film;
mod gizmo;
mod gpu;
mod panels;
mod pollock;
mod render;
mod rod;
mod roller;
mod script;
mod session;
mod sponge;
mod tools;

pub(crate) use app::{App, CanvasChoice, KNIFE_HOLD_PER_WIDTH, RUBBER_MAX, RUBBER_MIN};
use app::Pending;
use script::pollock_stats;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args: Vec<String> = std::env::args().collect();
    if let Some(n) = args.windows(2).find(|p| p[0] == "--pollock-stats").and_then(|p| p[1].parse::<u64>().ok()) {
        pollock_stats(n);
        return;
    }
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut pending = Pending { window: None, app: None };
    event_loop.run_app(&mut pending).expect("run");
}
