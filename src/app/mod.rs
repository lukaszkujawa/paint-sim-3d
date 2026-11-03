mod brushes;
mod canvas;
mod capture;
mod core;
mod events;
mod frame;
mod knife;
mod pointer;
mod stats;
mod tool_frame;

pub(crate) use self::core::*;
pub(crate) use self::events::*;
pub(crate) use self::knife::*;

pub(crate) const RUBBER_MIN: f64 = 0.002;
pub(crate) const RUBBER_MAX: f64 = 0.3;
pub(crate) const KNIFE_HOLD_PER_WIDTH: f64 = 1.2e-5;
pub(crate) const KNIFE_BANDS: usize = 8;
pub(crate) const PEN_BRUSH_PRESS: f64 = 0.25;
pub(crate) const PEN_KNIFE_CLOSE: f64 = 0.9;
