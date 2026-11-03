use super::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Params {
    pub size: [u32; 2],
    pub tiles: [u32; 2],
    pub dx: f32,
    pub dt: f32,
    pub inv_dx: f32,
    pub frame_dt: f32,
    pub g: [f32; 4],
    pub phys: [f32; 4],
    pub ein: [f32; 4],
    pub cure: [f32; 4],
    pub misc: [f32; 4],
    pub flags: [u32; 4],
    pub front: [f32; 4],
    pub front2: [f32; 4],
    pub front3: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct Contact {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub r: f32,
    pub zb: f32,
    pub bristle: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct RubberParams {
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub radius: f32,
    pub weave: f32,
    pub active: u32,
    pub pad: u32,
    pub refl: [u32; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct KnifeParams {
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub dir: [f32; 2],
    pub sweep: f32,
    pub height: f32,
    pub half_width: f32,
    pub active: u32,
    pub frame: u32,
    pub pad1: u32,
    pub bands: u32,
    pub pad2: u32,
    pub pad3: u32,
    pub pad4: u32,
    pub pig: [[u32; 8]; 8],
    pub fill: [f32; 8],
    pub smear: [f32; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct Scrap {
    pub cell: u32,
    pub vol: f32,
    pub frame: u32,
    pub pad1: u32,
    pub pig: [u32; 8],
}

#[derive(Clone, Debug)]
pub struct FilmSettings {
    pub width_m: f32,
    pub height_m: f32,
    pub rho: f32,
    pub sigma: f32,
    pub n_index: f32,
    pub m_cap: f32,
    pub contact_angle_deg: f32,
    pub hysteresis_deg: f32,
    pub h_star: f32,
    pub raw_canvas: bool,
    pub canvas_rgb: Option<[f32; 3]>,
    pub weave_um: f32,
    pub cure_rate: f32,
    pub tau_cure: f32,
    pub lnmu_cure: f32,
    pub v_cap: f32,
    pub tilt_deg: f32,
    pub tilt_dir_deg: f32,
    pub gravity: f32,
    pub medium_mu: f32,
    pub debug_view: u32,
    pub h_min: f32,
    pub m_cap_hydro: f32,
    pub max_substeps: u32,
    pub min_substeps: u32,
    pub implicit_iters: u32,
    pub implicit_hydro: bool,
    pub slump: bool,
    pub capillary: bool,
    pub capillary_implicit: bool,
}

impl Default for FilmSettings {
    fn default() -> Self {
        Self {
            width_m: 1.2,
            height_m: 0.8,
            rho: 1000.0,
            sigma: 0.035,
            n_index: 1.0,
            m_cap: 1.0e-8,
            contact_angle_deg: 45.0,
            hysteresis_deg: 15.0,
            h_star: 2.0e-5,
            raw_canvas: true,
            canvas_rgb: None,
            weave_um: 25.0,
            cure_rate: 1.0 / 600.0,
            tau_cure: 300.0,
            lnmu_cure: 5.0,
            v_cap: 1.0,
            tilt_deg: 0.0,
            tilt_dir_deg: -90.0,
            gravity: 9.81,
            medium_mu: 0.08,
            debug_view: 0,
            h_min: 1.5e-5,
            m_cap_hydro: 1.0e-7,
            max_substeps: 64,
            min_substeps: 4,
            implicit_iters: 40,
            implicit_hydro: true,
            slump: false,
            capillary: true,
            capillary_implicit: false,
        }
    }
}
