use crate::color;
use crate::gpu::{compute_pipeline, dispatch, BindBuilder, Gpu};
use bytemuck::{Pod, Zeroable};

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
pub struct Deposit {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub n0: [f32; 2],
    pub n1: [f32; 2],
    pub r0: f32,
    pub r1: f32,
    pub h0: f32,
    pub vol: f32,
    pub kind: u32,
    pub pad: [u32; 3],
    pub pig: [u32; 8],
}

pub const DEPOSIT_RECYCLED: u32 = 1;
pub const DEPOSIT_CAP_START: u32 = 2;
pub const DEPOSIT_CAP_END: u32 = 4;
const DEPOSIT_METHOD_SHIFT: u32 = 8;

#[derive(Clone, Copy)]
pub enum Method {
    Thread = 1,
    Drop = 2,
    Pour = 3,
    Brush = 4,
    Knife = 5,
    Roller = 6,
}

fn segment_dir(p0: [f32; 2], p1: [f32; 2]) -> ([f32; 2], f32) {
    let d = [p1[0] - p0[0], p1[1] - p0[1]];
    let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
    if l > 0.0 { ([d[0] / l, d[1] / l], l) } else { ([1.0, 0.0], 0.0) }
}

impl Deposit {
    pub fn ribbon(p0: [f32; 2], p1: [f32; 2], n0: [f32; 2], n1: [f32; 2], r0: f32, r1: f32, caps: u32, volume: f32, pig: [u32; 8], dx: f32) -> Self {
        let min_r = 0.8 * dx;
        let a = r0.max(min_r);
        let b = r1.max(min_r);
        let mut d = Self { p0, p1, n0, n1, r0: a, r1: b, h0: a, vol: volume, kind: caps, pad: [0; 3], pig };
        let lattice = 0.25 * dx;
        let sampled = d.sampled_volume(lattice);
        let rm = a.max(b);
        let disc = std::f32::consts::PI * rm * rm * rm / 2.0;
        if sampled < 0.25 * disc {
            let mid = [0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])];
            d = Self { p0: mid, p1: mid, n0, n1, r0: rm, r1: rm, h0: rm, vol: volume, kind: caps, pad: [0; 3], pig };
            d.h0 = volume / d.sampled_volume(lattice) * rm;
            return d;
        }
        d.h0 = volume / sampled * a;
        d
    }

    pub fn capsule(p0: [f32; 2], p1: [f32; 2], r: f32, volume: f32, pig: [u32; 8], dx: f32) -> Self {
        let (d, _) = segment_dir(p0, p1);
        Self::ribbon(p0, p1, d, d, r, r, DEPOSIT_CAP_START | DEPOSIT_CAP_END, volume, pig, dx)
    }

    pub fn stroke(p0: [f32; 2], p1: [f32; 2], r: f32, volume: f32, pig: [u32; 8], dx: f32) -> Self {
        let (d, l) = segment_dir(p0, p1);
        if l < r.max(0.8 * dx) {
            return Self::capsule(p0, p1, r, volume, pig, dx);
        }
        Self::ribbon(p0, p1, d, d, r, r, 0, volume, pig, dx)
    }

    pub fn laid_by(mut self, method: Method) -> Self {
        self.kind |= (method as u32) << DEPOSIT_METHOD_SHIFT;
        self
    }

    pub fn recycled(mut self) -> Self {
        self.kind |= DEPOSIT_RECYCLED;
        self
    }

    pub fn max_radius(&self) -> f32 {
        self.r0.max(self.r1)
    }

    pub fn max_height(&self) -> f32 {
        self.h0 * self.max_radius() / self.r0
    }

    fn sampled_volume(&self, lattice: f32) -> f32 {
        self.sampled_volume_in(lattice, [f32::NEG_INFINITY, f32::NEG_INFINITY], [f32::INFINITY, f32::INFINITY])
    }

    fn sampled_volume_in(&self, lattice: f32, lo: [f32; 2], hi: [f32; 2]) -> f32 {
        let rm = self.max_radius() + lattice;
        let x0 = ((self.p0[0].min(self.p1[0]) - rm).max(lo[0]) / lattice).floor() as i32;
        let x1 = ((self.p0[0].max(self.p1[0]) + rm).min(hi[0]) / lattice).ceil() as i32;
        let y0 = ((self.p0[1].min(self.p1[1]) - rm).max(lo[1]) / lattice).floor() as i32;
        let y1 = ((self.p0[1].max(self.p1[1]) + rm).min(hi[1]) / lattice).ceil() as i32;
        let mut sum = 0.0;
        for i in x0..x1 {
            for j in y0..y1 {
                sum += self.height_at([(i as f32 + 0.5) * lattice, (j as f32 + 0.5) * lattice]);
            }
        }
        sum * lattice * lattice
    }

    pub fn inside_fraction(&self, w: f32, h: f32) -> f32 {
        let rb = self.max_radius();
        let full = self.p0[0].min(self.p1[0]) - rb >= 0.0 && self.p0[1].min(self.p1[1]) - rb >= 0.0 && self.p0[0].max(self.p1[0]) + rb <= w && self.p0[1].max(self.p1[1]) + rb <= h;
        if full {
            return 1.0;
        }
        let lattice = 0.25 * self.r0.min(self.r1).max(1e-4);
        let all = self.sampled_volume(lattice);
        if all <= 0.0 {
            return 0.0;
        }
        (self.sampled_volume_in(lattice, [0.0, 0.0], [w, h]) / all).clamp(0.0, 1.0)
    }

    pub fn height_at(&self, p: [f32; 2]) -> f32 {
        let (dir, l) = segment_dir(self.p0, self.p1);
        let q0 = [p[0] - self.p0[0], p[1] - self.p0[1]];
        let q1 = [p[0] - self.p1[0], p[1] - self.p1[1]];
        let len = |q: [f32; 2]| (q[0] * q[0] + q[1] * q[1]).sqrt();
        if l <= 0.0 {
            let x = len(q0) / self.r0;
            return if x < 1.0 { self.h0 * (1.0 - x * x) } else { 0.0 };
        }
        let cap_start = self.kind & DEPOSIT_CAP_START != 0;
        let cap_end = self.kind & DEPOSIT_CAP_END != 0;
        if !cap_start && q0[0] * self.n0[0] + q0[1] * self.n0[1] < 0.0 {
            return 0.0;
        }
        if !cap_end && q1[0] * self.n1[0] + q1[1] * self.n1[1] >= 0.0 {
            return 0.0;
        }
        let u = q0[0] * dir[0] + q0[1] * dir[1];
        if u < -self.r0 || u > l + self.r1 {
            return 0.0;
        }
        let mut dist = (q0[0] * dir[1] - q0[1] * dir[0]).abs();
        if cap_start && u < 0.0 {
            dist = len(q0);
        }
        if cap_end && u > l {
            dist = len(q1);
        }
        let rt = self.r0 + (self.r1 - self.r0) * (u / l).clamp(0.0, 1.0);
        let x = dist / rt;
        if x < 1.0 { self.h0 * (rt / self.r0) * (1.0 - x * x) } else { 0.0 }
    }
}

pub fn substrate_refl(n: usize, raw: bool) -> Vec<u32> {
    let bands: Vec<f32> = (0..16)
        .map(|b| {
            let t = b as f32 / 15.0;
            if raw { 0.50 + 0.16 * t } else { 0.84 + 0.04 * t }
        })
        .collect();
    let mut packed = [0u32; 8];
    for k in 0..8 {
        let a = half::f16::from_f32(bands[2 * k]).to_bits() as u32;
        let b = half::f16::from_f32(bands[2 * k + 1]).to_bits() as u32;
        packed[k] = a | (b << 16);
    }
    let mut v = Vec::with_capacity(n * 8);
    for _ in 0..n {
        v.extend_from_slice(&packed);
    }
    v
}

pub fn weave(w: usize, h: usize, dx: f32, amp: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; w * h];
    if amp <= 0.0 {
        return v;
    }
    let period = 1.4e-3;
    let k = std::f32::consts::TAU / period;
    for j in 0..h {
        let y = (j as f32 + 0.5) * dx;
        for i in 0..w {
            let x = (i as f32 + 0.5) * dx;
            let mut hsh = (i as u32).wrapping_mul(73856093) ^ (j as u32).wrapping_mul(19349663);
            hsh ^= hsh >> 13;
            hsh = hsh.wrapping_mul(0x5bd1e995);
            hsh ^= hsh >> 15;
            let r = (hsh & 0xffff) as f32 / 65535.0 * 2.0 - 1.0;
            let s = (k * x).sin() * (k * y).sin();
            v[j * w + i] = amp * (1.0 + 0.6 * s + 0.4 * r);
        }
    }
    v
}

pub const MAX_CONTACTS: usize = 8192;
pub const MAX_BRISTLES: usize = 512;

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
pub struct KnifeParams {
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub dir: [f32; 2],
    pub sweep: f32,
    pub height: f32,
    pub half_width: f32,
    pub active: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub fill: f32,
    pub pad2: u32,
    pub pad3: u32,
    pub pad4: u32,
    pub pig: [u32; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct Scrap {
    pub cell: u32,
    pub vol: f32,
    pub pad0: u32,
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
            weave_um: 25.0,
            cure_rate: 1.0 / 600.0,
            tau_cure: 300.0,
            lnmu_cure: 5.0,
            v_cap: 1.0,
            tilt_deg: 0.0,
            tilt_dir_deg: 0.0,
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

const FULL_SHADE_EVERY: u32 = 1;

pub struct Film {
    pub w: u32,
    pub h: u32,
    pub dx: f32,
    pub tiles: [u32; 2],
    pub settings: FilmSettings,
    pub params: Params,
    pub params_buf: wgpu::Buffer,
    pub h_buf: wgpu::Buffer,
    pub m_buf: wgpu::Buffer,
    pub rheo: wgpu::Buffer,
    pub lap: wgpu::Buffer,
    pub flux: wgpu::Buffer,
    pub flux_s: wgpu::Buffer,
    pub alpha: wgpu::Buffer,
    pub delta: wgpu::Buffer,
    pub tmp: wgpu::Buffer,
    pub acc_s: wgpu::Buffer,
    pub strata_h: wgpu::Buffer,
    pub strata_pig: wgpu::Buffer,
    pub strata_cure: wgpu::Buffer,
    pub sub_refl: wgpu::Buffer,
    pub pigtab: wgpu::Buffer,
    pub cmf: wgpu::Buffer,
    pub strata_h2: wgpu::Buffer,
    pub strata_pig2: wgpu::Buffer,
    pub alpha_s: wgpu::Buffer,
    pub stats_buf: wgpu::Buffer,
    pub tile_flags: wgpu::Buffer,
    pub tile_list: wgpu::Buffer,
    pub tile_indirect: wgpu::Buffer,
    pub active_count: wgpu::Buffer,
    pub tiles16: [u32; 2],
    bg_tiles: wgpu::BindGroup,
    p_mark: wgpu::ComputePipeline,
    p_compact: wgpu::ComputePipeline,
    p_copy_back: wgpu::ComputePipeline,
    p_shade_full: wgpu::ComputePipeline,
    pub needs_full_shade: bool,
    pub active_tiles: u32,
    frame: u32,
    pub white_refl: Vec<u32>,
    pub knife_buf: wgpu::Buffer,
    pub scraps: wgpu::Buffer,
    pub scrap_count: wgpu::Buffer,
    pub scrap_read: [wgpu::Buffer; 2],
    pub scrap_pending: [std::sync::Arc<std::sync::atomic::AtomicU8>; 2],
    pub knife_active: bool,
    pub contacts: wgpu::Buffer,
    pub ctile_range: wgpu::Buffer,
    pub ctile_items: wgpu::Buffer,
    pub reservoir: wgpu::Buffer,
    pub res_read: [wgpu::Buffer; 2],
    pub res_pending: [std::sync::Arc<std::sync::atomic::AtomicU8>; 2],
    pub contact_count: usize,
    pub pool_buf: wgpu::Buffer,
    pub pool_read: [wgpu::Buffer; 2],
    pub pool_dims: [usize; 2],
    pub stats_read: [wgpu::Buffer; 2],
    pub stats_pending: [std::sync::Arc<std::sync::atomic::AtomicU8>; 2],
    pub stats_frame: usize,
    pub deposits: wgpu::Buffer,
    pub tile_start: wgpu::Buffer,
    pub tile_count: wgpu::Buffer,
    pub tile_items: wgpu::Buffer,
    pub color_tex: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub detail_tex: wgpu::Texture,
    pub detail_view: wgpu::TextureView,
    lay: wgpu::Buffer,
    bg0: wgpu::BindGroup,
    bg1: wgpu::BindGroup,
    bg_fac: wgpu::BindGroup,
    bg_dep: wgpu::BindGroup,
    bg_shade: wgpu::BindGroup,
    bg_knife: wgpu::BindGroup,
    p_knife: wgpu::ComputePipeline,
    bg_brush: wgpu::BindGroup,
    p_brush: wgpu::ComputePipeline,
    p_lap: wgpu::ComputePipeline,
    p_flux: wgpu::ComputePipeline,
    p_limit: wgpu::ComputePipeline,
    p_nose: wgpu::ComputePipeline,
    p_shed: wgpu::ComputePipeline,
    p_slump: wgpu::ComputePipeline,
    p_apply: wgpu::ComputePipeline,
    p_clear_acc: wgpu::ComputePipeline,
    p_split: wgpu::ComputePipeline,
    p_commit_tmp: wgpu::ComputePipeline,
    p_reduce: wgpu::ComputePipeline,
    p_pool: wgpu::ComputePipeline,
    p_strata_limit: wgpu::ComputePipeline,
    p_strata_update: wgpu::ComputePipeline,
    p_strata_fill: wgpu::ComputePipeline,
    p_cg_setup: wgpu::ComputePipeline,
    p_cg_init: wgpu::ComputePipeline,
    p_cg_ia: wgpu::ComputePipeline,
    p_cg_ib: wgpu::ComputePipeline,
    p_cg_ic: wgpu::ComputePipeline,
    p_cg_rinit: wgpu::ComputePipeline,
    p_cg_ra: wgpu::ComputePipeline,
    p_cg_rb: wgpu::ComputePipeline,
    p_cg_finish: wgpu::ComputePipeline,
    p_cg_rmass: wgpu::ComputePipeline,
    p_cg_rescale: wgpu::ComputePipeline,
    p_cg_lap_p: wgpu::ComputePipeline,
    p_cg_lap_h: wgpu::ComputePipeline,
    p_cg_lap_u: wgpu::ComputePipeline,
    cg_bufs: Vec<wgpu::Buffer>,
    pub cg_report: [f32; 4],
    pub implicit_steps: u32,
    p_deposit: wgpu::ComputePipeline,
    p_cure: wgpu::ComputePipeline,
    p_shade: wgpu::ComputePipeline,
    pub max_deposits: usize,
    pub h_max: f32,
    pub s_max: f32,
    pub mu_min: f32,
    pub h_max_pending: f32,
    pub substeps_last: u32,
    pub pigments: color::PigmentSet,
    pub cmf_table: Vec<f32>,
    pub deposit_count: usize,
    pub total_deposited: f64,
    pub weave_volume: f64,
}

impl Film {
    pub fn new(gpu: &Gpu, w: u32, h: u32, settings: FilmSettings) -> Self {
        let n = (w * h) as u64;
        let dx = settings.width_m / w as f32;
        let tiles = [(w + 63) / 64, (h + 63) / 64];
        let ntiles = (tiles[0] * tiles[1]) as u64;
        let max_deposits = 1 << 16;
        let params_buf = gpu.uniform("params", std::mem::size_of::<Params>() as u64);
        let h_buf = gpu.storage("h", n * 4);
        let m_buf = gpu.storage("m", n * 4);
        let rheo = gpu.storage("rheo", n * 8);
        let lap = gpu.storage("pres", n * 8);
        let flux = gpu.storage("flux", n * 16);
        let flux_s = gpu.storage("acc", n * 16);
        let alpha = gpu.storage("alpha", n * 4);
        let delta = gpu.storage("delta", n * 4);
        let tmp = gpu.storage("tmp", n * 4);
        let acc_s = gpu.storage("acc_s", n * 64);
        let strata_h = gpu.storage("strata_h", n * 16);
        let strata_pig = gpu.storage("strata_pig", n * 128);
        let strata_cure = gpu.storage("strata_cure", n * 16);
        let sub_refl = gpu.storage("sub_refl", n * 32);
        let strata_h2 = gpu.storage("strata_h2", n * 16);
        let strata_pig2 = gpu.storage("strata_pig2", n * 128);
        let alpha_s = gpu.storage("alpha_s", n * 16);
        let pigments = color::load_pigments();
        let pigtab_data = color::pigment_table(&pigments);
        let cmf_table = color::cmf_table();
        let pigtab = gpu.storage("pigtab", (pigtab_data.len() * 4) as u64);
        let cmf = gpu.storage("cmf", (cmf_table.len() * 4) as u64);
        gpu.queue.write_buffer(&pigtab, 0, bytemuck::cast_slice(&pigtab_data));
        gpu.queue.write_buffer(&cmf, 0, bytemuck::cast_slice(&cmf_table));
        let stats_buf = gpu.storage("stats", ntiles * 8);
        let tiles16 = [(w + 15) / 16, (h + 15) / 16];
        let nt16 = (tiles16[0] * tiles16[1]) as u64;
        let cg_bufs = vec![
            gpu.storage("cg_m", n * 16),
            gpu.storage("cg_d", n * 4),
            gpu.storage("cg_x", n * 4),
            gpu.storage("cg_r", n * 4),
            gpu.storage("cg_p", n * 4),
            gpu.storage("cg_q", n * 4),
            gpu.storage("cg_part", nt16 * 4),
            gpu.storage("cg_s", 32),
            gpu.storage("cg_part2", nt16 * 4),
            gpu.storage("h_frame", n * 4),
            gpu.storage("fnorm", n * 8),
            gpu.storage("cg_t", n * 4),
            gpu.storage("cg_t2", n * 4),
        ];
        let tile_flags = gpu.storage("tile_flags", nt16 * 4);
        let tile_list = gpu.storage("tile_list", nt16 * 4);
        let tile_indirect = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_indirect"),
            size: 16,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&tile_indirect, 0, bytemuck::cast_slice(&[0u32, 1, 1, 0]));
        let active_count = gpu.storage("active_count", 16);
        let tile_moved = gpu.storage("tile_moved", nt16 * 4);
        let max_scraps = 1u64 << 16;
        let knife_buf = gpu.uniform("knife", std::mem::size_of::<KnifeParams>() as u64);
        let scraps = gpu.storage("scraps", max_scraps * 48);
        let scrap_count = gpu.storage("scrap_count", 16);
        let scrap_read = [gpu.readback("scrap_r0", max_scraps * 48 + 16), gpu.readback("scrap_r1", max_scraps * 48 + 16)];
        let scrap_pending = [std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)), std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0))];
        let pool_dims = [((w + 3) / 4) as usize, ((h + 3) / 4) as usize];
        let npool = (pool_dims[0] * pool_dims[1]) as u64;
        let pool_buf = gpu.storage("pool", npool * 8);
        let pool_read = [gpu.readback("pool_r0", npool * 8), gpu.readback("pool_r1", npool * 8)];
        let contacts = gpu.storage("contacts", (MAX_CONTACTS * std::mem::size_of::<Contact>()) as u64);
        let ctile_range = gpu.storage("ctile_range", ntiles * 8);
        let ctile_items = gpu.storage("ctile_items", (MAX_CONTACTS * 16 * 4) as u64);
        let reservoir = gpu.storage("reservoir", (MAX_BRISTLES * 17 * 4) as u64);
        let res_read = [gpu.readback("res_r0", (MAX_BRISTLES * 17 * 4) as u64), gpu.readback("res_r1", (MAX_BRISTLES * 17 * 4) as u64)];
        let res_pending = [std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)), std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0))];
        let stats_read = [gpu.readback("stats_r0", ntiles * 8 + 48), gpu.readback("stats_r1", ntiles * 8 + 48)];
        let stats_pending = [std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)), std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0))];
        let deposits = gpu.storage("deposits", (max_deposits * std::mem::size_of::<Deposit>()) as u64);
        let tile_start = gpu.storage("tile_start", ntiles * 4);
        let tile_count = gpu.storage("tile_count", ntiles * 4);
        let tile_items = gpu.storage("tile_items", (max_deposits * 16 * 4) as u64);
        let color_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("color"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&Default::default());
        let detail_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("detail"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let detail_view = detail_tex.create_view(&Default::default());
        let lay = gpu.storage("lay", n * 4);

        let white = substrate_refl(n as usize, settings.raw_canvas);
        gpu.queue.write_buffer(&sub_refl, 0, bytemuck::cast_slice(&white));
        let wv = weave(w as usize, h as usize, dx, settings.weave_um * 1e-6);
        let weave_vol = wv.iter().map(|v| *v as f64).sum::<f64>() * (dx as f64) * (dx as f64);
        gpu.queue.write_buffer(&m_buf, 0, bytemuck::cast_slice(&wv));
        let white_refl = white;
        let ones: Vec<[f32; 2]> = vec![[1.0, 0.0]; n as usize];
        gpu.queue.write_buffer(&rheo, 0, bytemuck::cast_slice(&ones));

        let stages = wgpu::ShaderStages::COMPUTE;
        let (l0, bg0) = BindBuilder::new(stages)
            .uniform(&params_buf)
            .storage(&h_buf, false)
            .storage(&m_buf, false)
            .storage(&rheo, false)
            .storage(&lap, false)
            .storage(&flux, false)
            .storage(&flux_s, false)
            .storage(&alpha, false)
            .storage(&delta, false)
            .storage(&tmp, false)
            .storage(&acc_s, false)
            .storage(&strata_h, false)
            .build(&gpu.device, "film0");
        let (l1, bg1) = BindBuilder::new(stages)
            .storage(&strata_pig, false)
            .storage(&strata_cure, false)
            .storage(&sub_refl, false)
            .storage(&pigtab, true)
            .storage(&cmf, true)
            .storage(&strata_h2, false)
            .storage(&strata_pig2, false)
            .storage(&alpha_s, false)
            .build(&gpu.device, "film1");
        let (l_fac, bg_fac) = BindBuilder::new(stages)
            .storage(&stats_buf, false)
            .storage(&pool_buf, false)
            .storage(&cg_bufs[0], false)
            .storage(&cg_bufs[1], false)
            .storage(&cg_bufs[2], false)
            .storage(&cg_bufs[3], false)
            .storage(&cg_bufs[4], false)
            .storage(&cg_bufs[5], false)
            .storage(&cg_bufs[6], false)
            .storage(&cg_bufs[7], false)
            .storage(&cg_bufs[8], false)
            .storage(&cg_bufs[9], false)
            .storage(&cg_bufs[10], false)
            .storage(&cg_bufs[11], false)
            .storage(&cg_bufs[12], false)
            .build(&gpu.device, "fac");
        let (l_dep, bg_dep) = BindBuilder::new(stages)
            .storage(&deposits, true)
            .storage(&tile_start, true)
            .storage(&tile_count, true)
            .storage(&tile_items, true)
            .build(&gpu.device, "dep");
        let (l_shade, bg_shade) = BindBuilder::new(stages)
            .storage_texture(&color_view, wgpu::TextureFormat::Rgba16Float, wgpu::StorageTextureAccess::WriteOnly)
            .storage_texture(&detail_view, wgpu::TextureFormat::Rgba16Float, wgpu::StorageTextureAccess::WriteOnly)
            .build(&gpu.device, "shade");
        let (l_tiles, bg_tiles) = BindBuilder::new(stages)
            .storage(&tile_flags, false)
            .storage(&tile_list, false)
            .storage(&active_count, false)
            .storage(&tile_moved, false)
            .storage(&lay, false)
            .build(&gpu.device, "tiles");
        let (l_knife, bg_knife) = BindBuilder::new(stages)
            .uniform(&knife_buf)
            .storage(&scraps, false)
            .storage(&scrap_count, false)
            .build(&gpu.device, "knife");
        let (l_brush, bg_brush) = BindBuilder::new(stages)
            .storage(&contacts, true)
            .storage(&ctile_range, true)
            .storage(&ctile_items, true)
            .storage(&reservoir, false)
            .build(&gpu.device, "brush");

        let common = include_str!("../../shaders/common.wgsl");
        let film_src = format!("{}\n{}", common, include_str!("../../shaders/film.wgsl"));
        let strata_src = format!("{}\n{}", common, include_str!("../../shaders/strata.wgsl"));
        let deposit_src = format!("{}\n{}\n{}", common, include_str!("../../shaders/strata.wgsl"), include_str!("../../shaders/deposit.wgsl"));
        let cure_src = format!("{}\n{}\n{}\n{}", common, include_str!("../../shaders/strata.wgsl"), include_str!("../../shaders/km.wgsl"), include_str!("../../shaders/cure.wgsl"));
        let shade_src = format!("{}\n{}\n{}", common, include_str!("../../shaders/km.wgsl"), include_str!("../../shaders/shade.wgsl"));
        let knife_src = format!("{}\n{}\n{}", common, include_str!("../../shaders/strata.wgsl"), include_str!("../../shaders/knife.wgsl"));
        let sm_knife = gpu.shader("knife", &knife_src);
        let brush_src = format!("{}\n{}\n{}", common, include_str!("../../shaders/strata.wgsl"), include_str!("../../shaders/brush.wgsl"));
        let sm_brush = gpu.shader("brush", &brush_src);
        let sm_film = gpu.shader("film", &film_src);
        let sm_strata = gpu.shader("strata", &strata_src);
        let sm_dep = gpu.shader("deposit", &deposit_src);
        let sm_cure = gpu.shader("cure", &cure_src);
        let sm_shade = gpu.shader("shade", &shade_src);
        let d = &gpu.device;
        let lf: [&wgpu::BindGroupLayout; 4] = [&l0, &l1, &l_fac, &l_tiles];
        let ls: [&wgpu::BindGroupLayout; 4] = [&l0, &l1, &l_fac, &l_tiles];
        let p_lap = compute_pipeline(d, "laplacian", &sm_film, "laplacian", &lf);
        let p_flux = compute_pipeline(d, "fluxes", &sm_film, "fluxes", &lf);
        let p_limit = compute_pipeline(d, "limiter", &sm_film, "limiter", &lf);
        let p_nose = compute_pipeline(d, "nose", &sm_film, "nose", &lf);
        let p_shed = compute_pipeline(d, "shed", &sm_film, "shed_pass", &lf);
        let p_slump = compute_pipeline(d, "slump", &sm_film, "slump", &lf);
        let p_apply = compute_pipeline(d, "apply", &sm_film, "apply", &lf);
        let p_split = compute_pipeline(d, "strata_split", &sm_film, "strata_split", &lf);
        let p_clear_acc = compute_pipeline(d, "clear_acc", &sm_film, "clear_acc", &lf);
        let p_commit_tmp = compute_pipeline(d, "commit_tmp", &sm_film, "commit_tmp", &lf);
        let p_reduce = compute_pipeline(d, "reduce_tiles", &sm_film, "reduce_tiles", &lf);
        let p_pool = compute_pipeline(d, "pool8", &sm_film, "pool8", &lf);
        let p_mark = compute_pipeline(d, "mark_tiles", &sm_film, "mark_tiles", &lf);
        let p_compact = compute_pipeline(d, "compact_tiles", &sm_film, "compact_tiles", &lf);
        let p_strata_limit = compute_pipeline(d, "strata_limit", &sm_strata, "strata_limit", &ls);
        let p_strata_update = compute_pipeline(d, "strata_update", &sm_strata, "strata_update", &ls);
        let p_strata_fill = compute_pipeline(d, "strata_fill", &sm_strata, "strata_fill", &ls);
        let p_cg_setup = compute_pipeline(d, "cg_setup", &sm_film, "cg_setup", &lf);
        let p_cg_init = compute_pipeline(d, "cg_init", &sm_film, "cg_init", &lf);
        let p_cg_ia = compute_pipeline(d, "cg_iter_a", &sm_film, "cg_iter_a", &lf);
        let p_cg_ib = compute_pipeline(d, "cg_iter_b", &sm_film, "cg_iter_b", &lf);
        let p_cg_ic = compute_pipeline(d, "cg_iter_c", &sm_film, "cg_iter_c", &lf);
        let p_cg_rinit = compute_pipeline(d, "cg_reduce_init", &sm_film, "cg_reduce_init", &lf);
        let p_cg_ra = compute_pipeline(d, "cg_reduce_a", &sm_film, "cg_reduce_a", &lf);
        let p_cg_rb = compute_pipeline(d, "cg_reduce_b", &sm_film, "cg_reduce_b", &lf);
        let p_cg_finish = compute_pipeline(d, "cg_finish", &sm_film, "cg_finish", &lf);
        let p_cg_rmass = compute_pipeline(d, "cg_reduce_mass", &sm_film, "cg_reduce_mass", &lf);
        let p_cg_rescale = compute_pipeline(d, "cg_rescale", &sm_film, "cg_rescale", &lf);
        let p_cg_lap_p = compute_pipeline(d, "cg_lap_p", &sm_film, "cg_lap_p", &lf);
        let p_cg_lap_h = compute_pipeline(d, "cg_lap_h", &sm_film, "cg_lap_h", &lf);
        let p_cg_lap_u = compute_pipeline(d, "cg_lap_u", &sm_film, "cg_lap_u", &lf);
        let p_copy_back = compute_pipeline(d, "strata_copy_back", &sm_strata, "strata_copy_back", &ls);
        let p_deposit = compute_pipeline(d, "deposit", &sm_dep, "deposit", &[&l0, &l1, &l_dep, &l_tiles]);
        let p_cure = compute_pipeline(d, "cure", &sm_cure, "cure_step", &ls);
        let p_shade = compute_pipeline(d, "shade", &sm_shade, "shade", &[&l0, &l1, &l_shade, &l_tiles]);
        let p_shade_full = compute_pipeline(d, "shade_full", &sm_shade, "shade_full", &[&l0, &l1, &l_shade, &l_tiles]);
        let p_knife = compute_pipeline(d, "knife", &sm_knife, "knife", &[&l0, &l1, &l_knife, &l_tiles]);
        let p_brush = compute_pipeline(d, "brush_scrape", &sm_brush, "brush_scrape", &[&l0, &l1, &l_brush, &l_tiles]);

        let params = Params {
            size: [w, h],
            tiles,
            dx,
            dt: 1.0 / 60.0,
            inv_dx: 1.0 / dx,
            frame_dt: 1.0 / 60.0,
            g: [0.0; 4],
            phys: [0.0; 4],
            ein: [0.0; 4],
            cure: [0.0; 4],
            misc: [0.0; 4],
            flags: [0; 4],
            front: [0.0; 4],
            front2: [0.0; 4],
            front3: [0.0; 4],
        };
        let mut f = Self {
            w, h, dx, tiles, settings, params, params_buf,
            h_buf, m_buf, rheo, lap, flux, flux_s, alpha, delta, tmp, acc_s, strata_h, strata_pig, strata_cure,
            sub_refl, pigtab, cmf, strata_h2, strata_pig2, alpha_s, stats_buf, tile_flags, tile_list, tile_indirect, active_count, tiles16, bg_tiles, p_mark, p_compact, p_copy_back, p_shade_full, needs_full_shade: true, active_tiles: 0, frame: 0, white_refl, knife_buf, scraps, scrap_count, scrap_read, scrap_pending, knife_active: false, contacts, ctile_range, ctile_items, reservoir, res_read, res_pending, contact_count: 0, pool_buf, pool_read, pool_dims, stats_read, stats_pending, stats_frame: 0,
            deposits, tile_start, tile_count, tile_items,
            color_tex, color_view, detail_tex, detail_view, lay, bg0, bg1, bg_fac, bg_dep, bg_shade, bg_knife, p_knife, bg_brush, p_brush,
            p_lap, p_flux, p_limit, p_nose, p_shed, p_slump, p_apply, p_split, p_clear_acc, p_commit_tmp, p_reduce, p_pool,
            p_strata_limit, p_strata_update, p_strata_fill, p_cg_setup, p_cg_init, p_cg_ia, p_cg_ib, p_cg_ic, p_cg_rinit, p_cg_ra, p_cg_rb, p_cg_finish, p_cg_rmass, p_cg_rescale, p_cg_lap_p, p_cg_lap_h, p_cg_lap_u, cg_bufs, cg_report: [0.0; 4], implicit_steps: 1, p_deposit, p_cure, p_shade,
            max_deposits, h_max: 0.0, s_max: 0.0, mu_min: 1.0, h_max_pending: 0.0, substeps_last: 1, pigments, cmf_table, deposit_count: 0, total_deposited: 0.0, weave_volume: weave_vol,
        };
        f.update_params(gpu, 1.0 / 60.0, 1);
        f
    }

    pub fn gravity_vec(&self) -> [f32; 3] {
        let s = &self.settings;
        let tilt = s.tilt_deg.to_radians();
        let dir = s.tilt_dir_deg.to_radians();
        let gxy = s.gravity * tilt.sin();
        [gxy * dir.cos(), gxy * dir.sin(), s.gravity * tilt.cos()]
    }

    pub fn update_params(&mut self, gpu: &Gpu, frame_dt: f32, substeps: u32) {
        if std::env::var("PAINT_NO_IMPLICIT").is_ok() { self.settings.implicit_hydro = false; }
        let s = &self.settings;
        let g = self.gravity_vec();
        let theta = s.contact_angle_deg.to_radians();
        let pi_a = 2.0 * s.sigma * (1.0 - theta.cos()) / s.h_star;
        let dt = frame_dt / substeps as f32;
        let dx = self.dx;
        let m_cap_exp = 0.5 * dx * dx * dx * dx / (32.0 * s.sigma * dt);
        let m_cap = s.m_cap.min(m_cap_exp).max(1e-14);
        let m_exp = 0.1 * dx * dx / (dt * s.rho * g[2].max(0.1));
        self.params.dt = dt;
        self.params.frame_dt = frame_dt;
        self.params.g = [g[0], g[1], g[2], s.gravity];
        self.params.phys = [s.rho, s.sigma, s.n_index, if s.capillary && std::env::var("PAINT_NO_CAP").is_err() { m_cap } else { 0.0 }];
        let lc = (s.sigma / (s.rho * 9.81)).sqrt();
        self.params.ein = [theta.tan().max(0.05), lc * (0.5 * theta).sin(), pi_a, s.h_min];
        self.params.cure = [s.cure_rate, s.tau_cure, s.lnmu_cure, s.h_star];
        self.params.misc = [2.0e-6, 0.12, s.medium_mu, m_exp.min(s.m_cap_hydro)];
        self.params.flags = [s.debug_view, s.slump as u32, 0, 0];
        let lc_z = (s.sigma / (s.rho * g[2].max(0.1))).sqrt();
        let cap_explicit = m_exp.min(s.m_cap_hydro);
        let h_big = self.h_max.max(self.h_max_pending).max(1e-4);
        let m_phys = h_big * h_big * h_big / (3.0 * self.mu_min.max(1e-3)) - cap_explicit;
        let k_max = 40.0f32;
        let kappa_frame = frame_dt * s.rho * g[2].max(0.1) / (dx * dx);
        let n_i = if m_phys <= 0.0 { 0 } else { ((kappa_frame * m_phys / k_max).ceil().max(1.0) as u32).min(4) };
        let dt_i = frame_dt / n_i.max(1) as f32;
        let m_imp_max = k_max * dx * dx / (dt_i * s.rho * g[2].max(0.1));
        self.implicit_steps = n_i;
        let m_phys_cap = h_big * h_big * h_big / (3.0 * self.mu_min.max(1e-3));
        let cap_cond: f32 = std::env::var("PAINT_CAPCOND").ok().and_then(|v| v.parse().ok()).unwrap_or(80.0);
        let m_cap_impl = (cap_cond / 28.0) * dx.powi(4) / (dt_i * s.sigma);
        let m_c = if s.capillary_implicit { m_phys_cap.min(m_cap_impl) } else { 0.0 };
        let cc = dt_i * s.sigma * m_c / dx.powi(4);
        let kc = m_c * s.sigma / (dx * dx * dx);
        let fcap_exp: f32 = std::env::var("PAINT_FCAP").ok().and_then(|v| v.parse().ok()).unwrap_or(2.5);
        self.params.front3 = [cc, kc, m_c, fcap_exp];
        let no_hold = std::env::var("PAINT_NO_HOLD").is_ok();
        self.params.front = [if no_hold { 0.0 } else { 2.0 * lc_z * (0.5 * theta).sin() }, dx / lc_z, m_imp_max, dt_i];
        let kshape: f32 = std::env::var("PAINT_KSHAPE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        let no_transl = std::env::var("PAINT_NO_TRANSL").is_ok();
        let no_shape = std::env::var("PAINT_NO_SHAPE").is_ok();
        let kline: f32 = std::env::var("PAINT_KLINE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let flags = (no_transl as u32) | ((no_shape as u32) << 1);
        let hyst: f32 = std::env::var("PAINT_HYST").ok().and_then(|v| v.parse().ok()).unwrap_or(s.hysteresis_deg);
        self.params.front2 = [hyst.to_radians(), kshape, kline, flags as f32];
        gpu.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }

    pub fn choose_substeps(&self, frame_dt: f32) -> u32 {
        let s = &self.settings;
        let g = self.gravity_vec();
        let gxy = (g[0] * g[0] + g[1] * g[1]).sqrt();
        let h_max = self.h_max.max(self.h_max_pending).max(1e-5);
        let mu = self.mu_min.max(1e-3);
        let speed = (s.rho * gxy * h_max * h_max / mu).min(s.v_cap);
        let dt_cfl = if speed > 1e-6 { 0.5 * self.dx / speed } else { frame_dt };
        let mob = h_max * h_max * h_max / (3.0 * mu);
        let n_hyd = (frame_dt * s.rho * g[2] * mob.min(s.m_cap_hydro) / (0.2 * self.dx * self.dx)).ceil().max(1.0) as u32;
        let dx4 = self.dx.powi(4);
        let n_cap = (frame_dt * 32.0 * s.sigma * mob.min(s.m_cap) / dx4).ceil().max(1.0) as u32;
        let n_cfl = (frame_dt / dt_cfl).ceil().max(1.0) as u32;
        n_cfl.max(n_hyd).max(n_cap).clamp(s.min_substeps, s.max_substeps)
    }

    pub fn poll_stats(&mut self, gpu: &Gpu, coarse: &mut [f32], sub: &mut [f32]) -> bool {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        let mut got = false;
        for k in 0..2 {
            if self.stats_pending[k].load(std::sync::atomic::Ordering::Acquire) == 2 {
                let nt = (self.tiles[0] * self.tiles[1]) as usize;
                {
                    let view = self.stats_read[k].slice(..).get_mapped_range().expect("map");
                    let data: &[[f32; 2]] = bytemuck::cast_slice(&view[..nt * 8]);
                    let mut hm = 0.0f32;
                    let mut mm = 1e9f32;
                    for v in data {
                        hm = hm.max(v[0]);
                        mm = mm.min(v[1]);
                    }
                    self.h_max = hm;
                    self.mu_min = if mm > 1e8 { 1.0 } else { mm };
                    self.active_tiles = *bytemuck::from_bytes(&view[nt * 8..nt * 8 + 4]);
                    let s1: [f32; 4] = *bytemuck::from_bytes(&view[nt * 8 + 32..nt * 8 + 48]);
                    self.cg_report = s1;
                }
                {
                    let view = self.pool_read[k].slice(..).get_mapped_range().expect("map");
                    let data: &[f32] = bytemuck::cast_slice(&view);
                    let np = coarse.len().min(data.len() / 2);
                    coarse[..np].copy_from_slice(&data[..np]);
                    let ns = sub.len().min(np);
                    sub[..ns].copy_from_slice(&data[np..np + ns]);
                }
                self.stats_read[k].unmap();
                self.pool_read[k].unmap();
                self.stats_pending[k].store(0, std::sync::atomic::Ordering::Release);
                self.h_max_pending = 0.0;
                got = true;
            }
        }
        got
    }

    pub fn request_stats(&mut self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder) -> Option<usize> {
        let k = self.stats_frame % 2;
        self.stats_frame += 1;
        if self.stats_pending[k].load(std::sync::atomic::Ordering::Acquire) != 0 {
            return None;
        }
        let nt = (self.tiles[0] * self.tiles[1]) as u64;
        enc.copy_buffer_to_buffer(&self.stats_buf, 0, &self.stats_read[k], 0, nt * 8);
        enc.copy_buffer_to_buffer(&self.active_count, 0, &self.stats_read[k], nt * 8, 4);
        enc.copy_buffer_to_buffer(&self.cg_bufs[7], 0, &self.stats_read[k], nt * 8 + 16, 32);
        let np = (self.pool_dims[0] * self.pool_dims[1]) as u64;
        enc.copy_buffer_to_buffer(&self.pool_buf, 0, &self.pool_read[k], 0, np * 8);
        self.stats_pending[k].store(1, std::sync::atomic::Ordering::Release);
        let _ = gpu;
        Some(k)
    }

    pub fn map_stats(&mut self, k: usize) {
        let flag = self.stats_pending[k].clone();
        let flag2 = flag.clone();
        self.pool_read[k].slice(..).map_async(wgpu::MapMode::Read, move |r| {
            if r.is_err() {
                flag2.store(0, std::sync::atomic::Ordering::Release);
            }
        });
        self.stats_read[k].slice(..).map_async(wgpu::MapMode::Read, move |r| {
            if r.is_ok() {
                flag.store(2, std::sync::atomic::Ordering::Release);
            } else {
                flag.store(0, std::sync::atomic::Ordering::Release);
            }
        });
    }

    pub fn clear(&mut self, gpu: &Gpu) {
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("clear") });
        for b in [&self.h_buf, &self.m_buf, &self.lap, &self.flux, &self.flux_s, &self.alpha, &self.delta, &self.tmp, &self.acc_s, &self.strata_h, &self.strata_pig, &self.strata_cure, &self.strata_h2, &self.strata_pig2, &self.alpha_s, &self.tile_flags, &self.lay] {
            enc.clear_buffer(b, 0, None);
        }
        gpu.queue.submit(std::iter::once(enc.finish()));
        let n = (self.w * self.h) as usize;
        let ones: Vec<[f32; 2]> = vec![[1.0, 0.0]; n];
        gpu.queue.write_buffer(&self.rheo, 0, bytemuck::cast_slice(&ones));
        self.white_refl = substrate_refl(n, self.settings.raw_canvas);
        gpu.queue.write_buffer(&self.sub_refl, 0, bytemuck::cast_slice(&self.white_refl));
        let wv = weave(self.w as usize, self.h as usize, self.dx, self.settings.weave_um * 1e-6);
        self.weave_volume = wv.iter().map(|v| *v as f64).sum::<f64>() * (self.dx as f64) * (self.dx as f64);
        gpu.queue.write_buffer(&self.m_buf, 0, bytemuck::cast_slice(&wv));
        self.total_deposited = 0.0;
        self.h_max = 0.0;
        self.s_max = 0.0;
        self.h_max_pending = 0.0;
        self.needs_full_shade = true;
    }

    pub fn set_knife(&mut self, gpu: &Gpu, k: Option<KnifeParams>) {
        let p = k.unwrap_or_default();
        self.knife_active = p.active != 0;
        gpu.queue.write_buffer(&self.knife_buf, 0, bytemuck::bytes_of(&p));
    }

    pub fn upload_contacts(&mut self, gpu: &Gpu, cs: &[Contact]) {
        let count = cs.len().min(MAX_CONTACTS);
        self.contact_count = count;
        if count == 0 {
            return;
        }
        let nt = (self.tiles[0] * self.tiles[1]) as usize;
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nt];
        let tile_m = 64.0 * self.dx;
        for (ci, c) in cs.iter().enumerate().take(count) {
            let minx = c.p0[0].min(c.p1[0]) - c.r;
            let maxx = c.p0[0].max(c.p1[0]) + c.r;
            let miny = c.p0[1].min(c.p1[1]) - c.r;
            let maxy = c.p0[1].max(c.p1[1]) + c.r;
            if maxx < 0.0 || maxy < 0.0 || minx > self.settings.width_m || miny > self.settings.height_m || !(minx.is_finite() && miny.is_finite()) {
                continue;
            }
            let tx0 = ((minx / tile_m).floor().max(0.0) as i64).min(self.tiles[0] as i64 - 1);
            let tx1 = ((maxx / tile_m).floor().max(0.0) as i64).min(self.tiles[0] as i64 - 1);
            let ty0 = ((miny / tile_m).floor().max(0.0) as i64).min(self.tiles[1] as i64 - 1);
            let ty1 = ((maxy / tile_m).floor().max(0.0) as i64).min(self.tiles[1] as i64 - 1);
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    lists[(ty * self.tiles[0] as i64 + tx) as usize].push(ci as u32);
                }
            }
        }
        let mut range = vec![[0u32; 2]; nt];
        let mut items: Vec<u32> = Vec::new();
        for t in 0..nt {
            range[t] = [items.len() as u32, lists[t].len() as u32];
            items.extend_from_slice(&lists[t]);
        }
        items.truncate(MAX_CONTACTS * 16);
        gpu.queue.write_buffer(&self.contacts, 0, bytemuck::cast_slice(&cs[..count]));
        gpu.queue.write_buffer(&self.ctile_range, 0, bytemuck::cast_slice(&range));
        if !items.is_empty() {
            gpu.queue.write_buffer(&self.ctile_items, 0, bytemuck::cast_slice(&items));
        }
    }

    pub fn poll_reservoir(&mut self, gpu: &Gpu, out: &mut Vec<[u32; 17]>) {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        for k in 0..2 {
            if self.res_pending[k].load(std::sync::atomic::Ordering::Acquire) == 2 {
                {
                    let view = self.res_read[k].slice(..).get_mapped_range().expect("map");
                    let data: &[u32] = bytemuck::cast_slice(&view);
                    for b in 0..MAX_BRISTLES {
                        let mut r = [0u32; 17];
                        r.copy_from_slice(&data[b * 17..b * 17 + 17]);
                        if b < out.len() {
                            for i in 0..17 {
                                out[b][i] = out[b][i].saturating_add(r[i]);
                            }
                        } else {
                            out.push(r);
                        }
                    }
                }
                self.res_read[k].unmap();
                self.res_pending[k].store(0, std::sync::atomic::Ordering::Release);
            }
        }
    }

    pub fn request_reservoir(&mut self, enc: &mut wgpu::CommandEncoder) -> Option<usize> {
        let mut slot = None;
        for k in 0..2 {
            if self.res_pending[k].load(std::sync::atomic::Ordering::Acquire) == 0 {
                slot = Some(k);
                break;
            }
        }
        let Some(k) = slot else { return None };
        enc.copy_buffer_to_buffer(&self.reservoir, 0, &self.res_read[k], 0, (MAX_BRISTLES * 17 * 4) as u64);
        enc.clear_buffer(&self.reservoir, 0, None);
        self.res_pending[k].store(1, std::sync::atomic::Ordering::Release);
        Some(k)
    }

    pub fn map_reservoir(&mut self, k: usize) {
        let flag = self.res_pending[k].clone();
        self.res_read[k].slice(..).map_async(wgpu::MapMode::Read, move |r| {
            flag.store(if r.is_ok() { 2 } else { 0 }, std::sync::atomic::Ordering::Release);
        });
    }

    pub fn poll_scraps(&mut self, gpu: &Gpu, out: &mut Vec<Scrap>) {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        for k in 0..2 {
            if self.scrap_pending[k].load(std::sync::atomic::Ordering::Acquire) == 2 {
                {
                    let view = self.scrap_read[k].slice(..).get_mapped_range().expect("map");
                    let count: u32 = bytemuck::cast_slice::<u8, u32>(&view[..4])[0];
                    let n = (count as usize).min(1 << 16);
                    let data: &[Scrap] = bytemuck::cast_slice(&view[16..16 + n * 48]);
                    out.extend_from_slice(data);
                }
                self.scrap_read[k].unmap();
                self.scrap_pending[k].store(0, std::sync::atomic::Ordering::Release);
            }
        }
    }

    pub fn request_scraps(&mut self, enc: &mut wgpu::CommandEncoder) -> Option<usize> {
        let mut slot = None;
        for k in 0..2 {
            if self.scrap_pending[k].load(std::sync::atomic::Ordering::Acquire) == 0 {
                slot = Some(k);
                break;
            }
        }
        let Some(k) = slot else { return None };
        enc.copy_buffer_to_buffer(&self.scrap_count, 0, &self.scrap_read[k], 0, 16);
        enc.copy_buffer_to_buffer(&self.scraps, 0, &self.scrap_read[k], 16, (1u64 << 16) * 48);
        enc.clear_buffer(&self.scrap_count, 0, None);
        self.scrap_pending[k].store(1, std::sync::atomic::Ordering::Release);
        Some(k)
    }

    pub fn map_scraps(&mut self, k: usize) {
        let flag = self.scrap_pending[k].clone();
        self.scrap_read[k].slice(..).map_async(wgpu::MapMode::Read, move |r| {
            flag.store(if r.is_ok() { 2 } else { 0 }, std::sync::atomic::Ordering::Release);
        });
    }

    pub fn upload_deposits(&mut self, gpu: &Gpu, deps: &[Deposit]) {
        let count = deps.len().min(self.max_deposits);
        self.deposit_count = count;
        let nt = (self.tiles[0] * self.tiles[1]) as usize;
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nt];
        let tile_m = 64.0 * self.dx;
        for (di, d) in deps.iter().enumerate().take(count) {
            self.h_max_pending = self.h_max_pending.max(d.max_height() + self.h_max);
            let rb = d.max_radius();
            let inside = d.p0[0].max(d.p1[0]) + rb > 0.0 && d.p0[1].max(d.p1[1]) + rb > 0.0 && d.p0[0].min(d.p1[0]) - rb < self.settings.width_m && d.p0[1].min(d.p1[1]) - rb < self.settings.height_m;
            if !inside { continue; }
            if d.kind & DEPOSIT_RECYCLED == 0 {
                self.total_deposited += d.vol as f64 * d.inside_fraction(self.settings.width_m, self.settings.height_m) as f64;
            }
            let minx = d.p0[0].min(d.p1[0]) - rb;
            let maxx = d.p0[0].max(d.p1[0]) + rb;
            let miny = d.p0[1].min(d.p1[1]) - rb;
            let maxy = d.p0[1].max(d.p1[1]) + rb;
            let tx0 = ((minx / tile_m).floor().max(0.0) as i64).min(self.tiles[0] as i64 - 1);
            let tx1 = ((maxx / tile_m).floor().max(0.0) as i64).min(self.tiles[0] as i64 - 1);
            let ty0 = ((miny / tile_m).floor().max(0.0) as i64).min(self.tiles[1] as i64 - 1);
            let ty1 = ((maxy / tile_m).floor().max(0.0) as i64).min(self.tiles[1] as i64 - 1);
            if maxx < 0.0 || maxy < 0.0 { continue; }
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    lists[(ty * self.tiles[0] as i64 + tx) as usize].push(di as u32);
                }
            }
        }
        let mut start = vec![0u32; nt];
        let mut cnt = vec![0u32; nt];
        let mut items: Vec<u32> = Vec::new();
        for t in 0..nt {
            start[t] = items.len() as u32;
            cnt[t] = lists[t].len() as u32;
            items.extend_from_slice(&lists[t]);
        }
        let cap = self.max_deposits * 16;
        if items.len() > cap {
            items.truncate(cap);
        }
        if count > 0 {
            gpu.queue.write_buffer(&self.deposits, 0, bytemuck::cast_slice(&deps[..count]));
        }
        gpu.queue.write_buffer(&self.tile_start, 0, bytemuck::cast_slice(&start));
        gpu.queue.write_buffer(&self.tile_count, 0, bytemuck::cast_slice(&cnt));
        if !items.is_empty() {
            gpu.queue.write_buffer(&self.tile_items, 0, bytemuck::cast_slice(&items));
        }
    }

    fn groups(&self) -> (u32, u32) {
        ((self.w + 15) / 16, (self.h + 15) / 16)
    }

    pub fn dry_now(&mut self, gpu: &Gpu) {
        let saved = self.params.cure;
        self.params.cure[0] = 4.0 / self.params.frame_dt.max(1e-4);
        gpu.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        let (gx, gy) = self.groups();
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dry") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("dry"), timestamp_writes: None });
            dispatch(&mut pass, &self.p_cure, &[&self.bg0, &self.bg1, &self.bg_fac, &self.bg_tiles], gx, gy, 1);
        }
        gpu.queue.submit(std::iter::once(enc.finish()));
        self.params.cure = saved;
        gpu.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        self.needs_full_shade = true;
    }

    pub fn encode_frame(&mut self, enc: &mut wgpu::CommandEncoder, substeps: u32) {
        let (gx, gy) = self.groups();
        let nt16 = self.tiles16[0] * self.tiles16[1];
        enc.clear_buffer(&self.active_count, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pre"), timestamp_writes: None });
            let g = &self.bg_tiles;
            if self.contact_count > 0 {
                dispatch(&mut pass, &self.p_brush, &[&self.bg0, &self.bg1, &self.bg_brush, g], gx, gy, 1);
            }
            if self.deposit_count > 0 {
                dispatch(&mut pass, &self.p_deposit, &[&self.bg0, &self.bg1, &self.bg_dep, g], gx, gy, 1);
            }
            if self.knife_active {
                dispatch(&mut pass, &self.p_knife, &[&self.bg0, &self.bg1, &self.bg_knife, g], gx, gy, 1);
            }
            dispatch(&mut pass, &self.p_mark, &[&self.bg0, &self.bg1, &self.bg_fac, g], self.tiles16[0], self.tiles16[1], 1);
            dispatch(&mut pass, &self.p_compact, &[&self.bg0, &self.bg1, &self.bg_fac, g], (nt16 + 255) / 256, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.active_count, 0, &self.tile_indirect, 0, 4);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sim"), timestamp_writes: None });
            let g = &self.bg_tiles;
            let ind = |pass: &mut wgpu::ComputePass, p: &wgpu::ComputePipeline, groups: &[&wgpu::BindGroup]| {
                pass.set_pipeline(p);
                for (i, gg) in groups.iter().enumerate() {
                    pass.set_bind_group(i as u32, *gg, &[]);
                }
                pass.dispatch_workgroups_indirect(&self.tile_indirect, 0);
            };
            let gf: [&wgpu::BindGroup; 4] = [&self.bg0, &self.bg1, &self.bg_fac, g];
            dispatch(&mut pass, &self.p_cure, &gf, gx, gy, 1);
            ind(&mut pass, &self.p_clear_acc, &gf);
            if self.settings.implicit_hydro {
                for _ in 0..self.implicit_steps {
                    ind(&mut pass, &self.p_lap, &gf);
                    ind(&mut pass, &self.p_cg_setup, &gf);
                    ind(&mut pass, &self.p_cg_lap_h, &gf);
                    ind(&mut pass, &self.p_cg_init, &gf);
                    dispatch(&mut pass, &self.p_cg_rinit, &gf, 1, 1, 1);
                    for _ in 0..self.settings.implicit_iters {
                        ind(&mut pass, &self.p_cg_lap_p, &gf);
                        ind(&mut pass, &self.p_cg_ia, &gf);
                        dispatch(&mut pass, &self.p_cg_ra, &gf, 1, 1, 1);
                        ind(&mut pass, &self.p_cg_ib, &gf);
                        dispatch(&mut pass, &self.p_cg_rb, &gf, 1, 1, 1);
                        ind(&mut pass, &self.p_cg_ic, &gf);
                    }
                    ind(&mut pass, &self.p_cg_lap_u, &gf);
                    ind(&mut pass, &self.p_cg_finish, &gf);
                    dispatch(&mut pass, &self.p_cg_rmass, &gf, 1, 1, 1);
                    ind(&mut pass, &self.p_cg_rescale, &gf);
                }
            }
            for _ in 0..substeps {
                ind(&mut pass, &self.p_lap, &gf);
                ind(&mut pass, &self.p_flux, &gf);
                ind(&mut pass, &self.p_limit, &gf);
                ind(&mut pass, &self.p_apply, &gf);
                ind(&mut pass, &self.p_nose, &gf);
                if self.settings.slump {
                    ind(&mut pass, &self.p_shed, &gf);
                    ind(&mut pass, &self.p_slump, &gf);
                    ind(&mut pass, &self.p_commit_tmp, &gf);
                }
            }
            ind(&mut pass, &self.p_lap, &gf);
            ind(&mut pass, &self.p_split, &gf);
            dispatch(&mut pass, &self.p_reduce, &gf, self.tiles[0], self.tiles[1], 1);
            dispatch(&mut pass, &self.p_pool, &gf, (self.pool_dims[0] as u32 + 15) / 16, (self.pool_dims[1] as u32 + 15) / 16, 1);
            ind(&mut pass, &self.p_strata_limit, &gf);
            ind(&mut pass, &self.p_strata_update, &gf);
            ind(&mut pass, &self.p_copy_back, &gf);
            for _ in 0..6 {
                ind(&mut pass, &self.p_strata_fill, &gf);
            }
            if self.needs_full_shade || self.frame % FULL_SHADE_EVERY == 0 {
                dispatch(&mut pass, &self.p_shade_full, &[&self.bg0, &self.bg1, &self.bg_shade, g], gx, gy, 1);
                self.needs_full_shade = false;
            } else {
                ind(&mut pass, &self.p_shade, &[&self.bg0, &self.bg1, &self.bg_shade, g]);
            }
        }
        self.substeps_last = substeps;
        self.frame += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampled_volume(d: &Deposit) -> f32 {
        let r = d.max_radius();
        let step = r / 40.0;
        let lo = [d.p0[0].min(d.p1[0]) - 2.0 * r, d.p0[1].min(d.p1[1]) - 2.0 * r];
        let hi = [d.p0[0].max(d.p1[0]) + 2.0 * r, d.p0[1].max(d.p1[1]) + 2.0 * r];
        let mut sum = 0.0;
        let mut x = lo[0];
        while x < hi[0] {
            let mut y = lo[1];
            while y < hi[1] {
                sum += d.height_at([x, y]);
                y += step;
            }
            x += step;
        }
        sum * step * step
    }

    #[test]
    fn ribbon_segments_hold_their_volume() {
        let pig = [0u32; 8];
        let c = 0.3f32.cos();
        let s = 0.3f32.sin();
        let cases = [
            Deposit::ribbon([0.0, 0.0], [1e-3, 0.0], [1.0, 0.0], [1.0, 0.0], 4e-4, 6e-4, 0, 1e-9, pig, 3.9e-4),
            Deposit::ribbon([0.0, 0.0], [1e-3, 0.0], [c, -s], [c, s], 4e-4, 6e-4, 0, 1e-9, pig, 3.9e-4),
            Deposit::ribbon([0.0, 0.0], [3e-4, 0.0], [c, s], [c, -s], 8e-4, 8e-4, 0, 1e-9, pig, 3.9e-4),
            Deposit::capsule([0.0, 0.0], [2e-3, 1e-3], 5e-4, 1e-9, pig, 3.9e-4),
            Deposit::capsule([0.0, 0.0], [0.0, 0.0], 5e-4, 1e-9, pig, 3.9e-4),
            Deposit::stroke([0.0, 0.0], [4e-3, 0.0], 5e-4, 1e-9, pig, 3.9e-4),
        ];
        for (i, d) in cases.iter().enumerate() {
            let v = sampled_volume(d);
            assert!((v - d.vol).abs() < 0.03 * d.vol, "case {} sampled {} stored {}", i, v, d.vol);
        }
    }
}

