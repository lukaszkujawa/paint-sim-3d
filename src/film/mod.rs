mod deposit;
mod substrate;
mod journal;
mod settings;
mod readback;
pub use deposit::*;
pub use substrate::*;
pub use journal::*;
pub use settings::*;
use crate::color;
use crate::gpu::{compute_pipeline, dispatch, BindBuilder, Gpu, GpuClock, GpuProf};
use bytemuck::{Pod, Zeroable};

const FULL_SHADE_EVERY: u32 = 1;

pub const DEPOSIT_RECYCLED: u32 = 1;

pub const DEPOSIT_CAP_START: u32 = 2;

pub const DEPOSIT_CAP_END: u32 = 4;

pub const DEPOSIT_STRANDS: u32 = 8;

pub const MAX_CONTACTS: usize = 1 << 18;

pub const MAX_SCRAPS: usize = 1 << 18;

pub const MAX_BRISTLES: usize = 1024;

pub const KNIFE_JOURNAL_REACH: f32 = 0.08;

pub const JOURNAL_CELL_BYTES: u64 = 212;

pub struct Film {
    pub prof: Option<GpuProf>,
    pub clock: Option<GpuClock>,
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
    pub strata_h2: wgpu::Buffer,
    pub strata_pig2: wgpu::Buffer,
    pub alpha_s: wgpu::Buffer,
    pub stats_buf: wgpu::Buffer,
    pub tile_flags: wgpu::Buffer,
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
    rubber_buf: wgpu::Buffer,
    bg_rubber: wgpu::BindGroup,
    p_rubber: wgpu::ComputePipeline,
    rubber_active: bool,
    pub canvas_refl: [u32; 8],
    pub erased: bool,
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
    pub res_tool: [u8; 2],
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
    p_plug: wgpu::ComputePipeline,
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
    pub journal: Journal,
    pub tile_moved: wgpu::Buffer,
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
        let max_deposits = 1 << 17;
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
        let strata_cure2 = gpu.storage("strata_cure2", n * 16);
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
        let max_scraps = MAX_SCRAPS as u64;
        let knife_buf = gpu.uniform("knife", std::mem::size_of::<KnifeParams>() as u64);
        let rubber_buf = gpu.uniform("rubber", std::mem::size_of::<RubberParams>() as u64);
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

        let white = substrate_refl(n as usize, settings.raw_canvas, settings.canvas_rgb.map(|c| color::reflectance_of(color::canvas_albedo(c), &cmf_table)));
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
            .storage(&strata_cure2, false)
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
        let (l_rubber, bg_rubber) = BindBuilder::new(stages).uniform(&rubber_buf).build(&gpu.device, "rubber");
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
        let rubber_src = format!("{}\n{}", common, include_str!("../../shaders/rubber.wgsl"));
        let sm_rubber = gpu.shader("rubber", &rubber_src);
        let brush_src = format!("{}\n{}\n{}", common, include_str!("../../shaders/strata.wgsl"), include_str!("../../shaders/brush.wgsl"));
        let sm_brush = gpu.shader("brush", &brush_src);
        let sm_film = gpu.shader("film", &film_src);
        let sm_strata = gpu.shader("strata", &strata_src);
        let sm_dep = gpu.shader("deposit", &deposit_src);
        let sm_cure = gpu.shader("cure", &cure_src);
        let sm_shade = gpu.shader("shade", &shade_src);
        let d = &gpu.device;
        let journal = Journal::new(gpu, &common, [&l0, &l1, &l_tiles], tiles16[0] * tiles16[1]);
        let lf: [&wgpu::BindGroupLayout; 4] = [&l0, &l1, &l_fac, &l_tiles];
        let ls: [&wgpu::BindGroupLayout; 4] = [&l0, &l1, &l_fac, &l_tiles];
        let p_lap = compute_pipeline(d, "laplacian", &sm_film, "laplacian", &lf);
        let p_flux = compute_pipeline(d, "fluxes", &sm_film, "fluxes", &lf);
        let p_plug = compute_pipeline(d, "plug", &sm_film, "plug", &lf);
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
        let p_rubber = compute_pipeline(d, "rubber", &sm_rubber, "rubber", &[&l0, &l1, &l_rubber, &l_tiles]);
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
        let prof = GpuProf::new(gpu);
        let clock = if prof.is_none() { Some(GpuClock::new(gpu)) } else { None };
        let mut f = Self {
            prof,
            clock,
            w, h, dx, tiles, settings, params, params_buf,
            h_buf, m_buf, rheo, lap, flux, flux_s, alpha, delta, tmp, acc_s, strata_h, strata_pig, strata_cure,
            sub_refl, strata_h2, strata_pig2, alpha_s, stats_buf, tile_flags, tile_indirect, active_count, tiles16, bg_tiles, p_mark, p_compact, p_copy_back, p_shade_full, needs_full_shade: true, journal, tile_moved, active_tiles: 0, frame: 0, canvas_refl: first_refl(&white_refl), white_refl, knife_buf, rubber_buf, bg_rubber, p_rubber, rubber_active: false, erased: false, scraps, scrap_count, scrap_read, scrap_pending, knife_active: false, contacts, ctile_range, ctile_items, reservoir, res_read, res_pending, res_tool: [0; 2], contact_count: 0, pool_buf, pool_read, pool_dims, stats_read, stats_pending, stats_frame: 0,
            deposits, tile_start, tile_count, tile_items,
            color_tex, color_view, detail_tex, detail_view, lay, bg0, bg1, bg_fac, bg_dep, bg_shade, bg_knife, p_knife, bg_brush, p_brush,
            p_lap, p_plug, p_flux, p_limit, p_nose, p_shed, p_slump, p_apply, p_split, p_clear_acc, p_commit_tmp, p_reduce, p_pool,
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
        [gxy * dir.cos(), gxy * dir.sin(), (s.gravity * tilt.cos()).max(0.0)]
    }

    pub fn update_params(&mut self, gpu: &Gpu, frame_dt: f32, substeps: u32) {
        if std::env::var("PAINT_NO_IMPLICIT").is_ok() { self.settings.implicit_hydro = false; }
        let s = &self.settings;
        let g = self.gravity_vec();
        let theta = s.contact_angle_deg.to_radians();
        let pi_a = 3.0 * s.sigma * (1.0 - theta.cos()) / s.h_star;
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

    pub fn clear(&mut self, gpu: &Gpu) {
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("clear") });
        for b in [&self.h_buf, &self.m_buf, &self.lap, &self.flux, &self.flux_s, &self.alpha, &self.delta, &self.tmp, &self.acc_s, &self.strata_h, &self.strata_pig, &self.strata_cure, &self.strata_h2, &self.strata_pig2, &self.alpha_s, &self.tile_flags, &self.lay] {
            enc.clear_buffer(b, 0, None);
        }
        gpu.queue.submit(std::iter::once(enc.finish()));
        let n = (self.w * self.h) as usize;
        let ones: Vec<[f32; 2]> = vec![[1.0, 0.0]; n];
        gpu.queue.write_buffer(&self.rheo, 0, bytemuck::cast_slice(&ones));
        self.white_refl = substrate_refl(n, self.settings.raw_canvas, self.settings.canvas_rgb.map(|c| color::reflectance_of(color::canvas_albedo(c), &self.cmf_table)));
        self.canvas_refl = first_refl(&self.white_refl);
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
        if self.knife_active {
            let reach = KNIFE_JOURNAL_REACH + p.sweep;
            let minx = p.a[0].min(p.b[0]) - reach;
            let maxx = p.a[0].max(p.b[0]) + reach;
            let miny = p.a[1].min(p.b[1]) - reach;
            let maxy = p.a[1].max(p.b[1]) + reach;
            self.touch_rect(minx, miny, maxx, maxy);
        }
        gpu.queue.write_buffer(&self.knife_buf, 0, bytemuck::bytes_of(&p));
    }

    pub fn set_rubber(&mut self, gpu: &Gpu, seg: Option<([f32; 2], [f32; 2], f32)>) {
        let Some((a, b, radius)) = seg else {
            if self.rubber_active {
                self.rubber_active = false;
                self.needs_full_shade = true;
                gpu.queue.write_buffer(&self.rubber_buf, 0, bytemuck::bytes_of(&RubberParams::default()));
            }
            return;
        };
        self.rubber_active = true;
        self.erased = true;
        let reach = radius + self.dx;
        self.touch_rect(a[0].min(b[0]) - reach, a[1].min(b[1]) - reach, a[0].max(b[0]) + reach, a[1].max(b[1]) + reach);
        let p = RubberParams { a, b, radius, weave: self.settings.weave_um * 1e-6, active: 1, pad: 0, refl: self.canvas_refl };
        gpu.queue.write_buffer(&self.rubber_buf, 0, bytemuck::bytes_of(&p));
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
            self.touch_rect(minx, miny, maxx, maxy);
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
            let take = lists[t].len().min(MAX_CONTACTS * 16 - items.len());
            range[t] = [items.len() as u32, take as u32];
            items.extend_from_slice(&lists[t][..take]);
        }
        gpu.queue.write_buffer(&self.contacts, 0, bytemuck::cast_slice(&cs[..count]));
        gpu.queue.write_buffer(&self.ctile_range, 0, bytemuck::cast_slice(&range));
        if !items.is_empty() {
            gpu.queue.write_buffer(&self.ctile_items, 0, bytemuck::cast_slice(&items));
        }
    }

    pub fn upload_deposits(&mut self, gpu: &Gpu, deps: &[Deposit]) -> usize {
        let count = deps.len().min(self.max_deposits);
        self.deposit_count = count;
        let nt = (self.tiles[0] * self.tiles[1]) as usize;
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nt];
        let tile_m = 64.0 * self.dx;
        for (di, d) in deps.iter().enumerate().take(count) {
            self.h_max_pending = self.h_max_pending.max(d.max_height() + self.h_max);
            let rb = d.max_radius();
            self.touch_rect(d.p0[0].min(d.p1[0]) - rb, d.p0[1].min(d.p1[1]) - rb, d.p0[0].max(d.p1[0]) + rb, d.p0[1].max(d.p1[1]) + rb);
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
        let cap = self.max_deposits * 16;
        for t in 0..nt {
            let take = lists[t].len().min(cap - items.len());
            start[t] = items.len() as u32;
            cnt[t] = take as u32;
            items.extend_from_slice(&lists[t][..take]);
        }
        if count > 0 {
            gpu.queue.write_buffer(&self.deposits, 0, bytemuck::cast_slice(&deps[..count]));
        }
        gpu.queue.write_buffer(&self.tile_start, 0, bytemuck::cast_slice(&start));
        gpu.queue.write_buffer(&self.tile_count, 0, bytemuck::cast_slice(&cnt));
        if !items.is_empty() {
            gpu.queue.write_buffer(&self.tile_items, 0, bytemuck::cast_slice(&items));
        }
        count
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

    pub fn touch_rect(&mut self, minx: f32, miny: f32, maxx: f32, maxy: f32) {
        if self.journal.gen == 0 || !(minx.is_finite() && miny.is_finite() && maxx.is_finite() && maxy.is_finite()) {
            return;
        }
        let tile_m = 16.0 * self.dx;
        let tw = self.tiles16[0] as i64;
        let th = self.tiles16[1] as i64;
        let x0 = ((minx / tile_m).floor() as i64).clamp(0, tw - 1);
        let x1 = ((maxx / tile_m).floor() as i64).clamp(0, tw - 1);
        let y0 = ((miny / tile_m).floor() as i64).clamp(0, th - 1);
        let y1 = ((maxy / tile_m).floor() as i64).clamp(0, th - 1);
        if maxx < 0.0 || maxy < 0.0 {
            return;
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                self.journal.touch[(y * tw + x) as usize] = 1;
            }
        }
        self.journal.touched = true;
    }

    pub fn flush_journal(&mut self, gpu: &Gpu) {
        self.journal.flush(gpu);
    }

    pub fn restore_journal(&mut self, gpu: &Gpu, target: u32) {
        self.journal.restore(gpu, [&self.bg0, &self.bg1, &self.bg_tiles], target);
        self.needs_full_shade = true;
    }

    fn state_buffers(&self) -> [&wgpu::Buffer; 8] {
        [&self.h_buf, &self.m_buf, &self.rheo, &self.strata_h, &self.strata_pig, &self.strata_cure, &self.sub_refl, &self.lay]
    }

    pub fn encode_frame(&mut self, enc: &mut wgpu::CommandEncoder, substeps: u32) {
        let (gx, gy) = self.groups();
        let nt16 = self.tiles16[0] * self.tiles16[1];
        enc.clear_buffer(&self.active_count, 0, None);
        let mut prof = self.prof.take();
        let mut mk = |pass: &mut wgpu::ComputePass, l: &'static str| {
            if let Some(p) = prof.as_mut() {
                p.mark(pass, l);
            }
        };
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pre"), timestamp_writes: self.clock.as_ref().and_then(|c| c.compute(true)) });
            let g = &self.bg_tiles;
            if self.journal.gen != 0 {
                mk(&mut pass, "journal");
                dispatch(&mut pass, &self.journal.p_journal, &[&self.bg0, &self.bg1, &self.journal.bg, g], self.tiles16[0], self.tiles16[1], 1);
            }
            if self.contact_count > 0 {
                mk(&mut pass, "brush");
                dispatch(&mut pass, &self.p_brush, &[&self.bg0, &self.bg1, &self.bg_brush, g], gx, gy, 1);
            }
            if self.deposit_count > 0 {
                mk(&mut pass, "deposit");
                dispatch(&mut pass, &self.p_deposit, &[&self.bg0, &self.bg1, &self.bg_dep, g], gx, gy, 1);
            }
            if self.knife_active {
                mk(&mut pass, "knife");
                dispatch(&mut pass, &self.p_knife, &[&self.bg0, &self.bg1, &self.bg_knife, g], gx, gy, 1);
            }
            if self.rubber_active {
                mk(&mut pass, "rubber");
                dispatch(&mut pass, &self.p_rubber, &[&self.bg0, &self.bg1, &self.bg_rubber, g], gx, gy, 1);
            }
            mk(&mut pass, "mark");
            dispatch(&mut pass, &self.p_mark, &[&self.bg0, &self.bg1, &self.bg_fac, g], self.tiles16[0], self.tiles16[1], 1);
            mk(&mut pass, "compact");
            dispatch(&mut pass, &self.p_compact, &[&self.bg0, &self.bg1, &self.bg_fac, g], (nt16 + 255) / 256, 1, 1);
            mk(&mut pass, "");
        }
        enc.copy_buffer_to_buffer(&self.active_count, 0, &self.tile_indirect, 0, 4);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sim"), timestamp_writes: self.clock.as_ref().and_then(|c| c.compute(false)) });
            let g = &self.bg_tiles;
            let ind = |pass: &mut wgpu::ComputePass, p: &wgpu::ComputePipeline, groups: &[&wgpu::BindGroup]| {
                pass.set_pipeline(p);
                for (i, gg) in groups.iter().enumerate() {
                    pass.set_bind_group(i as u32, *gg, &[]);
                }
                pass.dispatch_workgroups_indirect(&self.tile_indirect, 0);
            };
            let gf: [&wgpu::BindGroup; 4] = [&self.bg0, &self.bg1, &self.bg_fac, g];
            mk(&mut pass, "cure");
            dispatch(&mut pass, &self.p_cure, &gf, gx, gy, 1);
            mk(&mut pass, "clear_acc");
            ind(&mut pass, &self.p_clear_acc, &gf);
            if self.settings.implicit_hydro {
                for _ in 0..self.implicit_steps {
                    mk(&mut pass, "lap");
                    ind(&mut pass, &self.p_lap, &gf);
                    mk(&mut pass, "cg_setup");
                    ind(&mut pass, &self.p_cg_setup, &gf);
                    mk(&mut pass, "cg_lap_h");
                    ind(&mut pass, &self.p_cg_lap_h, &gf);
                    mk(&mut pass, "cg_init");
                    ind(&mut pass, &self.p_cg_init, &gf);
                    mk(&mut pass, "cg_rinit");
                    dispatch(&mut pass, &self.p_cg_rinit, &gf, 1, 1, 1);
                    for _ in 0..self.settings.implicit_iters {
                        mk(&mut pass, "cg_lap_p");
                        ind(&mut pass, &self.p_cg_lap_p, &gf);
                        mk(&mut pass, "cg_ia");
                        ind(&mut pass, &self.p_cg_ia, &gf);
                        mk(&mut pass, "cg_ra");
                        dispatch(&mut pass, &self.p_cg_ra, &gf, 1, 1, 1);
                        mk(&mut pass, "cg_ib");
                        ind(&mut pass, &self.p_cg_ib, &gf);
                        mk(&mut pass, "cg_rb");
                        dispatch(&mut pass, &self.p_cg_rb, &gf, 1, 1, 1);
                        mk(&mut pass, "cg_ic");
                        ind(&mut pass, &self.p_cg_ic, &gf);
                    }
                    mk(&mut pass, "cg_lap_u");
                    ind(&mut pass, &self.p_cg_lap_u, &gf);
                    mk(&mut pass, "cg_finish");
                    ind(&mut pass, &self.p_cg_finish, &gf);
                    mk(&mut pass, "cg_rmass");
                    dispatch(&mut pass, &self.p_cg_rmass, &gf, 1, 1, 1);
                    mk(&mut pass, "cg_rescale");
                    ind(&mut pass, &self.p_cg_rescale, &gf);
                }
            }
            for _ in 0..substeps {
                mk(&mut pass, "lap");
                ind(&mut pass, &self.p_lap, &gf);
                mk(&mut pass, "plug");
                ind(&mut pass, &self.p_plug, &gf);
                mk(&mut pass, "flux");
                ind(&mut pass, &self.p_flux, &gf);
                mk(&mut pass, "limit");
                ind(&mut pass, &self.p_limit, &gf);
                mk(&mut pass, "apply");
                ind(&mut pass, &self.p_apply, &gf);
                mk(&mut pass, "nose");
                ind(&mut pass, &self.p_nose, &gf);
                if self.settings.slump {
                    mk(&mut pass, "shed");
                    ind(&mut pass, &self.p_shed, &gf);
                    mk(&mut pass, "slump");
                    ind(&mut pass, &self.p_slump, &gf);
                    mk(&mut pass, "commit_tmp");
                    ind(&mut pass, &self.p_commit_tmp, &gf);
                }
            }
            mk(&mut pass, "lap");
            ind(&mut pass, &self.p_lap, &gf);
            mk(&mut pass, "split");
            ind(&mut pass, &self.p_split, &gf);
            mk(&mut pass, "reduce");
            dispatch(&mut pass, &self.p_reduce, &gf, self.tiles[0], self.tiles[1], 1);
            mk(&mut pass, "pool");
            dispatch(&mut pass, &self.p_pool, &gf, (self.pool_dims[0] as u32 + 15) / 16, (self.pool_dims[1] as u32 + 15) / 16, 1);
            mk(&mut pass, "strata_limit");
            ind(&mut pass, &self.p_strata_limit, &gf);
            mk(&mut pass, "strata_update");
            ind(&mut pass, &self.p_strata_update, &gf);
            mk(&mut pass, "copy_back");
            ind(&mut pass, &self.p_copy_back, &gf);
            for _ in 0..6 {
                mk(&mut pass, "strata_fill");
                ind(&mut pass, &self.p_strata_fill, &gf);
            }
            if self.needs_full_shade || self.frame % FULL_SHADE_EVERY == 0 {
                mk(&mut pass, "shade_full");
                dispatch(&mut pass, &self.p_shade_full, &[&self.bg0, &self.bg1, &self.bg_shade, g], gx, gy, 1);
                self.needs_full_shade = false;
            } else {
                mk(&mut pass, "shade");
                ind(&mut pass, &self.p_shade, &[&self.bg0, &self.bg1, &self.bg_shade, g]);
            }
            mk(&mut pass, "");
        }
        self.prof = prof;
        self.substeps_last = substeps;
        self.frame += 1;
    }
}
