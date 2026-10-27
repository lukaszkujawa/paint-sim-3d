use crate::film::Film;
use crate::gpu::{compute_pipeline, BindBuilder, Gpu};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RefineParams {
    csize: [u32; 2],
    fsize: [u32; 2],
    dt: f32,
    wlo: f32,
    whi: f32,
    eps: f32,
}

pub struct Refine {
    pub fw: u32,
    pub fh: u32,
    pub dx: f32,
    pub h_fine: wgpu::Buffer,
    pub m_fine: wgpu::Buffer,
    scratch: wgpu::Buffer,
    weight: wgpu::Buffer,
    params: wgpu::Buffer,
    bg_h: wgpu::BindGroup,
    bg_m: wgpu::BindGroup,
    p_up: wgpu::ComputePipeline,
    p_weight: wgpu::ComputePipeline,
    p_ab: wgpu::ComputePipeline,
    p_ba: wgpu::ComputePipeline,
    pub bg_render: wgpu::BindGroup,
    pub iterations: u32,
}

impl Refine {
    pub fn new(gpu: &Gpu, film: &Film, render_layout: &wgpu::BindGroupLayout, cam_buf: &wgpu::Buffer) -> Self {
        let fw = film.w * 2;
        let fh = film.h * 2;
        let n = (fw as u64) * (fh as u64);
        let h_fine = gpu.storage("h_fine", n * 4);
        let m_fine = gpu.storage("m_fine", n * 4);
        let scratch = gpu.storage("refine_scratch", n * 4);
        let weight = gpu.storage("refine_weight", n * 4);
        let params = gpu.uniform("refine_params", std::mem::size_of::<RefineParams>() as u64);
        let stages = wgpu::ShaderStages::COMPUTE;
        let (layout, bg_h) = BindBuilder::new(stages)
            .uniform(&params)
            .storage(&film.h_buf, true)
            .storage(&h_fine, false)
            .storage(&scratch, false)
            .storage(&weight, false)
            .build(&gpu.device, "refine_h");
        let (_, bg_m) = BindBuilder::new(stages)
            .uniform(&params)
            .storage(&film.m_buf, true)
            .storage(&m_fine, false)
            .storage(&scratch, false)
            .storage(&weight, false)
            .build(&gpu.device, "refine_m");
        let module = gpu.shader("refine", include_str!("../../shaders/refine.wgsl"));
        let d = &gpu.device;
        let p_up = compute_pipeline(d, "refine_upsample", &module, "upsample", &[&layout]);
        let p_weight = compute_pipeline(d, "refine_weight", &module, "weight", &[&layout]);
        let p_ab = compute_pipeline(d, "refine_ab", &module, "curv_ab", &[&layout]);
        let p_ba = compute_pipeline(d, "refine_ba", &module, "curv_ba", &[&layout]);
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear_fine"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let bg_render = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_fine"),
            layout: render_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&film.color_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: h_fine.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: m_fine.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: film.tile_flags.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&film.detail_view) },
            ],
        });
        Self { fw, fh, dx: film.dx * 0.5, h_fine, m_fine, scratch, weight, params, bg_h, bg_m, p_up, p_weight, p_ab, p_ba, bg_render, iterations: 24 }
    }

    pub fn run(&self, gpu: &Gpu, film: &Film) {
        let p = RefineParams { csize: [film.w, film.h], fsize: [self.fw, self.fh], dt: 0.2, wlo: 0.4e-3, whi: 1.2e-3, eps: 2e-6 };
        gpu.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        let gx = (self.fw + 15) / 16;
        let gy = (self.fh + 15) / 16;
        for bg in [&self.bg_h, &self.bg_m] {
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("refine") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("refine"), timestamp_writes: None });
                pass.set_bind_group(0, bg, &[]);
                pass.set_pipeline(&self.p_up);
                pass.dispatch_workgroups(gx, gy, 1);
                pass.set_pipeline(&self.p_weight);
                pass.dispatch_workgroups(gx, gy, 1);
                let iters = self.iterations / 2 * 2;
                for k in 0..iters {
                    pass.set_pipeline(if k % 2 == 0 { &self.p_ab } else { &self.p_ba });
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
            gpu.queue.submit(std::iter::once(enc.finish()));
        }
        let _ = &self.scratch;
        let _ = &self.weight;
    }
}
