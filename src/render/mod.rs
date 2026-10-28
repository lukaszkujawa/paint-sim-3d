pub mod refine;
pub use refine::Refine;
use crate::film::Film;
use crate::gpu::{BindBuilder, Gpu};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CamUniform {
    pub inv_vp: [[f32; 4]; 4],
    pub eye: [f32; 4],
    pub canvas: [f32; 4],
    pub light: [f32; 4],
    pub size: [u32; 4],
    pub sky: [f32; 4],
    pub ground: [f32; 4],
    pub vp: [[f32; 4]; 4],
    pub jitter: [f32; 4],
    pub ortho: [f32; 4],
    pub basis_r: [f32; 4],
    pub basis_u: [f32; 4],
    pub lens: [f32; 4],
}

pub struct Camera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub fov: f32,
    pub aspect: f32,
}

impl Camera {
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(cp * cy, cp * sy, sp) * self.dist
    }

    pub fn view_proj(&self) -> Mat4 {
        let eye = self.eye();
        let view = glam::camera::rh::view::look_at_mat4(eye, self.target, Vec3::Z);
        let proj = glam::camera::rh::proj::directx::perspective(self.fov, self.aspect, 0.01, 20.0);
        proj * view
    }

    pub fn ray(&self, ndc: [f32; 2]) -> (Vec3, Vec3) {
        let inv = self.view_proj().inverse();
        let a = inv.project_point3(Vec3::new(ndc[0], ndc[1], 0.0));
        let b = inv.project_point3(Vec3::new(ndc[0], ndc[1], 1.0));
        (a, (b - a).normalize())
    }

    pub fn plane_hit(&self, ndc: [f32; 2], z: f32) -> Option<Vec3> {
        let (o, d) = self.ray(ndc);
        if d.z.abs() < 1e-6 {
            return None;
        }
        let t = (z - o.z) / d.z;
        if t < 0.0 {
            return None;
        }
        Some(o + d * t)
    }
}

fn pixel_footprint(cam: &Camera, viewport_height: u32) -> f32 {
    cam.dist * 2.0 * (0.5 * cam.fov).tan() / viewport_height.max(1) as f32
}

pub struct Renderer {
    pub cam_buf: wgpu::Buffer,
    pub bg: wgpu::BindGroup,
    pub layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
    pub light_dir: Vec3,
    pub light_intensity: f32,
    pub sky: [f32; 3],
    pub ground: [f32; 3],
    pub aperture_mm: f32,
    pub exposure: f32,
    pub bloom: f32,
    pub grain: f32,
    pub vignette: f32,
    pub sharpen: f32,
    pub agx: bool,
    pub key_elevation_deg: f32,
    pub key_azimuth_deg: f32,
    pub key_power: f32,
    pub fill: f32,
}

impl Renderer {
    pub fn new(gpu: &Gpu, film: &Film) -> Self {
        let cam_buf = gpu.uniform("cam", std::mem::size_of::<CamUniform>() as u64);
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let stages = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX;
        let (layout, bg) = BindBuilder::new(stages)
            .uniform(&cam_buf)
            .texture(&film.color_view, wgpu::TextureSampleType::Float { filterable: true })
            .sampler(&sampler, true)
            .storage(&film.h_buf, true)
            .storage(&film.m_buf, true)
            .storage(&film.tile_flags, true)
            .texture(&film.detail_view, wgpu::TextureSampleType::Float { filterable: true })
            .build(&gpu.device, "render");
        let module = gpu.shader("render", include_str!("../../shaders/render.wgsl"));
        let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render"),
            layout: Some(&pl),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: gpu.config.format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            cam_buf,
            bg,
            layout,
            pipeline,
            light_dir: Vec3::new(-0.5, 0.35, 0.8).normalize(),
            light_intensity: 2.6,
            sky: [0.55, 0.62, 0.75],
            ground: [0.22, 0.2, 0.18],
            aperture_mm: 0.8,
            exposure: -0.35,
            bloom: 0.25,
            grain: 0.006,
            vignette: 0.05,
            sharpen: 0.9,
            agx: false,
            key_elevation_deg: 40.0,
            key_azimuth_deg: 145.0,
            key_power: 3.2,
            fill: 0.95,
        }
    }

    pub fn uniform(&self, cam: &Camera, film: &Film) -> CamUniform {
        let vp = cam.view_proj();
        let eye = cam.eye();
        CamUniform {
            inv_vp: vp.inverse().to_cols_array_2d(),
            eye: [eye.x, eye.y, eye.z, 0.0],
            canvas: [film.settings.width_m, film.settings.height_m, film.dx, film.h_max.max(film.s_max) + 0.5e-3],
            light: [self.light_dir.x, self.light_dir.y, self.light_dir.z, self.light_intensity],
            size: [film.w, film.h, 0, 0],
            sky: [self.sky[0], self.sky[1], self.sky[2], 1.0],
            ground: [self.ground[0], self.ground[1], self.ground[2], 1.0],
            vp: vp.to_cols_array_2d(),
            jitter: [0.0; 4],
            ortho: [0.0, film.settings.contact_angle_deg.to_radians().tan().max(0.05), 0.0, 0.0],
            basis_r: [1.0, 0.0, 0.0, 0.0],
            basis_u: [0.0, 1.0, 0.0, 0.0],
            lens: [0.0; 4],
        }
    }

    pub fn update(&self, gpu: &Gpu, cam: &Camera, film: &Film) {
        let mut u = self.uniform(cam, film);
        u.jitter[3] = pixel_footprint(cam, gpu.config.height);
        gpu.queue.write_buffer(&self.cam_buf, 0, bytemuck::bytes_of(&u));
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

pub struct TubeRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub bg: wgpu::BindGroup,
    pub vbuf: wgpu::Buffer,
    pub ibuf: wgpu::Buffer,
    pub vcap: usize,
    pub icap: usize,
    pub count: u32,
    pub depth: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
}

impl TubeRenderer {
    pub fn new(gpu: &Gpu, cam_buf: &wgpu::Buffer) -> Self {
        let stages = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let (layout, bg) = BindBuilder::new(stages).uniform(cam_buf).build(&gpu.device, "tube");
        let module = gpu.shader("tube", include_str!("../../shaders/tube.wgsl"));
        let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tube"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let vlayout = wgpu::VertexBufferLayout {
            array_stride: 36,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 },
            ],
        };
        let pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tube"),
            layout: Some(&pl),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[Some(vlayout)] },
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: gpu.config.format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let vcap = 1 << 18;
        let icap = 1 << 20;
        let vbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("tube_v"), size: (vcap * 36) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let ibuf = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("tube_i"), size: (icap * 4) as u64, usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let (depth, depth_view) = Self::make_depth(gpu);
        Self { pipeline, bg, vbuf, ibuf, vcap, icap, count: 0, depth, depth_view }
    }

    fn make_depth(gpu: &Gpu) -> (wgpu::Texture, wgpu::TextureView) {
        let depth = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width: gpu.config.width.max(1), height: gpu.config.height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = depth.create_view(&Default::default());
        (depth, view)
    }

    pub fn resize(&mut self, gpu: &Gpu) {
        let (d, v) = Self::make_depth(gpu);
        self.depth = d;
        self.depth_view = v;
    }

    pub fn upload(&mut self, gpu: &Gpu, verts: &[crate::tools::TubeVertex], idx: &[u32]) {
        let nv = verts.len().min(self.vcap);
        let mut ni = idx.len().min(self.icap);
        ni -= ni % 3;
        if nv > 0 {
            gpu.queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&verts[..nv]));
        }
        if ni > 0 {
            gpu.queue.write_buffer(&self.ibuf, 0, bytemuck::cast_slice(&idx[..ni]));
        }
        self.count = if nv == verts.len() { ni as u32 } else { 0 };
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass) {
        if self.count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bg, &[]);
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.count, 0, 0..1);
    }
}

pub struct Beauty {
    pipeline: wgpu::RenderPipeline,
    resolve: wgpu::RenderPipeline,
    resolve_layout: wgpu::BindGroupLayout,
    res_buf: wgpu::Buffer,
}

impl Beauty {
    pub fn new(gpu: &Gpu, renderer: &Renderer) -> Self {
        let module = gpu.shader("render_beauty", include_str!("../../shaders/render.wgsl"));
        let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("beauty"),
            bind_group_layouts: &[Some(&renderer.layout)],
            immediate_size: 0,
        });
        let add = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
        let pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("beauty"),
            layout: Some(&pl),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: Some(wgpu::BlendState { color: add, alpha: add }), write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let resolve_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resolve"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let rmod = gpu.shader("resolve", include_str!("../../shaders/resolve.wgsl"));
        let rpl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resolve"),
            bind_group_layouts: &[Some(&resolve_layout)],
            immediate_size: 0,
        });
        let resolve = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("resolve"),
            layout: Some(&rpl),
            vertex: wgpu::VertexState { module: &rmod, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &rmod,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8Unorm, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let res_buf = gpu.uniform("resolve", 48);
        Self { pipeline, resolve, resolve_layout, res_buf }
    }

    pub fn render(&self, gpu: &Gpu, renderer: &Renderer, cam: &Camera, film: &Film, ortho: bool, width: u32, height: u32, spp: u32, fine: Option<&Refine>) -> image::RgbaImage {
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let acc = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("beauty_acc"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let out = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("beauty_out"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let acc_view = acc.create_view(&Default::default());
        let out_view = out.create_view(&Default::default());
        let view_cam = Camera { target: cam.target, yaw: cam.yaw, pitch: cam.pitch, dist: cam.dist, fov: cam.fov, aspect: width as f32 / height as f32 };
        let mut base = renderer.uniform(&view_cam, film);
        if let Some(f) = fine {
            base.size = [f.fw, f.fh, 0, 0];
            base.canvas[2] = f.dx;
            base.ortho[2] = film.w as f32 / f.fw as f32;
        }
        let bind = match fine {
            Some(f) => &f.bg_render,
            None => &renderer.bg,
        };
        let (el, az) = (renderer.key_elevation_deg.to_radians(), renderer.key_azimuth_deg.to_radians());
        let l = Vec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
        base.sky = [renderer.sky[0] * renderer.fill, renderer.sky[1] * renderer.fill, renderer.sky[2] * renderer.fill, 1.0];
        base.ground = [renderer.ground[0] * renderer.fill, renderer.ground[1] * renderer.fill, renderer.ground[2] * renderer.fill, 1.0];
        let u1 = if l.z.abs() < 0.9 { l.cross(Vec3::Z).normalize() } else { l.cross(Vec3::X).normalize() };
        let u2 = l.cross(u1);
        let eye = view_cam.eye();
        let fwd = (view_cam.target - eye).normalize();
        let right = fwd.cross(Vec3::Z).normalize();
        let up = right.cross(fwd);
        let aperture = renderer.aperture_mm * 1e-3;
        let grid = (spp as f32).sqrt().ceil().max(1.0) as u32;
        for i in 0..spp {
            let sx = ((i % grid) as f32 + 0.5) / grid as f32 - 0.5;
            let sy = ((i / grid) as f32 + 0.5) / grid as f32 - 0.5;
            let phi = (i as f32 * 2.399963) % (2.0 * std::f32::consts::PI);
            let rr = 0.12 * (((i as f32 + 0.5) / spp as f32).sqrt());
            let lj = (l + u1 * (rr * phi.cos()) + u2 * (rr * phi.sin())).normalize();
            let phi_l = (i as f32 * 2.399963 + 1.7) % (2.0 * std::f32::consts::PI);
            let r_l = aperture * ((i as f32 + 0.37) / spp as f32).sqrt();
            let mut u = base;
            let footprint = if ortho { film.settings.width_m / width as f32 } else { pixel_footprint(&view_cam, height) };
            u.jitter = [sx * 2.0 / width as f32, sy * 2.0 / height as f32, 1.0, footprint];
            u.ortho = [if ortho { 1.0 } else { 0.0 }, base.ortho[1], base.ortho[2], 0.0];
            u.light = [lj.x, lj.y, lj.z, renderer.key_power * std::f32::consts::PI];
            u.basis_r = [right.x, right.y, right.z, 0.0];
            u.basis_u = [up.x, up.y, up.z, 0.0];
            u.lens = [r_l * phi_l.cos(), r_l * phi_l.sin(), view_cam.dist, if !ortho && aperture > 0.0 { 1.0 } else { 0.0 }];
            gpu.queue.write_buffer(&renderer.cam_buf, 0, bytemuck::bytes_of(&u));
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("beauty") });
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("beauty"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &acc_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: if i == 0 { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) } else { wgpu::LoadOp::Load }, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.draw(0..3, 0..1);
            }
            gpu.queue.submit(std::iter::once(enc.finish()));
        }
        let res = [1.0f32 / spp as f32, renderer.exposure, renderer.bloom, renderer.grain, renderer.vignette, width as f32, height as f32, if renderer.agx { 1.0 } else { 0.0 }, renderer.sharpen, 0.0, 0.0, 0.0];
        gpu.queue.write_buffer(&self.res_buf, 0, bytemuck::bytes_of(&res));
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resolve"),
            layout: &self.resolve_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&acc_view) },
                wgpu::BindGroupEntry { binding: 1, resource: self.res_buf.as_entire_binding() },
            ],
        });
        let bpr = ((width * 4 + 255) / 256) * 256;
        let buf = gpu.readback("beauty_read", (bpr * height) as u64);
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("resolve") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &out_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.resolve);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &out, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(height) } },
            size,
        );
        gpu.queue.submit(std::iter::once(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range().expect("map");
        let mut img = image::RgbaImage::new(width, height);
        for y in 0..height {
            let row = &data[(y * bpr) as usize..(y * bpr + width * 4) as usize];
            for x in 0..width {
                let o = (x * 4) as usize;
                img.put_pixel(x, y, image::Rgba([row[o], row[o + 1], row[o + 2], 255]));
            }
        }
        drop(data);
        buf.unmap();
        img
    }
}
