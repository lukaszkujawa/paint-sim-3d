use std::sync::Arc;
use winit::window::Window;

pub struct Gpu {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
}

impl Gpu {
    pub fn new(window: Arc<Window>) -> Self {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc);
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("adapter");
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("paint-sim"),
            required_features: wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::SHADER_F16
                | wgpu::Features::FLOAT32_FILTERABLE,
            required_limits: limits,
            ..Default::default()
        }))
        .expect("device");
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| matches!(f, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm))
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            color_space: wgpu::SurfaceColorSpace::default(),
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        log::info!("adapter: {:?}", adapter.get_info());
        Self { surface, device, queue, config }
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn shader(&self, name: &str, src: &str) -> wgpu::ShaderModule {
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        })
    }

    pub fn storage(&self, name: &str, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn uniform(&self, name: &str, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: bytes.max(16),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn readback(&self, name: &str, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: bytes.max(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

pub struct BindBuilder<'a> {
    pub layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    pub entries: Vec<wgpu::BindGroupEntry<'a>>,
    pub stages: wgpu::ShaderStages,
}

impl<'a> BindBuilder<'a> {
    pub fn new(stages: wgpu::ShaderStages) -> Self {
        Self { layout_entries: vec![], entries: vec![], stages }
    }

    pub fn uniform(mut self, buf: &'a wgpu::Buffer) -> Self {
        let b = self.entries.len() as u32;
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: self.stages,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        self.entries.push(wgpu::BindGroupEntry { binding: b, resource: buf.as_entire_binding() });
        self
    }

    pub fn storage(mut self, buf: &'a wgpu::Buffer, read_only: bool) -> Self {
        let b = self.entries.len() as u32;
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: self.stages,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        self.entries.push(wgpu::BindGroupEntry { binding: b, resource: buf.as_entire_binding() });
        self
    }

    pub fn texture(mut self, view: &'a wgpu::TextureView, sample: wgpu::TextureSampleType) -> Self {
        let b = self.entries.len() as u32;
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: self.stages,
            ty: wgpu::BindingType::Texture {
                sample_type: sample,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        self.entries.push(wgpu::BindGroupEntry { binding: b, resource: wgpu::BindingResource::TextureView(view) });
        self
    }

    pub fn storage_texture(mut self, view: &'a wgpu::TextureView, format: wgpu::TextureFormat, access: wgpu::StorageTextureAccess) -> Self {
        let b = self.entries.len() as u32;
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: self.stages,
            ty: wgpu::BindingType::StorageTexture { access, format, view_dimension: wgpu::TextureViewDimension::D2 },
            count: None,
        });
        self.entries.push(wgpu::BindGroupEntry { binding: b, resource: wgpu::BindingResource::TextureView(view) });
        self
    }

    pub fn sampler(mut self, s: &'a wgpu::Sampler, filtering: bool) -> Self {
        let b = self.entries.len() as u32;
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: self.stages,
            ty: wgpu::BindingType::Sampler(if filtering {
                wgpu::SamplerBindingType::Filtering
            } else {
                wgpu::SamplerBindingType::NonFiltering
            }),
            count: None,
        });
        self.entries.push(wgpu::BindGroupEntry { binding: b, resource: wgpu::BindingResource::Sampler(s) });
        self
    }

    pub fn build(self, device: &wgpu::Device, name: &str) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(name),
            entries: &self.layout_entries,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(name),
            layout: &layout,
            entries: &self.entries,
        });
        (layout, group)
    }
}

pub fn compute_pipeline(device: &wgpu::Device, name: &str, module: &wgpu::ShaderModule, entry: &str, layouts: &[&wgpu::BindGroupLayout]) -> wgpu::ComputePipeline {
    let opts: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(|l| Some(*l)).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(name),
        bind_group_layouts: &opts,
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(name),
        layout: Some(&layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

pub fn dispatch(pass: &mut wgpu::ComputePass, pipeline: &wgpu::ComputePipeline, groups: &[&wgpu::BindGroup], x: u32, y: u32, z: u32) {
    pass.set_pipeline(pipeline);
    for (i, g) in groups.iter().enumerate() {
        pass.set_bind_group(i as u32, *g, &[]);
    }
    pass.dispatch_workgroups(x, y, z);
}
