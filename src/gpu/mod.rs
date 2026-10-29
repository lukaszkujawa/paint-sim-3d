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
            label: Some("canvaz"),
            required_features: wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::SHADER_F16
                | wgpu::Features::FLOAT32_FILTERABLE
                | (adapter.features() & wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
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

pub struct GpuProf {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    labels: Vec<&'static str>,
    totals: Vec<(&'static str, f64, u32)>,
    period: f64,
    frames: u32,
    cap: u32,
    pub every: u32,
}

impl GpuProf {
    pub const CAP: u32 = 4096;

    pub fn new(gpu: &Gpu) -> Option<Self> {
        let every: u32 = std::env::var("PAINT_GPUPROF").ok()?.parse().unwrap_or(120);
        if !gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES) {
            log::warn!("PAINT_GPUPROF needs TIMESTAMP_QUERY_INSIDE_PASSES");
            return None;
        }
        let cap = Self::CAP;
        let set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("prof"), ty: wgpu::QueryType::Timestamp, count: cap });
        let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("prof_resolve"), size: cap as u64 * 8, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = gpu.readback("prof_read", cap as u64 * 8);
        Some(Self { set, resolve, read, labels: Vec::new(), totals: Vec::new(), period: gpu.queue.get_timestamp_period() as f64, frames: 0, cap, every: every.max(1) })
    }

    pub fn mark(&mut self, pass: &mut wgpu::ComputePass, label: &'static str) {
        if (self.labels.len() as u32) < self.cap {
            pass.write_timestamp(&self.set, self.labels.len() as u32);
            self.labels.push(label);
        }
    }

    pub fn render_writes(&mut self, label: &'static str) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let i = self.labels.len() as u32;
        if i + 2 > self.cap {
            return None;
        }
        self.labels.push(label);
        self.labels.push("");
        Some(wgpu::RenderPassTimestampWrites { query_set: &self.set, beginning_of_pass_write_index: Some(i), end_of_pass_write_index: Some(i + 1) })
    }

    pub fn finish(&mut self, gpu: &Gpu, enc: wgpu::CommandEncoder) {
        let mut enc = enc;
        let n = self.labels.len() as u32;
        if n > 0 {
            enc.resolve_query_set(&self.set, 0..n, &self.resolve, 0);
            enc.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, n as u64 * 8);
        }
        gpu.queue.submit(std::iter::once(enc.finish()));
        if n == 0 {
            return;
        }
        self.read.slice(..n as u64 * 8).map_async(wgpu::MapMode::Read, |_| {});
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        {
            let view = self.read.slice(..n as u64 * 8).get_mapped_range().expect("map");
            let ts: &[u64] = bytemuck::cast_slice(&view);
            for k in 0..ts.len().saturating_sub(1) {
                let l = self.labels[k];
                if l.is_empty() {
                    continue;
                }
                let ms = ts[k + 1].saturating_sub(ts[k]) as f64 * self.period * 1e-6;
                match self.totals.iter_mut().find(|t| t.0 == l) {
                    Some(t) => {
                        t.1 += ms;
                        t.2 += 1;
                    }
                    None => self.totals.push((l, ms, 1)),
                }
            }
        }
        self.read.unmap();
        self.labels.clear();
        self.frames += 1;
        if self.frames % self.every == 0 {
            let mut rows = self.totals.clone();
            rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let sum: f64 = rows.iter().map(|r| r.1).sum();
            let line: Vec<String> = rows.iter().take(16).map(|r| format!("{} {:.2}", r.0, r.1 / self.every as f64)).collect();
            log::warn!("gpu ms/frame {:.2}: {}", sum / self.every as f64, line.join("  "));
            self.totals.clear();
        }
    }
}

pub struct GpuClock {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: [wgpu::Buffer; 3],
    state: [std::sync::Arc<std::sync::atomic::AtomicU32>; 3],
    slot: Option<usize>,
    next: usize,
    period: f64,
    pub sim_ms: f32,
    pub render_ms: f32,
    pub valid: bool,
}

impl GpuClock {
    pub fn new(gpu: &Gpu) -> Self {
        let set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("clock"), ty: wgpu::QueryType::Timestamp, count: 12 });
        let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("clock_resolve"), size: 3 * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = [gpu.readback("clock_read", 32), gpu.readback("clock_read", 32), gpu.readback("clock_read", 32)];
        let state = std::array::from_fn(|_| std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)));
        Self { set, resolve, read, state, slot: None, next: 0, period: gpu.queue.get_timestamp_period() as f64, sim_ms: 0.0, render_ms: 0.0, valid: false }
    }

    pub fn begin(&mut self) {
        let k = self.next;
        self.slot = if self.state[k].load(std::sync::atomic::Ordering::Acquire) == 0 { Some(k) } else { None };
        if self.slot.is_some() {
            self.next = (k + 1) % 3;
        }
    }

    fn write(&self, q: u32) -> Option<u32> {
        self.slot.map(|k| k as u32 * 4 + q)
    }

    pub fn compute(&self, start: bool) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let i = self.write(if start { 0 } else { 1 })?;
        Some(wgpu::ComputePassTimestampWrites { query_set: &self.set, beginning_of_pass_write_index: if start { Some(i) } else { None }, end_of_pass_write_index: if start { None } else { Some(i) } })
    }

    pub fn render(&self, start: bool) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let i = self.write(if start { 2 } else { 3 })?;
        Some(wgpu::RenderPassTimestampWrites { query_set: &self.set, beginning_of_pass_write_index: if start { Some(i) } else { None }, end_of_pass_write_index: if start { None } else { Some(i) } })
    }

    pub fn resolve(&self, enc: &mut wgpu::CommandEncoder) {
        let Some(k) = self.slot else { return };
        let q = k as u32 * 4;
        let at = k as u64 * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;
        enc.resolve_query_set(&self.set, q..q + 4, &self.resolve, at);
        enc.copy_buffer_to_buffer(&self.resolve, at, &self.read[k], 0, 32);
        self.state[k].store(1, std::sync::atomic::Ordering::Release);
    }

    pub fn map(&mut self) {
        let Some(k) = self.slot.take() else { return };
        if self.state[k].load(std::sync::atomic::Ordering::Acquire) != 1 {
            return;
        }
        let flag = self.state[k].clone();
        self.read[k].slice(..).map_async(wgpu::MapMode::Read, move |r| {
            flag.store(if r.is_ok() { 2 } else { 0 }, std::sync::atomic::Ordering::Release);
        });
    }

    pub fn poll(&mut self) {
        for k in 0..3 {
            if self.state[k].load(std::sync::atomic::Ordering::Acquire) != 2 {
                continue;
            }
            {
                let view = self.read[k].slice(..).get_mapped_range().expect("map");
                let ts: &[u64] = bytemuck::cast_slice(&view);
                let ms = |a: u64, b: u64| (b.saturating_sub(a) as f64 * self.period * 1e-6) as f32;
                let (sim, render) = (ms(ts[0], ts[1]), ms(ts[2], ts[3]));
                if sim < 1e3 && render < 1e3 {
                    self.sim_ms = sim;
                    self.render_ms = render;
                    self.valid = true;
                }
            }
            self.read[k].unmap();
            self.state[k].store(0, std::sync::atomic::Ordering::Release);
        }
    }
}
