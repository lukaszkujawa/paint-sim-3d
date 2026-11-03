use super::*;

pub struct Journal {
    pub gen: u32,
    pub cap: u32,
    pub touch: Vec<u32>,
    pub touched: bool,
    pub(super) params: wgpu::Buffer,
    pub(super) jhead: wgpu::Buffer,
    pub(super) touch_buf: wgpu::Buffer,
    pub(super) best: wgpu::Buffer,
    pub bg: wgpu::BindGroup,
    pub p_journal: wgpu::ComputePipeline,
    pub(super) p_scan: wgpu::ComputePipeline,
    pub(super) p_apply: wgpu::ComputePipeline,
    pub(super) head_read: wgpu::Buffer,
}

impl Journal {
    pub(super) fn new(gpu: &Gpu, common: &str, layouts: [&wgpu::BindGroupLayout; 3], ntiles: u32) -> Self {
        let tile_bytes = JOURNAL_CELL_BYTES * 256;
        let limit = (gpu.device.limits().max_storage_buffer_binding_size as u64).min(gpu.device.limits().max_buffer_size);
        let budget: u64 = std::env::var("PAINT_UNDO_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(384) * 1024 * 1024;
        let cap = (budget.min(limit) / tile_bytes).max(64) as u32;
        let params = gpu.uniform("journal_params", 16);
        let jmeta = gpu.storage("jmeta", cap as u64 * 16);
        let jdata = gpu.storage("jdata", cap as u64 * tile_bytes);
        let jhead = gpu.storage("jhead", 16);
        let touch_buf = gpu.storage("touch", ntiles as u64 * 4);
        let jmark = gpu.storage("jmark", ntiles as u64 * 4);
        let best = gpu.storage("best", ntiles as u64 * 4);
        let (lj, bg) = BindBuilder::new(wgpu::ShaderStages::COMPUTE)
            .uniform(&params)
            .storage(&jmeta, false)
            .storage(&jdata, false)
            .storage(&jhead, false)
            .storage(&touch_buf, true)
            .storage(&jmark, false)
            .storage(&best, false)
            .build(&gpu.device, "journal");
        let src = format!("{}\n{}", common, include_str!("../../shaders/journal.wgsl"));
        let sm = gpu.shader("journal", &src);
        let l: [&wgpu::BindGroupLayout; 4] = [layouts[0], layouts[1], &lj, layouts[2]];
        let p_journal = compute_pipeline(&gpu.device, "journal", &sm, "journal", &l);
        let p_scan = compute_pipeline(&gpu.device, "restore_scan", &sm, "restore_scan", &l);
        let p_apply = compute_pipeline(&gpu.device, "restore_apply", &sm, "restore_apply", &l);
        let head_read = gpu.readback("jhead_r", 16);
        Self { gen: 0, cap, touch: vec![0; ntiles as usize], touched: false, params, jhead, touch_buf, best, bg, p_journal, p_scan, p_apply, head_read }
    }

    pub(super) fn write_params(&self, gpu: &Gpu, target: u32) {
        gpu.queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&[self.gen, self.cap, target, 0u32]));
    }

    pub(super) fn flush(&mut self, gpu: &Gpu) {
        self.write_params(gpu, 0);
        gpu.queue.write_buffer(&self.touch_buf, 0, bytemuck::cast_slice(&self.touch));
        if self.touched {
            self.touch.iter_mut().for_each(|t| *t = 0);
            self.touched = false;
        }
    }

    pub fn head(&self, gpu: &Gpu) -> u32 {
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("jhead") });
        enc.copy_buffer_to_buffer(&self.jhead, 0, &self.head_read, 0, 16);
        gpu.queue.submit(std::iter::once(enc.finish()));
        self.head_read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let v: u32 = bytemuck::cast_slice::<u8, u32>(&self.head_read.slice(..).get_mapped_range().expect("map"))[0];
        self.head_read.unmap();
        v
    }

    pub(super) fn restore(&self, gpu: &Gpu, groups: [&wgpu::BindGroup; 3], target: u32) {
        self.write_params(gpu, target);
        let nt = self.touch.len();
        gpu.queue.write_buffer(&self.best, 0, bytemuck::cast_slice(&vec![u32::MAX; nt]));
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("restore") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("restore"), timestamp_writes: None });
            let g = [groups[0], groups[1], &self.bg, groups[2]];
            dispatch(&mut pass, &self.p_scan, &g, (self.cap + 63) / 64, 1, 1);
            dispatch(&mut pass, &self.p_apply, &g, self.cap.min(65535), (self.cap + 65534) / 65535, 1);
        }
        gpu.queue.submit(std::iter::once(enc.finish()));
        self.write_params(gpu, 0);
    }
}
