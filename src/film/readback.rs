use super::*;

impl Film {
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
                    self.s_max = data[..np].iter().fold(0.0f32, |a, &v| if v.is_finite() { a.max(v) } else { a });
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

    pub fn poll_reservoir(&mut self, gpu: &Gpu, outs: &mut [Vec<[u32; 17]>; 3]) {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        for k in 0..2 {
            if self.res_pending[k].load(std::sync::atomic::Ordering::Acquire) == 2 {
                {
                    let out = &mut outs[(self.res_tool[k] as usize).min(2)];
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

    pub fn request_reservoir(&mut self, enc: &mut wgpu::CommandEncoder, tool: u8) -> Option<usize> {
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
        self.res_tool[k] = tool;
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
                    let n = (count as usize).min(MAX_SCRAPS);
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

    pub fn read_mean_colour(&self, gpu: &Gpu, region: [u32; 4]) -> [f32; 3] {
        let (cw, ch) = (region[2] - region[0], region[3] - region[1]);
        let bpr = ((cw * 8 + 255) / 256) * 256;
        let buf = gpu.readback("colour_pick", (bpr * ch) as u64);
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("colour_pick") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &self.color_tex, mip_level: 0, origin: wgpu::Origin3d { x: region[0], y: region[1], z: 0 }, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: None } },
            wgpu::Extent3d { width: cw, height: ch, depth_or_array_layers: 1 },
        );
        gpu.queue.submit(std::iter::once(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range().expect("map");
        let halves: &[u16] = bytemuck::cast_slice(&data);
        let row = (bpr / 2) as usize;
        let mut sum = [0.0f32; 3];
        for y in 0..ch as usize {
            for x in 0..cw as usize {
                for c in 0..3 {
                    sum[c] += half::f16::from_bits(halves[y * row + x * 4 + c]).to_f32();
                }
            }
        }
        drop(data);
        buf.unmap();
        sum.map(|v| v / (cw * ch) as f32)
    }

    pub fn read_state(&self, gpu: &Gpu) -> Vec<Vec<u8>> {
        let bufs = self.state_buffers();
        let reads: Vec<wgpu::Buffer> = bufs.iter().map(|b| gpu.readback("state_r", b.size())).collect();
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("state") });
        for (b, r) in bufs.iter().zip(&reads) {
            enc.copy_buffer_to_buffer(b, 0, r, 0, b.size());
        }
        gpu.queue.submit(std::iter::once(enc.finish()));
        for r in &reads {
            r.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        let _ = gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        reads
            .iter()
            .map(|r| {
                let v = r.slice(..).get_mapped_range().expect("map").to_vec();
                r.unmap();
                v
            })
            .collect()
    }

    pub fn write_state(&mut self, gpu: &Gpu, parts: &[Vec<u8>]) -> bool {
        let bufs = self.state_buffers();
        if parts.len() != bufs.len() || parts.iter().zip(bufs.iter()).any(|(p, b)| p.len() as u64 != b.size()) {
            return false;
        }
        for (p, b) in parts.iter().zip(bufs.iter()) {
            gpu.queue.write_buffer(b, 0, p);
        }
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("state_w") });
        for b in [&self.lap, &self.flux, &self.flux_s, &self.alpha, &self.delta, &self.tmp, &self.acc_s, &self.strata_h2, &self.strata_pig2, &self.alpha_s] {
            enc.clear_buffer(b, 0, None);
        }
        gpu.queue.submit(std::iter::once(enc.finish()));
        let nt = (self.tiles16[0] * self.tiles16[1]) as usize;
        gpu.queue.write_buffer(&self.tile_moved, 0, bytemuck::cast_slice(&vec![1u32; nt]));
        self.needs_full_shade = true;
        true
    }
}
