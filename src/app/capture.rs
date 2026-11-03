use crate::app::*;
use std::time::Instant;

impl App {
    pub(crate) fn beauty_render(&mut self, ortho: bool) {
        let (w, h) = if ortho { (self.film.w * 2, self.film.h * 2) } else { (6144u32, 4096u32) };
        let t0 = Instant::now();
        self.refine.run(&self.gpu, &self.film);
        let img = self.beauty.render(&self.gpu, &self.renderer, &self.cam, &self.film, ortho, w, h, 64, Some(&self.refine));
        let path = format!("beauty_{}_{}.png", if ortho { "top" } else { "view" }, self.frame_index);
        let _ = img.save(&path);
        self.last_beauty = format!("Saved {} ({}x{}, {:.1} s)", path, w, h, t0.elapsed().as_secs_f32());
        log::warn!("saved {}", self.last_beauty);
    }

    pub(crate) fn save_screenshot(&mut self, path: &str, tex: &wgpu::Texture) {
        let w = self.gpu.config.width;
        let h = self.gpu.config.height;
        let bpr = ((w * 4 + 255) / 256) * 256;
        let buf = self.gpu.readback("shot", (bpr * h) as u64);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("shot") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range().expect("map");
        let mut img = image::RgbaImage::new(w, h);
        let bgra = matches!(self.gpu.config.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
        for y in 0..h {
            for x in 0..w {
                let o = (y * bpr + x * 4) as usize;
                let (r, g, b) = if bgra { (data[o + 2], data[o + 1], data[o]) } else { (data[o], data[o + 1], data[o + 2]) };
                img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
            }
        }
        drop(data);
        buf.unmap();
        let [vx, vy, vw, vh] = self.view_rect.map(|v| v.max(0.0) as u32);
        let img = if vw > 0 && vh > 0 && (vw < w || vh < h) && !self.auto.as_ref().map(|a| a.show_ui).unwrap_or(false) { image::imageops::crop_imm(&img, vx, vy, vw.min(w - vx.min(w)), vh.min(h - vy.min(h))).to_image() } else { img };
        let _ = img.save(path);
        log::warn!("saved {}", path);
    }

    pub(crate) fn read_texture_f16(&self, tex: &wgpu::Texture) -> Vec<f32> {
        let w = self.film.w;
        let h = self.film.h;
        let bpr = ((w * 8 + 255) / 256) * 256;
        let buf = self.gpu.readback("export", (bpr * h) as u64);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("export") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range().expect("map");
        let mut out = vec![0.0f32; (w * h * 4) as usize];
        for y in 0..h {
            let row = &data[(y * bpr) as usize..(y * bpr + w * 8) as usize];
            let halves: &[u16] = bytemuck::cast_slice(row);
            for (i, v) in halves.iter().enumerate() {
                out[(y * w * 4) as usize + i] = half::f16::from_bits(*v).to_f32();
            }
        }
        drop(data);
        buf.unmap();
        out
    }

    pub(crate) fn read_texture_raw(&self, tex: &wgpu::Texture) -> Vec<u8> {
        let w = self.film.w;
        let h = self.film.h;
        let bpr = ((w * 8 + 255) / 256) * 256;
        let buf = self.gpu.readback("snap_tex", (bpr * h) as u64);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("snap") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range().expect("map");
        let mut out = Vec::with_capacity((w * h * 8) as usize);
        for y in 0..h {
            out.extend_from_slice(&data[(y * bpr) as usize..(y * bpr + w * 8) as usize]);
        }
        drop(data);
        buf.unmap();
        out
    }

    pub(crate) fn read_buffer_raw(&self, src: &wgpu::Buffer) -> Vec<u8> {
        let size = src.size();
        let buf = self.gpu.readback("snap_buf", size);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("snap") });
        enc.copy_buffer_to_buffer(src, 0, &buf, 0, size);
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let out = buf.slice(..).get_mapped_range().expect("map").to_vec();
        buf.unmap();
        out
    }

    pub(crate) fn save_snapshot(&mut self, path: &str) {
        self.compute_stats();
        let parts = [
            self.read_buffer_raw(&self.film.h_buf),
            self.read_buffer_raw(&self.film.m_buf),
            self.read_buffer_raw(&self.film.tile_flags),
            self.read_texture_raw(&self.film.color_tex),
            self.read_texture_raw(&self.film.detail_tex),
        ];
        let mut out = Vec::new();
        out.extend_from_slice(bytemuck::bytes_of(&[self.film.w, self.film.h]));
        out.extend_from_slice(bytemuck::bytes_of(&[self.film.h_max, self.film.s_max]));
        for p in &parts {
            out.extend_from_slice(bytemuck::bytes_of(&(p.len() as u64)));
            out.extend_from_slice(p);
        }
        let _ = std::fs::write(path, out);
        log::warn!("saved snapshot {}", path);
    }

    pub(crate) fn load_snapshot(&mut self, path: &str) {
        let Ok(data) = std::fs::read(path) else {
            log::warn!("cannot read snapshot {}", path);
            return;
        };
        let dims: [u32; 2] = bytemuck::pod_read_unaligned(&data[0..8]);
        if dims != [self.film.w, self.film.h] {
            log::warn!("snapshot is {}x{}, film is {}x{}", dims[0], dims[1], self.film.w, self.film.h);
            return;
        }
        let maxes: [f32; 2] = bytemuck::pod_read_unaligned(&data[8..16]);
        let mut off = 16usize;
        let mut parts = Vec::new();
        for _ in 0..5 {
            let len: u64 = bytemuck::pod_read_unaligned(&data[off..off + 8]);
            off += 8;
            parts.push(&data[off..off + len as usize]);
            off += len as usize;
        }
        let q = &self.gpu.queue;
        q.write_buffer(&self.film.h_buf, 0, parts[0]);
        q.write_buffer(&self.film.m_buf, 0, parts[1]);
        q.write_buffer(&self.film.tile_flags, 0, parts[2]);
        let w = self.film.w;
        let h = self.film.h;
        for (tex, bytes) in [(&self.film.color_tex, parts[3]), (&self.film.detail_tex, parts[4])] {
            q.write_texture(
                wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                bytes,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
        self.film.h_max = maxes[0];
        self.film.s_max = maxes[1];
        self.paused = true;
        log::warn!("loaded snapshot {}", path);
    }

    pub(crate) fn export_blender(&mut self, dir: &str) {
        let _ = std::fs::create_dir_all(dir);
        let w = self.film.w;
        let h = self.film.h;
        let n = (w * h) as u64;
        let buf = self.gpu.readback("exp_h", n * 4);
        let bufm = self.gpu.readback("exp_m", n * 4);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("export") });
        enc.copy_buffer_to_buffer(&self.film.h_buf, 0, &buf, 0, n * 4);
        enc.copy_buffer_to_buffer(&self.film.m_buf, 0, &bufm, 0, n * 4);
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        bufm.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let hv: Vec<f32> = bytemuck::cast_slice(&buf.slice(..).get_mapped_range().expect("map")).to_vec();
        let mv: Vec<f32> = bytemuck::cast_slice(&bufm.slice(..).get_mapped_range().expect("map")).to_vec();
        buf.unmap();
        bufm.unmap();
        let color = self.read_texture_f16(&self.film.color_tex);
        let detail = self.read_texture_f16(&self.film.detail_tex);
        let mut height = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::new(w, h);
        let mut albedo = image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::new(w, h);
        let mut rough = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::new(w, h);
        let mut mask = image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::new(w, h);
        let q = |v: f32| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        let mut hmax = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                let c = (y * w + x) as usize;
                let total = hv[c].max(0.0) + mv[c].max(0.0);
                hmax = hmax.max(total);
                height.put_pixel(x, h - 1 - y, image::Luma([(total * 1e6).clamp(0.0, 65535.0) as u16]));
                let col = &color[c * 4..c * 4 + 4];
                albedo.put_pixel(x, h - 1 - y, image::Rgb([q(col[0]), q(col[1]), q(col[2])]));
                rough.put_pixel(x, h - 1 - y, image::Luma([q(col[3])]));
                let d = &detail[c * 4..c * 4 + 4];
                let method = d[0].floor();
                let angle = (d[0] - method).clamp(0.0, 1.0);
                let brushed = if (method - 5.0).abs() < 0.5 || (method - 6.0).abs() < 0.5 { 1.0 } else { 0.0 };
                let wet = (hv[c] / 0.5e-3).clamp(0.0, 1.0);
                mask.put_pixel(x, h - 1 - y, image::Rgb([q(wet), q(angle), q(brushed)]));
            }
        }
        let jobs: Vec<(String, Vec<u16>, image::ColorType)> = vec![
            (format!("{}/height.png", dir), height.into_raw(), image::ColorType::L16),
            (format!("{}/albedo.png", dir), albedo.into_raw(), image::ColorType::Rgb16),
            (format!("{}/roughness.png", dir), rough.into_raw(), image::ColorType::L16),
            (format!("{}/mask.png", dir), mask.into_raw(), image::ColorType::Rgb16),
        ];
        use rayon::prelude::*;
        jobs.par_iter().for_each(|(path, data, ct)| {
            let be: &[u8] = bytemuck::cast_slice(data);
            if let Ok(f) = std::fs::File::create(path) {
                let wr = std::io::BufWriter::new(f);
                let enc = image::codecs::png::PngEncoder::new_with_quality(wr, image::codecs::png::CompressionType::Fast, image::codecs::png::FilterType::NoFilter);
                let _ = image::ImageEncoder::write_image(enc, be, w, h, (*ct).into());
            }
        });
        let meta = format!(
            "{{\n  \"cells\": [{}, {}],\n  \"size_m\": [{}, {}],\n  \"height_scale_m\": 0.065535,\n  \"height_max_m\": {:.6},\n  \"frame\": {}\n}}\n",
            w, h, self.film.settings.width_m, self.film.settings.height_m, hmax, self.frame_index
        );
        let _ = std::fs::write(format!("{}/canvas.json", dir), meta);
        let _ = std::fs::write(format!("{}/render.py", dir), include_str!("../../blender/render.py"));
        self.last_beauty = format!("Exported {} ({}x{})", dir, w, h);
        log::warn!("exported Blender set to {}", dir);
    }
}
