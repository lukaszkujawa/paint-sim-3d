use crate::app::*;

impl App {
    pub(crate) fn compute_stats(&mut self) {
        let n = (self.film.w * self.film.h) as u64;
        let buf = self.gpu.readback("stats", n * 4);
        let bufm = self.gpu.readback("statsm", n * 4);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stats") });
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
        if let Ok(prefix) = std::env::var("PAINT_DUMP") {
            let dims = [self.film.w, self.film.h];
            let _ = std::fs::write(format!("{}_dims.bin", prefix), bytemuck::cast_slice::<u32, u8>(&dims));
            let _ = std::fs::write(format!("{}_h.bin", prefix), bytemuck::cast_slice::<f32, u8>(&hv));
            let _ = std::fs::write(format!("{}_m.bin", prefix), bytemuck::cast_slice::<f32, u8>(&mv));
            let bd = self.gpu.readback("delta_r", n * 4);
            let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dump") });
            enc.copy_buffer_to_buffer(&self.film.delta, 0, &bd, 0, n * 4);
            self.gpu.queue.submit(std::iter::once(enc.finish()));
            bd.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            let dv: Vec<u32> = bytemuck::cast_slice(&bd.slice(..).get_mapped_range().expect("map")).to_vec();
            bd.unmap();
            let _ = std::fs::write(format!("{}_d.bin", prefix), bytemuck::cast_slice::<u32, u8>(&dv));
        }
        let dx = self.film.dx as f64;
        let mut sum = 0.0f64;
        let mut summ = 0.0f64;
        let mut hmax = 0.0f32;
        let mut smax = 0.0f32;
        let mut wet = 0u64;
        let mut nan = 0u64;
        for (a, b) in hv.iter().zip(mv.iter()) {
            if !a.is_finite() { nan += 1; continue; }
            sum += *a as f64;
            summ += *b as f64;
            if *a > 0.0 { wet += 1; }
            hmax = hmax.max(*a);
            smax = smax.max(*a + *b);
        }
        self.film.s_max = smax;
        let vol = sum * dx * dx;
        let volm = summ * dx * dx - self.film.weave_volume;
        if self.film.erased {
            self.film.total_deposited = vol + volm + self.tool_scraped - self.tool_released;
            self.film.erased = false;
        }
        {
            let (imax, _) = hv.iter().enumerate().fold((0usize, 0.0f32), |a, (i, v)| if *v > a.1 { (i, *v) } else { a });
            let w = self.film.w as usize;
            let row = imax / w;
            let col = imax % w;
            let mut prof = String::new();
            let mut k = 0usize;
            while col + k < w && k <= 1024 {
                prof.push_str(&format!(" {:.0}", hv[row * w + col + k] * 1e6));
                k += 32;
            }
            log::warn!("profile um every {:.1} mm from max:{}", 32.0 * self.film.dx * 1e3, prof);
            let mut fine = String::new();
            let mut k = 0usize;
            while col + k < w && k <= 240 && (k < 2 || hv[row * w + col + k - 1] > 0.0) {
                fine.push_str(&format!(" {:.0}", hv[row * w + col + k] * 1e6));
                k += 1;
            }
            log::warn!("fine profile um per cell from max:{}", fine);
            let mut nb = String::new();
            for dj in -3i64..=3 {
                nb.push_str(" |");
                for di in -3i64..=3 {
                    let ii = col as i64 + di;
                    let jj = row as i64 + dj;
                    if ii >= 0 && jj >= 0 && (ii as usize) < w && (jj as usize) < self.film.h as usize {
                        nb.push_str(&format!(" {:.0}", hv[jj as usize * w + ii as usize] * 1e6));
                    }
                }
            }
            log::warn!("7x7 um around max:{}", nb);
            let mut art = String::new();
            for dj in -60i64..=60 {
                art.push_str("
");
                for di in -80i64..=80 {
                    let ii = col as i64 + di * 2;
                    let jj = row as i64 + dj * 2;
                    let mut ch = ' ';
                    if ii >= 0 && jj >= 0 && (ii as usize) < w && (jj as usize) < self.film.h as usize {
                        let v = hv[jj as usize * w + ii as usize];
                        ch = if v > 2e-3 { '#' } else if v > 1e-3 { '%' } else if v > 3e-4 { '+' } else if v > 0.0 { '.' } else { ' ' };
                    }
                    art.push(ch);
                }
            }
            log::warn!("planform around max (rows 0.39 mm, cols 0.39 mm):{}", art);
            let nn = (self.film.w * self.film.h) as u64;
            let br = self.gpu.readback("rheo_r", nn * 8);
            let bs = self.gpu.readback("sh_r", nn * 16);
            let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stats3") });
            enc.copy_buffer_to_buffer(&self.film.rheo, 0, &br, 0, nn * 8);
            enc.copy_buffer_to_buffer(&self.film.strata_h, 0, &bs, 0, nn * 16);
            self.gpu.queue.submit(std::iter::once(enc.finish()));
            br.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            bs.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            let rv: Vec<[f32; 2]> = bytemuck::cast_slice(&br.slice(..).get_mapped_range().expect("map")).to_vec();
            let sv: Vec<[f32; 4]> = bytemuck::cast_slice(&bs.slice(..).get_mapped_range().expect("map")).to_vec();
            br.unmap();
            bs.unmap();
            let mut pr = String::new();
            let mut k = 0usize;
            while col + k < w && k <= 512 {
                let c = row * w + col + k;
                let st = sv[c];
                pr.push_str(&format!(" [mu {:.2} ty {:.1} st {:.0}/{:.0}/{:.0}/{:.0}]", rv[c][0], rv[c][1], st[0] * 1e6, st[1] * 1e6, st[2] * 1e6, st[3] * 1e6));
                k += 32;
            }
            log::warn!("rheo/strata along profile:{}", pr);
            let mut low_ty = 0usize;
            let mut no_strata = 0usize;
            let mut wetc = 0usize;
            for c in 0..(nn as usize) {
                if hv[c] > 0.0 {
                    wetc += 1;
                    if rv[c][1] < 1.0 { low_ty += 1; }
                    let st = sv[c];
                    if st[0] + st[1] + st[2] + st[3] <= 0.0 { no_strata += 1; }
                }
            }
            log::warn!("wet {} low_ty {} no_strata {}", wetc, low_ty, no_strata);
        }
        let cr = self.film.cg_report;
        log::warn!("implicit: {} steps, residual {:.2e}, drift {:.2e} of {:.3e}", self.film.implicit_steps, if cr[0] > 0.0 { (cr[3] / cr[0]).sqrt() } else { 0.0 }, cr[1], cr[2]);
        self.stats = format!(
            "wet {:.4} mL  cured {:.4} mL  deposited {:.4} mL  brush held {:.4} mL  err {:.2}%  hmax {:.3} mm  wet cells {}  active tiles {}  nan {}  fps {:.0}  substeps {}",
            vol * 1e6, volm * 1e6, self.film.total_deposited * 1e6, self.brush.total_load() * 1e6,
            if self.film.total_deposited > 0.0 { (vol + volm + self.tool_scraped - self.tool_released - self.film.total_deposited) / self.film.total_deposited * 100.0 } else { 0.0 },
            hmax * 1e3, wet, self.film.active_tiles, nan, self.fps, self.film.substeps_last
        );
        log::warn!("{}", self.stats);
    }
}
