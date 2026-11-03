use super::*;

const FIT_FULL_EV: f32 = 1.0;

const FIT_MIN_EV: f32 = -3.0;

const FIT_MARGIN_L: f32 = 8.0;

const TONE_GAMMA: f32 = 1.3;

const TONE_MIN_SPAN: f32 = 20.0;

const TONE_HIGH: f32 = 0.995;

const TONE_LOW: f32 = 0.01;

const TENSOR_RADIUS: usize = 4;

const WARMUP_FRAMES: u32 = 3;

const CHANGED_DE: f32 = 4.0;

const SIZE_CHANGED: f64 = 0.01;

const ETF_ITERS: usize = 3;

const ETF_RADIUS: usize = 10;

const DETAIL_RADIUS: usize = 6;

const KIN_DE: f32 = 15.0;

const EFFICIENCY_RATE: f32 = 0.1;

const RETRY_SECS: f32 = 3.0;

const TILE_EFF_RATE: f32 = 0.3;

const EDGE_WEIGHT: f32 = 2.0;

const EVAL_WIDTH: u32 = 1536;

pub struct Evolver {
    pub(super) base: image::RgbaImage,
    pub(super) first: Vec<[f32; 3]>,
    pub prep: Prep,
    pub(super) gain: Option<[f32; 3]>,
    pub(super) offset: [f32; 3],
    pub(super) base_gain: [f32; 3],
    pub(super) fit_ev: f32,
    pub w: usize,
    pub h: usize,
    pub factor: u32,
    pub(super) cell: f64,
    pub(super) photo: Vec<[f32; 3]>,
    pub(super) white: [f32; 3],
    pub(super) black: [f32; 3],
    pub calibrated: bool,
    pub(super) palette: Arc<Vec<Swatch>>,
    pub(super) model: Option<Arc<Model>>,
    pub(super) mask: Vec<f32>,
    pub(super) focus: Option<Focus>,
    pub(super) current: Vec<[f32; 3]>,
    pub(super) err: Vec<f32>,
    pub(super) scratch: Vec<f32>,
    pub(super) rng: StdRng,
    pub(super) tool: Option<(ToolKind, BrushShape)>,
    pub(super) range: Option<ToolRange>,
    pub(super) reset: Option<f64>,
    pub(super) photo_gen: u64,
    pub(super) applied: Option<f64>,
    pub(super) seen: Option<f64>,
    pub size_hi: f64,
    pub(super) width_gain: f64,
    pub(super) opacity: f32,
    pub(super) efficiency: f32,
    pub(super) tile_eff: Arc<Vec<f32>>,
    pub(super) planned_ref: Option<Arc<Vec<[f32; 3]>>>,
    pub(super) sketch: Option<Sketch>,
    pub(super) job: Option<std::thread::JoinHandle<(Sketch, Batch)>>,
    pub(super) ready: Option<Batch>,
    pub(super) queue: VecDeque<Plan>,
    pub(super) synced: bool,
    pub(super) gen: u64,
    pub finished: bool,
    pub(super) dry_next: bool,
    pub(super) jobs: u32,
    pub(super) empty_batches: u32,
    pub(super) retry_at: Option<std::time::Instant>,
    pub tried: u64,
    pub painted: u64,
    pub kept: u64,
    pub pruned: u64,
    pub passes: u32,
    pub start_err: f32,
    pub mean_err: f32,
    pub status: String,
    pub phase: Phase,
}

impl Evolver {
    pub(super) fn linear(img: &image::RgbaImage, prep: Prep, cell_mm: f32) -> Vec<[f32; 3]> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let img = prepare(img, prep, cell_mm);
        (0..w * h)
            .into_par_iter()
            .map(|k| {
                let p = img.get_pixel((k % w) as u32, (h - 1 - k / w) as u32);
                [decode(p[0] as f32 / 255.0), decode(p[1] as f32 / 255.0), decode(p[2] as f32 / 255.0)]
            })
            .collect()
    }

    pub fn new(photo: &image::RgbaImage, film_w: u32, film_h: u32, dx: f32, set: &PigmentSet, cmf: &[f32], seed: u64, prep: Prep) -> Self {
        let factor = (film_w / EVAL_WIDTH).max(1);
        let (w, h) = ((film_w / factor) as usize, (film_h / factor) as usize);
        let img = image::imageops::resize(&cover_crop(photo, w as f32 / h as f32), w as u32, h as u32, image::imageops::FilterType::Triangle);
        let white = color::mix_rgb(set, cmf, &color::PaintMix::pure(WHITE).frac).map(decode);
        let black = color::mix_rgb(set, cmf, &color::PaintMix::pure(BLACK).frac).map(decode);
        let photo = Self::linear(&img, prep, dx * factor as f32 * 1e3);
        Self {
            base: img,
            first: Vec::new(),
            prep,
            gain: None,
            offset: [0.0; 3],
            base_gain: [1.0; 3],
            fit_ev: 0.0,
            w,
            h,
            factor,
            cell: dx as f64 * factor as f64,
            photo,
            white,
            black,
            calibrated: false,
            palette: Arc::new(swatches(set, cmf)),
            model: None,
            mask: vec![1.0; w * h],
            focus: None,
            current: vec![[0.0; 3]; w * h],
            err: vec![0.0; w * h],
            scratch: vec![0.0; w * h],
            rng: StdRng::seed_from_u64(seed),
            tool: None,
            range: None,
            reset: None,
            photo_gen: 0,
            applied: None,
            seen: None,
            size_hi: 0.0,
            width_gain: 1.0,
            opacity: 0.9,
            efficiency: 1.0,
            tile_eff: Arc::new(vec![1.0; w.div_ceil(TILE) * h.div_ceil(TILE)]),
            planned_ref: None,
            sketch: None,
            job: None,
            ready: None,
            queue: VecDeque::new(),
            synced: false,
            gen: 0,
            finished: false,
            empty_batches: 0,
            retry_at: None,
            dry_next: false,
            jobs: 0,
            tried: 0,
            painted: 0,
            kept: 0,
            pruned: 0,
            passes: 0,
            start_err: 0.0,
            mean_err: 0.0,
            status: String::new(),
            phase: Phase::Wait(WARMUP_FRAMES),
        }
    }

    pub fn set_prep(&mut self, prep: Prep) {
        self.prep = prep;
        self.planned_ref = None;
        self.photo = Self::linear(&self.base, prep, (self.cell * 1e3) as f32);
        if self.gain.is_some() {
            let gain = self.base_gain.map(|g| g * (-prep.lift_ev).exp2());
            self.calibrate(gain, self.offset);
        }
        self.sketch = None;
        self.gen += 1;
        self.queue.clear();
        self.ready = None;
        self.synced = false;
        self.finished = false;
        self.start_err = 0.0;
    }

    pub(super) fn range_fit(&self, gain: [f32; 3]) -> f32 {
        let mut sample: Vec<f32> = self.photo.iter().step_by(QUANTILE_STRIDE).map(|p| lab(*p)[0]).collect();
        sample.sort_by(|a, b| a.total_cmp(b));
        let want = sample[((sample.len() - 1) as f32 * TONE_HIGH) as usize] + FIT_MARGIN_L;
        let white_at = |ev: f32| view_lab(self.white, gain.map(|g| g * ev.exp2()))[0];
        if white_at(0.0) <= want {
            return 0.0;
        }
        let (mut lo, mut hi) = (FIT_MIN_EV, 0.0f32);
        for _ in 0..24 {
            let mid = 0.5 * (lo + hi);
            if white_at(mid) < want { lo = mid } else { hi = mid }
        }
        hi
    }

    pub fn calibrate(&mut self, gain: [f32; 3], offset: [f32; 3]) {
        self.gain = Some(gain);
        self.offset = offset;
        let (white, black) = (self.white, self.black);
        let show = self.base_gain;
        let (lb, lw) = (view_lab(black, show)[0], view_lab(white, show)[0]);
        let lift = [0, 1, 2].map(|c| show[c] / gain[c]);
        let shown: Vec<Swatch> = self.palette.iter().map(|s| Swatch { mix: s.mix, rgb: s.rgb, lab: view_lab(s.rgb, show) }).collect();
        let lifted: Vec<[f32; 3]> = self.photo.par_iter().map(|p| lab([0, 1, 2].map(|c| if lift[c] == 1.0 { p[c] } else { tonemap(tonemap_inverse(p[c]) * lift[c]) }))).collect();
        let mut sample: Vec<f32> = lifted.iter().step_by(QUANTILE_STRIDE).map(|v| v[0]).collect();
        sample.sort_by(|a, b| a.total_cmp(b));
        let at = |q: f32| sample[((sample.len() - 1) as f32 * q) as usize];
        let l0 = at(TONE_LOW);
        let lh = at(TONE_HIGH).max(lb + TONE_MIN_SPAN).min(lw);
        let gamma = TONE_GAMMA + (1.0 - TONE_GAMMA) * (-self.fit_ev / FIT_FULL_EV).clamp(0.0, 1.0);
        let mapped: Vec<[f32; 3]> = lifted
            .into_par_iter()
            .map(|v| {
                let l = if v[0] >= lh || lh <= l0 { v[0].max(lb) } else { lb + (lh - lb) * ((v[0] - l0) / (lh - l0)).clamp(0.0, 1.0).powf(gamma) };
                [l.min(lw), v[1], v[2]]
            })
            .collect();
        let target = gamut_map(mapped, &shown);
        let (tangent, edges) = structure(&target, self.w, self.h, TENSOR_RADIUS);
        let weight = edges.iter().map(|e| 1.0 + EDGE_WEIGHT * e).collect();
        let base = flow_field(&tangent, self.w, self.h);
        let flow = etf(&base, &edges, self.w, self.h, ETF_RADIUS, ETF_ITERS);
        let detail = box_blur(&edges, self.w, self.h, DETAIL_RADIUS);
        let (label, centres) = segment(&target, self.w, self.h, 7);
        let n = centres.len();
        let kin: Vec<bool> = (0..n * n).map(|k| de(centres[k / n], centres[k % n]) < KIN_DE).collect();
        let swatches = Arc::new(shown);
        self.model = Some(Arc::new(Model { w: self.w, h: self.h, cell: self.cell, target, flow, detail, label, kin, regions: n, weight, swatches }));
        self.calibrated = true;
        log::info!("display gain {:.2} {:.2} {:.2}", gain[0], gain[1], gain[2]);
    }

    pub(super) fn model(&self) -> &Model {
        self.model.as_ref().expect("calibrated")
    }

    pub fn observe(&mut self, lab: Vec<[f32; 3]>) {
        self.current = lab;
        let m = self.model.clone().expect("calibrated");
        self.err = self.current.par_iter().zip(&m.target).map(|(a, b)| de(*a, *b)).collect();
        self.mean_err = self.err.iter().sum::<f32>() / self.err.len().max(1) as f32;
        if self.first.is_empty() {
            self.first = self.current.clone();
        }
        if self.start_err == 0.0 {
            self.start_err = self.first.par_iter().zip(&m.target).map(|(a, b)| de(*a, *b)).sum::<f32>() / self.first.len().max(1) as f32;
        }
        self.synced = true;
    }

    pub fn synced(&self) -> bool {
        self.synced
    }

    pub fn set_tool(&mut self, kind: ToolKind, shape: BrushShape, size: f64) -> Option<f64> {
        let user_changed = self.applied.or(self.seen).map(|a| (size - a).abs() > SIZE_CHANGED * a).unwrap_or(true);
        if self.tool == Some((kind, shape)) && !user_changed {
            return None;
        }
        if self.tool != Some((kind, shape)) {
            self.tool = Some((kind, shape));
            self.range = tool_range(kind, shape);
            self.width_gain = 1.0;
            self.opacity = 0.9;
            self.efficiency = 1.0;
            self.tile_eff = Arc::new(vec![1.0; self.tile_eff.len()]);
        }
        self.applied = None;
        self.seen = Some(size);
        self.gen += 1;
        self.queue.clear();
        self.ready = None;
        self.finished = false;
        self.synced = false;
        let r = self.range?;
        self.size_hi = size.clamp(r.lo, r.hi);
        self.reset = Some(self.size_hi);
        if self.jobs > 0 {
            self.dry_next = true;
            self.passes += 1;
        }
        Some(size)
    }

    pub fn has_tool(&self) -> bool {
        self.range.is_some()
    }

    pub fn applied(&mut self, size: f64) {
        self.applied = Some(size);
    }

    pub fn resume(&mut self) {
        self.phase = Phase::Wait(WARMUP_FRAMES);
        self.tool = None;
        self.applied = None;
        self.seen = None;
        self.queue.clear();
        self.ready = None;
        self.synced = false;
        self.finished = false;
        self.gen += 1;
        self.status.clear();
    }

    pub fn progress(&self) -> f32 {
        let (mut sum, mut total, mut flat) = (0.0f64, 0.0f64, 0.0f64);
        let Some(m) = self.model.as_ref() else { return 0.0 };
        let mut mean = [0.0f64; 3];
        for (t, w) in m.target.iter().zip(&self.mask) {
            for c in 0..3 {
                mean[c] += (t[c] * *w) as f64;
            }
            total += *w as f64;
        }
        if total <= 0.0 {
            return 0.0;
        }
        let mean = mean.map(|v| (v / total) as f32);
        for ((t, e), w) in m.target.iter().zip(&self.err).zip(&self.mask) {
            sum += (*e * *w) as f64;
            flat += (de(*t, mean) * *w) as f64;
        }
        if flat <= 0.0 {
            return 0.0;
        }
        (1.0 - sum / flat).clamp(0.0, 1.0) as f32
    }

    pub(super) fn set_focus(&mut self, focus: Focus) {
        if self.focus != Some(focus) {
            self.focus = Some(focus);
            self.mask = self.model().mask(focus);
        }
    }

    pub(super) fn gain(&self, k: usize, delta: f32) -> f32 {
        if delta < 0.0 { delta * self.mask[k] } else { delta }
    }

    pub fn next(&mut self, candidates: usize, follow: f64, intent: f64, longest: f64, focus: Focus, paint: &[f32; 16]) -> Option<Plan> {
        if let Some(p) = self.queue.pop_front() {
            return Some(p);
        }
        if !self.synced {
            return None;
        }
        if self.job.as_ref().map(|j| j.is_finished()).unwrap_or(false) {
            let (sk, batch) = self.job.take().unwrap().join().expect("planner");
            let mut sk = sk;
            if let Some(m) = self.model.as_ref().filter(|m| !Arc::ptr_eq(m, &sk.m)) {
                sk.m = m.clone();
                sk.ref_hw = -1.0;
                sk.blobs.clear();
            }
            self.sketch = Some(sk);
            if batch.gen == self.gen {
                let mut batch = batch;
                if self.dry_next {
                    if let Some(p) = batch.plans.first_mut() {
                        p.dry = true;
                    }
                    self.dry_next = false;
                }
                self.tried += batch.planned as u64;
                self.pruned += batch.pruned as u64;
                self.size_hi = batch.sched.size_hi;
                if batch.plans.is_empty() {
                    self.empty_batches += 1;
                    self.finished = self.empty_batches >= EMPTY_BATCHES;
                    self.retry_at = Some(std::time::Instant::now() + std::time::Duration::from_secs_f32(RETRY_SECS));
                } else {
                    self.empty_batches = 0;
                    self.finished = false;
                }
                log::info!("batch planned {} pruned {} painting {} size {:.1} mm pass {} closer {:.0}%", batch.planned, batch.pruned, batch.plans.len(), batch.sched.size_hi * 1e3, self.passes, self.progress() * 100.0);
                self.planned_ref = Some(batch.reference.clone());
                self.ready = Some(batch);
            }
        }
        if self.job.is_some() || self.retry_at.map(|t| std::time::Instant::now() < t).unwrap_or(false) {
            return None;
        }
        self.retry_at = None;
        self.set_focus(focus);
        let pending = self.ready.take().map(|b| b.plans).unwrap_or_default();
        let range = self.range?;
        let kind = self.tool?.0;
        let m = self.model.clone()?;
        let size_hi = self.size_hi;
        let seed = self.rng.random();
        let mut sk = self.sketch.take().unwrap_or_else(|| Sketch::new(m, seed, kind, range, size_hi));
        let job = Job { kind, range, reset: self.reset.take(), candidates, follow, longest, focus, paint: *paint, width_gain: self.width_gain, opacity: self.opacity, efficiency: self.efficiency, tile_eff: self.tile_eff.clone(), intent: intent as f32, gen: self.gen };
        let real = self.current.clone();
        let ahead = pending.clone();
        self.jobs += 1;
        self.job = Some(std::thread::spawn(move || {
            sk.rebase(&real, &ahead);
            let b = sk.run(&job);
            (sk, b)
        }));
        if pending.is_empty() {
            return None;
        }
        self.synced = false;
        self.queue.extend(pending);
        self.queue.pop_front()
    }

    pub fn region(&self, plan: &Plan) -> [usize; 4] {
        self.model().bbox(&plan.pts, plan.hw * 1.5 + 4.0)
    }

    pub fn judge(&mut self, plan: &Plan, region: [usize; 4], after: &[[f32; 3]]) -> bool {
        let m = self.model.clone().expect("calibrated");
        let rw = region[2] - region[0];
        let mut covs = vec![0.0f32; rw * (region[3] - region[1])];
        for (k, c) in m.footprint(&mut self.scratch, &plan.pts, plan.hw, plan.kind == ToolKind::Brush) {
            let (i, j) = (k % self.w, k / self.w);
            if (region[0]..region[2]).contains(&i) && (region[1]..region[3]).contains(&j) {
                covs[(j - region[1]) * rw + (i - region[0])] = c;
            }
        }
        let judged = self.planned_ref.clone();
        let goal: &[[f32; 3]] = judged.as_deref().map(|v| v.as_slice()).unwrap_or(&m.target);
        let mut delta = 0.0f64;
        let (mut pred, mut changed) = (0u32, 0u32);
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for j in region[1]..region[3] {
            for i in region[0]..region[2] {
                let k = j * self.w + i;
                let a = after[(j - region[1]) * rw + (i - region[0])];
                let b = self.current[k];
                let w = m.weight[k];
                delta += self.gain(k, w * (de(a, goal[k]) - de(b, goal[k]))) as f64;
                let cov = covs[(j - region[1]) * rw + (i - region[0])];
                let reach = de(plan.lab, b);
                if reach > 2.0 * CHANGED_DE {
                    if cov > 0.5 {
                        pred += 1;
                    }
                    if de(a, b) > 0.35 * reach {
                        changed += 1;
                    }
                }
                if cov > 0.9 {
                    let d = [plan.lab[0] - b[0], plan.lab[1] - b[1], plan.lab[2] - b[2]];
                    num += (a[0] - b[0]) * d[0] + (a[1] - b[1]) * d[1] + (a[2] - b[2]) * d[2];
                    den += d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                }
            }
        }
        if pred > 20 {
            let ratio = (changed as f64 / pred as f64).clamp(0.5, 2.0);
            let seen = (plan.width_gain * ratio).clamp(0.15, 3.0);
            self.width_gain = (self.width_gain * (seen / self.width_gain).powf(0.25)).clamp(0.15, 3.0);
        }
        if den > 2000.0 {
            let o = (num / den).clamp(0.15, 1.0);
            self.opacity = 0.85 * self.opacity + 0.15 * o;
        }
        if plan.predicted < -MIN_GAIN {
            let ratio = ((delta as f32) / plan.predicted).clamp(0.0, 1.5);
            self.efficiency += EFFICIENCY_RATE * (ratio - self.efficiency);
            let mid = plan.pts[plan.pts.len() / 2];
            let tile = (mid[1].max(0.0) as usize / TILE).min(self.h.div_ceil(TILE) - 1) * self.w.div_ceil(TILE) + (mid[0].max(0.0) as usize / TILE).min(self.w.div_ceil(TILE) - 1);
            let eff = Arc::make_mut(&mut self.tile_eff);
            if let Some(e) = eff.get_mut(tile) {
                *e += TILE_EFF_RATE * (ratio - *e);
            }
        }
        self.painted += 1;
        let ok = delta < 0.0;
        if ok {
            self.store(region, after);
            self.kept += 1;
        }
        log::debug!("stroke {} predicted {:.0} real {:.0} kept {} gain {:.2} opacity {:.2}", self.painted, plan.predicted, delta, ok, self.width_gain, self.opacity);
        if self.painted % 50 == 0 {
            log::info!("painted {} kept {} planned {} pruned {} pass {} size {:.1} mm closer {:.0}% gain {:.2} opacity {:.2} efficiency {:.2}", self.painted, self.kept, self.tried, self.pruned, self.passes, self.size_hi * 1e3, self.progress() * 100.0, self.width_gain, self.opacity, self.efficiency);
        }
        ok
    }

    pub fn store(&mut self, region: [usize; 4], after: &[[f32; 3]]) {
        let m = self.model.clone().expect("calibrated");
        let rw = region[2] - region[0];
        for j in region[1]..region[3] {
            for i in region[0]..region[2] {
                let k = j * self.w + i;
                let a = after[(j - region[1]) * rw + (i - region[0])];
                let e = de(a, m.target[k]);
                self.mean_err += (e - self.err[k]) / self.err.len() as f32;
                self.current[k] = a;
                self.err[k] = e;
            }
        }
    }
}
