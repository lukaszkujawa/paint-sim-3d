use super::*;

const STRIDE_HW: f32 = 6.0;

const MAX_POINTS: f32 = 96.0;

pub(super) const TRACE_COHERENT: f32 = 0.1;

const TRACE_TURN: f32 = 0.3;

pub(super) const EDGE_PX: f32 = 2.0;

const SIZE_SPREAD: f64 = 0.75;

const RIDGE_MIN_HW: f32 = 8.0;

const LEN_WIDTHS: f32 = 20.0;

const ETF_SEED_EDGE: f32 = 0.2;

const ETF_PASS_ITERS: usize = 2;

const GATE_FULL_DE: f32 = 10.0;

const GATE_OPEN_DE: f32 = 6.0;

const GATE: f32 = 0.9;

const MEDIAN_COVER: f32 = 0.5;

const FLOW_JITTER: f32 = 0.8;

const OUTSIDE_DISCOUNT: f32 = 0.85;

const STRUCT_SIZE_LO: f64 = 0.7;

const RIDGE_FAN: [f32; 5] = [-0.5, -0.25, 0.0, 0.25, 0.5];

const RIDGE_LAMBDA: f32 = 0.8;

const ISO_MIN_STEPS: usize = 2;

const STOP_FINE: f32 = 0.5;

const ISO_STOP_DE: f32 = 22.0;

const BLOB_TRIES: usize = 3;

const VISIT_DECAY: f32 = 0.3;

const T_FOCUS: f32 = 3.0;

const REBLOB: usize = LOOKAHEAD;

const BLOBS_KEEP: usize = 4000;

const BLOB_WINDOW_MAX: usize = 512;

const BLOB_WINDOW_MIN: usize = 24;

const BLOB_WINDOW_HW: f32 = 8.0;

const MIN_BLOB: usize = 3;

const SEED_QUANTILE: f32 = 0.88;

const STROKE_COST: f32 = 2.0;

const T_MIN: f32 = 5.0;

const REF_RANGE_HW: f32 = 40.0;

const REF_RANGE_WIDE: f32 = 18.0;

const REF_RANGE: f32 = 12.0;

const REF_SIGMA: f32 = 0.5;

const TABU: f32 = 0.5;

const MISS_LIMIT: usize = 24;

const PRUNE_VISIBLE: f32 = 0.25;

const EXECUTE: usize = 48;

const STOPS: [f32; 4] = [f32::INFINITY, 30.0, 18.0, 10.0];

const MIN_HW: f64 = 0.75;

const REFINEMENTS: usize = 16;

pub(super) const APPROACH_MAX: f64 = 0.03;

pub(super) const APPROACH_MIN: f64 = 0.004;

pub(super) const APPROACH_HW: f64 = 3.0;

#[derive(Clone)]
pub struct Schedule {
    pub size_hi: f64,
    pub(super) start: f64,
    pub(super) misses: usize,
}

impl Schedule {
    pub(super) fn new(size_hi: f64) -> Self {
        Self { size_hi, start: size_hi, misses: 0 }
    }
}

pub(super) struct Job {
    pub(super) kind: ToolKind,
    pub(super) range: ToolRange,
    pub(super) reset: Option<f64>,
    pub(super) candidates: usize,
    pub(super) follow: f64,
    pub(super) longest: f64,
    pub(super) focus: Focus,
    pub(super) paint: [f32; 16],
    pub(super) width_gain: f64,
    pub(super) opacity: f32,
    pub(super) efficiency: f32,
    pub(super) tile_eff: Arc<Vec<f32>>,
    pub(super) intent: f32,
    pub(super) gen: u64,
}

pub struct Batch {
    pub(super) plans: Vec<Plan>,
    pub(super) sched: Schedule,
    pub(super) planned: usize,
    pub(super) pruned: usize,
    pub(super) gen: u64,
    pub(super) reference: Arc<Vec<[f32; 3]>>,
}

pub(super) struct Region {
    pub(super) window: [usize; 4],
    pub(super) cdf: Vec<f32>,
}

pub(super) struct Sketch {
    pub(super) m: Arc<Model>,
    pub(super) mask: Vec<f32>,
    pub(super) focus: Option<Focus>,
    pub(super) current: Vec<[f32; 3]>,
    pub(super) err: Vec<f32>,
    pub(super) reference: Arc<Vec<[f32; 3]>>,
    pub(super) tangent: Vec<[f32; 3]>,
    pub(super) ref_hw: f32,
    pub(super) smooth_err: Vec<f32>,
    pub(super) thresh: f32,
    pub(super) blobs: Vec<Blob>,
    pub(super) cursor: usize,
    pub(super) since_blob: usize,
    pub(super) tiles: Vec<f32>,
    pub(super) tabu: Vec<f32>,
    pub(super) tw: usize,
    pub(super) scratch: Vec<f32>,
    pub(super) rng: StdRng,
    pub(super) kind: ToolKind,
    pub(super) range: ToolRange,
    pub(super) longest: f32,
    pub(super) width_gain: f64,
    pub(super) opacity: f32,
    pub(super) efficiency: f32,
    pub(super) tile_eff: Arc<Vec<f32>>,
    pub(super) sched: Schedule,
    pub(super) started: bool,
    pub(super) last_tiles: Vec<usize>,
    pub(super) intent: f32,
    pub(super) follow: f32,
    pub(super) fine: f32,
    pub(super) gate: Vec<f32>,
}

impl Sketch {
    pub(super) fn new(m: Arc<Model>, seed: u64, kind: ToolKind, range: ToolRange, size_hi: f64) -> Self {
        let n = m.w * m.h;
        let (tw, th) = (m.w.div_ceil(TILE), m.h.div_ceil(TILE));
        Self {
            mask: vec![1.0; n],
            focus: None,
            current: vec![[0.0; 3]; n],
            err: vec![0.0; n],
            reference: Arc::new(m.target.clone()),
            tangent: Vec::new(),
            ref_hw: -1.0,
            smooth_err: Vec::new(),
            thresh: T_MIN,
            blobs: Vec::new(),
            cursor: 0,
            since_blob: 0,
            tiles: vec![0.0; tw * th],
            tabu: vec![1.0; tw * th],
            tw,
            scratch: vec![0.0; n],
            rng: StdRng::seed_from_u64(seed),
            kind,
            range,
            longest: 0.0,
            width_gain: 1.0,
            opacity: 0.9,
            efficiency: 1.0,
            tile_eff: Arc::new(Vec::new()),
            sched: Schedule::new(size_hi),
            started: false,
            last_tiles: Vec::new(),
            intent: 0.0,
            follow: 0.7,
            fine: 0.0,
            gate: vec![1.0; n],
            m,
        }
    }

    pub(super) fn refresh_reference(&mut self) {
        let hw = self.hw(self.sched.size_hi);
        if (hw - self.ref_hw).abs() <= 0.05 * hw.max(1.0) {
            return;
        }
        self.ref_hw = hw;
        let m = self.m.clone();
        let (w, h) = (m.w, m.h);
        let sigma = REF_SIGMA * hw;
        self.reference = Arc::new(if sigma < 0.6 { m.target.clone() } else { bilateral(&m.target, w, h, sigma, REF_RANGE + REF_RANGE_WIDE * (hw / REF_RANGE_HW).min(1.0)) });
        let radius = ((0.5 * hw).round() as usize).max(1);
        let (tangent, edges) = structure(&self.reference, w, h, radius);
        let seeded: Vec<[f32; 3]> = tangent.iter().zip(&edges).zip(&m.flow).map(|((t, e), f)| if *e > ETF_SEED_EDGE { *t } else { [f[0], f[1], f[2]] }).collect();
        self.tangent = etf(&seeded, &edges, w, h, radius.max(4), ETF_PASS_ITERS);
        self.err = self.current.par_iter().zip(self.reference.par_iter()).map(|(a, b)| de(*a, *b)).collect();
        self.refresh_tiles();
        self.blobs.clear();
    }

    pub(super) fn set_focus(&mut self, focus: Focus) {
        if self.focus != Some(focus) {
            self.focus = Some(focus);
            self.mask = self.m.mask(focus);
        }
    }

    pub(super) fn tile_of(&self, k: usize) -> usize {
        (k / self.m.w / TILE) * self.tw + (k % self.m.w) / TILE
    }

    pub(super) fn refresh_tiles(&mut self) {
        self.tiles.iter_mut().for_each(|t| *t = 0.0);
        for k in 0..self.err.len() {
            let t = self.tile_of(k);
            self.tiles[t] += self.err[k] * self.m.weight[k] * self.mask[k] * self.gate[k];
        }
        self.tabu.iter_mut().for_each(|t| *t = 1.0);
    }

    pub(super) fn rebase(&mut self, real: &[[f32; 3]], pending: &[Plan]) {
        let m = self.m.clone();
        self.current = real.to_vec();
        self.err = self.current.par_iter().zip(self.reference.par_iter()).map(|(a, b)| de(*a, *b)).collect();
        self.refresh_tiles();
        self.blobs.clear();
        for p in pending {
            let foot = m.footprint(&mut self.scratch, &p.pts, p.hw, p.kind == ToolKind::Brush);
            self.apply(&foot, p.lab);
        }
    }

    pub(super) fn apply(&mut self, foot: &[(usize, f32)], c: [f32; 3]) {
        let o = self.opacity;
        for &(k, a) in foot {
            let a = a * o;
            let cur = self.current[k];
            let new = [cur[0] + a * (c[0] - cur[0]), cur[1] + a * (c[1] - cur[1]), cur[2] + a * (c[2] - cur[2])];
            let e = de(new, self.reference[k]);
            let t = self.tile_of(k);
            self.tiles[t] += (e - self.err[k]) * self.m.weight[k] * self.mask[k] * self.gate[k];
            self.current[k] = new;
            self.err[k] = e;
        }
    }

    pub(super) fn attention(&self, k: usize) -> f32 {
        let g = self.gate[k];
        g + (1.0 - g) * ((self.err[k] - GATE_OPEN_DE) / GATE_FULL_DE).clamp(0.0, 1.0)
    }

    pub(super) fn min_gain(&self, _hw: f32) -> f32 {
        MIN_GAIN
    }

    pub(super) fn gain(&self, k: usize, delta: f32) -> f32 {
        if delta < 0.0 { delta * self.mask[k] } else { delta }
    }

    pub(super) fn size_lo(&self) -> f64 {
        self.range.lo.max(MIN_HW * self.m.cell / (self.range.half_width * self.width_gain))
    }

    pub(super) fn size_floor(&self) -> f64 {
        self.size_lo().max(SIZE_SPREAD * self.sched.size_hi)
    }

    pub(super) fn hw(&self, size: f64) -> f32 {
        ((size * self.range.half_width * self.width_gain) / self.m.cell).max(0.6) as f32
    }

    pub(super) fn len_range(&self, size: f64, hw: f32) -> (f32, f32) {
        let r = self.range;
        let lo = (hw * r.len.0).max(2.0);
        let hi = if r.long { self.longest.max(hw * r.len.1) } else { hw * r.len.1 };
        let hi = hi.min((r.reach * size / self.m.cell) as f32).min(2.0 * LEN_WIDTHS * hw);
        (lo, hi.max(lo * 1.2))
    }

    pub(super) fn window_region(&self, window: [usize; 4]) -> Option<Region> {
        let m = &self.m;
        let mut cdf = Vec::with_capacity((window[2] - window[0]) * (window[3] - window[1]));
        let mut acc = 0.0f32;
        for y in window[1]..window[3] {
            for x in window[0]..window[2] {
                let k = y * m.w + x;
                acc += self.err[k] * self.err[k] * m.weight[k] * self.mask[k] * self.attention(k);
                cdf.push(acc);
            }
        }
        if acc <= 0.0 {
            return None;
        }
        Some(Region { window, cdf })
    }

    pub(super) fn region_at_all(&self) -> Option<Region> {
        self.window_region([0, 0, self.m.w, self.m.h])
    }

    pub(super) fn blob_window(&self, b: &Blob) -> [usize; 4] {
        let pad = (2.0 * self.hw(self.sched.size_hi)).ceil() as usize;
        [b.window[0].saturating_sub(pad), b.window[1].saturating_sub(pad), (b.window[2] + pad).min(self.m.w), (b.window[3] + pad).min(self.m.h)]
    }

    pub(super) fn blob_stats(&self, cells: &[usize], e: &[f32], window: [usize; 4]) -> Option<Blob> {
        let w = self.m.w;
        let (mut mass, mut sx, mut sy, mut sxx, mut sxy, mut syy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut peak, mut peak_e) = (cells[0], -1.0f32);
        for &k in cells {
            let v = e[k] as f64;
            let (x, y) = ((k % w) as f64 + 0.5, (k / w) as f64 + 0.5);
            mass += v;
            sx += v * x;
            sy += v * y;
            sxx += v * x * x;
            sxy += v * x * y;
            syy += v * y * y;
            if e[k] > peak_e {
                peak_e = e[k];
                peak = k;
            }
        }
        if mass <= 0.0 {
            return None;
        }
        let (cx, cy) = (sx / mass, sy / mass);
        let (vxx, vxy, vyy) = (sxx / mass - cx * cx, sxy / mass - cx * cy, syy / mass - cy * cy);
        let tr = vxx + vyy;
        let det = vxx * vyy - vxy * vxy;
        let disc = (0.25 * tr * tr - det).max(0.0).sqrt();
        let (l1, l2) = (0.5 * tr + disc, (0.5 * tr - disc).max(0.0));
        let axis = if vxy.abs() > 1e-9 { (l1 - vxx).atan2(vxy) } else if vxx >= vyy { 0.0 } else { std::f64::consts::FRAC_PI_2 };
        Some(Blob {
            centre: [cx as f32, cy as f32],
            axis: axis as f32,
            along: (2.0 * (3.0 * l1).sqrt()).max(1.0) as f32,
            across: (2.0 * (3.0 * l2).sqrt()).max(1.0) as f32,
            peak: [(peak % w) as f32 + 0.5, (peak / w) as f32 + 0.5],
            mass: mass as f32,
            count: cells.len(),
            window,
            tile: self.tile_of(peak),
        })
    }

    pub(super) fn extract_blobs(&mut self) {
        let m = self.m.clone();
        let (w, h) = (m.w, m.h);
        let raw: Vec<f32> = (0..w * h).into_par_iter().map(|k| self.err[k] * m.weight[k] * self.mask[k] * self.attention(k)).collect();
        let e = box_blur(&raw, w, h, 1);
        let mut sample: Vec<f32> = e.iter().step_by(QUANTILE_STRIDE).copied().collect();
        let q = ((sample.len() - 1) as f32 * SEED_QUANTILE) as usize;
        let (_, quant, _) = sample.select_nth_unstable_by(q, |a, b| a.total_cmp(b));
        let thresh = quant.clamp(T_MIN, T_FOCUS * T_MIN);
        self.thresh = thresh;
        let hw = self.hw(self.sched.size_hi);
        let win = ((BLOB_WINDOW_HW * hw) as usize).clamp(BLOB_WINDOW_MIN, BLOB_WINDOW_MAX);
        let mut label = vec![u32::MAX; w * h];
        let mut blobs = Vec::new();
        let mut stack = Vec::new();
        let mut cells = Vec::new();
        let mut next = 0u32;
        for start in 0..w * h {
            if e[start] <= thresh || label[start] != u32::MAX {
                continue;
            }
            let id = next;
            next += 1;
            label[start] = id;
            stack.push(start);
            cells.clear();
            let mut bb = [w, h, 0, 0];
            while let Some(k) = stack.pop() {
                cells.push(k);
                let (x, y) = (k % w, k / w);
                bb = [bb[0].min(x), bb[1].min(y), bb[2].max(x + 1), bb[3].max(y + 1)];
                let nb = [(x > 0, k.wrapping_sub(1)), (x + 1 < w, k + 1), (y > 0, k.wrapping_sub(w)), (y + 1 < h, k + w)];
                for (ok, n) in nb {
                    if ok && e[n] > thresh && label[n] == u32::MAX {
                        label[n] = id;
                        stack.push(n);
                    }
                }
            }
            if cells.len() < MIN_BLOB {
                continue;
            }
            if bb[2] - bb[0] <= win && bb[3] - bb[1] <= win {
                if let Some(b) = self.blob_stats(&cells, &e, bb) {
                    blobs.push(b);
                }
                continue;
            }
            let step = win / 2;
            let mut y0 = bb[1];
            loop {
                let y1 = (y0 + win).min(bb[3]);
                let mut x0 = bb[0];
                loop {
                    let x1 = (x0 + win).min(bb[2]);
                    let part: Vec<usize> = (y0..y1).flat_map(|y| (x0..x1).map(move |x| y * w + x)).filter(|&k| label[k] == id).collect();
                    if part.len() >= MIN_BLOB {
                        if let Some(b) = self.blob_stats(&part, &e, [x0, y0, x1, y1]) {
                            blobs.push(b);
                        }
                    }
                    if x1 >= bb[2] {
                        break;
                    }
                    x0 += step;
                }
                if y1 >= bb[3] {
                    break;
                }
                y0 += step;
            }
        }
        let key = |b: &Blob| b.mass / (b.count as f32).sqrt() * self.tabu[b.tile] * self.tile_eff.get(b.tile).copied().unwrap_or(1.0).clamp(0.2, 1.5);
        blobs.sort_by(|a, b| key(b).total_cmp(&key(a)));
        blobs.truncate(BLOBS_KEEP);
        self.blobs = blobs;
        self.smooth_err = e;
        self.cursor = 0;
        self.since_blob = 0;
    }

    pub(super) fn sample_in(&mut self, reg: &Region) -> [f32; 2] {
        let total = *reg.cdf.last().unwrap();
        let pick = self.rng.random::<f32>() * total;
        let k = reg.cdf.partition_point(|c| *c < pick).min(reg.cdf.len() - 1);
        let rw = reg.window[2] - reg.window[0];
        [(reg.window[0] + k % rw) as f32 + self.rng.random::<f32>(), (reg.window[1] + k / rw) as f32 + self.rng.random::<f32>()]
    }

    pub(super) fn means(&self, foot: &[(usize, f32)]) -> ([f32; 3], [f32; 3]) {
        let mut acc = [0.0f32; 4];
        let mut chans: [Vec<f32>; 3] = [Vec::with_capacity(foot.len()), Vec::with_capacity(foot.len()), Vec::with_capacity(foot.len())];
        for &(k, a) in foot {
            let (t, c) = (self.reference[k], self.current[k]);
            for i in 0..3 {
                acc[i] += a * c[i];
            }
            acc[3] += a;
            if a >= MEDIAN_COVER {
                for i in 0..3 {
                    chans[i].push(t[i]);
                }
            }
        }
        if acc[3] <= 0.0 {
            return ([50.0, 0.0, 0.0], [50.0, 0.0, 0.0]);
        }
        if chans[0].is_empty() {
            for &(k, _) in foot {
                let t = self.reference[k];
                for i in 0..3 {
                    chans[i].push(t[i]);
                }
            }
        }
        let n = acc[3];
        let mid = |v: &mut Vec<f32>| {
            let q = v.len() / 2;
            *v.select_nth_unstable_by(q, |a, b| a.total_cmp(b)).1
        };
        let [c0, c1, c2] = &mut chans;
        ([mid(c0), mid(c1), mid(c2)], [acc[0] / n, acc[1] / n, acc[2] / n])
    }

    pub(super) fn pixel_delta(&self, k: usize, a: f32, c: [f32; 3], region: Option<u8>) -> f32 {
        let cover = a;
        let a = a * self.opacity;
        let cur = self.current[k];
        let new = [cur[0] + a * (c[0] - cur[0]), cur[1] + a * (c[1] - cur[1]), cur[2] + a * (c[2] - cur[2])];
        let d = self.gain(k, self.m.weight[k] * (de(new, self.reference[k]) - self.err[k])) + STROKE_COST * cover;
        match region {
            Some(r) if d < 0.0 && !self.m.same_shape(k, r) => d * (1.0 - OUTSIDE_DISCOUNT * self.intent),
            _ => d,
        }
    }

    pub(super) fn score(&self, foot: &[(usize, f32)], c: [f32; 3], region: Option<u8>) -> f32 {
        foot.iter().map(|&(k, a)| self.pixel_delta(k, a, c, region)).sum()
    }

    pub(super) fn stride(&self, hw: f32) -> usize {
        ((hw / STRIDE_HW).floor() as usize).max(1)
    }

    pub(super) fn trim(&mut self, pts: &[[f32; 2]], hw: f32, c: [f32; 3], region: Option<u8>) -> Vec<[f32; 2]> {
        if pts.len() < 4 {
            return pts.to_vec();
        }
        let st = self.stride(hw);
        let m = self.m.clone();
        let gains: Vec<f32> = pts
            .windows(2)
            .map(|w| m.footprint_at(&mut self.scratch, w, hw, false, st).iter().map(|&(k, a)| self.pixel_delta(k, a, c, region)).sum())
            .collect();
        let (mut best, mut best_range) = (0.0f32, (0, gains.len()));
        let (mut run, mut start) = (0.0f32, 0usize);
        for (i, g) in gains.iter().enumerate() {
            if run > 0.0 {
                run = 0.0;
                start = i;
            }
            run += g;
            if run < best {
                best = run;
                best_range = (start, i + 1);
            }
        }
        if best >= 0.0 {
            return pts.to_vec();
        }
        pts[best_range.0..=best_range.1].to_vec()
    }

    pub(super) fn bounded(&self, pts: Vec<[f32; 2]>, centre: [f32; 2], region: u8) -> Vec<[f32; 2]> {
        if pts.len() < 2 {
            return pts;
        }
        let m = &self.m;
        let ci = (0..pts.len()).min_by(|a, b| {
            let d = |p: [f32; 2]| (p[0] - centre[0]).powi(2) + (p[1] - centre[1]).powi(2);
            d(pts[*a]).total_cmp(&d(pts[*b]))
        }).unwrap();
        let inside = |p: [f32; 2]| m.same_shape(m.index(p), region);
        let mut a = ci;
        while a > 0 && inside(pts[a - 1]) {
            a -= 1;
        }
        let mut b = ci;
        while b + 1 < pts.len() && inside(pts[b + 1]) {
            b += 1;
        }
        pts[a..=b].to_vec()
    }

    pub(super) fn pick_colour(&mut self, foot: &[(usize, f32)], jitter: f32) -> (usize, [f32; 3]) {
        let (m, cur) = self.means(foot);
        let o = self.opacity.max(0.65);
        let mut want = [0.0f32; 3];
        for c in 0..3 {
            want[c] = cur[c] + (m[c] - cur[c]) / o + jitter * gauss(&mut self.rng) * if c == 0 { 1.0 } else { 1.4 };
        }
        want[0] = want[0].clamp(0.0, 100.0);
        (self.m.nearest(want), m)
    }

    pub(super) fn trace(&self, c: &Cand) -> Vec<[f32; 2]> {
        let m = &self.m;
        let step = (0.5 * c.hw).max(c.len / MAX_POINTS).max(1.0);
        let (xmax, ymax) = (m.w as f32 - EDGE_PX, m.h as f32 - EDGE_PX);
        let walk = |sign: f32, dist: f32| -> Vec<[f32; 2]> {
            let mut p = c.centre;
            let mut d = [sign * c.angle.cos(), sign * c.angle.sin()];
            let mut out = Vec::new();
            let mut gone = 0.0;
            while gone + step <= dist {
                let t = m.flow[m.index(p)];
                if t[2] > TRACE_COHERENT {
                    let flip = if t[0] * d[0] + t[1] * d[1] < 0.0 { -1.0 } else { 1.0 };
                    let n = [d[0] + TRACE_TURN * (flip * t[0] - d[0]), d[1] + TRACE_TURN * (flip * t[1] - d[1])];
                    let l = (n[0] * n[0] + n[1] * n[1]).sqrt().max(1e-6);
                    d = [n[0] / l, n[1] / l];
                }
                let q = [p[0] + d[0] * step, p[1] + d[1] * step];
                if q[0] < EDGE_PX || q[1] < EDGE_PX || q[0] > xmax || q[1] > ymax {
                    break;
                }
                p = q;
                out.push(p);
                gone += step;
            }
            out
        };
        let back = walk(-1.0, c.len * c.along);
        let fwd = walk(1.0, c.len * (1.0 - c.along));
        let mut pts: Vec<[f32; 2]> = back.into_iter().rev().collect();
        pts.push([c.centre[0].clamp(EDGE_PX, xmax), c.centre[1].clamp(EDGE_PX, ymax)]);
        pts.extend(fwd);
        pts
    }

    pub(super) fn seed_colour(&self, p: [f32; 2]) -> usize {
        let k = self.m.index(p);
        let (r, cur) = (self.reference[k], self.current[k]);
        let o = self.opacity.max(0.65);
        let mut want = [0.0f32; 3];
        for c in 0..3 {
            want[c] = cur[c] + (r[c] - cur[c]) / o;
        }
        want[0] = want[0].clamp(0.0, 100.0);
        self.m.nearest(want)
    }

    pub(super) fn stop_de(&self) -> f32 {
        ISO_STOP_DE * (1.0 - STOP_FINE * self.fine)
    }

    pub(super) fn start_dir(&self, c: &Cand) -> [f32; 2] {
        let t = self.tangent[self.m.index(c.centre)];
        if t[2] > TRACE_COHERENT { [t[0], t[1]] } else { [c.angle.cos(), c.angle.sin()] }
    }

    pub(super) fn iso(&self, c: &Cand) -> Vec<[f32; 2]> {
        let m = &self.m;
        let step = c.hw.max(1.0);
        let (xmax, ymax) = (m.w as f32 - EDGE_PX, m.h as f32 - EDGE_PX);
        let d0 = self.start_dir(c);
        let fc = self.follow;
        let walk = |sign: f32, dist: f32| -> Vec<[f32; 2]> {
            let mut p = c.centre;
            let mut d = [sign * d0[0], sign * d0[1]];
            let mut out = Vec::new();
            let mut gone = 0.0;
            while gone + step <= dist && out.len() < MAX_POINTS as usize {
                let q = [p[0] + d[0] * step, p[1] + d[1] * step];
                if q[0] < EDGE_PX || q[1] < EDGE_PX || q[0] > xmax || q[1] > ymax {
                    break;
                }
                let k = m.index(q);
                let r = self.reference[k];
                let away = de(r, c.hint);
                if away > self.stop_de() || (out.len() >= ISO_MIN_STEPS && de(r, self.current[k]) < away) {
                    break;
                }
                let t = self.tangent[k];
                if t[2] > TRACE_COHERENT {
                    let flip = if t[0] * d[0] + t[1] * d[1] < 0.0 { -1.0 } else { 1.0 };
                    let n = [fc * flip * t[0] + (1.0 - fc) * d[0], fc * flip * t[1] + (1.0 - fc) * d[1]];
                    let l = (n[0] * n[0] + n[1] * n[1]).sqrt().max(1e-6);
                    d = [n[0] / l, n[1] / l];
                }
                p = q;
                out.push(p);
                gone += step;
            }
            out
        };
        let back = walk(-1.0, c.len * c.along);
        let fwd = walk(1.0, c.len * (1.0 - c.along));
        let mut pts: Vec<[f32; 2]> = back.into_iter().rev().collect();
        pts.push([c.centre[0].clamp(EDGE_PX, xmax), c.centre[1].clamp(EDGE_PX, ymax)]);
        pts.extend(fwd);
        pts
    }

    pub(super) fn ridge(&self, c: &Cand) -> Vec<[f32; 2]> {
        let m = &self.m;
        if self.smooth_err.len() != m.w * m.h {
            return self.iso(c);
        }
        let step = c.hw.max(1.0);
        let (xmax, ymax) = (m.w as f32 - EDGE_PX, m.h as f32 - EDGE_PX);
        let d0 = self.start_dir(c);
        let walk = |sign: f32, dist: f32| -> Vec<[f32; 2]> {
            let mut p = c.centre;
            let mut d = [sign * d0[0], sign * d0[1]];
            let mut out = Vec::new();
            let mut gone = 0.0;
            while gone + step <= dist && out.len() < MAX_POINTS as usize {
                let mut best: Option<([f32; 2], f32)> = None;
                for a in RIDGE_FAN {
                    let (s, co) = a.sin_cos();
                    let nd = [co * d[0] - s * d[1], s * d[0] + co * d[1]];
                    let q = [p[0] + nd[0] * step, p[1] + nd[1] * step];
                    if q[0] < EDGE_PX || q[1] < EDGE_PX || q[0] > xmax || q[1] > ymax {
                        continue;
                    }
                    let v = self.smooth_err[m.index(q)];
                    if best.map(|b| v > b.1).unwrap_or(true) {
                        best = Some((nd, v));
                    }
                }
                let Some((nd, v)) = best else { break };
                if v < self.thresh && out.len() >= ISO_MIN_STEPS {
                    break;
                }
                let n = [RIDGE_LAMBDA * nd[0] + (1.0 - RIDGE_LAMBDA) * d[0], RIDGE_LAMBDA * nd[1] + (1.0 - RIDGE_LAMBDA) * d[1]];
                let l = (n[0] * n[0] + n[1] * n[1]).sqrt().max(1e-6);
                d = [n[0] / l, n[1] / l];
                let q = [p[0] + d[0] * step, p[1] + d[1] * step];
                if q[0] < EDGE_PX || q[1] < EDGE_PX || q[0] > xmax || q[1] > ymax {
                    break;
                }
                let k = m.index(q);
                if de(self.reference[k], c.hint) > self.stop_de() {
                    break;
                }
                p = q;
                out.push(p);
                gone += step;
            }
            out
        };
        let back = walk(-1.0, c.len * c.along);
        let fwd = walk(1.0, c.len * (1.0 - c.along));
        let mut pts: Vec<[f32; 2]> = back.into_iter().rev().collect();
        pts.push([c.centre[0].clamp(EDGE_PX, xmax), c.centre[1].clamp(EDGE_PX, ymax)]);
        pts.extend(fwd);
        pts
    }

    pub(super) fn bezier(&self, c: &Cand) -> Vec<[f32; 2]> {
        let (s, co) = c.angle.sin_cos();
        let start = [c.centre[0] - co * c.len * c.along, c.centre[1] - s * c.len * c.along];
        let end = [start[0] + co * c.len, start[1] + s * c.len];
        let ctrl = [(start[0] + end[0]) * 0.5 - s * c.bend * c.len, (start[1] + end[1]) * 0.5 + co * c.bend * c.len];
        let n = if c.len < 2.0 { 1 } else { (c.len / c.hw.max(2.0)).ceil().clamp(8.0, MAX_POINTS) as usize };
        let (xmax, ymax) = (self.m.w as f32 - EDGE_PX, self.m.h as f32 - EDGE_PX);
        (0..=n)
            .map(|k| {
                let t = k as f32 / n as f32;
                let u = 1.0 - t;
                let x = u * u * start[0] + 2.0 * u * t * ctrl[0] + t * t * end[0];
                let y = u * u * start[1] + 2.0 * u * t * ctrl[1] + t * t * end[1];
                [x.clamp(EDGE_PX, xmax), y.clamp(EDGE_PX, ymax)]
            })
            .collect()
    }

    pub(super) fn clip(&self, pts: Vec<[f32; 2]>, centre: [f32; 2], stop: f32) -> Vec<[f32; 2]> {
        if !stop.is_finite() || pts.len() < 2 {
            return pts;
        }
        let m = &self.m;
        let ci = (0..pts.len()).min_by(|a, b| {
            let d = |p: [f32; 2]| (p[0] - centre[0]).powi(2) + (p[1] - centre[1]).powi(2);
            d(pts[*a]).total_cmp(&d(pts[*b]))
        }).unwrap();
        let here = self.reference[m.index(pts[ci])];
        let inside = |p: [f32; 2]| de(self.reference[m.index(p)], here) <= stop;
        let mut a = ci;
        while a > 0 && inside(pts[a - 1]) {
            a -= 1;
        }
        let mut b = ci;
        while b + 1 < pts.len() && inside(pts[b + 1]) {
            b += 1;
        }
        pts[a..=b].to_vec()
    }

    pub(super) fn evaluate(&mut self, mut c: Cand, refit: bool, jitter: f32) -> Cand {
        c.size = c.size.clamp(self.size_floor(), self.sched.size_hi.max(self.size_floor()));
        c.hw = self.hw(c.size);
        let (lo, hi) = self.len_range(c.size, c.hw);
        c.len = c.len.clamp(lo, hi);
        let raw = match c.shape {
            Shape::Bezier => self.bezier(&c),
            Shape::Flow => self.trace(&c),
            Shape::Iso => self.iso(&c),
            Shape::Ridge => self.ridge(&c),
        };
        let m = self.m.clone();
        let tapered = self.kind == ToolKind::Brush;
        c.region = m.label[m.index(c.centre)];
        let region = c.bound.then_some(c.region);
        let stop = c.stop.min(self.stop_de());
        c.pts = if c.bound {
            self.clip(self.bounded(raw, c.centre, c.region), c.centre, stop)
        } else {
            self.clip(raw, c.centre, stop)
        };
        let st = self.stride(c.hw);
        let area = (st * st) as f32;
        let foot = m.footprint_at(&mut self.scratch, &c.pts, c.hw, tapered, st);
        if refit {
            (c.swatch, c.hint) = self.pick_colour(&foot, jitter);
        }
        let colour = m.swatches[c.swatch].lab;
        {
            let trimmed = self.trim(&c.pts, c.hw, colour, region);
            if trimmed.len() != c.pts.len() {
                c.pts = trimmed;
                let foot = m.footprint_at(&mut self.scratch, &c.pts, c.hw, tapered, st);
                c.score = self.score(&foot, colour, region) * area;
                return c;
            }
        }
        c.score = self.score(&foot, colour, region) * area;
        c
    }

    pub(super) fn random_stop(&mut self) -> f32 {
        if self.rng.random::<f32>() < self.intent {
            STOPS[self.rng.random_range(STOPS.len() - 2..STOPS.len())]
        } else {
            STOPS[self.rng.random_range(0..STOPS.len())]
        }
    }

    pub(super) fn candidate(&mut self, reg: &Region) -> Cand {
        let centre = self.sample_in(reg);
        let lo = self.size_floor();
        let hi = self.sched.size_hi.max(lo);
        let size = (lo.ln() + self.rng.random::<f64>() * (hi.ln() - lo.ln())).exp();
        let hw = self.hw(size);
        let (llo, lhi) = self.len_range(size, hw);
        let len = (llo.ln() + self.rng.random::<f32>() * (lhi.ln() - llo.ln())).exp();
        let f = self.m.flow[self.m.index(centre)];
        let shaped = self.rng.random::<f32>() < self.intent;
        let shape = if shaped && self.rng.random::<f32>() < self.follow { Shape::Flow } else { Shape::Bezier };
        let angle = f[1].atan2(f[0]) + FLOW_JITTER * (1.0 - self.intent) * gauss(&mut self.rng);
        let stop = self.random_stop();
        let c = Cand { centre, angle, len, bend: 0.12 * gauss(&mut self.rng).clamp(-2.0, 2.0), along: self.rng.random(), shape, stop, bound: shaped, structured: false, region: 0, size, hw, hint: [50.0, 0.0, 0.0], pts: Vec::new(), swatch: 0, score: 0.0 };
        self.evaluate(c, true, 3.0)
    }

    pub(super) fn structured(&mut self, b: &Blob) -> Vec<Cand> {
        let lo = self.size_floor();
        let hi = self.sched.size_hi.max(lo);
        let size = hi * (STRUCT_SIZE_LO + (1.0 - STRUCT_SIZE_LO) * self.rng.random::<f64>());
        let hw = self.hw(size);
        let (_, lhi) = self.len_range(size, hw);
        let seed = self.seed_colour(b.peak);
        let hint = self.reference[self.m.index(b.peak)];
        let base = Cand { centre: b.peak, angle: b.axis, len: lhi, bend: 0.0, along: 0.5, shape: Shape::Iso, stop: f32::INFINITY, bound: false, structured: true, region: 0, size, hw, hint, pts: Vec::new(), swatch: seed, score: 0.0 };
        let mut out = Vec::with_capacity(5);
        out.push(self.evaluate(base.clone(), false, 0.0));
        out.push(self.evaluate(base.clone(), true, 0.0));
        if hw >= RIDGE_MIN_HW {
            out.push(self.evaluate(Cand { shape: Shape::Ridge, ..base.clone() }, true, 0.0));
        }
        out.push(self.evaluate(Cand { shape: Shape::Flow, ..base.clone() }, true, 0.0));
        let across = (b.across * 0.5 * self.m.cell as f32) as f64 / (self.range.half_width * self.width_gain);
        let ribbon = Cand { centre: b.centre, len: b.along, size: across.clamp(lo, hi), shape: Shape::Bezier, ..base };
        out.push(self.evaluate(ribbon, true, 0.0));
        out
    }

    pub(super) fn mutate(&mut self, best: &Cand) -> Cand {
        let mut c = best.clone();
        match self.rng.random_range(0..6) {
            0 => {
                c.size *= (0.3 * gauss(&mut self.rng)).exp() as f64;
                self.evaluate(c, true, 0.0)
            }
            1 => {
                let sw = self.m.swatches[c.swatch].lab;
                let want = [sw[0] + 5.0 * gauss(&mut self.rng), sw[1] + 7.0 * gauss(&mut self.rng), sw[2] + 7.0 * gauss(&mut self.rng)];
                c.swatch = self.m.nearest(want);
                self.evaluate(c, false, 0.0)
            }
            2 => {
                c.stop = self.random_stop();
                self.evaluate(c, true, 0.0)
            }
            3 => {
                c.len *= (0.5 * gauss(&mut self.rng)).exp();
                c.along = (c.along + 0.15 * gauss(&mut self.rng)).clamp(0.0, 1.0);
                if c.shape == Shape::Bezier {
                    c.angle += 0.15 * (1.0 - 0.8 * self.intent) * gauss(&mut self.rng);
                }
                self.evaluate(c, true, 0.0)
            }
            4 => {
                c.size *= (0.3 * gauss(&mut self.rng)).exp() as f64;
                self.evaluate(c, true, 4.0)
            }
            _ => {
                c.centre[0] += 0.3 * c.hw * gauss(&mut self.rng);
                c.centre[1] += 0.3 * c.hw * gauss(&mut self.rng);
                self.evaluate(c, true, 0.0)
            }
        }
    }

    pub(super) fn consider(best: &mut Option<Cand>, c: Cand) {
        if best.as_ref().map(|b| c.score < b.score).unwrap_or(true) {
            *best = Some(c);
        }
    }

    pub(super) fn plan_one(&mut self, job: &Job) -> Option<(Plan, Vec<(usize, f32)>)> {
        self.refresh_reference();
        if self.blobs.is_empty() || self.cursor >= self.blobs.len() || self.since_blob >= REBLOB {
            self.extract_blobs();
        }
        let n_rand = (job.candidates as f32 * (1.0 - self.intent) * (1.0 - self.fine)).round() as usize;
        let floor = self.min_gain(self.hw(self.sched.size_hi));
        let mut best: Option<Cand> = None;
        let mut tried = Vec::new();
        for _ in 0..BLOB_TRIES {
            if self.cursor >= self.blobs.len() {
                break;
            }
            let b = self.blobs[self.cursor].clone();
            self.cursor += 1;
            tried.push(b.tile);
            for c in self.structured(&b) {
                Self::consider(&mut best, c);
            }
            if n_rand > 0 {
                if let Some(reg) = self.window_region(self.blob_window(&b)) {
                    for _ in 0..n_rand.div_ceil(2) {
                        let c = self.candidate(&reg);
                        Self::consider(&mut best, c);
                    }
                }
            }
            if best.as_ref().map(|b| b.score < -floor).unwrap_or(false) {
                break;
            }
        }
        if n_rand > 0 {
            if let Some(reg) = self.region_at_all() {
                for _ in 0..n_rand / 2 {
                    let c = self.candidate(&reg);
                    Self::consider(&mut best, c);
                }
            }
        }
        self.last_tiles = tried;
        let mut best = best?;
        for _ in 0..REFINEMENTS {
            let c = self.mutate(&best);
            if c.score < best.score {
                best = c;
            }
        }
        if best.score > -floor {
            log::debug!("miss: best {:.0} floor {:.0} blobs {} cursor {} thresh {:.1}", -best.score, floor, self.blobs.len(), self.cursor, self.thresh);
            return None;
        }
        self.since_blob += 1;
        if log::log_enabled!(log::Level::Trace) {
            let (a, b) = (best.pts[0], best.pts[best.pts.len() - 1]);
            let len: f32 = best.pts.windows(2).map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt()).sum();
            log::trace!("stroke shape {:?} struct {} len/hw {:.1} angle {:.0} gain {:.0} hw {:.1} at {:.0},{:.0}", best.shape as u8, best.structured, len / best.hw.max(1.0), (b[1] - a[1]).atan2(b[0] - a[0]).to_degrees(), -best.score, best.hw, best.centre[0], best.centre[1]);
        }
        let m = self.m.clone();
        let mut mix = m.swatches[best.swatch].mix;
        mix[12] = job.paint[12];
        mix[13] = job.paint[13];
        mix[14] = job.paint[14];
        let path = best.pts.iter().map(|p| DVec2::new(p[0] as f64 * m.cell, p[1] as f64 * m.cell)).collect();
        let lead = if self.kind == ToolKind::Brush { (APPROACH_HW * best.hw as f64 * m.cell).clamp(APPROACH_MIN, APPROACH_MAX) } else { 0.0 };
        let foot = m.footprint(&mut self.scratch, &best.pts, best.hw, self.kind == ToolKind::Brush);
        let plan = Plan { path, lead, size: best.size, mix, speed: self.range.speed, kind: self.kind, dry: false, pts: best.pts, hw: best.hw, lab: m.swatches[best.swatch].lab, predicted: best.score, width_gain: self.width_gain };
        Some((plan, foot))
    }

    pub(super) fn hit(&mut self) -> Vec<usize> {
        self.sched.misses = 0;
        let tiles = std::mem::take(&mut self.last_tiles);
        for &t in &tiles {
            self.tabu[t] *= VISIT_DECAY;
        }
        tiles
    }

    pub(super) fn miss(&mut self) -> bool {
        for t in std::mem::take(&mut self.last_tiles) {
            self.tabu[t] *= TABU;
        }
        self.sched.misses += 1;
        self.sched.misses >= MISS_LIMIT
    }

    pub(super) fn run(&mut self, job: &Job) -> Batch {
        if self.kind != job.kind || self.range.lo != job.range.lo || self.range.hi != job.range.hi {
            self.kind = job.kind;
            self.range = job.range;
        }
        if let Some(s) = job.reset {
            let start = if self.started { self.sched.start.max(s) } else { s };
            self.started = true;
            self.sched = Schedule::new(s);
            self.sched.start = start;
            self.tabu.iter_mut().for_each(|t| *t = 1.0);
            self.blobs.clear();
        }
        self.set_focus(job.focus);
        self.intent = job.intent;
        self.follow = (job.follow as f32).clamp(0.05, 1.0);
        let fine = (1.0 - self.sched.size_hi / self.sched.start.max(1e-9)).clamp(0.0, 1.0) as f32;
        self.fine = fine;
        let a = GATE * fine;
        let m = self.m.clone();
        self.gate = m.detail.par_iter().map(|d| 1.0 - a + a * d).collect();
        self.refresh_tiles();
        self.longest = (job.longest / self.m.cell) as f32;
        self.width_gain = job.width_gain;
        self.opacity = job.opacity;
        self.efficiency = job.efficiency;
        self.tile_eff = job.tile_eff.clone();
        self.refresh_reference();
        self.blobs.clear();
        self.sched.misses = 0;
        let (mut plans, mut feet, mut scheds, mut cooled) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        while plans.len() < LOOKAHEAD {
            match self.plan_one(job) {
                Some((plan, foot)) => {
                    self.apply(&foot, plan.lab);
                    cooled.push(self.hit());
                    plans.push(plan);
                    feet.push(foot);
                    scheds.push(self.sched.clone());
                }
                None => {
                    if self.miss() {
                        break;
                    }
                }
            }
        }
        let planned = plans.len();
        let o = self.opacity;
        let mut keep = vec![false; planned];
        let mut shade = std::mem::take(&mut self.scratch);
        for i in (0..planned).rev() {
            let (mut vis, mut area) = (0.0f32, 0.0f32);
            for &(k, a) in &feet[i] {
                let a = a * o;
                let w = self.m.weight[k];
                vis += a * (1.0 - shade[k]) * w;
                area += a * w;
                shade[k] = 1.0 - (1.0 - shade[k]) * (1.0 - a);
            }
            let f = vis / area.max(1e-6);
            keep[i] = f >= PRUNE_VISIBLE && plans[i].predicted * f < -MIN_GAIN;
        }
        for foot in &feet {
            for &(k, _) in foot {
                shade[k] = 0.0;
            }
        }
        self.scratch = shade;
        let (mut out, mut pruned, mut last) = (Vec::new(), 0, None);
        for (i, mut p) in plans.into_iter().enumerate() {
            if out.len() >= EXECUTE {
                break;
            }
            if keep[i] {
                p.dry = false;
                out.push(p);
                last = Some(i);
            } else {
                pruned += 1;
            }
        }
        if out.len() >= EXECUTE {
            if let Some(i) = last {
                self.sched = scheds[i].clone();
            }
        }
        let executed = last.map(|i| i + 1).unwrap_or(0);
        for (i, tiles) in cooled.iter().enumerate() {
            if i >= executed || !keep[i] {
                for &t in tiles {
                    self.tabu[t] = (self.tabu[t] / VISIT_DECAY).min(1.0);
                }
            }
        }
        Batch { plans: out, sched: self.sched.clone(), planned, pruned, gen: job.gen, reference: self.reference.clone() }
    }
}
