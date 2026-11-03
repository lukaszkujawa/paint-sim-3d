use super::*;

const PRESS_RAMP_SIZES: f64 = 2.0;

#[derive(Clone, Copy)]
pub(super) struct ToolRange {
    pub(super) lo: f64,
    pub(super) hi: f64,
    pub(super) half_width: f64,
    pub(super) len: (f32, f32),
    pub(super) speed: f64,
    pub(super) long: bool,
    pub(super) reach: f64,
}

pub(super) fn tool_range(kind: ToolKind, shape: BrushShape) -> Option<ToolRange> {
    let reach = if kind == ToolKind::Brush { 30.0 } else { f64::INFINITY };
    let r = |lo, hi, half_width, len, speed| Some(ToolRange { lo, hi, half_width, len, speed, long: !matches!(kind, ToolKind::Sponge | ToolKind::Knife), reach });
    match kind {
        ToolKind::Pour => r(0.0008, 0.02, 1.15, (2.0, 10.0), 0.5),
        ToolKind::Brush if shape == BrushShape::Flat => r(0.0001, 0.2, 0.5, (2.0, 8.0), 0.3),
        ToolKind::Brush => r(0.0001, 0.03, 0.55, (2.0, 8.0), 0.3),
        ToolKind::Roller => r(0.04, 0.25, 0.5, (1.0, 4.0), 0.35),
        ToolKind::Sponge => r(0.02, 0.15, 0.4, (0.0, 1.2), 0.12),
        ToolKind::Stick => r(0.003, 0.03, 0.2, (6.0, 30.0), 0.4),
        ToolKind::Bottle => r(0.0005, 0.006, 1.5, (6.0, 30.0), 0.2),
        ToolKind::Bucket => r(0.003, 0.03, 1.0, (6.0, 30.0), 0.4),
        ToolKind::Knife => r(0.005, 0.15, 0.5, (1.5, 7.0), 0.15),
        ToolKind::Rubber => None,
    }
}

pub(super) fn is_thread(kind: ToolKind) -> bool {
    matches!(kind, ToolKind::Stick | ToolKind::Bottle | ToolKind::Bucket)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Shape {
    Bezier,
    Flow,
    Iso,
    Ridge,
}

#[derive(Clone)]
pub(super) struct Cand {
    pub(super) centre: [f32; 2],
    pub(super) angle: f32,
    pub(super) len: f32,
    pub(super) bend: f32,
    pub(super) along: f32,
    pub(super) shape: Shape,
    pub(super) stop: f32,
    pub(super) bound: bool,
    pub(super) structured: bool,
    pub(super) region: u8,
    pub(super) size: f64,
    pub(super) hw: f32,
    pub(super) hint: [f32; 3],
    pub(super) pts: Vec<[f32; 2]>,
    pub(super) swatch: usize,
    pub(super) score: f32,
}

#[derive(Clone)]
pub(super) struct Blob {
    pub(super) centre: [f32; 2],
    pub(super) axis: f32,
    pub(super) along: f32,
    pub(super) across: f32,
    pub(super) peak: [f32; 2],
    pub(super) mass: f32,
    pub(super) count: usize,
    pub(super) window: [usize; 4],
    pub(super) tile: usize,
}

#[derive(Clone)]
pub struct Plan {
    pub path: Vec<DVec2>,
    pub lead: f64,
    pub size: f64,
    pub mix: [f32; 16],
    pub speed: f64,
    pub kind: ToolKind,
    pub dry: bool,
    pub(super) pts: Vec<[f32; 2]>,
    pub(super) hw: f32,
    pub(super) lab: [f32; 3],
    pub(super) predicted: f32,
    pub(super) width_gain: f64,
}

impl Plan {
    pub(super) fn heading(a: DVec2, b: DVec2) -> DVec2 {
        let d = b - a;
        if d.length() > 1e-9 { d.normalize() } else { DVec2::X }
    }

    pub fn entry(&self) -> (DVec2, DVec2) {
        let n = self.path.len();
        let dir = if n > 1 { Self::heading(self.path[0], self.path[1]) } else { DVec2::X };
        (self.path[0] - dir * self.lead, dir)
    }

    pub fn ahead(&self, s: f64) -> DVec2 {
        let total = self.length();
        if s <= total {
            self.at(s)
        } else {
            *self.path.last().unwrap() + self.exit_dir() * (s - total)
        }
    }

    pub fn exit_dir(&self) -> DVec2 {
        let n = self.path.len();
        if n > 1 { Self::heading(self.path[n - 2], self.path[n - 1]) } else { DVec2::X }
    }

    pub fn length(&self) -> f64 {
        self.path.windows(2).map(|w| (w[1] - w[0]).length()).sum()
    }

    pub fn at(&self, s: f64) -> DVec2 {
        let mut left = s.max(0.0);
        for w in self.path.windows(2) {
            let l = (w[1] - w[0]).length();
            if left <= l && l > 0.0 {
                return w[0] + (w[1] - w[0]) * (left / l);
            }
            left -= l;
        }
        *self.path.last().unwrap()
    }
}

pub enum Phase {
    Wait(u32),
    Observe,
    Plan,
    Lower(Plan, u32),
    Approach(Plan, f64),
    Stroke(Plan, f64),
    Exit(Plan, f64),
    Lift(Plan, u32),
    Settle(Plan, u32),
}

pub(super) struct Model {
    pub(super) w: usize,
    pub(super) h: usize,
    pub(super) cell: f64,
    pub(super) target: Vec<[f32; 3]>,
    pub(super) flow: Vec<[f32; 3]>,
    pub(super) detail: Vec<f32>,
    pub(super) label: Vec<u8>,
    pub(super) kin: Vec<bool>,
    pub(super) regions: usize,
    pub(super) weight: Vec<f32>,
    pub(super) swatches: Arc<Vec<Swatch>>,
}

impl Model {
    pub(super) fn same_shape(&self, k: usize, region: u8) -> bool {
        self.kin[self.label[k] as usize * self.regions + region as usize]
    }

    pub(super) fn index(&self, p: [f32; 2]) -> usize {
        (p[1].max(0.0) as usize).min(self.h - 1) * self.w + (p[0].max(0.0) as usize).min(self.w - 1)
    }

    pub(super) fn bbox(&self, pts: &[[f32; 2]], pad: f32) -> [usize; 4] {
        let mut b = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        for p in pts {
            b[0] = b[0].min(p[0]);
            b[1] = b[1].min(p[1]);
            b[2] = b[2].max(p[0]);
            b[3] = b[3].max(p[1]);
        }
        [
            (b[0] - pad).floor().max(0.0) as usize,
            (b[1] - pad).floor().max(0.0) as usize,
            ((b[2] + pad).ceil().max(0.0) as usize).min(self.w),
            ((b[3] + pad).ceil().max(0.0) as usize).min(self.h),
        ]
    }

    pub(super) fn footprint(&self, cov: &mut [f32], pts: &[[f32; 2]], hw: f32, tapered: bool) -> Vec<(usize, f32)> {
        self.footprint_at(cov, pts, hw, tapered, 1)
    }

    pub(super) fn footprint_at(&self, cov: &mut [f32], pts: &[[f32; 2]], hw: f32, tapered: bool, stride: usize) -> Vec<(usize, f32)> {
        let mut touched = Vec::new();
        let segs: Vec<[[f32; 2]; 2]> = if pts.len() < 2 { vec![[pts[0], pts[0]]] } else { pts.windows(2).map(|w| [w[0], w[1]]).collect() };
        let n = segs.len();
        for (si, [a, e]) in segs.into_iter().enumerate() {
            let hw = if tapered && n > 2 { hw * taper((si as f32 + 0.5) / n as f32) } else { hw };
            let bb = self.bbox(&[a, e], hw + 1.0);
            let (ex, ey) = (e[0] - a[0], e[1] - a[1]);
            let l2 = ex * ex + ey * ey;
            let j0 = bb[1].div_ceil(stride) * stride;
            let i0 = bb[0].div_ceil(stride) * stride;
            for j in (j0..bb[3]).step_by(stride) {
                for i in (i0..bb[2]).step_by(stride) {
                    let (x, y) = (i as f32 + 0.5, j as f32 + 0.5);
                    let t = if l2 > 0.0 { (((x - a[0]) * ex + (y - a[1]) * ey) / l2).clamp(0.0, 1.0) } else { 0.0 };
                    let (dx, dy) = (x - a[0] - t * ex, y - a[1] - t * ey);
                    let c = (hw - (dx * dx + dy * dy).sqrt() + 0.5).clamp(0.0, 1.0);
                    if c <= 0.0 {
                        continue;
                    }
                    let k = j * self.w + i;
                    if cov[k] == 0.0 {
                        touched.push(k);
                    }
                    cov[k] = cov[k].max(c);
                }
            }
        }
        let out = touched.iter().map(|&k| (k, cov[k])).collect();
        for k in touched {
            cov[k] = 0.0;
        }
        out
    }

    pub(super) fn nearest(&self, want: [f32; 3]) -> usize {
        let mut best = (f32::MAX, 0);
        for (k, s) in self.swatches.iter().enumerate() {
            let d = (s.lab[0] - want[0]).powi(2) + (s.lab[1] - want[1]).powi(2) + (s.lab[2] - want[2]).powi(2);
            if d < best.0 {
                best = (d, k);
            }
        }
        best.1
    }

    pub(super) fn mask(&self, focus: Focus) -> Vec<f32> {
        let (w, h) = (self.w, self.h);
        let aspect = w as f32 / h as f32;
        (0..w * h).into_par_iter().map(|k| focus.weight(((k % w) as f32 + 0.5) / w as f32, 1.0 - ((k / w) as f32 + 0.5) / h as f32, aspect)).collect()
    }
}

pub(super) fn press_profile(s: f64, total: f64, size: f64) -> f64 {
    let ramp = (TAPER_END as f64 * total).min(PRESS_RAMP_SIZES * size).max(1e-6);
    let t = (s.min(total - s) / ramp).clamp(0.0, 1.0);
    PRESS_START + (1.0 - PRESS_START) * t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(path: Vec<DVec2>) -> Plan {
        Plan { path, lead: 0.01, size: 0.01, mix: [0.0; 16], speed: 0.3, kind: ToolKind::Brush, dry: false, pts: vec![[0.0, 0.0]], hw: 1.0, lab: [0.0; 3], predicted: 0.0, width_gain: 1.0 }
    }

    #[test]
    fn walks_along_the_path() {
        let p = plan(vec![DVec2::ZERO, DVec2::new(1.0, 0.0), DVec2::new(1.0, 1.0)]);
        assert_eq!(p.length(), 2.0);
        assert_eq!(p.at(0.5), DVec2::new(0.5, 0.0));
        assert_eq!(p.at(1.5), DVec2::new(1.0, 0.5));
        assert_eq!(p.at(9.0), DVec2::new(1.0, 1.0));
    }

    #[test]
    fn enters_behind_the_start_and_exits_past_the_end() {
        let p = plan(vec![DVec2::ZERO, DVec2::new(1.0, 0.0)]);
        let (from, dir) = p.entry();
        assert_eq!(dir, DVec2::X);
        assert!((from.x + 0.01).abs() < 1e-12);
        assert_eq!(p.ahead(1.5), DVec2::new(1.5, 0.0));
    }

    #[test]
    fn single_point_path_is_safe() {
        let p = plan(vec![DVec2::new(0.2, 0.3)]);
        assert_eq!(p.length(), 0.0);
        assert_eq!(p.exit_dir(), DVec2::X);
        assert_eq!(p.at(1.0), DVec2::new(0.2, 0.3));
    }

    #[test]
    fn pressure_ramps_in_and_out() {
        let (total, size) = (1.0, 0.02);
        let (start, middle, end) = (press_profile(0.0, total, size), press_profile(0.5, total, size), press_profile(total, total, size));
        assert!((middle - 1.0).abs() < 1e-9);
        assert!(start < middle && end < middle);
        assert!((start - end).abs() < 1e-9);
    }

    #[test]
    fn every_painting_tool_has_a_range_and_rubber_does_not() {
        assert!(tool_range(ToolKind::Rubber, BrushShape::Round).is_none());
        for kind in [ToolKind::Brush, ToolKind::Roller, ToolKind::Knife, ToolKind::Stick, ToolKind::Bottle, ToolKind::Bucket, ToolKind::Pour, ToolKind::Sponge] {
            let r = tool_range(kind, BrushShape::Flat).unwrap();
            assert!(r.lo < r.hi && r.speed > 0.0);
        }
    }
}
