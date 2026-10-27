use glam::DVec2;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
struct Node {
    p: DVec2,
    s: f64,
    v: f64,
    w: f64,
    q: DVec2,
    speed: f64,
    spread: f64,
    r: f64,
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub p0: DVec2,
    pub p1: DVec2,
    pub n0: DVec2,
    pub n1: DVec2,
    pub r0: f64,
    pub r1: f64,
    pub thread_r: f64,
    pub vol: f64,
    pub speed: f64,
    pub cap_start: bool,
    pub cap_end: bool,
}

#[derive(Default, Clone, Debug)]
pub struct Ribbon {
    nodes: Vec<Node>,
    cursor: f64,
    emitted: f64,
    started: bool,
}

const JOIN_TOL: f64 = 1e-6;
const PILE_DENSITY: f64 = 3.0;

fn pile_segment(p: DVec2, vol: f64, rt: f64, spread: f64, speed: f64) -> Segment {
    let rt = rt.max(1e-5);
    let r = (1.5 * spread.max(1.0) * rt).max((2.0 * vol / PI).cbrt());
    Segment { p0: p, p1: p, n0: DVec2::X, n1: DVec2::X, r0: r, r1: r, thread_r: rt, vol, speed, cap_start: true, cap_end: true }
}

impl Ribbon {
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[cfg(test)]
    pub fn total_volume(&self) -> f64 {
        self.nodes.last().map(|n| n.v).unwrap_or(0.0)
    }

    pub fn push(&mut self, p0: DVec2, p1: DVec2, vol: f64, speed: f64, spread: f64, r: f64) -> Option<Ribbon> {
        let mut finished = None;
        let continues = self.nodes.last().map(|n| n.p.distance(p0) <= JOIN_TOL).unwrap_or(false);
        if !continues {
            if !self.nodes.is_empty() {
                finished = Some(std::mem::take(self));
            }
            self.nodes.push(Node { p: p0, s: 0.0, v: 0.0, w: 0.0, q: DVec2::ZERO, speed, spread, r });
        }
        let last = *self.nodes.last().unwrap();
        let l = last.p.distance(p1);
        if l <= JOIN_TOL {
            let n = self.nodes.last_mut().unwrap();
            n.v += vol;
            n.speed = n.speed.max(speed);
            n.spread = n.spread.max(spread);
            n.r = n.r.max(r);
        } else {
            let v = last.v + vol;
            self.nodes.push(Node { p: p1, s: last.s + l, v, w: last.w + 0.5 * (last.v + v) * l, q: last.q + 0.5 * (last.p + p1) * l, speed, spread, r });
        }
        finished
    }

    fn pile(&mut self, step: f64, out: &mut Vec<Segment>) -> bool {
        let last = *self.nodes.last().unwrap();
        let pending = last.v - self.emitted;
        let rt = last.r.max(1e-5);
        if pending <= 4.0 * PI * rt * rt * rt {
            return false;
        }
        let len = last.s - self.cursor;
        if pending / len.max(step) < PILE_DENSITY * PI * rt * rt {
            return false;
        }
        let p = if len > step && self.nodes.len() > 1 { (self.position_integral(last.s) - self.position_integral(self.cursor)) / len } else { last.p };
        out.push(pile_segment(p, pending, rt, last.spread, last.speed));
        self.nodes.clear();
        self.nodes.push(Node { p: last.p, s: 0.0, v: 0.0, w: 0.0, q: DVec2::ZERO, speed: last.speed, spread: last.spread, r: last.r });
        self.cursor = 0.0;
        self.emitted = 0.0;
        self.started = false;
        true
    }

    pub fn drain(&mut self, step: f64, flush: bool, out: &mut Vec<Segment>) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }
        if !flush && self.pile(step, out) {
            return;
        }
        if n == 1 {
            if flush {
                let nd = self.nodes[0];
                let pending = nd.v - self.emitted;
                if pending > 0.0 {
                    out.push(pile_segment(nd.p, pending, nd.r, nd.spread, nd.speed));
                }
                *self = Ribbon::default();
            }
            return;
        }
        let s_end = self.nodes[n - 1].s;
        let s_avail = if flush { s_end } else { self.nodes[n - 2].s };
        loop {
            let s0 = self.cursor;
            if s0 >= s_end - JOIN_TOL {
                break;
            }
            let span = self.window(s0, step, s_avail);
            let s1 = if flush && s_end - s0 < 1.5 * span { s_end } else { (s0 + span).min(s_end) };
            if !flush && s1 + self.window(s1, step, s_avail) > s_avail {
                break;
            }
            let last = flush && s1 >= s_end - JOIN_TOL;
            let p0 = self.point(s0, step, flush, s_avail);
            let p1 = self.point(s1, step, flush, s_avail);
            let dir = (p1 - p0).normalize_or_zero();
            let piled = p0.distance(p1) < 0.5 * span;
            let cum = self.smoothed_cumulative(s1, step, flush, s_avail);
            let vol = (cum - self.emitted).max(0.0);
            self.emitted = self.emitted.max(cum);
            let r0 = self.footprint(s0, step, flush, s_avail);
            let r1 = self.footprint(s1, step, flush, s_avail);
            let n0 = if self.started && !piled { self.joint(s0, step, flush, s_avail) } else { dir };
            let n1 = if last || piled { dir } else { self.joint(s1, step, flush, s_avail) };
            let mid = 0.5 * (s0 + s1);
            let thread_r = (self.density(mid, step, flush, s_avail) / PI).sqrt();
            out.push(Segment { p0, p1, n0, n1, r0, r1, thread_r, vol, speed: self.speed(mid), cap_start: !self.started || piled, cap_end: last || piled });
            self.started = true;
            self.cursor = s1;
            if last {
                break;
            }
        }
        if flush {
            *self = Ribbon::default();
        } else {
            let keep_from = self.cursor - 2.0 * self.window(self.cursor, step, s_avail) - 2.0 * step;
            let drop = self.nodes.iter().take_while(|nd| nd.s < keep_from).count().saturating_sub(2);
            if drop > 0 {
                self.nodes.drain(..drop);
            }
        }
    }

    fn interval(&self, s: f64) -> usize {
        let n = self.nodes.len();
        self.nodes.partition_point(|nd| nd.s <= s).clamp(1, n - 1) - 1
    }

    fn cumulative(&self, s: f64) -> f64 {
        let i = self.interval(s);
        let (a, b) = (self.nodes[i], self.nodes[i + 1]);
        let t = ((s - a.s) / (b.s - a.s)).clamp(0.0, 1.0);
        a.v + (b.v - a.v) * t
    }

    fn integral(&self, s: f64) -> f64 {
        let i = self.interval(s);
        let (a, b) = (self.nodes[i], self.nodes[i + 1]);
        let l = b.s - a.s;
        let d = (s - a.s).clamp(0.0, l);
        a.w + a.v * d + (b.v - a.v) * d * d / (2.0 * l)
    }

    fn raw_density(&self, s: f64, half: f64, s_hi: f64) -> f64 {
        let lo = (s - half).max(0.0);
        let hi = (s + half).min(s_hi);
        if hi - lo <= JOIN_TOL {
            return 0.0;
        }
        (self.cumulative(hi) - self.cumulative(lo)) / (hi - lo)
    }

    fn window(&self, s: f64, step: f64, s_hi: f64) -> f64 {
        let lambda = self.raw_density(s, step, s_hi);
        (self.spread(s) * (lambda / PI).sqrt()).max(step)
    }

    fn half_window(&self, s: f64, step: f64, flush: bool, s_hi: f64) -> f64 {
        let w = self.window(s, step, s_hi).min(s);
        if flush { w.min(s_hi - s) } else { w }
    }

    fn smoothed_cumulative(&self, s: f64, step: f64, flush: bool, s_hi: f64) -> f64 {
        let w = self.half_window(s, step, flush, s_hi);
        if w <= JOIN_TOL {
            return self.cumulative(s);
        }
        (self.integral(s + w) - self.integral(s - w)) / (2.0 * w)
    }

    fn density(&self, s: f64, step: f64, flush: bool, s_hi: f64) -> f64 {
        let w = self.half_window(s, step, flush, s_hi);
        if w <= 0.5 * step {
            return self.raw_density(s, step, s_hi);
        }
        (self.cumulative(s + w) - self.cumulative(s - w)) / (2.0 * w)
    }

    fn footprint(&self, s: f64, step: f64, flush: bool, s_hi: f64) -> f64 {
        self.spread(s) * (self.density(s, step, flush, s_hi) / PI).sqrt()
    }

    fn lerp_field(&self, s: f64, f: impl Fn(&Node) -> f64) -> f64 {
        let i = self.interval(s);
        let (a, b) = (&self.nodes[i], &self.nodes[i + 1]);
        let t = ((s - a.s) / (b.s - a.s)).clamp(0.0, 1.0);
        f(a) + (f(b) - f(a)) * t
    }

    fn speed(&self, s: f64) -> f64 {
        self.lerp_field(s, |n| n.speed)
    }

    fn spread(&self, s: f64) -> f64 {
        self.lerp_field(s, |n| n.spread)
    }

    fn raw_point(&self, s: f64) -> DVec2 {
        let i = self.interval(s);
        let (a, b) = (self.nodes[i], self.nodes[i + 1]);
        let t = ((s - a.s) / (b.s - a.s)).clamp(0.0, 1.0);
        a.p.lerp(b.p, t)
    }

    fn position_integral(&self, s: f64) -> DVec2 {
        let i = self.interval(s);
        let (a, b) = (self.nodes[i], self.nodes[i + 1]);
        let l = b.s - a.s;
        let d = (s - a.s).clamp(0.0, l);
        a.q + a.p * d + (b.p - a.p) * (d * d / (2.0 * l))
    }

    fn point(&self, s: f64, step: f64, flush: bool, s_hi: f64) -> DVec2 {
        let w = self.half_window(s, step, flush, s_hi);
        if w <= JOIN_TOL {
            return self.raw_point(s);
        }
        (self.position_integral(s + w) - self.position_integral(s - w)) / (2.0 * w)
    }

    fn joint(&self, s: f64, step: f64, flush: bool, s_hi: f64) -> DVec2 {
        let w = self.half_window(s, step, flush, s_hi).max(0.5 * step);
        (self.raw_point(s + w) - self.raw_point(s - w)).normalize_or_zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lay(points: &[(f64, f64)], vols: &[f64]) -> Ribbon {
        let mut r = Ribbon::default();
        for e in 0..points.len() - 1 {
            let a = DVec2::new(points[e].0, points[e].1);
            let b = DVec2::new(points[e + 1].0, points[e + 1].1);
            assert!(r.push(a, b, vols[e], 0.5, 1.5, 3e-4).is_none());
        }
        r
    }

    #[test]
    fn conserves_volume_across_incremental_drains() {
        let step = 4e-4;
        let mut r = Ribbon::default();
        let mut out = Vec::new();
        let mut total = 0.0;
        for e in 0..60 {
            let vol = if e % 3 == 0 { 0.072e-9 } else { 0.009e-9 };
            let a = DVec2::new(0.1 + 0.3e-3 * e as f64, 0.2 + 0.1e-3 * (e as f64 * 0.7).sin());
            let b = DVec2::new(0.1 + 0.3e-3 * (e + 1) as f64, 0.2 + 0.1e-3 * ((e + 1) as f64 * 0.7).sin());
            r.push(a, b, vol, 0.5, 1.5, 3e-4);
            total += vol;
            r.drain(step, false, &mut out);
        }
        r.drain(step, true, &mut out);
        let emitted: f64 = out.iter().map(|s| s.vol).sum();
        assert!((emitted - total).abs() < 1e-12 * total.max(1e-12), "emitted {} total {}", emitted, total);
        assert!(out.first().unwrap().cap_start && out.last().unwrap().cap_end);
        assert!(out.iter().take(out.len() - 1).all(|s| !s.cap_end));
        assert!(r.is_empty());
    }

    #[test]
    fn segments_are_continuous() {
        let pts: Vec<(f64, f64)> = (0..40).map(|i| (0.3 + 1e-3 * i as f64, 0.3 + 2e-3 * (i as f64 * 0.3).sin())).collect();
        let vols = vec![0.05e-9; 39];
        let mut r = lay(&pts, &vols);
        let mut out = Vec::new();
        r.drain(4e-4, true, &mut out);
        for w in out.windows(2) {
            assert!(w[0].p1.distance(w[1].p0) < 1e-12);
            assert!(w[0].n1.distance(w[1].n0) < 1e-12);
            assert!((w[0].r1 - w[1].r0).abs() < 1e-12);
        }
    }

    #[test]
    fn smooths_alternating_edge_volumes() {
        let pts: Vec<(f64, f64)> = (0..80).map(|i| (0.2 + 3e-4 * i as f64, 0.2)).collect();
        let vols: Vec<f64> = (0..79).map(|i| if i % 2 == 0 { 0.08e-9 } else { 0.01e-9 }).collect();
        let mut r = lay(&pts, &vols);
        let mut out = Vec::new();
        r.drain(3e-4, true, &mut out);
        let inner: Vec<f64> = out[6..out.len() - 6].iter().map(|s| s.vol / s.p0.distance(s.p1)).collect();
        let mean = inner.iter().sum::<f64>() / inner.len() as f64;
        let spread = inner.iter().map(|v| (v - mean).abs()).fold(0.0, f64::max) / mean;
        assert!(spread < 0.15, "density spread {}", spread);
    }

    #[test]
    fn jump_starts_a_new_ribbon() {
        let mut r = Ribbon::default();
        assert!(r.push(DVec2::new(0.0, 0.0), DVec2::new(1e-3, 0.0), 1e-10, 0.5, 1.0, 2e-4).is_none());
        let done = r.push(DVec2::new(5e-3, 0.0), DVec2::new(6e-3, 0.0), 1e-10, 0.5, 1.0, 2e-4);
        assert!(done.is_some());
        assert!((done.unwrap().total_volume() - 1e-10).abs() < 1e-20);
        assert!((r.total_volume() - 1e-10).abs() < 1e-20);
    }
}
