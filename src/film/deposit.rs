use super::*;

const DEPOSIT_METHOD_SHIFT: u32 = 8;

const STRAND_DRY: f32 = 0.85;

const STRAND_WET: f32 = 0.12;

const CLUMP_AMP: f32 = 0.5;

const CLUMP_W: f32 = 3.0e-3;

const STRAND_FLOOR: f32 = 0.0;

const STRAND_SOFT: f32 = 0.12;

const STRAND_CELLS: f32 = 1.0;

const STRAND_MIN_W: f32 = 0.18e-3;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct Deposit {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub n0: [f32; 2],
    pub n1: [f32; 2],
    pub r0: f32,
    pub r1: f32,
    pub h0: f32,
    pub vol: f32,
    pub kind: u32,
    pub pad: [u32; 3],
    pub pig: [u32; 8],
}

#[derive(Clone, Copy)]
pub enum Method {
    Thread = 1,
    Drop = 2,
    Pour = 3,
    Brush = 4,
    Knife = 5,
    Roller = 6,
}

pub(super) fn strand_hash(i: i32, seed: u32) -> f32 {
    let mut h = (i as u32).wrapping_mul(73856093) ^ seed.wrapping_mul(19349663);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1e995);
    h ^= h >> 15;
    (h & 0xffff) as f32 / 65535.0
}

pub(super) fn strand_noise(x: f32, seed: u32) -> f32 {
    let i = x.floor();
    let f = x - i;
    let t = f * f * (3.0 - 2.0 * f);
    let a = strand_hash(i as i32, seed);
    let b = strand_hash(i as i32 + 1, seed);
    a + (b - a) * t
}

pub(super) fn strand_mask(s: f32, seed: u32, thr: f32, width: f32) -> f32 {
    let fine = strand_noise(s / width, seed);
    let clump = strand_noise(s / CLUMP_W, seed ^ 40503);
    let t = thr + CLUMP_AMP * (clump - 0.5) * thr;
    let v = ((fine - t + STRAND_SOFT) / (2.0 * STRAND_SOFT)).clamp(0.0, 1.0);
    STRAND_FLOOR + (1.0 - STRAND_FLOOR) * v
}

pub(super) fn strand_duty(thr: f32) -> f32 {
    static TABLE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        (0..256)
            .map(|k| {
                let t = k as f32 / 255.0;
                let n = 40000;
                let sum: f32 = (0..n).map(|i| strand_mask(i as f32 * 0.37e-3 + 1.0e-5, 7 + (i / 2000) as u32, t, 0.39e-3)).sum();
                sum / n as f32
            })
            .collect()
    });
    table[((thr * 255.0) as usize).min(255)]
}

pub(super) fn segment_dir(p0: [f32; 2], p1: [f32; 2]) -> ([f32; 2], f32) {
    let d = [p1[0] - p0[0], p1[1] - p0[1]];
    let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
    if l > 0.0 { ([d[0] / l, d[1] / l], l) } else { ([1.0, 0.0], 0.0) }
}

impl Deposit {
    pub fn ribbon(p0: [f32; 2], p1: [f32; 2], n0: [f32; 2], n1: [f32; 2], r0: f32, r1: f32, caps: u32, volume: f32, pig: [u32; 8], dx: f32) -> Self {
        let min_r = 0.8 * dx;
        let a = r0.max(min_r);
        let b = r1.max(min_r);
        let mut d = Self { p0, p1, n0, n1, r0: a, r1: b, h0: a, vol: volume, kind: caps, pad: [0; 3], pig };
        let lattice = 0.25 * dx;
        let sampled = d.sampled_volume(lattice);
        let rm = a.max(b);
        let disc = std::f32::consts::PI * rm * rm * rm / 2.0;
        if sampled < 0.25 * disc {
            let mid = [0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])];
            d = Self { p0: mid, p1: mid, n0, n1, r0: rm, r1: rm, h0: rm, vol: volume, kind: caps, pad: [0; 3], pig };
            d.h0 = volume / d.sampled_volume(lattice) * rm;
            return d;
        }
        d.h0 = volume / sampled * a;
        d
    }

    pub fn capsule(p0: [f32; 2], p1: [f32; 2], r: f32, volume: f32, pig: [u32; 8], dx: f32) -> Self {
        let (d, _) = segment_dir(p0, p1);
        Self::ribbon(p0, p1, d, d, r, r, DEPOSIT_CAP_START | DEPOSIT_CAP_END, volume, pig, dx)
    }

    pub fn stroke(p0: [f32; 2], p1: [f32; 2], r: f32, volume: f32, pig: [u32; 8], dx: f32) -> Self {
        let (d, l) = segment_dir(p0, p1);
        if l < r.max(0.8 * dx) {
            return Self::capsule(p0, p1, r, volume, pig, dx);
        }
        Self::ribbon(p0, p1, d, d, r, r, 0, volume, pig, dx)
    }

    pub fn laid_by(mut self, method: Method) -> Self {
        self.kind |= (method as u32) << DEPOSIT_METHOD_SHIFT;
        self
    }

    pub fn strands(mut self, seed: u32, offset: f32, fill: f32, dx: f32) -> Self {
        self.kind |= DEPOSIT_STRANDS;
        let thr = (1.0 - fill).clamp(STRAND_WET, STRAND_DRY);
        self.pad[0] = (seed << 8) | (thr * 255.0) as u32;
        self.pad[1] = offset.to_bits();
        self.pad[2] = STRAND_MIN_W.max(STRAND_CELLS * dx).to_bits();
        self.h0 /= strand_duty((self.pad[0] & 255) as f32 / 255.0);
        self
    }

    pub fn recycled(mut self) -> Self {
        self.kind |= DEPOSIT_RECYCLED;
        self
    }

    pub fn max_radius(&self) -> f32 {
        self.r0.max(self.r1)
    }

    pub fn max_height(&self) -> f32 {
        self.h0 * self.max_radius() / self.r0
    }

    pub(super) fn sampled_volume(&self, lattice: f32) -> f32 {
        self.sampled_volume_in(lattice, [f32::NEG_INFINITY, f32::NEG_INFINITY], [f32::INFINITY, f32::INFINITY])
    }

    pub(super) fn sampled_volume_in(&self, lattice: f32, lo: [f32; 2], hi: [f32; 2]) -> f32 {
        let rm = self.max_radius() + lattice;
        let x0 = ((self.p0[0].min(self.p1[0]) - rm).max(lo[0]) / lattice).floor() as i32;
        let x1 = ((self.p0[0].max(self.p1[0]) + rm).min(hi[0]) / lattice).ceil() as i32;
        let y0 = ((self.p0[1].min(self.p1[1]) - rm).max(lo[1]) / lattice).floor() as i32;
        let y1 = ((self.p0[1].max(self.p1[1]) + rm).min(hi[1]) / lattice).ceil() as i32;
        let height = self.profile();
        let mut sum = 0.0;
        for i in x0..x1 {
            for j in y0..y1 {
                sum += height([(i as f32 + 0.5) * lattice, (j as f32 + 0.5) * lattice]);
            }
        }
        sum * lattice * lattice
    }

    pub fn inside_fraction(&self, w: f32, h: f32) -> f32 {
        let rb = self.max_radius();
        let full = self.p0[0].min(self.p1[0]) - rb >= 0.0 && self.p0[1].min(self.p1[1]) - rb >= 0.0 && self.p0[0].max(self.p1[0]) + rb <= w && self.p0[1].max(self.p1[1]) + rb <= h;
        if full {
            return 1.0;
        }
        let lattice = 0.25 * self.r0.min(self.r1).max(1e-4);
        let all = self.sampled_volume(lattice);
        if all <= 0.0 {
            return 0.0;
        }
        (self.sampled_volume_in(lattice, [0.0, 0.0], [w, h]) / all).clamp(0.0, 1.0)
    }

    #[cfg(test)]
    pub fn height_at(&self, p: [f32; 2]) -> f32 {
        self.profile()(p)
    }

    pub(super) fn profile(&self) -> impl Fn([f32; 2]) -> f32 + '_ {
        let (dir, l) = segment_dir(self.p0, self.p1);
        let cap_start = self.kind & DEPOSIT_CAP_START != 0;
        let cap_end = self.kind & DEPOSIT_CAP_END != 0;
        let len = |q: [f32; 2]| (q[0] * q[0] + q[1] * q[1]).sqrt();
        let strands = self.kind & DEPOSIT_STRANDS != 0;
        let (seed, thr, offset, width) = (self.pad[0] >> 8, (self.pad[0] & 255) as f32 / 255.0, f32::from_bits(self.pad[1]), f32::from_bits(self.pad[2]));
        let mask = move |q: [f32; 2], d: [f32; 2]| if strands { strand_mask(offset - (q[0] * d[1] - q[1] * d[0]), seed, thr, width) } else { 1.0 };
        move |p: [f32; 2]| {
            let q0 = [p[0] - self.p0[0], p[1] - self.p0[1]];
            let q1 = [p[0] - self.p1[0], p[1] - self.p1[1]];
            if l <= 0.0 {
                let x = len(q0) / self.r0;
                return if x < 1.0 { self.h0 * (1.0 - x * x) * mask(q0, self.n0) } else { 0.0 };
            }
            if !cap_start && q0[0] * self.n0[0] + q0[1] * self.n0[1] < 0.0 {
                return 0.0;
            }
            if !cap_end && q1[0] * self.n1[0] + q1[1] * self.n1[1] >= 0.0 {
                return 0.0;
            }
            let u = q0[0] * dir[0] + q0[1] * dir[1];
            if u < -self.r0 || u > l + self.r1 {
                return 0.0;
            }
            let mut dist = (q0[0] * dir[1] - q0[1] * dir[0]).abs();
            if cap_start && u < 0.0 {
                dist = len(q0);
            }
            if cap_end && u > l {
                dist = len(q1);
            }
            let rt = self.r0 + (self.r1 - self.r0) * (u / l).clamp(0.0, 1.0);
            let x = dist / rt;
            if x < 1.0 { self.h0 * (rt / self.r0) * (1.0 - x * x) * mask(q0, dir) } else { 0.0 }
        }
    }
}

#[cfg(test)]
mod strand_tests {
    use super::*;

    #[test]
    fn strands_conserve_paint_on_average() {
        let dx = 0.39e-3f32;
        for fill in [1.0f32, 0.6, 0.3, 0.1] {
            let (mut laid, mut asked) = (0.0f64, 0.0f64);
            for k in 0..600 {
                let vol = 2.0e-9f32;
                let plain = Deposit::stroke([0.1, 0.1], [0.11, 0.1], 0.0004, vol, [0; 8], dx);
                let d = plain.strands(3 + (k / 40) as u32, k as f32 * 0.37e-3, fill, dx);
                laid += d.sampled_volume(0.25 * dx) as f64;
                asked += vol as f64;
            }
            let ratio = laid / asked;
            assert!((ratio - 1.0).abs() < 0.12, "fill {} laid/asked {}", fill, ratio);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampled_volume(d: &Deposit) -> f32 {
        let r = d.max_radius();
        let step = r / 40.0;
        let lo = [d.p0[0].min(d.p1[0]) - 2.0 * r, d.p0[1].min(d.p1[1]) - 2.0 * r];
        let hi = [d.p0[0].max(d.p1[0]) + 2.0 * r, d.p0[1].max(d.p1[1]) + 2.0 * r];
        let mut sum = 0.0;
        let mut x = lo[0];
        while x < hi[0] {
            let mut y = lo[1];
            while y < hi[1] {
                sum += d.height_at([x, y]);
                y += step;
            }
            x += step;
        }
        sum * step * step
    }

    fn reference_height(d: &Deposit, p: [f32; 2]) -> f32 {
        let (dir, l) = segment_dir(d.p0, d.p1);
        let q0 = [p[0] - d.p0[0], p[1] - d.p0[1]];
        let q1 = [p[0] - d.p1[0], p[1] - d.p1[1]];
        let len = |q: [f32; 2]| (q[0] * q[0] + q[1] * q[1]).sqrt();
        if l <= 0.0 {
            let x = len(q0) / d.r0;
            return if x < 1.0 { d.h0 * (1.0 - x * x) } else { 0.0 };
        }
        let cap_start = d.kind & DEPOSIT_CAP_START != 0;
        let cap_end = d.kind & DEPOSIT_CAP_END != 0;
        if !cap_start && q0[0] * d.n0[0] + q0[1] * d.n0[1] < 0.0 {
            return 0.0;
        }
        if !cap_end && q1[0] * d.n1[0] + q1[1] * d.n1[1] >= 0.0 {
            return 0.0;
        }
        let u = q0[0] * dir[0] + q0[1] * dir[1];
        if u < -d.r0 || u > l + d.r1 {
            return 0.0;
        }
        let mut dist = (q0[0] * dir[1] - q0[1] * dir[0]).abs();
        if cap_start && u < 0.0 {
            dist = len(q0);
        }
        if cap_end && u > l {
            dist = len(q1);
        }
        let rt = d.r0 + (d.r1 - d.r0) * (u / l).clamp(0.0, 1.0);
        let x = dist / rt;
        if x < 1.0 { d.h0 * (rt / d.r0) * (1.0 - x * x) } else { 0.0 }
    }

    #[test]
    fn lattice_volume_matches_reference() {
        let pig = [0u32; 8];
        let dx = 3.9e-4f32;
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        for _ in 0..300 {
            let p0 = [0.1 + 0.05 * rnd(), 0.1 + 0.05 * rnd()];
            let p1 = [p0[0] + 4e-3 * (rnd() - 0.5), p0[1] + 4e-3 * (rnd() - 0.5)];
            let r = 2e-4 + 2e-3 * rnd();
            let d = Deposit::stroke(p0, p1, r, 1e-9, pig, dx);
            let lattice = 0.25 * dx;
            let rm = d.max_radius() + lattice;
            let x0 = ((d.p0[0].min(d.p1[0]) - rm) / lattice).floor() as i32;
            let x1 = ((d.p0[0].max(d.p1[0]) + rm) / lattice).ceil() as i32;
            let y0 = ((d.p0[1].min(d.p1[1]) - rm) / lattice).floor() as i32;
            let y1 = ((d.p0[1].max(d.p1[1]) + rm) / lattice).ceil() as i32;
            let mut sum = 0.0;
            for i in x0..x1 {
                for j in y0..y1 {
                    sum += reference_height(&d, [(i as f32 + 0.5) * lattice, (j as f32 + 0.5) * lattice]);
                }
            }
            assert_eq!((sum * lattice * lattice).to_bits(), d.sampled_volume(lattice).to_bits());
            for _ in 0..50 {
                let p = [p0[0] + 6e-3 * (rnd() - 0.5), p0[1] + 6e-3 * (rnd() - 0.5)];
                assert_eq!(reference_height(&d, p).to_bits(), d.height_at(p).to_bits());
            }
        }
    }

    #[test]
    fn ribbon_segments_hold_their_volume() {
        let pig = [0u32; 8];
        let c = 0.3f32.cos();
        let s = 0.3f32.sin();
        let cases = [
            Deposit::ribbon([0.0, 0.0], [1e-3, 0.0], [1.0, 0.0], [1.0, 0.0], 4e-4, 6e-4, 0, 1e-9, pig, 3.9e-4),
            Deposit::ribbon([0.0, 0.0], [1e-3, 0.0], [c, -s], [c, s], 4e-4, 6e-4, 0, 1e-9, pig, 3.9e-4),
            Deposit::ribbon([0.0, 0.0], [3e-4, 0.0], [c, s], [c, -s], 8e-4, 8e-4, 0, 1e-9, pig, 3.9e-4),
            Deposit::capsule([0.0, 0.0], [2e-3, 1e-3], 5e-4, 1e-9, pig, 3.9e-4),
            Deposit::capsule([0.0, 0.0], [0.0, 0.0], 5e-4, 1e-9, pig, 3.9e-4),
            Deposit::stroke([0.0, 0.0], [4e-3, 0.0], 5e-4, 1e-9, pig, 3.9e-4),
        ];
        for (i, d) in cases.iter().enumerate() {
            let v = sampled_volume(d);
            assert!((v - d.vol).abs() < 0.03 * d.vol, "case {} sampled {} stored {}", i, v, d.vol);
        }
    }
}
