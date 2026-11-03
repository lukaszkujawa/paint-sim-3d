use crate::color;
use crate::film::{Deposit, KnifeParams, Method, Scrap};
use crate::tools::ToolKind;
use glam::DVec3;
use crate::app::*;

const KNIFE_CAPACITY_PER_WIDTH: f64 = 5.0e-5;

#[derive(Clone, Copy)]
pub(crate) struct KnifeBand {
    pub(crate) vol: f64,
    pub(crate) comp: [f32; 16],
}

const KNIFE_BAND_MIX_PER_M: f64 = 1.2;
const KNIFE_SMEAR: f64 = 0.35;
const KNIFE_PENDING_DECAY: f64 = 0.5;
const KNIFE_POSE_HISTORY: usize = 8;
const KNIFE_END_RIDGE: f64 = 0.8;
const KNIFE_SPILL_PER_M: f64 = 2.5;
const KNIFE_RESERVE_STEPS: f64 = 8.0;
const KNIFE_RIDGE_OVERLAP: f64 = 0.35;
const KNIFE_TURN_DIST: f64 = 0.04;
const KNIFE_TURN_MIN: f64 = 2.0e-4;
impl KnifeBand {
    pub(crate) fn blend(&mut self, vol: f64, comp: &[f32; 16]) {
        let total = self.vol + vol;
        if total <= 0.0 {
            return;
        }
        for i in 0..16 {
            self.comp[i] = ((self.comp[i] as f64 * self.vol + comp[i] as f64 * vol) / total) as f32;
        }
        self.vol = total;
    }
}

impl App {
    pub(crate) fn load_knife(&mut self) {
        let each = self.tool.knife_width * KNIFE_CAPACITY_PER_WIDTH / KNIFE_BANDS as f64;
        for b in &mut self.knife_bands {
            b.vol = each;
            b.comp = self.paint.frac;
        }
    }

    pub(crate) fn knife_load(&self) -> f64 {
        self.knife_bands.iter().map(|b| b.vol).sum()
    }

    pub(crate) fn knife_mix(&self) -> [f32; 16] {
        let mut m = KnifeBand { vol: 0.0, comp: [0.0; 16] };
        for b in &self.knife_bands {
            m.blend(b.vol, &b.comp);
        }
        m.comp
    }

    pub(crate) fn wipe_knife(&mut self) {
        for b in &mut self.knife_bands {
            b.vol = 0.0;
        }
    }

    pub(crate) fn knife_capacity(&self) -> f64 {
        self.tool.knife_width * KNIFE_CAPACITY_PER_WIDTH
    }
}

impl App {
    pub(crate) fn knife_frame(&mut self, pen: Option<f64>, _frame_dt: f32, deposits: &mut Vec<Deposit>) -> Option<KnifeParams> {
        let mut knife = None;
        if self.tool.kind == ToolKind::Knife {
            let here = DVec3::new(self.tool.pos.x, self.tool.pos.y, 0.0);
            let moved = self.knife_track.map(|q| here - q).unwrap_or(DVec3::ZERO);
            let dist = moved.length();
            if dist > KNIFE_TURN_MIN {
                let k = (dist / KNIFE_TURN_DIST).min(1.0);
                let flip = if moved.x * self.knife_dir[0] + moved.y * self.knife_dir[1] < 0.0 { -1.0 } else { 1.0 };
                let nx = self.knife_dir[0] + (flip * moved.x / dist - self.knife_dir[0]) * k;
                let ny = self.knife_dir[1] + (flip * moved.y / dist - self.knife_dir[1]) * k;
                let l = (nx * nx + ny * ny).sqrt().max(1e-9);
                self.knife_dir = [flip * nx / l, flip * ny / l];
            }
            self.knife_track = Some(here);
            let d = DVec3::new(self.knife_dir[0], self.knife_dir[1], 0.0);
            let blade_dir = DVec3::new(-d.y, d.x, 0.0);
            let hw = self.tool.knife_width * 0.5;
            let p = DVec3::new(self.tool.pos.x, self.tool.pos.y, 0.0);
            let gap = self.tool.knife_height * pen.map(|p| 1.0 - PEN_KNIFE_CLOSE * p).unwrap_or(1.0);
            let hold = self.tool.knife_width * KNIFE_HOLD_PER_WIDTH;
            let band_w = self.tool.knife_width / KNIFE_BANDS as f64;
            if self.tool.flowing && !self.paused {
                let step = self.knife_prev.map(|q| p - q).unwrap_or(DVec3::ZERO);
                let ahead = step.length();
                let across = if ahead > 0.0 { (step / ahead).dot(d) } else { 0.0 };
                if ahead > 1e-6 && across > 0.2 {
                    let mdir = step / ahead;
                    let a = p - blade_dir * hw;
                    let b = p + blade_dir * hw;
                    let need = gap.max(1e-6) * band_w * (ahead + self.film.dx as f64);
                    let mut pig = [[0u32; 8]; KNIFE_BANDS];
                    let mut fill = [0.0f32; KNIFE_BANDS];
                    let mut smear = [0.0f32; KNIFE_BANDS];
                    let reserve = need * KNIFE_RESERVE_STEPS;
                    for (i, band) in self.knife_bands.iter().enumerate() {
                        let avail = (band.vol - self.knife_pending[i]).max(0.0);
                        let ratio = (avail / reserve).min(1.0);
                        fill[i] = ratio as f32;
                        smear[i] = (KNIFE_SMEAR * ratio) as f32;
                        pig[i] = color::PaintMix { frac: band.comp }.normalized().packed();
                        self.knife_pending[i] += need * (1.5 * ratio).min(1.0) * ratio.powf(0.6);
                    }
                    self.knife_poses.push_back((self.frame_index as u32, p, blade_dir));
                    while self.knife_poses.len() > KNIFE_POSE_HISTORY {
                        self.knife_poses.pop_front();
                    }
                    knife = Some(KnifeParams {
                        a: [a.x as f32, a.y as f32],
                        b: [b.x as f32, b.y as f32],
                        dir: [mdir.x as f32, mdir.y as f32],
                        sweep: ahead as f32,
                        height: gap as f32,
                        half_width: hw as f32,
                        active: 1,
                        frame: self.frame_index as u32,
                        pad1: 0,
                        bands: KNIFE_BANDS as u32,
                        pad2: 0,
                        pad3: 0,
                        pad4: 0,
                        pig,
                        fill,
                        smear,
                    });
                    let keep = (hold / KNIFE_BANDS as f64).max(reserve);
                    for i in 0..KNIFE_BANDS {
                        let bow = (self.knife_bands[i].vol - self.knife_pending[i] - keep).max(0.0);
                        if bow <= 0.0 {
                            continue;
                        }
                        self.knife_bands[i].vol -= bow;
                        self.tool_released += bow;
                        let r = (0.75 * bow / band_w).sqrt().max(self.film.dx as f64);
                        let q = p + mdir * (r + self.film.dx as f64);
                        let a = q + blade_dir * (-hw + (i as f64 - KNIFE_RIDGE_OVERLAP) * band_w);
                        let b = q + blade_dir * (-hw + (i as f64 + 1.0 + KNIFE_RIDGE_OVERLAP) * band_w);
                        let pig = color::PaintMix { frac: self.knife_bands[i].comp }.normalized().packed();
                        let bn = [blade_dir.x as f32, blade_dir.y as f32];
                        deposits.push(Deposit::ribbon([a.x as f32, a.y as f32], [b.x as f32, b.y as f32], bn, bn, r as f32, r as f32, 0, bow as f32, pig, self.film.dx).laid_by(Method::Knife).recycled());
                    }
                }
                if let Some(q) = self.knife_prev {
                    let dist = (p - q).length();
                    let bow = self.knife_load() - hold;
                    if bow > 0.0 && dist > 0.0 {
                        let spill = bow * (dist * KNIFE_SPILL_PER_M).min(1.0);
                        for (side, bi) in [(-1.0, 0usize), (1.0, KNIFE_BANDS - 1)] {
                            let take = (0.5 * spill).min(self.knife_bands[bi].vol);
                            if take <= 0.0 {
                                continue;
                            }
                            self.knife_bands[bi].vol -= take;
                            let pig = color::PaintMix { frac: self.knife_bands[bi].comp }.normalized().packed();
                            let r = (0.75 * take / dist).sqrt().max(self.film.dx as f64) as f32;
                            let e0 = q + blade_dir * (side * (hw + r as f64 * 0.6));
                            let e1 = p + blade_dir * (side * (hw + r as f64 * 0.6));
                            self.tool_released += take;
                            let dn = [d.x as f32, d.y as f32];
                            deposits.push(Deposit::ribbon([e0.x as f32, e0.y as f32], [e1.x as f32, e1.y as f32], dn, dn, r, r, 0, take as f32, pig, self.film.dx).laid_by(Method::Knife).recycled());
                        }
                    }
                    let mix = (dist * KNIFE_BAND_MIX_PER_M).min(0.3);
                    if mix > 0.0 {
                        let old = self.knife_bands;
                        for i in 0..KNIFE_BANDS {
                            let neighbours = [i.checked_sub(1), Some(i + 1).filter(|&n| n < KNIFE_BANDS)];
                            let given = neighbours.iter().flatten().count() as f64 * 0.5 * mix * old[i].vol;
                            let mut b = KnifeBand { vol: old[i].vol - given, comp: old[i].comp };
                            for j in neighbours.iter().flatten() {
                                b.blend(0.5 * mix * old[*j].vol, &old[*j].comp);
                            }
                            self.knife_bands[i] = b;
                        }
                    }
                }
                self.knife_prev = Some(p);
            } else if !self.paused {
                if let Some(last) = self.knife_prev {
                    let keep = hold / KNIFE_BANDS as f64;
                    for i in 0..KNIFE_BANDS {
                        let vol = (self.knife_bands[i].vol - keep).max(0.0) * KNIFE_END_RIDGE;
                        if vol <= 0.0 {
                            continue;
                        }
                        self.knife_bands[i].vol -= vol;
                        let r = (0.75 * vol / band_w).sqrt().max(self.film.dx as f64);
                        let q = last + d * r;
                        let a = q + blade_dir * (-hw + (i as f64 - KNIFE_RIDGE_OVERLAP) * band_w);
                        let b = q + blade_dir * (-hw + (i as f64 + 1.0 + KNIFE_RIDGE_OVERLAP) * band_w);
                        let pig = color::PaintMix { frac: self.knife_bands[i].comp }.normalized().packed();
                        self.tool_released += vol;
                        let bn = [blade_dir.x as f32, blade_dir.y as f32];
                        deposits.push(Deposit::ribbon([a.x as f32, a.y as f32], [b.x as f32, b.y as f32], bn, bn, r as f32, r as f32, 0, vol as f32, pig, self.film.dx).laid_by(Method::Knife).recycled());
                    }
                }
                self.knife_prev = None;
            }
        } else {
            self.knife_prev = None;
            self.knife_track = None;
        }
        knife
    }

    pub(crate) fn absorb_scraps(&mut self, scraps: &[Scrap]) {
        self.scrap_total += scraps.iter().map(|s| s.vol.max(0.0) as f64).sum::<f64>();
        self.tool_scraped += scraps.iter().map(|s| s.vol.max(0.0) as f64).sum::<f64>();
        self.tool_released += scraps.iter().map(|s| (-s.vol).max(0.0) as f64).sum::<f64>();
        for p in &mut self.knife_pending {
            *p *= KNIFE_PENDING_DECAY;
        }
        if !scraps.is_empty() {
            let width = self.tool.knife_width.max(1e-6);
            let fallback = (DVec3::new(self.tool.pos.x, self.tool.pos.y, 0.0), DVec3::new(-self.knife_dir[1], self.knife_dir[0], 0.0));
            for s in scraps {
                let (centre, blade) = self.knife_poses.iter().rev().find(|q| q.0 == s.frame).map(|q| (q.1, q.2)).unwrap_or(fallback);
                let i = (s.cell % self.film.w) as f64 + 0.5;
                let j = (s.cell / self.film.w) as f64 + 0.5;
                let cell = DVec3::new(i * self.film.dx as f64, j * self.film.dx as f64, 0.0);
                let along = (cell - centre).dot(blade) / width + 0.5;
                let bi = ((along * KNIFE_BANDS as f64).floor().max(0.0) as usize).min(KNIFE_BANDS - 1);
                let vol = s.vol as f64;
                if vol < 0.0 {
                    self.knife_bands[bi].vol = (self.knife_bands[bi].vol + vol).max(0.0);
                    self.knife_pending[bi] = (self.knife_pending[bi] + vol).max(0.0);
                    continue;
                }
                let mut comp = [0.0f32; 16];
                for k in 0..8 {
                    comp[2 * k] = half::f16::from_bits((s.pig[k] & 0xffff) as u16).to_f32();
                    comp[2 * k + 1] = half::f16::from_bits((s.pig[k] >> 16) as u16).to_f32();
                }
                self.knife_bands[bi].blend(vol, &comp);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::KnifeBand;

    #[test]
    fn blending_weights_paint_by_volume() {
        let mut comp = [0.0f32; 16];
        comp[0] = 1.0;
        let mut band = KnifeBand { vol: 1.0, comp };
        let mut other = [0.0f32; 16];
        other[0] = 0.0;
        other[1] = 1.0;
        band.blend(3.0, &other);
        assert_eq!(band.vol, 4.0);
        assert!((band.comp[0] - 0.25).abs() < 1e-6);
        assert!((band.comp[1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn blending_nothing_changes_nothing() {
        let mut band = KnifeBand { vol: 0.0, comp: [0.5; 16] };
        band.blend(0.0, &[1.0; 16]);
        assert_eq!(band.comp, [0.5; 16]);
    }
}
