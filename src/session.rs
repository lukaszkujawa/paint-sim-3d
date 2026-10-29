use crate::{tools::Airborne, App};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAGIC: &[u8; 4] = b"PSIM";
const VERSION: u32 = 1;
const CHUNK: usize = 32 << 20;
const MAX_LEVELS: usize = 64;
const MAX_FULL: usize = 3;
const AUTOSAVE_SECS: f32 = 180.0;
const THUMB_W: u32 = 360;
const THUMB_WHITE: f32 = 0.8;
pub const PAINTINGS_DIR: &str = "paintings";
pub const AUTOSAVE_NAME: &str = "autosave";

pub enum LevelKind {
    Journal { start: u32 },
    Full(Vec<u8>),
}

pub struct Level {
    pub gen: u32,
    pub kind: LevelKind,
}

pub struct History {
    pub gen: u32,
    pub levels: Vec<Level>,
    pub saved_gen: u32,
    pub autosaved_gen: u32,
    pub last_autosave: Instant,
    pub name: String,
}

impl History {
    pub fn new() -> Self {
        Self { gen: 0, levels: Vec::new(), saved_gen: 0, autosaved_gen: 0, last_autosave: Instant::now(), name: String::new() }
    }

    fn trim(&mut self) {
        while self.levels.len() > MAX_LEVELS {
            self.levels.remove(0);
        }
        let fulls: Vec<usize> = self.levels.iter().enumerate().filter(|(_, l)| matches!(l.kind, LevelKind::Full(_))).map(|(i, _)| i).collect();
        if fulls.len() > MAX_FULL {
            let cut = fulls[fulls.len() - MAX_FULL - 1];
            self.levels.drain(..=cut);
        }
    }
}

pub struct Painting {
    pub name: String,
    pub path: PathBuf,
    pub modified: std::time::SystemTime,
}

fn pack(parts: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    for p in parts {
        let chunks: Vec<Vec<u8>> = p.par_chunks(CHUNK).map(lz4_flex::compress_prepend_size).collect();
        out.extend_from_slice(&(p.len() as u64).to_le_bytes());
        out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
        for c in &chunks {
            out.extend_from_slice(&(c.len() as u64).to_le_bytes());
            out.extend_from_slice(c);
        }
    }
    out
}

fn unpack(data: &[u8]) -> Option<Vec<Vec<u8>>> {
    let mut off = 0usize;
    let take = |off: &mut usize, n: usize| -> Option<&[u8]> {
        let s = data.get(*off..*off + n)?;
        *off += n;
        Some(s)
    };
    let count = u32::from_le_bytes(take(&mut off, 4)?.try_into().ok()?) as usize;
    let mut parts = Vec::with_capacity(count);
    for _ in 0..count {
        let len = u64::from_le_bytes(take(&mut off, 8)?.try_into().ok()?) as usize;
        let nchunks = u32::from_le_bytes(take(&mut off, 4)?.try_into().ok()?) as usize;
        let mut slices = Vec::with_capacity(nchunks);
        for _ in 0..nchunks {
            let clen = u64::from_le_bytes(take(&mut off, 8)?.try_into().ok()?) as usize;
            slices.push(take(&mut off, clen)?);
        }
        let decoded: Vec<Vec<u8>> = slices.par_iter().map(|c| lz4_flex::decompress_size_prepended(c).unwrap_or_default()).collect();
        let joined: Vec<u8> = decoded.concat();
        if joined.len() != len {
            return None;
        }
        parts.push(joined);
    }
    Some(parts)
}

pub fn painting_path(name: &str) -> PathBuf {
    Path::new(PAINTINGS_DIR).join(format!("{}.canvaz", name))
}

pub fn thumb_path(path: &Path) -> PathBuf {
    path.with_extension("png")
}

pub fn list_paintings() -> Vec<Painting> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(PAINTINGS_DIR) {
        for e in rd.flatten() {
            let path = e.path();
            if path.extension().map(|x| x == "canvaz" || x == "psim").unwrap_or(false) {
                let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                let modified = e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
                out.push(Painting { name, path, modified });
            }
        }
    }
    out.sort_by(|a, b| b.modified.cmp(&a.modified));
    out
}

impl App {
    fn capture(&self) -> Vec<u8> {
        pack(&self.film.read_state(&self.gpu))
    }

    fn apply(&mut self, blob: &[u8]) -> bool {
        let Some(parts) = unpack(blob) else { return false };
        let ok = self.film.write_state(&self.gpu, &parts);
        if ok {
            self.air = Airborne::new();
        }
        ok
    }

    fn next_gen(&mut self) -> u32 {
        self.history.gen += 1;
        self.film.journal.gen = self.history.gen;
        self.history.gen
    }

    fn history_on(&self) -> bool {
        self.auto.as_ref().map(|a| a.script.starts_with("undo") || a.script.starts_with("evolv")).unwrap_or(true)
    }

    pub(crate) fn begin_stroke(&mut self) {
        if !self.history_on() {
            return;
        }
        let start = self.film.journal.head(&self.gpu);
        let gen = self.next_gen();
        self.history.levels.push(Level { gen, kind: LevelKind::Journal { start } });
        self.history.trim();
    }

    pub(crate) fn begin_full(&mut self) {
        if !self.history_on() {
            return;
        }
        let blob = self.capture();
        let gen = self.next_gen();
        self.history.levels.push(Level { gen, kind: LevelKind::Full(blob) });
        self.history.trim();
    }

    pub(crate) fn can_undo(&self) -> bool {
        !self.history.levels.is_empty()
    }

    pub(crate) fn undo(&mut self) {
        if self.history.levels.is_empty() {
            self.notice("Nothing to undo");
            return;
        }
        if let Some(p) = self.painter.as_mut() {
            p.stop();
        }
        self.stop_evolve();
        match self.rollback() {
            Ok(()) => self.notice("Undone"),
            Err(msg) => self.notice(msg),
        }
    }

    pub(crate) fn rollback(&mut self) -> Result<(), &'static str> {
        let Some(level) = self.history.levels.pop() else {
            return Err("Nothing to undo");
        };
        match level.kind {
            LevelKind::Journal { start } => {
                let head = self.film.journal.head(&self.gpu);
                if head.wrapping_sub(start) > self.film.journal.cap {
                    self.history.levels.retain(|l| matches!(l.kind, LevelKind::Full(_)));
                    return Err("That stroke changed too much of the canvas to undo");
                }
                self.film.restore_journal(&self.gpu, level.gen);
                self.air = Airborne::new();
            }
            LevelKind::Full(blob) => {
                if !self.apply(&blob) {
                    return Err("Undo failed");
                }
            }
        }
        self.next_gen();
        Ok(())
    }

    pub(crate) fn notice(&mut self, msg: &str) {
        self.last_beauty = msg.to_string();
    }

    fn thumbnail(&self, path: &Path) {
        let raw = self.read_texture_raw(&self.film.color_tex);
        let (w, h) = (self.film.w, self.film.h);
        let tw = THUMB_W;
        let th = (THUMB_W as f32 * h as f32 / w as f32).round() as u32;
        let mut img = image::RgbImage::new(tw, th);
        let mut lum: Vec<f32> = (0..(w as usize * h as usize)).step_by(97).map(|i| {
            let o = i * 8;
            let ch = |k: usize| half::f16::from_le_bytes([raw[o + 2 * k], raw[o + 2 * k + 1]]).to_f32();
            0.2126 * ch(0) + 0.7152 * ch(1) + 0.0722 * ch(2)
        }).collect();
        lum.sort_by(|a, b| a.total_cmp(b));
        let white = lum.get(lum.len() * 95 / 100).copied().unwrap_or(1.0).max(0.05);
        let gain = THUMB_WHITE / white;
        let enc = |c: f32| {
            let c = c * gain;
            let c = c.clamp(0.0, 1.0);
            let s = if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
            (s * 255.0).round() as u8
        };
        for ty in 0..th {
            for tx in 0..tw {
                let x = (tx as u64 * w as u64 / tw as u64) as usize;
                let y = (h - 1 - (ty as u64 * h as u64 / th as u64) as u32) as usize;
                let o = (y * w as usize + x) * 8;
                let ch = |k: usize| half::f16::from_le_bytes([raw[o + 2 * k], raw[o + 2 * k + 1]]).to_f32();
                img.put_pixel(tx, ty, image::Rgb([enc(ch(0)), enc(ch(1)), enc(ch(2))]));
            }
        }
        let _ = img.save(thumb_path(path));
    }

    pub(crate) fn save_painting(&mut self, name: &str) -> bool {
        let _ = std::fs::create_dir_all(PAINTINGS_DIR);
        let path = painting_path(name);
        let t0 = Instant::now();
        let blob = self.capture();
        let s = &self.film.settings;
        let mut out = Vec::with_capacity(blob.len() + 64);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.film.w.to_le_bytes());
        out.extend_from_slice(&self.film.h.to_le_bytes());
        for v in [s.tilt_deg, s.tilt_dir_deg, s.cure_rate, self.film.h_max, self.film.s_max] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.film.total_deposited.to_le_bytes());
        out.extend_from_slice(&blob);
        let ok = std::fs::write(&path, &out).is_ok();
        if ok {
            self.thumbnail(&path);
            self.history.saved_gen = self.history.gen;
            if name != AUTOSAVE_NAME {
                self.history.name = name.to_string();
                self.notice(&format!("Saved {} ({:.0} MB, {:.1} s)", name, out.len() as f64 / 1e6, t0.elapsed().as_secs_f32()));
            }
        } else {
            self.notice(&format!("Could not save {}", path.display()));
        }
        ok
    }

    pub(crate) fn open_painting(&mut self, path: &Path) {
        let Ok(data) = std::fs::read(path) else {
            self.notice("Could not read the painting");
            return;
        };
        if data.len() < 44 || &data[0..4] != MAGIC {
            self.notice("Not a painting file");
            return;
        }
        let u = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
        let f = |o: usize| f32::from_le_bytes(data[o..o + 4].try_into().unwrap());
        if u(4) != VERSION {
            self.notice("That painting was saved by an incompatible version");
            return;
        }
        if u(8) != self.film.w || u(12) != self.film.h {
            self.rebuild_canvas(u(8));
            if u(8) != self.film.w || u(12) != self.film.h {
                self.notice(&format!("That painting was made at {}x{} cells and this canvas cannot match it", u(8), u(12)));
                return;
            }
        }
        self.begin_full();
        if !self.apply(&data[44..]) {
            self.notice("The painting file is damaged");
            return;
        }
        let s = &mut self.film.settings;
        s.tilt_deg = f(16);
        s.tilt_dir_deg = f(20);
        s.cure_rate = f(24);
        self.film.h_max = f(28);
        self.film.s_max = f(32);
        self.film.total_deposited = f64::from_le_bytes(data[36..44].try_into().unwrap());
        let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        self.history.name = if name == AUTOSAVE_NAME { String::new() } else { name.clone() };
        self.history.saved_gen = self.history.gen;
        self.painter = None;
        self.notice(&format!("Opened {}", name));
    }

    pub(crate) fn autosave_tick(&mut self, force: bool) {
        if self.auto.is_some() || self.history.gen == self.history.autosaved_gen {
            return;
        }
        if force || self.history.last_autosave.elapsed().as_secs_f32() > AUTOSAVE_SECS {
            let keep = self.history.saved_gen;
            if self.save_painting(AUTOSAVE_NAME) {
                self.history.saved_gen = keep;
            }
            self.history.autosaved_gen = self.history.gen;
            self.history.last_autosave = Instant::now();
        }
    }
}
