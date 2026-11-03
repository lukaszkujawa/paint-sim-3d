
pub fn substrate_refl(n: usize, raw: bool, colour: Option<[f32; 16]>) -> Vec<u32> {
    let bands: Vec<f32> = (0..16)
        .map(|b| {
            let t = b as f32 / 15.0;
            match colour {
                Some(c) => c[b],
                None if raw => 0.50 + 0.16 * t,
                None => 0.84 + 0.04 * t,
            }
        })
        .collect();
    let mut packed = [0u32; 8];
    for k in 0..8 {
        let a = half::f16::from_f32(bands[2 * k]).to_bits() as u32;
        let b = half::f16::from_f32(bands[2 * k + 1]).to_bits() as u32;
        packed[k] = a | (b << 16);
    }
    let mut v = Vec::with_capacity(n * 8);
    for _ in 0..n {
        v.extend_from_slice(&packed);
    }
    v
}

pub(super) fn first_refl(v: &[u32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    out.copy_from_slice(&v[..8]);
    out
}

pub fn weave(w: usize, h: usize, dx: f32, amp: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; w * h];
    if amp <= 0.0 {
        return v;
    }
    let period = 1.4e-3;
    let k = std::f32::consts::TAU / period;
    for j in 0..h {
        let y = (j as f32 + 0.5) * dx;
        for i in 0..w {
            let x = (i as f32 + 0.5) * dx;
            let mut hsh = (i as u32).wrapping_mul(73856093) ^ (j as u32).wrapping_mul(19349663);
            hsh ^= hsh >> 13;
            hsh = hsh.wrapping_mul(0x5bd1e995);
            hsh ^= hsh >> 15;
            let r = (hsh & 0xffff) as f32 / 65535.0 * 2.0 - 1.0;
            let s = (k * x).sin() * (k * y).sin();
            v[j * w + i] = amp * (1.0 + 0.6 * s + 0.4 * r);
        }
    }
    v
}
