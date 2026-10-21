struct RP {
    csize: vec2<u32>,
    fsize: vec2<u32>,
    dt: f32,
    wlo: f32,
    whi: f32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> R: RP;
@group(0) @binding(1) var<storage, read> coarse: array<f32>;
@group(0) @binding(2) var<storage, read_write> fa: array<f32>;
@group(0) @binding(3) var<storage, read_write> fb: array<f32>;
@group(0) @binding(4) var<storage, read_write> wgt: array<f32>;

fn cat(i: i32, j: i32) -> f32 {
    let ii = u32(clamp(i, 0, i32(R.csize.x) - 1));
    let jj = u32(clamp(j, 0, i32(R.csize.y) - 1));
    return coarse[jj * R.csize.x + ii];
}

fn fidx(i: i32, j: i32) -> u32 {
    let ii = u32(clamp(i, 0, i32(R.fsize.x) - 1));
    let jj = u32(clamp(j, 0, i32(R.fsize.y) - 1));
    return jj * R.fsize.x + ii;
}

fn bsw(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    let u = 1.0 - t;
    return vec4<f32>(u * u * u, 3.0 * t3 - 6.0 * t2 + 4.0, -3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0, t3) / 6.0;
}

@compute @workgroup_size(16, 16)
fn upsample(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= R.fsize.x || id.y >= R.fsize.y { return; }
    let scale = f32(R.csize.x) / f32(R.fsize.x);
    let g = (vec2<f32>(id.xy) + vec2<f32>(0.5)) * scale - vec2<f32>(0.5);
    let f = floor(g);
    let t = g - f;
    let i = i32(f.x);
    let j = i32(f.y);
    let wx = bsw(t.x);
    let wy = bsw(t.y);
    var s = 0.0;
    for (var b = -1; b <= 2; b++) {
        var row = 0.0;
        for (var a = -1; a <= 2; a++) {
            row += wx[a + 1] * cat(i + a, j + b);
        }
        s += wy[b + 1] * row;
    }
    fa[id.y * R.fsize.x + id.x] = max(s, 0.0);
}

@compute @workgroup_size(16, 16)
fn weight(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= R.fsize.x || id.y >= R.fsize.y { return; }
    let i = i32(id.x);
    let j = i32(id.y);
    var m = 0.0;
    for (var b = -3; b <= 3; b++) {
        for (var a = -3; a <= 3; a++) {
            m = max(m, fa[fidx(i + a, j + b)]);
        }
    }
    wgt[id.y * R.fsize.x + id.x] = smoothstep(R.wlo, R.whi, m);
}

fn flow(t: f32, l: f32, r: f32, d: f32, u: f32, ld: f32, rd: f32, lu: f32, ru: f32) -> f32 {
    let gx = 0.5 * (r - l);
    let gy = 0.5 * (u - d);
    let gxx = r - 2.0 * t + l;
    let gyy = u - 2.0 * t + d;
    let gxy = 0.25 * (ru - rd - lu + ld);
    let g2 = gx * gx + gy * gy;
    return (gxx * gy * gy - 2.0 * gxy * gx * gy + gyy * gx * gx) / (g2 + R.eps * R.eps);
}

@compute @workgroup_size(16, 16)
fn curv_ab(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= R.fsize.x || id.y >= R.fsize.y { return; }
    let i = i32(id.x);
    let j = i32(id.y);
    let c = id.y * R.fsize.x + id.x;
    let w = wgt[c];
    let t = fa[c];
    if w <= 0.0 {
        fb[c] = t;
        return;
    }
    let k = flow(t, fa[fidx(i - 1, j)], fa[fidx(i + 1, j)], fa[fidx(i, j - 1)], fa[fidx(i, j + 1)], fa[fidx(i - 1, j - 1)], fa[fidx(i + 1, j - 1)], fa[fidx(i - 1, j + 1)], fa[fidx(i + 1, j + 1)]);
    fb[c] = max(t + R.dt * w * k, 0.0);
}

@compute @workgroup_size(16, 16)
fn curv_ba(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= R.fsize.x || id.y >= R.fsize.y { return; }
    let i = i32(id.x);
    let j = i32(id.y);
    let c = id.y * R.fsize.x + id.x;
    let w = wgt[c];
    let t = fb[c];
    if w <= 0.0 {
        fa[c] = t;
        return;
    }
    let k = flow(t, fb[fidx(i - 1, j)], fb[fidx(i + 1, j)], fb[fidx(i, j - 1)], fb[fidx(i, j + 1)], fb[fidx(i - 1, j - 1)], fb[fidx(i + 1, j - 1)], fb[fidx(i - 1, j + 1)], fb[fidx(i + 1, j + 1)]);
    fa[c] = max(t + R.dt * w * k, 0.0);
}
