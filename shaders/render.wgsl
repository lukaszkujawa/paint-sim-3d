struct Cam {
    inv_vp: mat4x4<f32>,
    eye: vec4<f32>,
    canvas: vec4<f32>,
    light: vec4<f32>,
    size: vec4<u32>,
    sky: vec4<f32>,
    ground: vec4<f32>,
    vp: mat4x4<f32>,
    jitter: vec4<f32>,
    ortho: vec4<f32>,
}

@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read> hbuf: array<f32>;
@group(0) @binding(4) var<storage, read> mbuf: array<f32>;
@group(0) @binding(5) var<storage, read> tile_flags: array<u32>;
@group(0) @binding(6) var detail_tex: texture_2d<f32>;

const TAU: f32 = 6.2831853;
const PI: f32 = 3.14159265;
const METHOD_BRUSH: f32 = 4.0;
const METHOD_KNIFE: f32 = 5.0;
const F0_PAINT: f32 = 0.04;
const BRISTLE_SPACING: f32 = 0.35e-3;
const BRISTLE_DEPTH: f32 = 25e-6;
const KNIFE_SPACING: f32 = 0.09e-3;
const KNIFE_DEPTH: f32 = 6e-6;
const GROOVE_FULL_DEPTH_AT: f32 = 0.3e-3;
const WRINKLE_WAVELENGTH: f32 = 0.8e-3;
const WRINKLE_HEIGHT: f32 = 8e-6;
const THIN_FILM_MATTE: f32 = 0.12e-3;
const SIGMA_PAINT: f32 = 0.035;
const KEY_HALF_ANGLE: f32 = 0.12;
const SOFTBOX_HALF_W: f32 = 0.45;
const SOFTBOX_HALF_H: f32 = 0.3;
const SOFTBOX_RADIANCE: f32 = 5.0;
const HORIZON_GLOW: vec3<f32> = vec3<f32>(0.62, 0.6, 0.56);
const AO_DIRS: i32 = 6;
const AO_STEPS: i32 = 4;

fn near_paint(p: vec2<f32>) -> bool {
    let tw = i32((cam.size.x + 15u) / 16u);
    let th = i32((cam.size.y + 15u) / 16u);
    let g = vec2<i32>(floor(p / (cam.canvas.z * 16.0)));
    for (var dj = -2; dj <= 2; dj++) {
        for (var di = -2; di <= 2; di++) {
            let x = g.x + di;
            let y = g.y + dj;
            if x >= 0 && y >= 0 && x < tw && y < th && (tile_flags[x + y * tw] & 1u) != 0u { return true; }
        }
    }
    return false;
}

fn normal_bilinear(p: vec2<f32>) -> vec3<f32> {
    let e = 0.7 * cam.canvas.z;
    let sx = surf_bilinear(p + vec2<f32>(e, 0.0)) - surf_bilinear(p - vec2<f32>(e, 0.0));
    let sy = surf_bilinear(p + vec2<f32>(0.0, e)) - surf_bilinear(p - vec2<f32>(0.0, e));
    return normalize(vec3<f32>(-sx / (2.0 * e), -sy / (2.0 * e), 1.0));
}

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var o: VOut;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    o.pos = vec4<f32>(x, y, 0.0, 1.0);
    o.ndc = vec2<f32>(x, y);
    return o;
}

fn surf_at(i: i32, j: i32) -> f32 {
    let w = i32(cam.size.x);
    let hh = i32(cam.size.y);
    let ii = u32(clamp(i, 0, w - 1));
    let jj = u32(clamp(j, 0, hh - 1));
    let c = jj * cam.size.x + ii;
    return hbuf[c] + mbuf[c];
}

fn surf_bilinear(p: vec2<f32>) -> f32 {
    if p.x < 0.0 || p.y < 0.0 || p.x > cam.canvas.x || p.y > cam.canvas.y { return 0.0; }
    let g = p / cam.canvas.z - vec2<f32>(0.5);
    let f = floor(g);
    let t = g - f;
    let i = i32(f.x);
    let j = i32(f.y);
    let a = surf_at(i, j);
    let b = surf_at(i + 1, j);
    let c = surf_at(i, j + 1);
    let d = surf_at(i + 1, j + 1);
    return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}

fn cr_w(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    let u = 1.0 - t;
    return vec4<f32>(u * u * u, 3.0 * t3 - 6.0 * t2 + 4.0, -3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0, t3) / 6.0;
}

fn surf_cubic(p: vec2<f32>) -> f32 {
    if p.x < 0.0 || p.y < 0.0 || p.x > cam.canvas.x || p.y > cam.canvas.y { return 0.0; }
    let g = p / cam.canvas.z - vec2<f32>(0.5);
    let f = floor(g);
    let t = g - f;
    let i = i32(f.x);
    let j = i32(f.y);
    let wx = cr_w(t.x);
    let wy = cr_w(t.y);
    var s = 0.0;
    for (var b = -1; b <= 2; b++) {
        var row = 0.0;
        for (var a = -1; a <= 2; a++) {
            row += wx[a + 1] * surf_at(i + a, j + b);
        }
        s += wy[b + 1] * row;
    }
    return max(s, 0.0);
}

fn wet_at(i: i32, j: i32) -> f32 {
    let w = i32(cam.size.x);
    let hh = i32(cam.size.y);
    let ii = u32(clamp(i, 0, w - 1));
    let jj = u32(clamp(j, 0, hh - 1));
    return hbuf[jj * cam.size.x + ii];
}

fn surf_wide(p: vec2<f32>) -> f32 {
    if p.x < 0.0 || p.y < 0.0 || p.x > cam.canvas.x || p.y > cam.canvas.y { return 0.0; }
    let g = p / cam.canvas.z - vec2<f32>(0.5);
    let f = floor(g);
    let t = g - f;
    let i = i32(f.x);
    let j = i32(f.y);
    let s2 = 2.0 * 1.0 * 1.0;
    var wx: array<f32, 6>;
    var wy: array<f32, 6>;
    var nx = 0.0;
    var ny = 0.0;
    for (var a = 0; a < 6; a++) {
        let da = f32(a - 2) - t.x;
        let db = f32(a - 2) - t.y;
        wx[a] = exp(-da * da / s2);
        wy[a] = exp(-db * db / s2);
        nx += wx[a];
        ny += wy[a];
    }
    var s = 0.0;
    for (var b = 0; b < 6; b++) {
        var row = 0.0;
        for (var a = 0; a < 6; a++) {
            row += wx[a] * surf_at(i + a - 2, j + b - 2);
        }
        s += wy[b] * row;
    }
    return s / (nx * ny);
}

fn wet_bilinear(p: vec2<f32>) -> f32 {
    if p.x < 0.0 || p.y < 0.0 || p.x > cam.canvas.x || p.y > cam.canvas.y { return 0.0; }
    let g = p / cam.canvas.z - vec2<f32>(0.5);
    let f = floor(g);
    let t = g - f;
    let i = i32(f.x);
    let j = i32(f.y);
    let a = wet_at(i, j);
    let b = wet_at(i + 1, j);
    let c = wet_at(i, j + 1);
    let d = wet_at(i + 1, j + 1);
    return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}

struct Detail {
    dir: vec2<f32>,
    brush: f32,
    knife: f32,
    cure: f32,
    yield_pa: f32,
    thickness: f32,
}

fn detail_texel(i: i32, j: i32) -> vec4<f32> {
    let w = i32(cam.size.x);
    let hh = i32(cam.size.y);
    var sc = cam.ortho.z;
    if sc <= 0.0 { sc = 1.0; }
    let ci = i32(floor((f32(clamp(i, 0, w - 1)) + 0.5) * sc));
    let cj = i32(floor((f32(clamp(j, 0, hh - 1)) + 0.5) * sc));
    return textureLoad(detail_tex, vec2<i32>(ci, cj), 0);
}

fn detail_params(p: vec2<f32>) -> Detail {
    let g = p / cam.canvas.z - vec2<f32>(0.5);
    let f = floor(g);
    let t = g - f;
    let i = i32(f.x);
    let j = i32(f.y);
    let wts = vec4<f32>((1.0 - t.x) * (1.0 - t.y), t.x * (1.0 - t.y), (1.0 - t.x) * t.y, t.x * t.y);
    var d: Detail;
    var dir = vec2<f32>(0.0);
    d.brush = 0.0;
    d.knife = 0.0;
    d.cure = 0.0;
    d.yield_pa = 0.0;
    d.thickness = 0.0;
    for (var k = 0; k < 4; k++) {
        let a = detail_texel(i + (k & 1), j + (k >> 1));
        let wk = wts[k];
        let method = floor(a.x);
        let ang = (fract(a.x) - 0.5) * TAU;
        dir += wk * vec2<f32>(cos(ang), sin(ang));
        d.brush += wk * select(0.0, 1.0, abs(method - METHOD_BRUSH) < 0.5);
        d.knife += wk * select(0.0, 1.0, abs(method - METHOD_KNIFE) < 0.5);
        d.cure += wk * a.y;
        d.yield_pa += wk * a.z;
        d.thickness += wk * a.w;
    }
    d.dir = select(normalize(dir), vec2<f32>(1.0, 0.0), length(dir) < 1e-4);
    return d;
}

fn groove_profile(across: f32, along: f32, spacing: f32) -> f32 {
    let k = TAU / spacing;
    let drift = 0.6 * sin(along * TAU / 0.021);
    let drift2 = 0.4 * sin(along * TAU / 0.013 + 1.7);
    return 0.5 * cos(across * k + drift) + 0.3 * cos(across * k * 1.62 + drift2 + 2.1) + 0.2 * cos(across * k * 0.57 + 0.5 * drift);
}

fn wrinkle_profile(p: vec2<f32>, wavelength: f32) -> f32 {
    let k = TAU / wavelength;
    let q1 = dot(p, vec2<f32>(0.93, 0.37)) * k * 0.13;
    let q2 = dot(p, vec2<f32>(-0.21, 0.98)) * k * 0.11;
    let q3 = dot(p, vec2<f32>(0.6, -0.8)) * k * 0.17;
    let ph1 = 2.4 * sin(q2) + 1.1 * sin(q3 + 1.0);
    let ph2 = 2.4 * sin(q3) + 1.1 * sin(q1 + 2.0);
    let ph3 = 2.4 * sin(q1) + 1.1 * sin(q2 + 3.0);
    let d0 = vec2<f32>(1.0, 0.0);
    let d1 = vec2<f32>(0.5, 0.8660254);
    let d2 = vec2<f32>(-0.5, 0.8660254);
    let s = cos(dot(p, d0) * k + ph1) + cos(dot(p, d1) * k + ph2) + cos(dot(p, d2) * k + ph3);
    let patches = 0.6 + 0.4 * sin(q1 * 0.7 + 1.0) * sin(q2 * 0.8);
    return s / 3.0 * patches;
}

fn groove_depth(nominal: f32, spacing: f32, yield_pa: f32, footprint: f32) -> f32 {
    let k = TAU / spacing;
    let held = yield_pa / (SIGMA_PAINT * k * k);
    let resolved = 1.0 - smoothstep(0.2, 0.5, footprint / spacing);
    return min(nominal, held) * resolved;
}

fn micro(p: vec2<f32>, d: Detail) -> f32 {
    if cam.jitter.w >= 0.5 * WRINKLE_WAVELENGTH { return 0.0; }
    let hp = d.thickness;
    if hp <= 2e-5 { return 0.0; }
    let across = dot(p, vec2<f32>(-d.dir.y, d.dir.x));
    let along = dot(p, d.dir);
    let carry = min(hp / GROOVE_FULL_DEPTH_AT, 1.0);
    let footprint = cam.jitter.w;
    var z = 0.0;
    if d.brush > 0.0 {
        z += d.brush * groove_depth(BRISTLE_DEPTH, BRISTLE_SPACING, d.yield_pa, footprint) * carry * groove_profile(across, along, BRISTLE_SPACING);
    }
    if d.knife > 0.0 {
        z += d.knife * groove_depth(KNIFE_DEPTH, KNIFE_SPACING, d.yield_pa, footprint) * carry * groove_profile(across, along, KNIFE_SPACING);
    }
    let skin = d.cure * smoothstep(0.5e-3, 1.5e-3, hp) * (1.0 - smoothstep(0.2, 0.5, footprint / WRINKLE_WAVELENGTH));
    if skin > 0.0 {
        z += WRINKLE_HEIGHT * skin * wrinkle_profile(p, WRINKLE_WAVELENGTH);
    }
    return z;
}

fn micro_slope(p: vec2<f32>, d: Detail) -> vec2<f32> {
    let e = 0.06 * cam.canvas.z;
    let mx = micro(p + vec2<f32>(e, 0.0), d) - micro(p - vec2<f32>(e, 0.0), d);
    let my = micro(p + vec2<f32>(0.0, e), d) - micro(p - vec2<f32>(0.0, e), d);
    return vec2<f32>(mx, my) / (2.0 * e);
}

fn edge_factor(p: vec2<f32>) -> f32 {
    let g = p / cam.canvas.z - vec2<f32>(0.5);
    let f = floor(g);
    let i = i32(f.x);
    let j = i32(f.y);
    var hmax = 0.0;
    var hmin = 1e9;
    for (var b = -1; b <= 2; b++) {
        for (var a = -1; a <= 2; a++) {
            let v = wet_at(i + a, j + b);
            hmax = max(hmax, v);
            hmin = min(hmin, v);
        }
    }
    if hmax <= 1e-6 { return 0.0; }
    return smoothstep(0.3, 0.8, 1.0 - hmin / hmax);
}

fn surf_smooth(p: vec2<f32>) -> f32 {
    if p.x < 0.0 || p.y < 0.0 || p.x > cam.canvas.x || p.y > cam.canvas.y { return 0.0; }
    let dx = cam.canvas.z;
    let g = p / dx - vec2<f32>(0.5);
    let f = floor(g);
    let t = g - f;
    let i = i32(f.x);
    let j = i32(f.y);
    let sig2 = 1.0;
    var wx: array<f32, 6>;
    var wy: array<f32, 6>;
    var dwx: array<f32, 6>;
    var dwy: array<f32, 6>;
    var nx = 0.0;
    var ny = 0.0;
    var dnx = 0.0;
    var dny = 0.0;
    for (var a = 0; a < 6; a++) {
        let da = f32(a - 2) - t.x;
        let db = f32(a - 2) - t.y;
        wx[a] = exp(-da * da / (2.0 * sig2));
        wy[a] = exp(-db * db / (2.0 * sig2));
        dwx[a] = wx[a] * da / sig2;
        dwy[a] = wy[a] * db / sig2;
        nx += wx[a];
        ny += wy[a];
        dnx += dwx[a];
        dny += dwy[a];
    }
    var num = 0.0;
    var num2 = 0.0;
    var dnumx = 0.0;
    var dnumy = 0.0;
    for (var b = 0; b < 6; b++) {
        var row = 0.0;
        var row2 = 0.0;
        var drow = 0.0;
        for (var a = 0; a < 6; a++) {
            let sv = surf_at(i + a - 2, j + b - 2);
            row += wx[a] * sv;
            row2 += wx[a] * sv * sv;
            drow += dwx[a] * sv;
        }
        num += wy[b] * row;
        num2 += wy[b] * row2;
        dnumx += wy[b] * drow;
        dnumy += dwy[b] * row;
    }
    let nn = nx * ny;
    let hb = num / nn;
    if hb <= 1e-7 { return max(hb, 0.0); }
    let hin = max(num2 / max(num, 1e-12), hb);
    let gx = (dnumx * nx - num * dnx) / (nx * nn);
    let gy = (dnumy * ny - num * dny) / (ny * nn);
    let gm = max(length(vec2<f32>(gx, gy)) / dx, 1e-3);
    let d = (hb - 0.35 * hin) / gm;
    if d <= 0.0 { return 0.0; }
    let lc = 2.0e-3;
    let scale = min(hin, 2.0 * lc);
    let cap = hin * tanh(d * cam.ortho.y / scale);
    return min(hb, cap);
}

fn surf_march(p: vec2<f32>) -> f32 {
    if cam.jitter.z > 0.5 { return surf_cubic(p); }
    return surf_bilinear(p);
}

fn surf_hit(p: vec2<f32>, d: Detail) -> f32 {
    if cam.jitter.z > 0.5 { return surf_smooth(p) + micro(p, d); }
    return surf_bilinear(p) + micro(p, d);
}

fn base_slope(p: vec2<f32>) -> vec2<f32> {
    let e = 0.7 * cam.canvas.z;
    if cam.jitter.z > 0.5 {
        let sx = surf_smooth(p + vec2<f32>(e, 0.0)) - surf_smooth(p - vec2<f32>(e, 0.0));
        let sy = surf_smooth(p + vec2<f32>(0.0, e)) - surf_smooth(p - vec2<f32>(0.0, e));
        return vec2<f32>(sx, sy) / (2.0 * e);
    }
    let sx = surf_cubic(p + vec2<f32>(e, 0.0)) - surf_cubic(p - vec2<f32>(e, 0.0));
    let sy = surf_cubic(p + vec2<f32>(0.0, e)) - surf_cubic(p - vec2<f32>(0.0, e));
    return vec2<f32>(sx, sy) / (2.0 * e);
}

fn normal_from_slope(s: vec2<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(-s, 1.0));
}

fn col_at(i: i32, j: i32) -> vec4<f32> {
    let w = i32(cam.size.x);
    let hh = i32(cam.size.y);
    var sc = cam.ortho.z;
    if sc <= 0.0 { sc = 1.0; }
    let ci = i32(floor((f32(clamp(i, 0, w - 1)) + 0.5) * sc));
    let cj = i32(floor((f32(clamp(j, 0, hh - 1)) + 0.5) * sc));
    return textureLoad(color_tex, vec2<i32>(ci, cj), 0);
}

fn color_sharp(p: vec2<f32>) -> vec4<f32> {
    let g = p / cam.canvas.z - vec2<f32>(0.5);
    let f = floor(g);
    let t = g - f;
    let i = i32(f.x);
    let j = i32(f.y);
    let wb = vec4<f32>((1.0 - t.x) * (1.0 - t.y), t.x * (1.0 - t.y), (1.0 - t.x) * t.y, t.x * t.y);
    let hw = vec4<f32>(wet_at(i, j), wet_at(i + 1, j), wet_at(i, j + 1), wet_at(i + 1, j + 1));
    let c0 = col_at(i, j);
    let c1 = col_at(i + 1, j);
    let c2 = col_at(i, j + 1);
    let c3 = col_at(i + 1, j + 1);
    let cbil = wb.x * c0 + wb.y * c1 + wb.z * c2 + wb.w * c3;
    let hmax = max(max(hw.x, hw.y), max(hw.z, hw.w));
    let hmin = min(min(hw.x, hw.y), min(hw.z, hw.w));
    if hmax <= 1e-6 { return cbil; }
    let e = 1.0 - hmin / hmax;
    let edge = smoothstep(0.3, 0.6, e);
    if edge <= 0.0 { return cbil; }
    let wx = cr_w(t.x);
    let wy = cr_w(t.y);
    var hb = 0.0;
    var hb2 = 0.0;
    for (var b = -1; b <= 2; b++) {
        for (var a = -1; a <= 2; a++) {
            let wv = wx[a + 1] * wy[b + 1];
            let hv = wet_at(i + a, j + b);
            hb += wv * hv;
            hb2 += wv * hv * hv;
        }
    }
    hb = max(hb, 0.0);
    let hin = hb2 / max(hb, 1e-9);
    let win = wb * hw;
    let wout = wb * (vec4<f32>(hmax) - hw);
    let sin = max(win.x + win.y + win.z + win.w, 1e-12);
    let sout = max(wout.x + wout.y + wout.z + wout.w, 1e-12);
    let cin = (win.x * c0 + win.y * c1 + win.z * c2 + win.w * c3) / sin;
    let cout = (wout.x * c0 + wout.y * c1 + wout.z * c2 + wout.w * c3) / sout;
    let s = smoothstep(0.25, 0.45, hb / max(hin, 1e-9));
    let cedge = mix(cout, cin, s);
    return vec4<f32>(mix(cbil, cedge, edge).rgb, cbil.a);
}

fn shadow_at(p: vec2<f32>, z0: f32, l: vec3<f32>) -> f32 {
    if l.z <= 0.05 { return 1.0; }
    let step = cam.canvas.z * 1.5;
    let spread = tan(KEY_HALF_ANGLE);
    var sh = 1.0;
    for (var k = 1; k <= 20; k++) {
        let d = f32(k) * step;
        let q = p + l.xy / l.z * d;
        let zs = surf_march(q);
        let zr = z0 + d;
        if zr > cam.canvas.w { break; }
        let penumbra = d * spread / l.z + 0.5 * step;
        sh = min(sh, 1.0 - clamp((zs - zr) / penumbra, 0.0, 1.0));
    }
    return sh;
}

fn ambient_occlusion(p: vec2<f32>, z0: f32, d: Detail) -> f32 {
    let step = cam.canvas.z * 2.0;
    var open = 0.0;
    for (var k = 0; k < AO_DIRS; k++) {
        let ang = (f32(k) + 0.5) * TAU / f32(AO_DIRS);
        let dir = vec2<f32>(cos(ang), sin(ang));
        var horizon = 0.0;
        for (var i = 1; i <= AO_STEPS; i++) {
            let dist = f32(i) * step;
            let rise = surf_hit(p + dir * dist, d) - z0;
            horizon = max(horizon, rise / dist);
        }
        open += 1.0 - horizon / sqrt(1.0 + horizon * horizon);
    }
    return open / f32(AO_DIRS);
}

fn env_radiance(dir: vec3<f32>, rough: f32) -> vec3<f32> {
    let sky = cam.sky.rgb;
    let floor = cam.ground.rgb;
    let soft = 0.08 + 0.9 * rough;
    let up = smoothstep(-soft, soft, dir.z);
    let band = exp(-abs(dir.z) / (0.12 + 0.6 * rough));
    var e = mix(floor, sky, up) + HORIZON_GLOW * 0.5 * band;
    let l = normalize(cam.light.xyz);
    let side = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), l));
    let upv = cross(l, side);
    let cos_l = dot(dir, l);
    if cos_l > 0.0 {
        let u = atan2(dot(dir, side), cos_l);
        let v = atan2(dot(dir, upv), cos_l);
        let edge = 0.03 + 0.7 * rough;
        let inside = (1.0 - smoothstep(SOFTBOX_HALF_W - edge, SOFTBOX_HALF_W + edge, abs(u))) * (1.0 - smoothstep(SOFTBOX_HALF_H - edge, SOFTBOX_HALF_H + edge, abs(v)));
        let fill = (SOFTBOX_HALF_W * SOFTBOX_HALF_H) / ((SOFTBOX_HALF_W + edge) * (SOFTBOX_HALF_H + edge));
        e += vec3<f32>(SOFTBOX_RADIANCE * cam.light.w / 2.6) * inside * fill;
    }
    return e;
}

fn d_ggx(ndh: f32, a2: f32) -> f32 {
    let d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn v_smith(ndv: f32, ndl: f32, a2: f32) -> f32 {
    let gv = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2);
    let gl = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

fn spec_lobe(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32, f0: f32) -> f32 {
    let hvec = normalize(v + l);
    let a = max(rough * rough, 0.002);
    let a_wide = min(a + sin(KEY_HALF_ANGLE), 1.0);
    let a2 = a_wide * a_wide;
    let energy = (a / a_wide) * (a / a_wide);
    let ndh = max(dot(n, hvec), 0.0);
    let ndv = max(dot(n, v), 1e-3);
    let ndl = max(dot(n, l), 0.0);
    let f = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, hvec), 0.0), 5.0);
    return d_ggx(ndh, a2) * energy * v_smith(ndv, ndl, a_wide * a_wide) * f;
}

fn env_dfg(f0: f32, rough: f32, ndv: f32) -> f32 {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
    let scale = a004 * -1.04 + r.z;
    let bias = a004 * 1.04 + r.w;
    return f0 * scale + bias;
}

fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let cc = 2.43; let d = 0.59; let e = 0.14;
    return clamp((c * (a * c + b)) / (c * (cc * c + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let ndc = in.ndc + cam.jitter.xy;
    let p0 = cam.inv_vp * vec4<f32>(ndc, 0.0, 1.0);
    let p1 = cam.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
    var a = p0.xyz / p0.w;
    var dir = normalize(p1.xyz / p1.w - a);
    if cam.ortho.x > 0.5 {
        let pxy = (ndc * 0.5 + vec2<f32>(0.5)) * vec2<f32>(cam.canvas.x, cam.canvas.y);
        a = vec3<f32>(pxy, 1.0);
        dir = vec3<f32>(0.0, 0.0, -1.0);
    }
    let bg = mix(cam.ground.rgb, cam.sky.rgb, clamp(dir.z * 0.5 + 0.5, 0.0, 1.0));
    if abs(dir.z) < 1e-5 { return vec4<f32>(to_srgb(tonemap(bg)), 1.0); }
    let t1 = -a.z / dir.z;
    if t1 < 0.0 { return vec4<f32>(to_srgb(tonemap(bg)), 1.0); }
    let top = cam.canvas.w;
    var t0 = 0.0;
    if a.z > top { t0 = (top - a.z) / dir.z; }
    let span = max(t1 - t0, 0.0);
    let horiz = span * length(dir.xy);
    let nsteps = clamp(i32(ceil(horiz / cam.canvas.z)), 1, 96);
    let step = span / f32(nsteps);
    var ta = t0;
    var tb = t1;
    var found = false;
    for (var k = 1; k <= nsteps; k++) {
        let tk = t0 + f32(k) * step;
        let q = a + dir * tk;
        if q.z <= surf_march(q.xy) {
            tb = tk;
            found = true;
            break;
        }
        ta = tk;
    }
    var t = t1;
    let detail = detail_params((a + dir * tb).xy);
    if found {
        for (var r = 0; r < 8; r++) {
            let tm = 0.5 * (ta + tb);
            let q = a + dir * tm;
            if q.z <= surf_hit(q.xy, detail) { tb = tm; } else { ta = tm; }
        }
        t = tb;
    }
    let hit = a + dir * t;
    let p = hit.xy;
    let W = cam.canvas.x;
    let H = cam.canvas.y;
    if p.x < 0.0 || p.y < 0.0 || p.x > W || p.y > H {
        let fl = cam.ground.rgb * 0.6;
        return vec4<f32>(to_srgb(tonemap(fl)), 1.0);
    }
    let v = -dir;
    let l = normalize(cam.light.xyz);
    if !near_paint(p) && cam.jitter.z <= 0.5 {
        let cs = color_sharp(p);
        return vec4<f32>(to_srgb(tonemap(lit(normal_bilinear(p), v, l, 1.0, 1.0, cs.rgb, cs.a, 0.0))), 1.0);
    }
    let ms = micro_slope(p, detail);
    let n = normal_from_slope(base_slope(p) + ms);
    let slope = length(n.xy) / max(n.z, 1e-3);
    var pc = p;
    if slope > 0.4 {
        let up = normalize(-n.xy);
        pc = p + up * cam.canvas.z * (0.5 + clamp(slope, 0.0, 3.0) * 0.5);
        pc = clamp(pc, vec2<f32>(0.0), vec2<f32>(W, H));
    }
    let col = color_sharp(pc);
    let thin = 0.3 * (1.0 - smoothstep(0.0, THIN_FILM_MATTE, detail.thickness)) * smoothstep(0.0, 1e-5, detail.thickness);
    let rough = clamp(col.a + 0.3 * length(ms) + thin, 0.03, 0.95);
    let z0 = surf_hit(p, detail);
    let sh = shadow_at(p, z0, l);
    let ao = ambient_occlusion(p, z0, detail);
    let c = lit(n, v, l, sh, ao, col.rgb, rough, detail.cure);
    if cam.jitter.z > 0.5 { return vec4<f32>(c, 1.0); }
    return vec4<f32>(to_srgb(tonemap(c)), 1.0);
}

fn lit(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, sh: f32, ao: f32, albedo: vec3<f32>, rough: f32, cure: f32) -> vec3<f32> {
    let ndl = max(dot(n, l), 0.0);
    let ndv = max(dot(n, v), 1e-3);
    let sun = cam.light.w;
    let ambient = env_radiance(n, 1.0) * ao;
    let diffuse = albedo * (sun * ndl * sh + ambient);
    var spec = spec_lobe(n, v, l, rough, F0_PAINT);
    spec += 0.35 * cure * spec_lobe(n, v, l, max(rough, 0.55), F0_PAINT);
    let direct = spec * sun * ndl * sh;
    let rv = reflect(-v, n);
    let occlusion = clamp(pow(ndv + ao, exp2(-16.0 * rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
    let env = env_radiance(rv, rough) * env_dfg(F0_PAINT, rough, ndv) * occlusion;
    return diffuse + vec3<f32>(direct) + env;
}
