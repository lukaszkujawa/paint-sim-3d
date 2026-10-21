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

fn surf_hit(p: vec2<f32>) -> f32 {
    if cam.jitter.z > 0.5 { return surf_smooth(p); }
    return surf_bilinear(p);
}

fn normal_at(p: vec2<f32>) -> vec3<f32> {
    let e = 0.7 * cam.canvas.z;
    if cam.jitter.z > 0.5 {
        let sx = surf_smooth(p + vec2<f32>(e, 0.0)) - surf_smooth(p - vec2<f32>(e, 0.0));
        let sy = surf_smooth(p + vec2<f32>(0.0, e)) - surf_smooth(p - vec2<f32>(0.0, e));
        return normalize(vec3<f32>(-sx / (2.0 * e), -sy / (2.0 * e), 1.0));
    }
    let sx = surf_cubic(p + vec2<f32>(e, 0.0)) - surf_cubic(p - vec2<f32>(e, 0.0));
    let sy = surf_cubic(p + vec2<f32>(0.0, e)) - surf_cubic(p - vec2<f32>(0.0, e));
    return normalize(vec3<f32>(-sx / (2.0 * e), -sy / (2.0 * e), 1.0));
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
    return mix(cbil, cedge, edge);
}

fn shadow_at(p: vec2<f32>, z0: f32, l: vec3<f32>) -> f32 {
    if l.z <= 0.05 { return 1.0; }
    let step = cam.canvas.z * 1.5;
    var sh = 1.0;
    for (var k = 1; k <= 20; k++) {
        let d = f32(k) * step;
        let q = p + l.xy / l.z * d;
        let zs = surf_march(q);
        let zr = z0 + d;
        let occ = (zs - zr) / step;
        sh = min(sh, 1.0 - clamp(occ * 2.0, 0.0, 1.0));
    }
    return sh;
}

fn ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32, f0: f32) -> f32 {
    let hvec = normalize(v + l);
    let a = max(rough * rough, 0.002);
    let a2 = a * a;
    let ndh = max(dot(n, hvec), 0.0);
    let d = a2 / (3.14159265 * pow(ndh * ndh * (a2 - 1.0) + 1.0, 2.0));
    let ndv = max(dot(n, v), 1e-3);
    let ndl = max(dot(n, l), 0.0);
    let k = a / 2.0;
    let g = (ndv / (ndv * (1.0 - k) + k)) * (ndl / (ndl * (1.0 - k) + k));
    let f = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, hvec), 0.0), 5.0);
    return d * g * f / max(4.0 * ndv * ndl, 1e-3);
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
    if found {
        for (var r = 0; r < 8; r++) {
            let tm = 0.5 * (ta + tb);
            let q = a + dir * tm;
            if q.z <= surf_hit(q.xy) { tb = tm; } else { ta = tm; }
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
    let n = normal_at(p);
    let slope = length(n.xy) / max(n.z, 1e-3);
    var pc = p;
    if slope > 0.4 {
        let up = normalize(-n.xy);
        pc = p + up * cam.canvas.z * (0.5 + clamp(slope, 0.0, 3.0) * 0.5);
        pc = clamp(pc, vec2<f32>(0.0), vec2<f32>(W, H));
    }
    let col = color_sharp(pc);
    let v = -dir;
    let l = normalize(cam.light.xyz);
    let z0 = surf_hit(p);
    let sh = shadow_at(p, z0, l);
    let ndl = max(dot(n, l), 0.0);
    let albedo = col.rgb;
    let rough = col.a;
    let skyc = cam.sky.rgb;
    let diffuse = albedo * (cam.light.w * ndl * sh + skyc * (0.5 + 0.5 * n.z));
    let f0 = 0.038;
    let spec = ggx(n, v, l, rough, f0) * cam.light.w * ndl * sh;
    let rv = reflect(-v, n);
    let envc = mix(cam.ground.rgb, skyc, clamp(rv.z * 0.5 + 0.5, 0.0, 1.0));
    let fres = f0 + (1.0 - f0) * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    let env = envc * fres * (1.0 - rough * 0.7);
    let c = diffuse + vec3<f32>(spec) + env;
    if cam.jitter.z > 0.5 { return vec4<f32>(c, 1.0); }
    return vec4<f32>(to_srgb(tonemap(c)), 1.0);
}
