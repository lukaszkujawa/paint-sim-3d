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
    basis_r: vec4<f32>,
    basis_u: vec4<f32>,
    lens: vec4<f32>,
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
const METHOD_ROLLER: f32 = 6.0;
const ROLLER_STIPPLE_DEPTH: f32 = 14e-6;
const ROLLER_STIPPLE_SCALE: f32 = 0.7e-3;
const ROLLER_STIPPLE_STRETCH: f32 = 1.6;
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
const FILL_RADIANCE: f32 = 0.35;
const FILL_TINT: vec3<f32> = vec3<f32>(0.86, 0.92, 1.0);
const COAT_ROUGH: f32 = 0.07;
const GRAIN_SCALE: f32 = 0.6e-3;
const WEAVE_PERIOD: f32 = 1.4e-3;
const LAMP_DIST: f32 = 1.3;
const HWARP_FINE: f32 = 0.06e-3;
const HWARP_COARSE: f32 = 0.12e-3;
const CWARP_ISO: f32 = 0.22e-3;
const CWARP_STREAK: f32 = 0.45e-3;
const STREAK_WIDTH: f32 = 0.11e-3;
const STREAK_LENGTH: f32 = 2.8e-3;
const TELEGRAPH_DEPTH: f32 = 0.09e-3;
const IMPASTO_FRACTION: f32 = 0.05;
const IMPASTO_MAX: f32 = 25e-6;
const ZENITH_DIM: f32 = 0.2;
const PINHOLE_CELL: f32 = 0.7e-3;
const PINHOLE_RATE: f32 = 0.05;
const BEAUTY_SOFTBOX_SCALE: f32 = 1.5;
const BEAUTY_SOFTBOX_RADIANCE: f32 = 1.6;
const PAINT_HUE_DRIFT: f32 = 0.035;
const PAINT_TONE_DRIFT: f32 = 0.06;
const BEAUTY_MIN_ALPHA: f32 = 0.035;
const FIREFLY_CLAMP: f32 = 40.0;
const BEAUTY_PENUMBRA: f32 = 0.03;
const SHADOW_BIAS: f32 = 2e-6;
const CLUMP_WIDTH: f32 = 1.1e-3;
const CLUMP_ELONGATION: f32 = 14.0;
const CLUMP_DEPTH: f32 = 14e-6;
const DRY_GLOSS_FLOOR: f32 = 0.06;
const DRY_GLOSS_SCALE: f32 = 0.5;
const COLOR_CRISP: f32 = 2.2;
const MARBLE_R1: f32 = 1.2e-3;
const MARBLE_R2: f32 = 2.6e-3;
const MARBLE_MIN_CONTRAST: f32 = 0.03;
const MARBLE_SCALE: f32 = 1.6e-3;
const MARBLE_SOFT: f32 = 0.06;
const MARBLE_STRENGTH: f32 = 0.85;
const MARBLE_SELECT: f32 = 6.0;
const EDGE_MIN_STEP: f32 = 15e-6;
const EDGE_ISO: f32 = 0.3;
const EDGE_ISO_SOFT: f32 = 0.08;

fn weave(p: vec2<f32>) -> vec3<f32> {
    let q = p / WEAVE_PERIOD;
    let wx = 1.0 + 0.35 * vnoise(vec2<f32>(floor(q.x) * 0.37, q.y * 0.11));
    let wy = 1.0 + 0.35 * vnoise(vec2<f32>(q.x * 0.11, floor(q.y) * 0.37) + vec2<f32>(5.0, 3.0));
    let ox = 0.25 * vnoise(vec2<f32>(floor(q.y) * 0.5, 1.0));
    let oy = 0.25 * vnoise(vec2<f32>(floor(q.x) * 0.5, 7.0));
    let fx = fract(q.x + ox + 0.08 * vnoise(p / 6.0e-3));
    let fy = fract(q.y + oy + 0.08 * vnoise(p / 6.0e-3 + vec2<f32>(2.0, 4.0)));
    let warp = wx * (1.0 - pow(abs(fx - 0.5) * 2.0, 1.6));
    let weft = wy * (1.0 - pow(abs(fy - 0.5) * 2.0, 1.6));
    let over = step(0.5, fract(0.5 * floor(q.x) + 0.5 * floor(q.y)));
    let top = mix(warp, weft, over);
    let fibres = vnoise(p / 0.22e-3 + vec2<f32>(fx, fy)) * vnoise(p / 0.7e-3 + vec2<f32>(4.0, 2.0));
    return vec3<f32>(top, max(warp, weft), fibres);
}

fn canvas_albedo(p: vec2<f32>, base: vec3<f32>) -> vec3<f32> {
    let w = weave(p);
    let thread = 0.16 * (w.y - 0.6) + 0.05 * w.z;
    let slub = 0.08 * max(vnoise(p / 2.3e-3 + vec2<f32>(5.0, 1.0)) - 0.5, 0.0) / 0.5;
    let drift = 0.05 * vnoise(p / 0.08) + 0.03 * vnoise(p / 0.025 + vec2<f32>(11.0, 4.0));
    let tint = vec3<f32>(1.0 + 0.02 * vnoise(p / 0.05 + vec2<f32>(2.0, 9.0)), 1.0, 1.0 - 0.025 * vnoise(p / 0.06 + vec2<f32>(8.0, 3.0)));
    return base * (1.0 + thread + slub + drift) * tint;
}

fn canvas_bump(p: vec2<f32>) -> f32 {
    let w = weave(p);
    return 14e-6 * (w.x - 0.5) + 4e-6 * w.z;
}

fn canvas_bump_levelled(p: vec2<f32>, hp: f32) -> f32 {
    let q = p / WEAVE_PERIOD;
    let ox = 0.25 * vnoise(vec2<f32>(floor(q.y) * 0.5, 1.0));
    let oy = 0.25 * vnoise(vec2<f32>(floor(q.x) * 0.5, 7.0));
    let fx = q.x + ox + 0.08 * vnoise(p / 6.0e-3);
    let fy = q.y + oy + 0.08 * vnoise(p / 6.0e-3 + vec2<f32>(2.0, 4.0));
    let warp = 0.5 - 0.5 * cos(TAU * fx);
    let weft = 0.5 - 0.5 * cos(TAU * fy);
    let over = step(0.5, fract(0.5 * floor(q.x) + 0.5 * floor(q.y)));
    let sharp = exp(-hp / (0.4 * TELEGRAPH_DEPTH));
    let top = mix(warp, weft, over);
    return mix(14e-6 * (warp * weft - 0.25), canvas_bump(p), sharp) * exp(-hp / TELEGRAPH_DEPTH) + 3e-6 * (top - 0.5) * (1.0 - sharp) * exp(-hp / TELEGRAPH_DEPTH);
}

fn pinholes(p: vec2<f32>, hp: f32) -> f32 {
    let g = floor(p / PINHOLE_CELL);
    var z = 0.0;
    for (var b = -1; b <= 1; b++) {
        for (var a = -1; a <= 1; a++) {
            let c = g + vec2<f32>(f32(a), f32(b));
            let r = hash22(c * 0.731 + vec2<f32>(0.17, 0.61));
            let r2 = hash21(c * 0.413 + vec2<f32>(0.93, 0.29));
            if r2 > PINHOLE_RATE { continue; }
            let centre = (c + vec2<f32>(0.15) + 0.7 * r) * PINHOLE_CELL;
            let radius = mix(0.025e-3, 0.09e-3, r.x * r.y) * min(hp / 0.4e-3, 1.0);
            let d = length(p - centre) / max(radius, 1e-6);
            let crater = -(1.0 - smoothstep(0.0, 1.0, d)) + 0.35 * exp(-(d - 1.15) * (d - 1.15) * 8.0);
            z += crater * radius * 0.45;
        }
    }
    return z;
}

fn specks(p: vec2<f32>) -> vec2<f32> {
    let cell = 0.35e-3;
    let g = floor(p / cell);
    let r = hash21(g * 0.173 + vec2<f32>(0.31, 0.77));
    let r2 = hash21(g * 0.291 + vec2<f32>(0.59, 0.13));
    let centre = (g + vec2<f32>(0.2 + 0.6 * r, 0.2 + 0.6 * r2)) * cell;
    let d = length(p - centre) / (cell * (0.08 + 0.12 * r2));
    let hit = 1.0 - smoothstep(0.6, 1.0, d);
    let dust = select(0.0, hit, r > 0.985);
    let dark = select(0.0, hit, r < 0.012);
    return vec2<f32>(dust, dark);
}

fn lamp_falloff(p: vec2<f32>) -> f32 {
    let l = normalize(cam.light.xyz);
    let centre = vec3<f32>(0.5 * cam.canvas.x, 0.5 * cam.canvas.y, 0.0);
    let lamp = centre + l * LAMP_DIST;
    let d = lamp - vec3<f32>(p, 0.0);
    let dl = length(d);
    let cosine = max(dot(d / dl, l), 0.0);
    return (LAMP_DIST * LAMP_DIST) / (dl * dl) * pow(cosine, 1.5);
}

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(0.1031, 0.1030));
    q += dot(q, q.yx + vec2<f32>(33.33));
    return fract((q.x + q.y) * q.x);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var q = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yzx + vec3<f32>(33.33));
    return fract((q.xx + q.yz) * q.zy);
}

fn gnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let ga = hash22(i) * 2.0 - vec2<f32>(1.0);
    let gb = hash22(i + vec2<f32>(1.0, 0.0)) * 2.0 - vec2<f32>(1.0);
    let gc = hash22(i + vec2<f32>(0.0, 1.0)) * 2.0 - vec2<f32>(1.0);
    let gd = hash22(i + vec2<f32>(1.0, 1.0)) * 2.0 - vec2<f32>(1.0);
    let a = dot(ga, f);
    let b = dot(gb, f - vec2<f32>(1.0, 0.0));
    let c = dot(gc, f - vec2<f32>(0.0, 1.0));
    let d = dot(gd, f - vec2<f32>(1.0, 1.0));
    return 1.4 * mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>, octaves: i32) -> f32 {
    var q = p;
    var amp = 0.5;
    var sum = 0.0;
    var norm = 0.0;
    for (var k = 0; k < octaves; k++) {
        sum += amp * gnoise(q);
        norm += amp;
        q = vec2<f32>(1.6 * q.x - 1.2 * q.y, 1.2 * q.x + 1.6 * q.y) + vec2<f32>(3.1, 1.7);
        amp *= 0.5;
    }
    return sum / norm;
}

fn fbm_filtered(p: vec2<f32>, scale: f32, octaves: i32) -> f32 {
    var q = p / scale;
    var amp = 0.5;
    var sum = 0.0;
    var norm = 0.0;
    var size = scale;
    for (var k = 0; k < octaves; k++) {
        let keep = 1.0 - smoothstep(0.5, 1.5, cam.jitter.w / size);
        sum += amp * gnoise(q) * keep;
        norm += amp;
        q = vec2<f32>(1.6 * q.x - 1.2 * q.y, 1.2 * q.x + 1.6 * q.y) + vec2<f32>(3.1, 1.7);
        amp *= 0.5;
        size *= 0.5;
    }
    return sum / norm;
}

fn beauty_mode() -> bool {
    return cam.jitter.z > 0.5;
}

fn hwarp(p: vec2<f32>) -> vec2<f32> {
    if !beauty_mode() { return p; }
    let a = vec2<f32>(fbm(p / 0.7e-3, 3), fbm(p / 0.7e-3 + vec2<f32>(17.1, 3.7), 3));
    let b = vec2<f32>(gnoise(p / 3.5e-3 + vec2<f32>(5.2, 1.3)), gnoise(p / 3.5e-3 + vec2<f32>(9.4, 2.8)));
    return p + HWARP_FINE * a + HWARP_COARSE * b;
}

fn streak_coords(p: vec2<f32>, dir: vec2<f32>) -> vec2<f32> {
    let n = vec2<f32>(-dir.y, dir.x);
    return vec2<f32>(dot(p, n), dot(p, dir));
}

fn granulation(p: vec2<f32>) -> f32 {
    return 0.6 * vnoise(p / GRAIN_SCALE) + 0.3 * vnoise(p / (GRAIN_SCALE * 0.41) + vec2<f32>(7.3, 2.1)) + 0.1 * vnoise(p / (GRAIN_SCALE * 3.7) + vec2<f32>(1.7, 9.2));
}

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
    roller: f32,
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
    d.roller = 0.0;
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
        d.roller += wk * select(0.0, 1.0, abs(method - METHOD_ROLLER) < 0.5);
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
    if hp <= 2e-5 { return canvas_bump(p) * (1.0 - smoothstep(0.0, 0.5 * WRINKLE_WAVELENGTH, cam.jitter.w)); }
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
    if d.roller > 0.0 {
        let resolved = 1.0 - smoothstep(0.2, 0.6, footprint / ROLLER_STIPPLE_SCALE);
        let peel = fbm_filtered(vec2<f32>(across, along / ROLLER_STIPPLE_STRETCH), ROLLER_STIPPLE_SCALE, 3);
        z += d.roller * ROLLER_STIPPLE_DEPTH * min(hp / 4e-5, 1.0) * resolved * peel;
    }
    let skin = d.cure * smoothstep(0.5e-3, 1.5e-3, hp) * (1.0 - smoothstep(0.2, 0.5, footprint / WRINKLE_WAVELENGTH));
    if skin > 0.0 {
        z += WRINKLE_HEIGHT * skin * wrinkle_profile(p, WRINKLE_WAVELENGTH);
    }
    if beauty_mode() {
        z += canvas_bump_levelled(p, hp) * (1.0 - smoothstep(0.0, 0.5 * WRINKLE_WAVELENGTH, footprint));
        let amp = clamp(IMPASTO_FRACTION * hp, 1.0e-6, IMPASTO_MAX) * mix(0.45, 1.0, d.cure);
        let tool = clamp(d.brush + d.knife, 0.0, 1.0);
        let sc = streak_coords(p, d.dir);
        let aniso = fbm_filtered(vec2<f32>(sc.x / 0.6, sc.y / 3.0), 0.25e-3, 4);
        let iso = fbm_filtered(p, 0.35e-3, 4);
        z += amp * mix(iso, aniso, 0.35 + 0.65 * tool);
        let clump = fbm_filtered(vec2<f32>(sc.x, sc.y / CLUMP_ELONGATION), CLUMP_WIDTH, 4);
        z += tool * carry * CLUMP_DEPTH * clump;
        let pin_fade = smoothstep(0.08e-3, 0.3e-3, hp) * (1.0 - smoothstep(0.02e-3, 0.06e-3, footprint));
        if pin_fade > 0.0 { z += pin_fade * pinholes(p, hp); }
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
    let sig2 = 0.45;
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
    if cam.jitter.z > 0.5 { return surf_cubic(hwarp(p)); }
    return surf_bilinear(p);
}

fn surf_hit(p: vec2<f32>, d: Detail) -> f32 {
    if cam.jitter.z > 0.5 { return surf_smooth(hwarp(p)) + micro(p, d); }
    return surf_bilinear(p) + micro(p, d);
}

fn base_slope(p: vec2<f32>) -> vec2<f32> {
    let e = 0.7 * cam.canvas.z;
    if cam.jitter.z > 0.5 {
        let sx = surf_smooth(hwarp(p + vec2<f32>(e, 0.0))) - surf_smooth(hwarp(p - vec2<f32>(e, 0.0)));
        let sy = surf_smooth(hwarp(p + vec2<f32>(0.0, e))) - surf_smooth(hwarp(p - vec2<f32>(0.0, e)));
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

fn color_beauty(p: vec2<f32>) -> vec4<f32> {
    var sc = cam.ortho.z;
    if sc <= 0.0 { sc = 1.0; }
    let cdx = cam.canvas.z / sc;
    let g = p / cdx - vec2<f32>(0.5);
    let f = floor(g);
    let t = g - f;
    let i = i32(f.x);
    let j = i32(f.y);
    var hs: array<f32, 16>;
    var hmin = 1e9;
    var hmax = 0.0;
    for (var k = 0; k < 16; k++) {
        let a = k % 4 - 1;
        let b = k / 4 - 1;
        let centre = (vec2<f32>(f32(i + a), f32(j + b)) + vec2<f32>(0.5)) * cdx;
        let hv = surf_bilinear(centre);
        hs[k] = hv;
        hmin = min(hmin, hv);
        hmax = max(hmax, hv);
    }
    let crisp = color_crisp(p);
    if hmax <= 1e-7 { return crisp; }
    let range = hmax - hmin;
    let edge = smoothstep(0.3, 0.6, range / hmax) * smoothstep(EDGE_MIN_STEP, 2.0 * EDGE_MIN_STEP, range);
    if edge <= 0.0 { return crisp; }
    let wx = cr_w(t.x);
    let wy = cr_w(t.y);
    var cin = vec4<f32>(0.0);
    var cout = vec4<f32>(0.0);
    var win = 1e-12;
    var wout = 1e-12;
    for (var k = 0; k < 16; k++) {
        let a = k % 4;
        let b = k / 4;
        let w = wx[a] * wy[b] + 0.02;
        let rel = (hs[k] - hmin) / range;
        let c = coarse_col(i + a - 1, j + b - 1);
        let w_in = w * rel * rel * rel;
        let w_out = w * (1.0 - rel) * (1.0 - rel) * (1.0 - rel);
        cin += w_in * c;
        cout += w_out * c;
        win += w_in;
        wout += w_out;
    }
    cin /= win;
    cout /= wout;
    let hb = (surf_cubic(p) - hmin) / range;
    let s = smoothstep(EDGE_ISO - EDGE_ISO_SOFT, EDGE_ISO + EDGE_ISO_SOFT, hb);
    return mix(crisp, mix(cout, cin, s), edge);
}

fn color_sharp(p: vec2<f32>) -> vec4<f32> {
    if beauty_mode() { return color_beauty(p); }
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
    var cbil = wb.x * c0 + wb.y * c1 + wb.z * c2 + wb.w * c3;
    if beauty_mode() { cbil = color_crisp(p); }
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

fn coarse_col(i: i32, j: i32) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(color_tex));
    return textureLoad(color_tex, clamp(vec2<i32>(i, j), vec2<i32>(0), dims - vec2<i32>(1)), 0);
}

fn crisp_t(t: f32) -> f32 {
    let a = pow(t, COLOR_CRISP);
    return a / (a + pow(1.0 - t, COLOR_CRISP));
}

fn color_crisp(p: vec2<f32>) -> vec4<f32> {
    var sc = cam.ortho.z;
    if sc <= 0.0 { sc = 1.0; }
    let cdx = cam.canvas.z / sc;
    let g = p / cdx - vec2<f32>(0.5);
    let f = floor(g);
    let t = g - f;
    let i = i32(f.x);
    let j = i32(f.y);
    let tx = crisp_t(t.x);
    let ty = crisp_t(t.y);
    let a = coarse_col(i, j);
    let b = coarse_col(i + 1, j);
    let c = coarse_col(i, j + 1);
    let d = coarse_col(i + 1, j + 1);
    return mix(mix(a, b, tx), mix(c, d, tx), ty);
}

fn cwarp(p: vec2<f32>, d: Detail) -> vec2<f32> {
    let q = hwarp(p);
    if !beauty_mode() { return q; }
    let free = 1.0 - edge_factor(q);
    let sc = streak_coords(p, d.dir);
    let streak = fbm(vec2<f32>(sc.x / STREAK_WIDTH, sc.y / STREAK_LENGTH), 4);
    let n = vec2<f32>(-d.dir.y, d.dir.x);
    let tool = clamp(d.brush + d.knife, 0.0, 1.0);
    let iso = vec2<f32>(fbm(p / 1.3e-3 + vec2<f32>(4.1, 8.3), 4), fbm(p / 1.3e-3 + vec2<f32>(1.9, 6.6), 4));
    return q + free * (n * streak * CWARP_STREAK * (0.35 + 0.65 * tool) + iso * CWARP_ISO);
}

fn marble(p: vec2<f32>, c: vec3<f32>, d: Detail) -> vec3<f32> {
    let inv = vec2<f32>(1.0 / cam.canvas.x, 1.0 / cam.canvas.y);
    var mean = c;
    var cols: array<vec3<f32>, 16>;
    for (var k = 0; k < 16; k++) {
        let ring = select(MARBLE_R1, MARBLE_R2, k >= 8);
        let ang = (f32(k) + select(0.0, 0.5, k >= 8)) * TAU / 8.0;
        let q = p + vec2<f32>(cos(ang), sin(ang)) * ring;
        cols[k] = textureSampleLevel(color_tex, samp, q * inv, 0.0).rgb;
        mean += cols[k];
    }
    mean /= 17.0;
    var spread = 1e-6;
    for (var k = 0; k < 16; k++) {
        spread = max(spread, dot(cols[k] - mean, cols[k] - mean));
    }
    var a = vec3<f32>(0.0);
    var wa = 0.0;
    for (var k = 0; k < 16; k++) {
        let w = exp(MARBLE_SELECT * (dot(cols[k] - mean, cols[k] - mean) / spread - 1.0));
        a += w * cols[k];
        wa += w;
    }
    a /= wa;
    var spread_a = 1e-6;
    for (var k = 0; k < 16; k++) {
        spread_a = max(spread_a, dot(cols[k] - a, cols[k] - a));
    }
    var b = vec3<f32>(0.0);
    var wb = 0.0;
    for (var k = 0; k < 16; k++) {
        let w = exp(MARBLE_SELECT * (dot(cols[k] - a, cols[k] - a) / spread_a - 1.0));
        b += w * cols[k];
        wb += w;
    }
    b /= wb;
    let ab = b - a;
    let len2 = dot(ab, ab);
    if len2 < MARBLE_MIN_CONTRAST * MARBLE_MIN_CONTRAST { return c; }
    let f = clamp(dot(c - a, ab) / len2, 0.0, 1.0);
    let resid = c - (a + ab * f);
    let on_line = 1.0 - smoothstep(0.08, 0.25, length(resid) / sqrt(len2));
    let scp = streak_coords(p, d.dir);
    let tool = clamp(d.brush + d.knife, 0.0, 1.0);
    let n_iso = fbm(p / MARBLE_SCALE + vec2<f32>(2.3, 7.7), 5);
    let n_str = fbm(vec2<f32>(scp.x / (0.35 * MARBLE_SCALE), scp.y / (4.0 * MARBLE_SCALE)), 5);
    let n = mix(n_iso, n_str, 0.4 + 0.6 * tool);
    let thr = 0.5 + 0.55 * n;
    let soft = MARBLE_SOFT + 0.6 * cam.jitter.w / MARBLE_SCALE;
    let fs = smoothstep(thr - soft, thr + soft, f);
    let fm = mix(f, fs, MARBLE_STRENGTH * on_line);
    return max(a + ab * fm + resid, vec3<f32>(0.0));
}

fn shadow_at(p: vec2<f32>, z0: f32, l: vec3<f32>) -> f32 {
    if l.z <= 0.05 { return 1.0; }
    let step = cam.canvas.z * 1.5;
    var spread = tan(KEY_HALF_ANGLE);
    var zstart = z0;
    if beauty_mode() {
        spread = BEAUTY_PENUMBRA;
        zstart = max(z0, surf_march(p)) + SHADOW_BIAS;
    }
    var sh = 1.0;
    for (var k = 1; k <= 20; k++) {
        let d = f32(k) * step;
        let q = p + l.xy / l.z * d;
        let zs = surf_march(q);
        let zr = zstart + d;
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
    var zenith = 1.0;
    if beauty_mode() { zenith = mix(1.0, ZENITH_DIM, smoothstep(0.2, 0.9, dir.z) * (1.0 - rough)); }
    var e = mix(floor, sky * zenith, up) + HORIZON_GLOW * 0.5 * band;
    let l = normalize(cam.light.xyz);
    let box_scale = select(1.0, BEAUTY_SOFTBOX_SCALE, beauty_mode());
    let box_radiance = select(SOFTBOX_RADIANCE * cam.light.w / 2.6, BEAUTY_SOFTBOX_RADIANCE * cam.light.w / PI, beauty_mode());
    e += softbox(dir, l, rough, SOFTBOX_HALF_W * box_scale, SOFTBOX_HALF_H * box_scale) * vec3<f32>(box_radiance);
    let lf = normalize(vec3<f32>(-l.x * 0.9, -l.y * 0.9, max(l.z * 0.7, 0.25)));
    e += softbox(dir, lf, rough, SOFTBOX_HALF_W * 1.4, SOFTBOX_HALF_H * 1.6) * FILL_TINT * (FILL_RADIANCE * SOFTBOX_RADIANCE * cam.light.w / 2.6);
    return e;
}

fn softbox(dir: vec3<f32>, l: vec3<f32>, rough: f32, hw: f32, hh: f32) -> f32 {
    let side = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), l));
    let upv = cross(l, side);
    let cos_l = dot(dir, l);
    if cos_l <= 0.0 { return 0.0; }
    let u = atan2(dot(dir, side), cos_l);
    let v = atan2(dot(dir, upv), cos_l);
    let edge = 0.03 + 0.7 * rough;
    let inside = (1.0 - smoothstep(hw - edge, hw + edge, abs(u))) * (1.0 - smoothstep(hh - edge, hh + edge, abs(v)));
    let fill = (hw * hh) / ((hw + edge) * (hh + edge));
    return inside * fill;
}

fn d_ggx_aniso(n: vec3<f32>, t: vec3<f32>, b: vec3<f32>, hvec: vec3<f32>, at: f32, ab: f32) -> f32 {
    let th = dot(t, hvec) / at;
    let bh = dot(b, hvec) / ab;
    let nh = dot(n, hvec);
    let d = th * th + bh * bh + nh * nh;
    return 1.0 / (PI * at * ab * d * d);
}

fn spec_lobe_aniso(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32, f0: f32, aniso: f32, tdir: vec2<f32>) -> f32 {
    if aniso <= 0.01 { return spec_lobe(n, v, l, rough, f0); }
    let hvec = normalize(v + l);
    let a = max(rough * rough, 0.002);
    let a_wide = select(min(a + sin(KEY_HALF_ANGLE), 1.0), max(a, BEAUTY_MIN_ALPHA), beauty_mode());
    let energy = select((a / a_wide) * (a / a_wide), 1.0, beauty_mode());
    var t = vec3<f32>(tdir, 0.0);
    t = normalize(t - n * dot(t, n));
    let b = cross(n, t);
    let at = max(a_wide * (1.0 - 0.6 * aniso), 0.002);
    let ab = min(a_wide * (1.0 + 1.2 * aniso), 1.0);
    let ndv = max(dot(n, v), 1e-3);
    let ndl = max(dot(n, l), 0.0);
    let f = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, hvec), 0.0), 5.0);
    let a2 = at * ab;
    return d_ggx_aniso(n, t, b, hvec, at, ab) * energy * v_smith(ndv, ndl, a2) * f;
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
    let a_wide = select(min(a + sin(KEY_HALF_ANGLE), 1.0), max(a, BEAUTY_MIN_ALPHA), beauty_mode());
    let a2 = a_wide * a_wide;
    let energy = select((a / a_wide) * (a / a_wide), 1.0, beauty_mode());
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
    } else if cam.lens.w > 0.5 {
        let fwd = cross(cam.basis_u.xyz, cam.basis_r.xyz);
        let fd = cam.lens.z / max(dot(dir, fwd), 1e-3);
        let fp = a + dir * fd;
        a = a + cam.basis_r.xyz * cam.lens.x + cam.basis_u.xyz * cam.lens.y;
        dir = normalize(fp - a);
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
        return vec4<f32>(to_srgb(tonemap(lit(normal_bilinear(p), v, l, 1.0, 1.0, cs.rgb, cs.a, 0.0, 0.0, 0.0, vec2<f32>(1.0, 0.0), 1.0))), 1.0);
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
    let pcw = cwarp(pc, detail);
    var col = color_sharp(pcw);
    if beauty_mode() && detail.thickness > 0.02e-3 {
        col = vec4<f32>(marble(pcw, col.rgb, detail), col.a);
    }
    let thin = 0.3 * (1.0 - smoothstep(0.0, THIN_FILM_MATTE, detail.thickness)) * smoothstep(0.0, 1e-5, detail.thickness);
    var rough = clamp(col.a + 0.3 * length(ms) + thin, 0.03, 0.95);
    if beauty_mode() {
        let paint_cover = smoothstep(0.0, 0.08e-3, detail.thickness);
        let gloss = DRY_GLOSS_FLOOR + (col.a - DRY_GLOSS_FLOOR) * DRY_GLOSS_SCALE;
        let patchy = 0.75 + 0.5 * (0.5 + 0.5 * fbm(p / 2.5e-3 + vec2<f32>(6.1, 2.4), 3));
        rough = clamp(mix(col.a, gloss * patchy, paint_cover) + thin, 0.03, 0.95);
    }
    let z0 = surf_hit(p, detail);
    let sh = shadow_at(p, z0, l);
    let ao = ambient_occlusion(p, z0, detail);
    var albedo = col.rgb;
    let painted = smoothstep(0.0, 0.08e-3, detail.thickness);
    if painted > 0.0 {
        let g = granulation(p);
        albedo *= 1.0 + 0.045 * g * painted;
        let rim = edge_factor(p);
        albedo *= 1.0 - 0.10 * rim * painted;
    }
    if beauty_mode() && painted > 0.0 {
        let sc = streak_coords(p, detail.dir);
        let tool = clamp(detail.brush + detail.knife, 0.0, 1.0);
        let st = fbm_filtered(vec2<f32>(sc.x, sc.y / 25.0), 0.09e-3, 4);
        albedo *= 1.0 + 0.16 * st * (0.3 + 0.7 * tool) * painted;
        let cl = fbm_filtered(vec2<f32>(sc.x, sc.y / CLUMP_ELONGATION), CLUMP_WIDTH, 3);
        albedo *= 1.0 + 0.10 * cl * tool * painted;
        let thin_paint = 1.0 - smoothstep(0.03e-3, 0.25e-3, detail.thickness);
        let wv = weave(p);
        albedo *= 1.0 + 0.22 * thin_paint * (wv.x - 0.55) * painted;
    }
    let bare = 1.0 - smoothstep(0.0, 0.15e-3, detail.thickness);
    if bare > 0.0 {
        albedo = mix(albedo, canvas_albedo(p, albedo), bare);
    }
    if beauty_mode() && painted > 0.0 {
        let hue = vec3<f32>(fbm(p / 6e-3 + vec2<f32>(1.3, 5.1), 3), fbm(p / 6e-3 + vec2<f32>(7.7, 2.2), 3), fbm(p / 6e-3 + vec2<f32>(4.4, 9.9), 3));
        let tone = fbm(p / 2.2e-3 + vec2<f32>(3.3, 0.7), 4);
        albedo *= (vec3<f32>(1.0) + PAINT_HUE_DRIFT * hue) * (1.0 + PAINT_TONE_DRIFT * tone * painted);
    }
    let sp = specks(p);
    albedo = mix(albedo, vec3<f32>(0.85, 0.83, 0.8), 0.7 * sp.x * (0.4 + 0.6 * bare));
    albedo *= 1.0 - 0.6 * sp.y;
    let rough_var = 1.0 + 0.18 * vnoise(p / 0.45e-3 + vec2<f32>(3.0, 8.0)) * painted;
    let rough_s = clamp(rough * rough_var, 0.03, 0.95);
    let wetness = wet_bilinear(p);
    let coat = smoothstep(0.02e-3, 0.25e-3, wetness) * (1.0 - 0.7 * detail.cure) * (1.0 - thin / 0.3);
    let aniso = clamp(0.7 * detail.brush + 0.5 * detail.knife, 0.0, 0.8) * smoothstep(0.0, 0.15e-3, detail.thickness);
    let falloff = select(1.0, lamp_falloff(p), cam.jitter.z > 0.5);
    let c = lit(n, v, l, sh, ao, albedo, rough_s, detail.cure, coat, aniso, detail.dir, falloff);
    if cam.jitter.z > 0.5 { return vec4<f32>(select(min(c, vec3<f32>(FIREFLY_CLAMP)), vec3<f32>(0.0), c != c), 1.0); }
    return vec4<f32>(to_srgb(tonemap(c)), 1.0);
}

fn lit(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, sh: f32, ao: f32, albedo: vec3<f32>, rough: f32, cure: f32, coat: f32, aniso: f32, tdir: vec2<f32>, falloff: f32) -> vec3<f32> {
    let ndl = max(dot(n, l), 0.0);
    let ndv = max(dot(n, v), 1e-3);
    let sun = cam.light.w * falloff;
    let f_env = env_dfg(F0_PAINT, rough, ndv);
    let kd = 1.0 - f_env;
    let ambient = env_radiance(n, 1.0) * ao;
    let lambert = select(1.0, 1.0 / PI, beauty_mode());
    let diffuse = albedo * kd * (sun * ndl * sh * lambert + ambient);
    var spec = spec_lobe_aniso(n, v, l, rough, F0_PAINT, aniso, tdir);
    spec += 0.35 * cure * spec_lobe(n, v, l, max(rough, 0.55), F0_PAINT);
    let direct = spec * sun * ndl * sh;
    let rv = reflect(-v, n);
    let occlusion = clamp(pow(ndv + ao, exp2(-16.0 * rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
    let env = env_radiance(rv, rough) * f_env * occlusion;
    let base_spec = select(1.0, 1.0 - coat, beauty_mode());
    var c = diffuse + (vec3<f32>(direct) + env) * base_spec;
    if coat > 0.0 {
        let fc = env_dfg(F0_PAINT, COAT_ROUGH, ndv);
        let coat_direct = spec_lobe(n, v, l, COAT_ROUGH, F0_PAINT) * sun * ndl * sh;
        let occ_c = clamp(pow(ndv + ao, exp2(-16.0 * COAT_ROUGH - 1.0)) - 1.0 + ao, 0.0, 1.0);
        let coat_env = env_radiance(rv, COAT_ROUGH) * fc * occ_c;
        c = c * (1.0 - coat * fc) + coat * (vec3<f32>(coat_direct) + coat_env);
    }
    return c;
}
