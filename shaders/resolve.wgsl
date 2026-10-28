@group(0) @binding(0) var acc: texture_2d<f32>;
@group(0) @binding(1) var<uniform> R: Res;

struct Res {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
}

struct VOut {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var o: VOut;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    o.pos = vec4<f32>(x, y, 0.0, 1.0);
    return o;
}

fn aces(c: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let cc = 2.43; let d = 0.59; let e = 0.14;
    return clamp((c * (a * c + b)) / (c * (cc * c + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn agx(c: vec3<f32>) -> vec3<f32> {
    let inset = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104));
    let outset = mat3x3<f32>(
        vec3<f32>(1.1968790051, -0.0528968518, -0.0529716355),
        vec3<f32>(-0.0980208811, 1.1519031299, -0.0980434501),
        vec3<f32>(-0.0990297441, -0.0989611768, 1.1510736726));
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var v = inset * max(c, vec3<f32>(1e-10));
    v = clamp(log2(v), vec3<f32>(min_ev), vec3<f32>(max_ev));
    v = (v - vec3<f32>(min_ev)) / (max_ev - min_ev);
    let x2 = v * v;
    let x4 = x2 * x2;
    v = 15.5 * x4 * x2 - 40.14 * x4 * v + 31.96 * x4 - 6.868 * x2 * v + 0.4298 * x2 + 0.1191 * v - 0.00232;
    v = clamp(v, vec3<f32>(0.0), vec3<f32>(1.0));
    let luma = dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
    v = pow(v, vec3<f32>(1.35));
    v = vec3<f32>(luma) + 1.4 * (v - vec3<f32>(luma));
    v = outset * clamp(v, vec3<f32>(0.0), vec3<f32>(1.0));
    return pow(clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
}

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn hash12(p: vec2<f32>) -> f32 {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973).xy);
    q += dot(q, q.yx + vec2<f32>(33.33));
    return fract((q.x + q.y) * q.x);
}

fn lum(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn load(p: vec2<i32>) -> vec3<f32> {
    let w = i32(R.b.y);
    let h = i32(R.b.z);
    let q = clamp(p, vec2<i32>(0), vec2<i32>(w - 1, h - 1));
    return textureLoad(acc, q, 0).rgb * R.a.x;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.pos.xy);
    let exposure = exp2(R.a.y);
    var c = load(px) * exposure;
    let bloom = R.a.z;
    if bloom > 0.0 {
        let radius = max(R.b.z * 0.02, 4.0);
        var glow = vec3<f32>(0.0);
        var wsum = 0.0;
        for (var k = 0; k < 32; k++) {
            let fk = f32(k) + 0.5;
            let r = radius * sqrt(fk / 32.0);
            let ang = fk * 2.399963;
            let o = vec2<i32>(vec2<f32>(cos(ang), sin(ang)) * r);
            let s = load(px + o) * exposure;
            let l = lum(s);
            let wk = 1.0 - fk / 33.0;
            glow += s * max(l - 1.6, 0.0) / max(l, 1e-4) * wk;
            wsum += wk;
        }
        c += bloom * 0.35 * glow / wsum;
    }
    let sharpen = R.c.x;
    if sharpen > 0.0 {
        let blur = (load(px + vec2<i32>(1, 0)) + load(px + vec2<i32>(-1, 0)) + load(px + vec2<i32>(0, 1)) + load(px + vec2<i32>(0, -1))) * exposure * 0.25;
        let base = load(px) * exposure;
        c += sharpen * (base - blur);
        c = max(c, vec3<f32>(0.0));
    }
    let uv = in.pos.xy / vec2<f32>(R.b.y, R.b.z) - vec2<f32>(0.5);
    let vig = 1.0 - R.b.x * smoothstep(0.15, 0.75, dot(uv, uv) * 2.0);
    c *= vig;
    var mapped = agx(c);
    if R.b.w < 0.5 { mapped = aces(c); }
    let grain = R.a.w;
    if grain > 0.0 {
        let g = (hash12(in.pos.xy + vec2<f32>(0.37, 0.11)) - 0.5) * 2.0;
        let shade = 1.0 - smoothstep(0.0, 1.0, lum(mapped));
        mapped = clamp(mapped + vec3<f32>(g * grain * (0.35 + 0.65 * shade)), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return vec4<f32>(to_srgb(mapped), 1.0);
}
