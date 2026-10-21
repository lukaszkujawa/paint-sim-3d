@group(0) @binding(0) var acc: texture_2d<f32>;
@group(0) @binding(1) var<uniform> R: vec4<f32>;

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
    let c = textureLoad(acc, vec2<i32>(in.pos.xy), 0).rgb * R.x;
    return vec4<f32>(to_srgb(tonemap(c)), 1.0);
}
