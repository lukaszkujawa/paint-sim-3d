struct Cam {
    inv_vp: mat4x4<f32>,
    eye: vec4<f32>,
    canvas: vec4<f32>,
    light: vec4<f32>,
    size: vec4<u32>,
    sky: vec4<f32>,
    ground: vec4<f32>,
    vp: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> cam: Cam;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) col: vec3<f32>,
}

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) wpos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) col: vec3<f32>,
}

@vertex
fn vs(in: VIn) -> VOut {
    var o: VOut;
    o.pos = cam.vp * vec4<f32>(in.pos, 1.0);
    o.wpos = in.pos;
    o.nrm = in.nrm;
    o.col = in.col;
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
    let n = normalize(in.nrm);
    let v = normalize(cam.eye.xyz - in.wpos);
    let l = normalize(cam.light.xyz);
    let ndl = max(dot(n, l), 0.0);
    let hv = normalize(v + l);
    let spec = pow(max(dot(n, hv), 0.0), 120.0) * 0.6;
    let fres = 0.04 + 0.96 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    let env = mix(cam.ground.rgb, cam.sky.rgb, clamp(reflect(-v, n).z * 0.5 + 0.5, 0.0, 1.0));
    let c = in.col * (cam.light.w * ndl + cam.sky.rgb * (0.5 + 0.5 * n.z)) + vec3<f32>(spec * cam.light.w) + env * fres;
    return vec4<f32>(to_srgb(tonemap(c)), 1.0);
}
