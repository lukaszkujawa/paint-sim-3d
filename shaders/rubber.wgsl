struct RubberParams {
    a: vec2<f32>,
    b: vec2<f32>,
    radius: f32,
    weave: f32,
    enabled: u32,
    pad: u32,
    refl: array<vec4<u32>, 2>,
}

@group(2) @binding(0) var<uniform> R: RubberParams;

const WEAVE_PERIOD_M: f32 = 1.4e-3;
const RUBBER_TAU: f32 = 6.2831853;

fn weave_at(i: u32, j: u32) -> f32 {
    if R.weave <= 0.0 { return 0.0; }
    var s = (i * 73856093u) ^ (j * 19349663u);
    s ^= s >> 13u;
    s *= 0x5bd1e995u;
    s ^= s >> 15u;
    let r = f32(s & 0xffffu) / 65535.0 * 2.0 - 1.0;
    let k = RUBBER_TAU / WEAVE_PERIOD_M;
    let x = (f32(i) + 0.5) * P.dx;
    let y = (f32(j) + 0.5) * P.dx;
    return R.weave * (1.0 + 0.6 * sin(k * x) * sin(k * y) + 0.4 * r);
}

@compute @workgroup_size(16, 16)
fn rubber(@builtin(global_invocation_id) id: vec3<u32>) {
    if !in_grid(id) { return; }
    if R.enabled == 0u { return; }
    let p = (vec2<f32>(id.xy) + vec2<f32>(0.5)) * P.dx;
    let ab = R.b - R.a;
    let t = clamp(dot(p - R.a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    if length(p - R.a - ab * t) > R.radius { return; }
    let c = idx(id.x, id.y);
    h[c] = 0.0;
    m[c] = weave_at(id.x, id.y);
    rheo[c] = vec2<f32>(1.0, 0.0);
    alpha[c] = 0.0;
    alpha_s[c] = vec4<f32>(0.0);
    delta[c] = 0.0;
    tmp[c] = 0.0;
    flux[c] = vec4<f32>(0.0);
    acc[c] = vec4<f32>(0.0);
    acc_s[c] = vec4<f32>(0.0);
    strata_h[c] = vec4<f32>(0.0);
    strata_h2[c] = vec4<f32>(0.0);
    strata_cure[c] = vec4<f32>(0.0);
    strata_cure2[c] = vec4<f32>(0.0);
    for (var k = 0u; k < 8u; k++) {
        strata_pig[c * 8u + k] = vec4<u32>(0u);
        strata_pig2[c * 8u + k] = vec4<u32>(0u);
        sub_refl[c * 8u + k] = R.refl[k / 4u][k % 4u];
    }
    lay[c] = 0u;
    mark_moved(id.x, id.y);
}
