const KNIFE_BANDS: u32 = 8u;
const KNIFE_BLADE_LEN: f32 = 0.07;
const KNIFE_SLOPE: f32 = 0.25;
const KNIFE_FLAT: f32 = 0.01;
const KNIFE_COVER: f32 = 1.5;
const KNIFE_FLEX: f32 = 0.004;
const KNIFE_FLEX_TAPS: i32 = 4;
const KNIFE_TOOTH: f32 = 0.8;
const KNIFE_TOOTH_ALONG: f32 = 0.008;
const KNIFE_TOOTH_ACROSS: f32 = 0.0012;

struct KnifeParams {
    a: vec2<f32>,
    b: vec2<f32>,
    dir: vec2<f32>,
    sweep: f32,
    height: f32,
    half_width: f32,
    enabled: u32,
    frame: u32,
    pad1: u32,
    bands: u32,
    pad2: u32,
    pad3: u32,
    pad4: u32,
    pig: array<vec4<u32>, 16>,
    fill: array<vec4<f32>, 2>,
    smear: array<vec4<f32>, 2>,
}

struct Scrap {
    cell: u32,
    vol: f32,
    frame: u32,
    pad1: u32,
    pig0: vec4<u32>,
    pig1: vec4<u32>,
}

@group(2) @binding(0) var<uniform> K: KnifeParams;
@group(2) @binding(1) var<storage, read_write> scraps: array<Scrap>;
@group(2) @binding(2) var<storage, read_write> scrap_count: atomic<u32>;

fn cell_hash(x: u32, y: u32) -> f32 {
    var h = x * 0x8da6b343u ^ y * 0xd8163841u;
    h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
    h = (h ^ (h >> 12u)) * 0x297a2d39u;
    h = h ^ (h >> 15u);
    return f32(h & 0xffffu) / 65535.0;
}

fn lattice_noise(q: vec2<f32>) -> f32 {
    let i = floor(q);
    let f = fract(q);
    let u = f * f * (3.0 - 2.0 * f);
    let o = vec2<i32>(i) + vec2<i32>(1 << 20);
    let x = u32(o.x);
    let y = u32(o.y);
    let a = cell_hash(x, y);
    let b = cell_hash(x + 1u, y);
    let c = cell_hash(x, y + 1u);
    let d = cell_hash(x + 1u, y + 1u);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn tooth(p: vec2<f32>, md: vec2<f32>, along: vec2<f32>) -> f32 {
    let q = vec2<f32>(dot(p, md) / KNIFE_TOOTH_ALONG, dot(p, along) / KNIFE_TOOTH_ACROSS);
    return 0.65 * lattice_noise(q) + 0.35 * lattice_noise(2.3 * q + vec2<f32>(17.0, 5.0));
}

fn emit(c: u32, vol: f32, pig: array<f32, 16>) {
    let slot = atomicAdd(&scrap_count, 1u);
    if slot < arrayLength(&scraps) {
        var r: Scrap;
        r.cell = c;
        r.vol = vol;
        r.frame = K.frame;
        r.pad1 = 0u;
        r.pig0 = pack_pig_a(pig);
        r.pig1 = pack_pig_b(pig);
        scraps[slot] = r;
    }
}

@compute @workgroup_size(16, 16)
fn knife(@builtin(global_invocation_id) id: vec3<u32>) {
    if !in_grid(id) { return; }
    if K.enabled == 0u { return; }
    let c = idx(id.x, id.y);
    let p = (vec2<f32>(id.xy) + vec2<f32>(0.5)) * P.dx;
    let ab = K.b - K.a;
    let mv = K.dir * K.sweep;
    let det = ab.x * mv.y - ab.y * mv.x;
    if abs(det) < 1e-12 { return; }
    let q = p - K.a;
    let t = (q.x * mv.y - q.y * mv.x) / det;
    let u = -(ab.x * q.y - ab.y * q.x) / det;
    if t < 0.0 || t > 1.0 { return; }
    let behind = u * K.sweep;
    if behind < -P.dx || behind > KNIFE_BLADE_LEN { return; }
    let at_edge = u <= 1.0;
    let nb = max(min(K.bands, KNIFE_BANDS), 1u);
    let bi = min(u32(t * f32(nb)), nb - 1u);
    let ratio = select(0.0, K.fill[bi / 4u][bi % 4u], at_edge);
    let smear_b = select(0.0, K.smear[bi / 4u][bi % 4u], at_edge);
    let band_pig = unpack_pig(K.pig[2u * bi], K.pig[2u * bi + 1u]);
    let hv = h[c];
    let along = ab / max(length(ab), 1e-9);
    var support = m[c];
    for (var k = -KNIFE_FLEX_TAPS; k <= KNIFE_FLEX_TAPS; k++) {
        let o = p + along * (f32(k) * KNIFE_FLEX / f32(KNIFE_FLEX_TAPS));
        let cx = clamp(i32(o.x / P.dx), 0, i32(P.size.x) - 1);
        let cy = clamp(i32(o.y / P.dx), 0, i32(P.size.y) - 1);
        support = max(support, m[idx(u32(cx), u32(cy))]);
    }
    let blade = support + K.height + max(behind - KNIFE_FLAT, 0.0) * KNIFE_SLOPE;
    let cap = max(blade - m[c], 0.0);
    let keep = min(hv, cap);
    let carried = K.height * KNIFE_COVER * ratio;
    let reach = blade - carried;
    let soft = max(0.15 * carried, 2e-6);
    let rough = K.height * KNIFE_TOOTH * tooth(p, normalize(mv + vec2<f32>(1e-9, 0.0)), along);
    let contact = select(0.0, smoothstep(reach - soft, reach + soft, m[c] + keep + rough), carried > 0.0);
    let level = keep + (cap - keep) * contact;
    let swap = keep * smear_b;
    if hv <= cap && hv >= level && swap <= 0.0 { return; }
    var s = stack_load(c);
    var excess = (hv - cap) + swap;
    var removed = 0.0;
    var pig: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { pig[i] = 0.0; }
    var k = 3u;
    loop {
        if excess <= 0.0 { break; }
        let th = s.th[k];
        if th > 0.0 {
            let take = min(th, excess);
            pig = pig_mix(pig, removed, s.pg[k], take);
            removed += take;
            excess -= take;
            s.th[k] = th - take;
        }
        if k == 0u { break; }
        k--;
    }
    var hn = hv - removed;
    let add = max(max(level, keep) - hn, 0.0);
    if add > 0.0 {
        stack_push_top(&s, add, band_pig, 0.0);
        hn += add;
    }
    h[c] = hn;
    lay[c] = lay_pack(METHOD_KNIFE, normalize(mv + vec2<f32>(1e-9, 0.0)));
    mark_moved(id.x, id.y);
    stack_compact(&s);
    stack_store(c, s);
    if removed > 0.0 {
        emit(c, removed * P.dx * P.dx, pig);
    }
    if add > 0.0 {
        var none: array<f32, 16>;
        for (var i = 0u; i < 16u; i++) { none[i] = 0.0; }
        emit(c, -add * P.dx * P.dx, none);
    }
}
