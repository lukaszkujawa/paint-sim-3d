struct KnifeParams {
    a: vec2<f32>,
    b: vec2<f32>,
    dir: vec2<f32>,
    sweep: f32,
    height: f32,
    half_width: f32,
    enabled: u32,
    pad0: u32,
    pad1: u32,
}

struct Scrap {
    cell: u32,
    vol: f32,
    pad0: u32,
    pad1: u32,
    pig0: vec4<u32>,
    pig1: vec4<u32>,
}

@group(2) @binding(0) var<uniform> K: KnifeParams;
@group(2) @binding(1) var<storage, read_write> scraps: array<Scrap>;
@group(2) @binding(2) var<storage, read_write> scrap_count: atomic<u32>;

@compute @workgroup_size(16, 16)
fn knife(@builtin(global_invocation_id) id: vec3<u32>) {
    if !in_grid(id) { return; }
    if K.enabled == 0u { return; }
    let c = idx(id.x, id.y);
    let hv = h[c];
    let cap = max(K.height - m[c], 0.0);
    if hv <= cap { return; }
    let p = (vec2<f32>(id.xy) + vec2<f32>(0.5)) * P.dx;
    let ab = K.b - K.a;
    let l2 = max(dot(ab, ab), 1e-12);
    let t = dot(p - K.a, ab) / l2;
    if t < 0.0 || t > 1.0 { return; }
    let foot = K.a + ab * t;
    let off = dot(p - foot, K.dir);
    if off > 0.0 || off < -(K.sweep + P.dx) { return; }
    var s = stack_load(c);
    var excess = hv - cap;
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
    if removed <= 0.0 { return; }
    h[c] = hv - removed;
    mark_moved(id.x, id.y);
    stack_compact(&s);
    stack_store(c, s);
    let slot = atomicAdd(&scrap_count, 1u);
    if slot < arrayLength(&scraps) {
        var r: Scrap;
        r.cell = c;
        r.vol = removed * P.dx * P.dx;
        r.pad0 = 0u;
        r.pad1 = 0u;
        r.pig0 = pack_pig_a(pig);
        r.pig1 = pack_pig_b(pig);
        scraps[slot] = r;
    }
}
