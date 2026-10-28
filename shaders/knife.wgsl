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
    fill: f32,
    pad2: u32,
    pad3: u32,
    pad4: u32,
    pig0: vec4<u32>,
    pig1: vec4<u32>,
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
    let level = min(K.fill, cap);
    if hv <= cap && hv >= level { return; }
    let p = (vec2<f32>(id.xy) + vec2<f32>(0.5)) * P.dx;
    let ab = K.b - K.a;
    let mv = K.dir * K.sweep;
    let det = ab.x * mv.y - ab.y * mv.x;
    if abs(det) < 1e-12 { return; }
    let q = p - K.a;
    let t = (q.x * mv.y - q.y * mv.x) / det;
    let u = -(ab.x * q.y - ab.y * q.x) / det;
    if t < 0.0 || t > 1.0 { return; }
    if u < -P.dx / max(K.sweep, 1e-9) || u > 1.0 { return; }
    let lay_dir = normalize(mv + vec2<f32>(1e-9, 0.0));
    var s = stack_load(c);
    if hv < level {
        let add = level - hv;
        stack_push_top(&s, add, unpack_pig(K.pig0, K.pig1), 0.0);
        h[c] = level;
        lay[c] = lay_pack(METHOD_KNIFE, lay_dir);
        mark_moved(id.x, id.y);
        stack_compact(&s);
        stack_store(c, s);
        let slot = atomicAdd(&scrap_count, 1u);
        if slot < arrayLength(&scraps) {
            var r: Scrap;
            r.cell = c;
            r.vol = -add * P.dx * P.dx;
            r.pad0 = 0u;
            r.pad1 = 0u;
            r.pig0 = vec4<u32>(0u);
            r.pig1 = vec4<u32>(0u);
            scraps[slot] = r;
        }
        return;
    }
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
    lay[c] = lay_pack(METHOD_KNIFE, lay_dir);
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
