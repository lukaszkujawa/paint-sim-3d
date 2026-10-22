struct Contact {
    p0: vec2<f32>,
    p1: vec2<f32>,
    r: f32,
    zb: f32,
    bristle: u32,
    pad: u32,
}

@group(2) @binding(0) var<storage, read> contacts: array<Contact>;
@group(2) @binding(1) var<storage, read> ctile_start: array<u32>;
@group(2) @binding(2) var<storage, read> ctile_count: array<u32>;
@group(2) @binding(3) var<storage, read> ctile_items: array<u32>;
@group(2) @binding(4) var<storage, read_write> reservoir: array<atomic<u32>>;

const RES_UNIT: f32 = 1e15;
const LEAVE_FILM: f32 = 8e-5;

fn seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let l2 = dot(ab, ab);
    var t = 0.0;
    if l2 > 0.0 {
        t = clamp(dot(p - a, ab) / l2, 0.0, 1.0);
    }
    return length(p - (a + ab * t));
}

@compute @workgroup_size(16, 16)
fn brush_scrape(@builtin(global_invocation_id) id: vec3<u32>) {
    if !in_grid(id) { return; }
    let tile = (id.x / TILE) + (id.y / TILE) * P.tiles.x;
    let n = ctile_count[tile];
    if n == 0u { return; }
    let c = idx(id.x, id.y);
    let hv = h[c];
    if hv <= LEAVE_FILM { return; }
    let start = ctile_start[tile];
    let p = (vec2<f32>(id.xy) + vec2<f32>(0.5)) * P.dx;
    var cap = 1e9;
    var who = 0xffffffffu;
    for (var q = 0u; q < n; q++) {
        let k = contacts[ctile_items[start + q]];
        if seg_dist(p, k.p0, k.p1) < k.r {
            let cc = max(k.zb - m[c], LEAVE_FILM);
            if cc < cap {
                cap = cc;
                who = k.bristle;
            }
        }
    }
    if who == 0xffffffffu || hv <= cap { return; }
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
    let vol = removed * P.dx * P.dx;
    let base = who * 17u;
    atomicAdd(&reservoir[base], u32(vol * RES_UNIT + 0.5));
    for (var i = 0u; i < 16u; i++) {
        var sc = 1.0;
        if i >= 13u { sc = 0.01; }
        atomicAdd(&reservoir[base + 1u + i], u32(max(vol * pig[i] * sc, 0.0) * RES_UNIT + 0.5));
    }
}
