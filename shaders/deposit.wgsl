@group(2) @binding(0) var<storage, read> deposits: array<Deposit>;
@group(2) @binding(1) var<storage, read> tile_start: array<u32>;
@group(2) @binding(2) var<storage, read> tile_count: array<u32>;
@group(2) @binding(3) var<storage, read> tile_items: array<u32>;

fn seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, open_start: bool) -> vec2<f32> {
    let ab = b - a;
    let l2 = dot(ab, ab);
    var t = 0.0;
    if l2 > 0.0 {
        let tr = dot(p - a, ab) / l2;
        if open_start && (tr < 0.0 || tr > 1.0) { return vec2<f32>(1e9, 0.0); }
        t = clamp(tr, 0.0, 1.0);
    }
    return vec2<f32>(length(p - (a + ab * t)), t);
}

@compute @workgroup_size(16, 16)
fn deposit(@builtin(global_invocation_id) id: vec3<u32>) {
    if !in_grid(id) { return; }
    let tile = (id.x / TILE) + (id.y / TILE) * P.tiles.x;
    let n = tile_count[tile];
    if n == 0u { return; }
    let start = tile_start[tile];
    let c = idx(id.x, id.y);
    let p = (vec2<f32>(id.xy) + vec2<f32>(0.5)) * P.dx;
    var s = stack_load(c);
    var added = 0.0;
    for (var q = 0u; q < n; q++) {
        let d = deposits[tile_items[start + q]];
        var r1 = d.r;
        if d.pad != 0u { r1 = bitcast<f32>(d.pad); }
        var add = 0.0;
        for (var sy = 0u; sy < 4u; sy++) {
            for (var sx = 0u; sx < 4u; sx++) {
                let ps = p + (vec2<f32>(f32(sx), f32(sy)) - vec2<f32>(1.5)) * (0.25 * P.dx);
                let dt = seg_dist(ps, d.p0, d.p1, (d.kind & 2u) != 0u);
                let rt = mix(d.r, r1, dt.y);
                if dt.x < rt {
                    let x = dt.x / rt;
                    add += 0.0625 * d.h0 * (rt / d.r) * (1.0 - x * x);
                }
            }
        }
        if add > 0.0 {
            let pig = unpack_pig(d.pig0, d.pig1);
            stack_push_top(&s, add, pig, 0.0);
            added += add;
        }
    }
    if added > 0.0 {
        h[c] = h[c] + added;
        stack_compact(&s);
        stack_store(c, s);
    }
}
