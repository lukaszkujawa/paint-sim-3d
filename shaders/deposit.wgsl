@group(2) @binding(0) var<storage, read> deposits: array<Deposit>;
@group(2) @binding(1) var<storage, read> tile_start: array<u32>;
@group(2) @binding(2) var<storage, read> tile_count: array<u32>;
@group(2) @binding(3) var<storage, read> tile_items: array<u32>;

fn ribbon_height(p: vec2<f32>, d: Deposit) -> f32 {
    let ab = d.p1 - d.p0;
    let l2 = dot(ab, ab);
    if l2 <= 0.0 {
        let x = length(p - d.p0) / d.r0;
        return select(0.0, d.h0 * (1.0 - x * x), x < 1.0);
    }
    let l = sqrt(l2);
    let dir = ab / l;
    let q0 = p - d.p0;
    let q1 = p - d.p1;
    let cap_start = (d.kind & DEPOSIT_CAP_START) != 0u;
    let cap_end = (d.kind & DEPOSIT_CAP_END) != 0u;
    if !cap_start && dot(q0, d.n0) < 0.0 { return 0.0; }
    if !cap_end && dot(q1, d.n1) >= 0.0 { return 0.0; }
    let u = dot(q0, dir);
    if u < -d.r0 || u > l + d.r1 { return 0.0; }
    var dist = abs(q0.x * dir.y - q0.y * dir.x);
    if cap_start && u < 0.0 { dist = length(q0); }
    if cap_end && u > l { dist = length(q1); }
    let rt = mix(d.r0, d.r1, clamp(u / l, 0.0, 1.0));
    let x = dist / rt;
    return select(0.0, d.h0 * (rt / d.r0) * (1.0 - x * x), x < 1.0);
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
    var top_add = 0.0;
    var top_lay = 0u;
    for (var q = 0u; q < n; q++) {
        let d = deposits[tile_items[start + q]];
        var add = 0.0;
        for (var sy = 0u; sy < 4u; sy++) {
            for (var sx = 0u; sx < 4u; sx++) {
                let ps = p + (vec2<f32>(f32(sx), f32(sy)) - vec2<f32>(1.5)) * (0.25 * P.dx);
                add += 0.0625 * ribbon_height(ps, d);
            }
        }
        if add > 0.0 {
            let pig = unpack_pig(d.pig0, d.pig1);
            stack_push_top(&s, add, pig, 0.0);
            added += add;
            if add > top_add {
                top_add = add;
                top_lay = lay_pack(d.kind >> DEPOSIT_METHOD_SHIFT, normalize(d.p1 - d.p0 + vec2<f32>(1e-9, 0.0)));
            }
        }
    }
    if added > 0.0 {
        h[c] = h[c] + added;
        lay[c] = top_lay;
        mark_moved(id.x, id.y);
        stack_compact(&s);
        stack_store(c, s);
    }
}
