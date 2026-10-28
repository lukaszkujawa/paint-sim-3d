const CELL_WORDS: u32 = 53u;
const TILE_CELLS: u32 = 256u;

struct JournalParams {
    gen: u32,
    cap: u32,
    target_gen: u32,
    pad: u32,
}

@group(2) @binding(0) var<uniform> J: JournalParams;
@group(2) @binding(1) var<storage, read_write> jmeta: array<vec4<u32>>;
@group(2) @binding(2) var<storage, read_write> jdata: array<u32>;
@group(2) @binding(3) var<storage, read_write> jhead: array<atomic<u32>>;
@group(2) @binding(4) var<storage, read> touch: array<u32>;
@group(2) @binding(5) var<storage, read_write> jmark: array<u32>;
@group(2) @binding(6) var<storage, read_write> best: array<atomic<u32>>;

var<workgroup> wg_slot: u32;

fn tile_busy(t: u32) -> bool {
    let tw = tiles16();
    let tx = i32(t % tw.x);
    let ty = i32(t / tw.x);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let x = tx + dx;
            let y = ty + dy;
            if x < 0 || y < 0 || x >= i32(tw.x) || y >= i32(tw.y) { continue; }
            if (tile_flags[u32(x) + u32(y) * tw.x] & 32u) != 0u { return true; }
        }
    }
    return false;
}

fn save_cell(c: u32, base: u32) {
    jdata[base] = bitcast<u32>(h[c]);
    jdata[base + 1u] = bitcast<u32>(m[c]);
    let r = rheo[c];
    jdata[base + 2u] = bitcast<u32>(r.x);
    jdata[base + 3u] = bitcast<u32>(r.y);
    let sh = strata_h[c];
    for (var k = 0u; k < 4u; k++) { jdata[base + 4u + k] = bitcast<u32>(sh[k]); }
    for (var k = 0u; k < 8u; k++) {
        let v = strata_pig[c * 8u + k];
        for (var e = 0u; e < 4u; e++) { jdata[base + 8u + k * 4u + e] = v[e]; }
    }
    let cu = strata_cure[c];
    for (var k = 0u; k < 4u; k++) { jdata[base + 40u + k] = bitcast<u32>(cu[k]); }
    for (var k = 0u; k < 8u; k++) { jdata[base + 44u + k] = sub_refl[c * 8u + k]; }
    jdata[base + 52u] = lay[c];
}

fn load_cell(c: u32, base: u32) {
    h[c] = bitcast<f32>(jdata[base]);
    m[c] = bitcast<f32>(jdata[base + 1u]);
    rheo[c] = vec2<f32>(bitcast<f32>(jdata[base + 2u]), bitcast<f32>(jdata[base + 3u]));
    var sh = vec4<f32>(0.0);
    for (var k = 0u; k < 4u; k++) { sh[k] = bitcast<f32>(jdata[base + 4u + k]); }
    strata_h[c] = sh;
    for (var k = 0u; k < 8u; k++) {
        var v = vec4<u32>(0u);
        for (var e = 0u; e < 4u; e++) { v[e] = jdata[base + 8u + k * 4u + e]; }
        strata_pig[c * 8u + k] = v;
    }
    var cu = vec4<f32>(0.0);
    for (var k = 0u; k < 4u; k++) { cu[k] = bitcast<f32>(jdata[base + 40u + k]); }
    strata_cure[c] = cu;
    for (var k = 0u; k < 8u; k++) { sub_refl[c * 8u + k] = jdata[base + 44u + k]; }
    lay[c] = jdata[base + 52u];
}

fn slot_valid(s: u32) -> bool {
    let mt = jmeta[s];
    if mt.w != 1u { return false; }
    let head = atomicLoad(&jhead[0]);
    return mt.z < head && head - mt.z <= J.cap;
}

@compute @workgroup_size(16, 16)
fn journal(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let tw = tiles16();
    let t = wid.x + wid.y * tw.x;
    if li == 0u {
        wg_slot = 0xffffffffu;
        if J.gen != 0u && jmark[t] != J.gen && (touch[t] != 0u || tile_busy(t)) {
            let seq = atomicAdd(&jhead[0], 1u);
            let slot = seq % J.cap;
            jmeta[slot] = vec4<u32>(t, J.gen, seq, 1u);
            jmark[t] = J.gen;
            wg_slot = slot;
        }
    }
    let slot = workgroupUniformLoad(&wg_slot);
    if slot == 0xffffffffu { return; }
    let i = wid.x * 16u + lid.x;
    let j = wid.y * 16u + lid.y;
    if i >= P.size.x || j >= P.size.y { return; }
    save_cell(idx(i, j), (slot * TILE_CELLS + li) * CELL_WORDS);
}

@compute @workgroup_size(64)
fn restore_scan(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = id.x;
    if s >= J.cap || !slot_valid(s) { return; }
    let mt = jmeta[s];
    if mt.y < J.target_gen { return; }
    atomicMin(&best[mt.x], mt.z);
}

@compute @workgroup_size(16, 16)
fn restore_apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let s = wid.x + wid.y * 65535u;
    if s >= J.cap || !slot_valid(s) { return; }
    let mt = jmeta[s];
    if mt.y < J.target_gen || atomicLoad(&best[mt.x]) != mt.z { return; }
    let tw = tiles16();
    let i = (mt.x % tw.x) * 16u + lid.x;
    let j = (mt.x / tw.x) * 16u + lid.y;
    if li == 0u { tile_moved[mt.x] = 1u; }
    if i >= P.size.x || j >= P.size.y { return; }
    load_cell(idx(i, j), (s * TILE_CELLS + li) * CELL_WORDS);
}
