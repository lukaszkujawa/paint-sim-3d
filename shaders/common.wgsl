struct Params {
    size: vec2<u32>,
    tiles: vec2<u32>,
    dx: f32,
    dt: f32,
    inv_dx: f32,
    frame_dt: f32,
    g: vec4<f32>,
    phys: vec4<f32>,
    ein: vec4<f32>,
    cure: vec4<f32>,
    misc: vec4<f32>,
    flags: vec4<u32>,
    front: vec4<f32>,
    front2: vec4<f32>,
    front3: vec4<f32>,
}

struct Deposit {
    p0: vec2<f32>,
    p1: vec2<f32>,
    n0: vec2<f32>,
    n1: vec2<f32>,
    r0: f32,
    r1: f32,
    h0: f32,
    vol: f32,
    kind: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    pig0: vec4<u32>,
    pig1: vec4<u32>,
}

const DEPOSIT_CAP_START: u32 = 2u;
const DEPOSIT_CAP_END: u32 = 4u;
const DEPOSIT_METHOD_SHIFT: u32 = 8u;
const METHOD_THREAD: u32 = 1u;
const METHOD_DROP: u32 = 2u;
const METHOD_POUR: u32 = 3u;
const METHOD_BRUSH: u32 = 4u;
const METHOD_KNIFE: u32 = 5u;

const STRATA: u32 = 4u;
const NPIG: u32 = 12u;
const NBAND: u32 = 16u;
const PIG_STRIDE: u32 = 40u;
const TILE: u32 = 64u;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> rheo: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> pres: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> flux: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> acc: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(8) var<storage, read_write> delta: array<f32>;
@group(0) @binding(9) var<storage, read_write> tmp: array<f32>;
@group(0) @binding(10) var<storage, read_write> acc_s: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> strata_h: array<vec4<f32>>;

@group(1) @binding(0) var<storage, read_write> strata_pig: array<vec4<u32>>;
@group(1) @binding(1) var<storage, read_write> strata_cure: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> sub_refl: array<u32>;
@group(1) @binding(3) var<storage, read> pigtab: array<f32>;
@group(1) @binding(4) var<storage, read> cmf: array<f32>;
@group(1) @binding(5) var<storage, read_write> strata_h2: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> strata_pig2: array<vec4<u32>>;
@group(1) @binding(7) var<storage, read_write> alpha_s: array<vec4<f32>>;

@group(3) @binding(0) var<storage, read_write> tile_flags: array<u32>;
@group(3) @binding(1) var<storage, read_write> tile_list: array<u32>;
@group(3) @binding(2) var<storage, read_write> active_count: array<atomic<u32>>;
@group(3) @binding(3) var<storage, read_write> tile_moved: array<u32>;
@group(3) @binding(4) var<storage, read_write> lay: array<u32>;

fn lay_pack(method: u32, dir: vec2<f32>) -> u32 {
    let turn = atan2(dir.y, dir.x) / 6.2831853 + 0.5;
    return (method & 0xffu) | (u32(clamp(turn, 0.0, 1.0) * 65535.0) << 8u);
}

fn lay_method(v: u32) -> u32 {
    return v & 0xffu;
}

fn lay_dir(v: u32) -> vec2<f32> {
    let ang = (f32((v >> 8u) & 0xffffu) / 65535.0 - 0.5) * 6.2831853;
    return vec2<f32>(cos(ang), sin(ang));
}

const MOVED: u32 = 64u;
const W_AXIS: f32 = 0.6666667;
const W_DIAG_FLUX: f32 = 0.23570226;
const W_DIAG_LAP: f32 = 0.16666667;
const MOVE_EPS: f32 = 1e-6;
const FRAME_EPS: f32 = 2e-6;

fn tile16_of(i: u32, j: u32) -> u32 {
    return (i / 16u) + (j / 16u) * tiles16().x;
}

fn mark_moved(i: u32, j: u32) {
    tile_moved[tile16_of(i, j)] = 1u;
}

var<workgroup> wg_moved: atomic<u32>;

fn begin_motion(li: u32) {
    if li == 0u { atomicStore(&wg_moved, 0u); }
    workgroupBarrier();
}

fn end_motion(moved: bool, wid: vec3<u32>, li: u32) {
    if moved { atomicStore(&wg_moved, 1u); }
    workgroupBarrier();
    if li == 0u && atomicLoad(&wg_moved) != 0u { tile_moved[tile_list[wid.x]] = 1u; }
}

fn tiles16() -> vec2<u32> {
    return vec2<u32>((P.size.x + 15u) / 16u, (P.size.y + 15u) / 16u);
}

fn cell_active(i: u32, j: u32) -> bool {
    let t = (i / 16u) + (j / 16u) * tiles16().x;
    return (tile_flags[t] & 32u) != 0u;
}

fn tile_cell(wid: vec3<u32>, lid: vec3<u32>) -> vec3<u32> {
    let t = tile_list[wid.x];
    let tw = tiles16().x;
    let tx = t % tw;
    let ty = t / tw;
    return vec3<u32>(tx * 16u + lid.x, ty * 16u + lid.y, 0u);
}

fn idx(i: u32, j: u32) -> u32 {
    return j * P.size.x + i;
}

fn cell_of(id: vec3<u32>) -> vec2<u32> {
    return id.xy;
}

fn in_grid(id: vec3<u32>) -> bool {
    return id.x < P.size.x && id.y < P.size.y;
}

fn refl_i(i: i32) -> u32 {
    let w = i32(P.size.x);
    var r = i;
    if r < 0 { r = -r - 1; }
    if r >= w { r = 2 * w - r - 1; }
    return u32(clamp(r, 0, w - 1));
}

fn refl_j(j: i32) -> u32 {
    let hh = i32(P.size.y);
    var r = j;
    if r < 0 { r = -r - 1; }
    if r >= hh { r = 2 * hh - r - 1; }
    return u32(clamp(r, 0, hh - 1));
}

fn surf(i: i32, j: i32) -> f32 {
    let c = idx(refl_i(i), refl_j(j));
    return h[c] + m[c];
}

fn unpack_pig(a: vec4<u32>, b: vec4<u32>) -> array<f32, 16> {
    var o: array<f32, 16>;
    let a0 = unpack2x16float(a.x); let a1 = unpack2x16float(a.y); let a2 = unpack2x16float(a.z); let a3 = unpack2x16float(a.w);
    let b0 = unpack2x16float(b.x); let b1 = unpack2x16float(b.y); let b2 = unpack2x16float(b.z); let b3 = unpack2x16float(b.w);
    o[0] = a0.x; o[1] = a0.y; o[2] = a1.x; o[3] = a1.y; o[4] = a2.x; o[5] = a2.y; o[6] = a3.x; o[7] = a3.y;
    o[8] = b0.x; o[9] = b0.y; o[10] = b1.x; o[11] = b1.y; o[12] = b2.x; o[13] = b2.y; o[14] = b3.x; o[15] = b3.y;
    return o;
}

fn pack_pig_a(o: array<f32, 16>) -> vec4<u32> {
    return vec4<u32>(pack2x16float(vec2<f32>(o[0], o[1])), pack2x16float(vec2<f32>(o[2], o[3])), pack2x16float(vec2<f32>(o[4], o[5])), pack2x16float(vec2<f32>(o[6], o[7])));
}

fn pack_pig_b(o: array<f32, 16>) -> vec4<u32> {
    return vec4<u32>(pack2x16float(vec2<f32>(o[8], o[9])), pack2x16float(vec2<f32>(o[10], o[11])), pack2x16float(vec2<f32>(o[12], o[13])), pack2x16float(vec2<f32>(o[14], o[15])));
}

fn disjoining(hv: f32) -> f32 {
    let hs = P.cure.w;
    let a = P.ein.z;
    let x = max(1.0 - hv / hs, 0.0);
    return a * x * x;
}
