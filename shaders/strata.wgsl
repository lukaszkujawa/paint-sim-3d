struct Stack {
    th: array<f32, 4>,
    cu: array<f32, 4>,
    pg: array<array<f32, 16>, 4>,
}

fn stack_load(c: u32) -> Stack {
    var s: Stack;
    let t = strata_h[c];
    let u = strata_cure[c];
    s.th[0] = t.x; s.th[1] = t.y; s.th[2] = t.z; s.th[3] = t.w;
    s.cu[0] = u.x; s.cu[1] = u.y; s.cu[2] = u.z; s.cu[3] = u.w;
    for (var k = 0u; k < 4u; k++) {
        s.pg[k] = unpack_pig(strata_pig[(c * 4u + k) * 2u], strata_pig[(c * 4u + k) * 2u + 1u]);
    }
    return s;
}

fn stack_store2(c: u32, s: Stack) {
    strata_h2[c] = vec4<f32>(s.th[0], s.th[1], s.th[2], s.th[3]);
    strata_cure[c] = vec4<f32>(s.cu[0], s.cu[1], s.cu[2], s.cu[3]);
    for (var k = 0u; k < 4u; k++) {
        strata_pig2[(c * 4u + k) * 2u] = pack_pig_a(s.pg[k]);
        strata_pig2[(c * 4u + k) * 2u + 1u] = pack_pig_b(s.pg[k]);
    }
}

fn stack_store(c: u32, s: Stack) {
    strata_h[c] = vec4<f32>(s.th[0], s.th[1], s.th[2], s.th[3]);
    strata_cure[c] = vec4<f32>(s.cu[0], s.cu[1], s.cu[2], s.cu[3]);
    for (var k = 0u; k < 4u; k++) {
        strata_pig[(c * 4u + k) * 2u] = pack_pig_a(s.pg[k]);
        strata_pig[(c * 4u + k) * 2u + 1u] = pack_pig_b(s.pg[k]);
    }
}

fn pig_dist(a: array<f32, 16>, b: array<f32, 16>) -> f32 {
    var d = 0.0;
    for (var i = 0u; i < 13u; i++) {
        d += abs(a[i] - b[i]);
    }
    return d;
}

fn pig_mix(a: array<f32, 16>, wa: f32, b: array<f32, 16>, wb: f32) -> array<f32, 16> {
    var o: array<f32, 16>;
    let w = max(wa + wb, 1e-20);
    for (var i = 0u; i < 16u; i++) {
        o[i] = (a[i] * wa + b[i] * wb) / w;
    }
    let va = select(0.0, wa, a[14] > 0.0);
    let vb = select(0.0, wb, b[14] > 0.0);
    if va + vb > 0.0 {
        o[14] = (a[14] * va + b[14] * vb) / (va + vb);
    } else {
        o[14] = 0.0;
    }
    return o;
}

fn stack_count(s: Stack) -> u32 {
    var n = 0u;
    for (var k = 0u; k < 4u; k++) {
        if s.th[k] > 0.0 { n = k + 1u; }
    }
    return n;
}

fn stack_merge(sp: ptr<function, Stack>, a: u32) {
    let b = a + 1u;
    let ta = (*sp).th[a];
    let tb = (*sp).th[b];
    (*sp).pg[a] = pig_mix((*sp).pg[a], ta, (*sp).pg[b], tb);
    (*sp).cu[a] = ((*sp).cu[a] * ta + (*sp).cu[b] * tb) / max(ta + tb, 1e-20);
    (*sp).th[a] = ta + tb;
    for (var k = b; k < 3u; k++) {
        (*sp).th[k] = (*sp).th[k + 1u];
        (*sp).cu[k] = (*sp).cu[k + 1u];
        (*sp).pg[k] = (*sp).pg[k + 1u];
    }
    (*sp).th[3] = 0.0;
    (*sp).cu[3] = 0.0;
}

fn stack_compact(sp: ptr<function, Stack>) {
    for (var sweep = 0u; sweep < 4u; sweep++) {
        for (var k = 0u; k < 3u; k++) {
            if (*sp).th[k] <= 0.0 && (*sp).th[k + 1u] > 0.0 {
                (*sp).th[k] = (*sp).th[k + 1u];
                (*sp).cu[k] = (*sp).cu[k + 1u];
                (*sp).pg[k] = (*sp).pg[k + 1u];
                (*sp).th[k + 1u] = 0.0;
                (*sp).cu[k + 1u] = 0.0;
            }
        }
    }
    let tiny = P.misc.x;
    for (var k = 0u; k < 3u; k++) {
        if (*sp).th[k] > 0.0 && (*sp).th[k] < tiny && (*sp).th[k + 1u] > 0.0 {
            stack_merge(sp, k);
        }
    }
}

fn stack_push_top(sp: ptr<function, Stack>, vol: f32, pig: array<f32, 16>, cure: f32) {
    if vol <= 0.0 { return; }
    let n = stack_count(*sp);
    if n > 0u {
        let t = n - 1u;
        let close = pig_dist((*sp).pg[t], pig) < P.misc.y || (*sp).th[t] < P.misc.x;
        if close {
            (*sp).pg[t] = pig_mix((*sp).pg[t], (*sp).th[t], pig, vol);
            (*sp).cu[t] = ((*sp).cu[t] * (*sp).th[t] + cure * vol) / ((*sp).th[t] + vol);
            (*sp).th[t] += vol;
            return;
        }
    }
    if n == 4u {
        var best = 0u;
        var bd = 1e30;
        for (var k = 0u; k < 3u; k++) {
            let d = pig_dist((*sp).pg[k], (*sp).pg[k + 1u]);
            if d < bd { bd = d; best = k; }
        }
        stack_merge(sp, best);
    }
    let t = stack_count(*sp);
    (*sp).th[t] = vol;
    (*sp).pg[t] = pig;
    (*sp).cu[t] = cure;
}

fn stack_add_band(sp: ptr<function, Stack>, zlo: f32, zhi: f32, vol: f32, pig: array<f32, 16>, cure: f32) {
    if vol <= 0.0 { return; }
    let span = max(zhi - zlo, 1e-12);
    var z0 = 0.0;
    var above = vol;
    for (var k = 0u; k < 4u; k++) {
        let tk = (*sp).th[k];
        if tk <= 0.0 { break; }
        let z1 = z0 + tk;
        let ov = max(min(zhi, z1) - max(zlo, z0), 0.0);
        let part = vol * ov / span;
        if part > 0.0 {
            (*sp).pg[k] = pig_mix((*sp).pg[k], tk, pig, part);
            (*sp).cu[k] = ((*sp).cu[k] * tk + cure * part) / (tk + part);
            (*sp).th[k] = tk + part;
            above -= part;
        }
        z0 = z1;
    }
    if above > 1e-14 {
        stack_push_top(sp, above, pig, cure);
    }
}

@compute @workgroup_size(16, 16)
fn strata_limit(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    var out = max(acc_s[4u * c], vec4<f32>(0.0)) + max(acc_s[4u * c + 1u], vec4<f32>(0.0)) + max(acc_s[4u * c + 2u], vec4<f32>(0.0)) + max(acc_s[4u * c + 3u], vec4<f32>(0.0));
    if id.x > 0u && cell_active(id.x - 1u, id.y) { out += max(-acc_s[4u * idx(id.x - 1u, id.y)], vec4<f32>(0.0)); }
    if id.y > 0u && cell_active(id.x, id.y - 1u) { out += max(-acc_s[4u * idx(id.x, id.y - 1u) + 1u], vec4<f32>(0.0)); }
    if id.x > 0u && id.y > 0u && cell_active(id.x - 1u, id.y - 1u) { out += max(-acc_s[4u * idx(id.x - 1u, id.y - 1u) + 2u], vec4<f32>(0.0)); }
    if id.x > 0u && id.y + 1u < P.size.y && cell_active(id.x - 1u, id.y + 1u) { out += max(-acc_s[4u * idx(id.x - 1u, id.y + 1u) + 3u], vec4<f32>(0.0)); }
    let th = strata_h[c];
    var a = vec4<f32>(1.0);
    for (var k = 0u; k < 4u; k++) {
        if out[k] > 0.0 { a[k] = min(1.0, th[k] * P.dx / out[k]); }
    }
    alpha_s[c] = a;
}

fn inflow_from(sp: ptr<function, Stack>, cn: u32, f: vec4<f32>) {
    let an = alpha_s[cn];
    let thn = strata_h[cn];
    var z0 = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let z1 = z0 + thn[k];
        let v = f[k] * an[k] * P.inv_dx;
        if v > 0.0 && thn[k] > 0.0 {
            let pig = unpack_pig(strata_pig[(cn * 4u + k) * 2u], strata_pig[(cn * 4u + k) * 2u + 1u]);
            stack_add_band(sp, z0, z1, v, pig, strata_cure[cn][k]);
        }
        z0 = z1;
    }
}

@compute @workgroup_size(16, 16)
fn strata_update(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    var s = stack_load(c);
    let a = alpha_s[c];
    let fr = acc_s[4u * c];
    let fu = acc_s[4u * c + 1u];
    let fdg = acc_s[4u * c + 2u];
    let feg = acc_s[4u * c + 3u];
    var fl = vec4<f32>(0.0);
    var fd = vec4<f32>(0.0);
    var fll = vec4<f32>(0.0);
    var flu = vec4<f32>(0.0);
    var cl = c;
    var cd = c;
    var cll = c;
    var clu = c;
    if id.x > 0u && cell_active(id.x - 1u, id.y) { cl = idx(id.x - 1u, id.y); fl = acc_s[4u * cl]; }
    if id.y > 0u && cell_active(id.x, id.y - 1u) { cd = idx(id.x, id.y - 1u); fd = acc_s[4u * cd + 1u]; }
    if id.x > 0u && id.y > 0u && cell_active(id.x - 1u, id.y - 1u) { cll = idx(id.x - 1u, id.y - 1u); fll = acc_s[4u * cll + 2u]; }
    if id.x > 0u && id.y + 1u < P.size.y && cell_active(id.x - 1u, id.y + 1u) { clu = idx(id.x - 1u, id.y + 1u); flu = acc_s[4u * clu + 3u]; }
    let out = max(fr, vec4<f32>(0.0)) + max(fu, vec4<f32>(0.0)) + max(fdg, vec4<f32>(0.0)) + max(feg, vec4<f32>(0.0)) + max(-fl, vec4<f32>(0.0)) + max(-fd, vec4<f32>(0.0)) + max(-fll, vec4<f32>(0.0)) + max(-flu, vec4<f32>(0.0));
    for (var k = 0u; k < 4u; k++) {
        s.th[k] = max(s.th[k] - out[k] * a[k] * P.inv_dx, 0.0);
    }
    if id.x + 1u < P.size.x {
        let cr = idx(id.x + 1u, id.y);
        inflow_from(&s, cr, -fr);
    }
    if id.y + 1u < P.size.y {
        let cu = idx(id.x, id.y + 1u);
        inflow_from(&s, cu, -fu);
    }
    if id.x + 1u < P.size.x && id.y + 1u < P.size.y {
        let cq = idx(id.x + 1u, id.y + 1u);
        inflow_from(&s, cq, -fdg);
    }
    if id.x + 1u < P.size.x && id.y > 0u {
        let cq = idx(id.x + 1u, id.y - 1u);
        inflow_from(&s, cq, -feg);
    }
    if cl != c { inflow_from(&s, cl, fl); }
    if cd != c { inflow_from(&s, cd, fd); }
    if cll != c { inflow_from(&s, cll, fll); }
    if clu != c { inflow_from(&s, clu, flu); }
    let hc = h[c];
    var sum = s.th[0] + s.th[1] + s.th[2] + s.th[3];
    if hc <= 0.0 {
        for (var k = 0u; k < 4u; k++) { s.th[k] = 0.0; s.cu[k] = 0.0; }
    } else if sum > 0.0 {
        let k = hc / sum;
        for (var q = 0u; q < 4u; q++) { s.th[q] *= k; }
    }
    stack_compact(&s);
    stack_store2(c, s);
}

@compute @workgroup_size(16, 16)
fn strata_copy_back(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    strata_h[c] = strata_h2[c];
    for (var k = 0u; k < 8u; k++) {
        strata_pig[c * 8u + k] = strata_pig2[c * 8u + k];
    }
}

@compute @workgroup_size(16, 16)
fn strata_fill(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    let hc = h[c];
    if hc <= 0.0 { return; }
    let th = strata_h[c];
    if th.x + th.y + th.z + th.w > 0.0 { return; }
    var src = c;
    var best = 0.0;
    var cands: array<u32, 4>;
    var nc = 0u;
    if id.x > 0u { cands[nc] = idx(id.x - 1u, id.y); nc++; }
    if id.y > 0u { cands[nc] = idx(id.x, id.y - 1u); nc++; }
    if id.x + 1u < P.size.x { cands[nc] = idx(id.x + 1u, id.y); nc++; }
    if id.y + 1u < P.size.y { cands[nc] = idx(id.x, id.y + 1u); nc++; }
    for (var q = 0u; q < nc; q++) {
        let cn = cands[q];
        let tn = strata_h[cn];
        let tsum = tn.x + tn.y + tn.z + tn.w;
        if tsum > best { best = tsum; src = cn; }
    }
    if src == c { return; }
    var ks = 0u;
    let tn = strata_h[src];
    for (var k = 0u; k < 4u; k++) { if tn[k] > 0.0 { ks = k; } }
    strata_h[c] = vec4<f32>(hc, 0.0, 0.0, 0.0);
    strata_pig[c * 8u] = strata_pig[(src * 4u + ks) * 2u];
    strata_pig[c * 8u + 1u] = strata_pig[(src * 4u + ks) * 2u + 1u];
    for (var k = 1u; k < 4u; k++) {
        strata_pig[c * 8u + 2u * k] = vec4<u32>(0u);
        strata_pig[c * 8u + 2u * k + 1u] = vec4<u32>(0u);
    }
    strata_cure[c] = vec4<f32>(strata_cure[src][ks], 0.0, 0.0, 0.0);
}
