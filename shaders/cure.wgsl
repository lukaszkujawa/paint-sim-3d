const OIL_SHRINK: f32 = 0.04;

fn composition_rheo(s: Stack) -> vec2<f32> {
    var total = 0.0;
    var lnmu = 0.0;
    var tau = 0.0;
    var thin = 0.0;
    var wsum = 0.0;
    var cure = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let t = s.th[k];
        if t <= 0.0 { continue; }
        total += t;
        cure += t * s.cu[k];
        var csum = 0.0;
        var lm = 0.0;
        var ty = 0.0;
        for (var i = 0u; i < NPIG; i++) {
            let c = s.pg[k][i];
            if c > 0.0 {
                csum += c;
                lm += c * log(max(pigtab[i * PIG_STRIDE + 32u], 1e-3));
                ty += c * s.pg[k][13];
            }
        }
        if s.pg[k][14] > 0.0 {
            lm = csum * (s.pg[k][14] - 10.0);
        }
        if csum > 0.0 {
            lnmu += t * lm / csum;
            tau += t * ty / csum;
            wsum += t;
        }
        thin += t * s.pg[k][12];
    }
    if total <= 0.0 { return vec2<f32>(1.0, 0.0); }
    var mu = log(P.misc.z);
    var ty = 0.0;
    if wsum > 0.0 {
        mu = lnmu / wsum;
        ty = tau / wsum;
    }
    let tf = clamp(thin / total, 0.0, 0.95);
    let cb = cure / total;
    mu += 3.0 * log(1.0 - tf) + P.cure.z * cb * cb;
    ty = ty * pow(1.0 - tf, 4.0) + P.cure.y * cb * cb;
    return vec2<f32>(exp(mu), ty);
}

fn layer_lnmu(a: vec4<u32>, b: vec4<u32>) -> f32 {
    let p = unpack_pig(a, b);
    var lm = 0.0;
    for (var i = 0u; i < NPIG; i++) {
        let c = p[i];
        if c > 0.0 {
            lm += c * log(max(pigtab[i * PIG_STRIDE + 32u], 1e-3));
        }
    }
    return lm;
}

fn packed_rheo(c: u32, t: vec4<f32>, u: vec4<f32>) -> vec2<f32> {
    var total = 0.0;
    var lnmu = 0.0;
    var tau = 0.0;
    var thin = 0.0;
    var wsum = 0.0;
    var cure = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let tk = t[k];
        if tk <= 0.0 { continue; }
        total += tk;
        cure += tk * u[k];
        let a = strata_pig[(c * 4u + k) * 2u];
        let b = strata_pig[(c * 4u + k) * 2u + 1u];
        let v0 = unpack2x16float(a.x);
        let v1 = unpack2x16float(a.y);
        let v2 = unpack2x16float(a.z);
        let v3 = unpack2x16float(a.w);
        let v4 = unpack2x16float(b.x);
        let v5 = unpack2x16float(b.y);
        let ex = unpack2x16float(b.z);
        let mu14 = unpack2x16float(b.w).x;
        let pc = array<f32, 12>(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, v3.x, v3.y, v4.x, v4.y, v5.x, v5.y);
        var csum = 0.0;
        var ty = 0.0;
        for (var i = 0u; i < NPIG; i++) {
            let cc = pc[i];
            if cc > 0.0 {
                csum += cc;
                ty += cc * ex.y;
            }
        }
        if csum > 0.0 {
            if mu14 > 0.0 {
                lnmu += tk * (csum * (mu14 - 10.0)) / csum;
            } else {
                lnmu += tk * layer_lnmu(a, b) / csum;
            }
            tau += tk * ty / csum;
            wsum += tk;
        }
        thin += tk * ex.x;
    }
    if total <= 0.0 { return vec2<f32>(1.0, 0.0); }
    var mu = log(P.misc.z);
    var ty = 0.0;
    if wsum > 0.0 {
        mu = lnmu / wsum;
        ty = tau / wsum;
    }
    let tf = clamp(thin / total, 0.0, 0.95);
    let cb = cure / total;
    mu += 3.0 * log(1.0 - tf) + P.cure.z * cb * cb;
    ty = ty * pow(1.0 - tf, 4.0) + P.cure.y * cb * cb;
    return vec2<f32>(exp(mu), ty);
}

@compute @workgroup_size(16, 16)
fn cure_step(@builtin(global_invocation_id) id: vec3<u32>) {
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    if h[c] <= 0.0 {
        rheo[c] = vec2<f32>(1.0, 0.0);
        return;
    }
    let rate = P.cure.x * P.frame_dt;
    let t = strata_h[c];
    let u = select(strata_cure[c], min(strata_cure[c] + vec4<f32>(rate), vec4<f32>(1.0)), t > vec4<f32>(0.0));
    if !(t.x > 0.0 && u.x >= 1.0) {
        strata_cure[c] = u;
        rheo[c] = packed_rheo(c, t, u);
        return;
    }
    var s = stack_load(c);
    s.cu[0] = u.x; s.cu[1] = u.y; s.cu[2] = u.z; s.cu[3] = u.w;
    var committed = 0.0;
    var guard = 0u;
    while s.th[0] > 0.0 && s.cu[0] >= 1.0 && guard < 4u {
        guard++;
        var r = load_refl(c);
        for (var b = 0u; b < NBAND; b++) {
            let ks = layer_ks(s.pg[0], b);
            let rt = km_layer(ks.x, ks.y, s.th[0]);
            r[b] = composite(rt, r[b]);
        }
        store_refl(c, r);
        committed += s.th[0];
        m[c] = m[c] + s.th[0] * (1.0 - OIL_SHRINK) * (1.0 - clamp(s.pg[0][12], 0.0, 0.95));
        s.th[0] = 0.0;
        s.cu[0] = 0.0;
        stack_compact(&s);
    }
    if committed > 0.0 {
        h[c] = max(h[c] - committed, 0.0);
    }
    rheo[c] = composition_rheo(s);
    stack_store(c, s);
}
