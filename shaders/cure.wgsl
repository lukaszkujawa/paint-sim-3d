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

@compute @workgroup_size(16, 16)
fn cure_step(@builtin(global_invocation_id) id: vec3<u32>) {
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    if h[c] <= 0.0 {
        rheo[c] = vec2<f32>(1.0, 0.0);
        return;
    }
    var s = stack_load(c);
    let rate = P.cure.x * P.frame_dt;
    for (var k = 0u; k < 4u; k++) {
        if s.th[k] > 0.0 { s.cu[k] = min(s.cu[k] + rate, 1.0); }
    }
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
