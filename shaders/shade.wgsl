@group(2) @binding(0) var color_out: texture_storage_2d<rgba16float, write>;

fn stack_count_dbg(th: vec4<f32>) -> u32 {
    var n = 0u;
    for (var k = 0u; k < 4u; k++) { if th[k] > 0.0 { n = k + 1u; } }
    return n;
}

@compute @workgroup_size(16, 16)
fn shade(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    shade_cell(id);
}

@compute @workgroup_size(16, 16)
fn shade_full(@builtin(global_invocation_id) id: vec3<u32>) {
    shade_cell(id);
}

fn shade_cell(id: vec3<u32>) {
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    var r = load_refl(c);
    let hv = h[c];
    var wet = 0.0;
    if hv > 0.0 {
        let th = strata_h[c];
        let cu = strata_cure[c];
        for (var k = 0u; k < 4u; k++) {
            let t = th[k];
            if t <= 0.0 { continue; }
            let pig = unpack_pig(strata_pig[(c * 4u + k) * 2u], strata_pig[(c * 4u + k) * 2u + 1u]);
            for (var b = 0u; b < NBAND; b++) {
                let ks = layer_ks(pig, b);
                let rt = km_layer(ks.x, ks.y, t);
                r[b] = composite(rt, r[b]);
            }
            wet = 1.0 - cu[k];
        }
    }
    let k1 = 0.04;
    let k2 = mix(0.4, 0.6, wet);
    for (var b = 0u; b < NBAND; b++) {
        r[b] = (1.0 - k1) * (1.0 - k2) * r[b] / max(1.0 - k2 * r[b], 1e-3);
    }
    var rgb = spectrum_to_rgb(r);
    let rough = mix(0.32, 0.07, wet);
    if P.flags.x == 1u && hv > 0.0 {
        let th = strata_h[c];
        let ssum = th.x + th.y + th.z + th.w;
        var csum = 0.0;
        var bad = false;
        for (var k = 0u; k < 4u; k++) {
            if th[k] <= 0.0 { continue; }
            let pig = unpack_pig(strata_pig[(c * 4u + k) * 2u], strata_pig[(c * 4u + k) * 2u + 1u]);
            var cs = 0.0;
            for (var i = 0u; i < 13u; i++) { cs += pig[i]; }
            if !(cs > 0.0) { bad = true; }
            csum += cs;
        }
        if ssum <= 0.0 { rgb = vec3<f32>(0.0, 0.0, 1.0); }
        else if bad { rgb = vec3<f32>(1.0, 0.0, 0.0); }
        else if abs(ssum - hv) > 0.01 * hv { rgb = vec3<f32>(0.0, 1.0, 0.0); }
        else { rgb = vec3<f32>(0.3, 0.3, 0.3) + 0.15 * f32(stack_count_dbg(th)); }
    }
    textureStore(color_out, vec2<i32>(id.xy), vec4<f32>(rgb, rough));
}
