fn km_layer(k: f32, s: f32, t: f32) -> vec2<f32> {
    if k + s < 1e-6 || t <= 0.0 { return vec2<f32>(0.0, 1.0); }
    let bs = sqrt(max(k * (k + 2.0 * s), 0.0));
    let arg = min(bs * t, 20.0);
    let e = exp(arg);
    let sh = 0.5 * (e - 1.0 / e);
    let ch = 0.5 * (e + 1.0 / e);
    let den = (k + s) * sh + bs * ch;
    if den < 1e-12 { return vec2<f32>(0.0, 1.0); }
    return vec2<f32>(s * sh / den, bs / den);
}

fn composite(rt: vec2<f32>, rb: f32) -> f32 {
    return rt.x + rt.y * rt.y * rb / max(1.0 - rt.x * rb, 1e-4);
}

fn layer_ks(pig: array<f32, 16>, band: u32) -> vec2<f32> {
    var k = 0.0;
    var s = 0.0;
    for (var i = 0u; i < NPIG; i++) {
        let c = pig[i];
        if c > 0.0 {
            k += c * pigtab[i * PIG_STRIDE + band];
            s += c * pigtab[i * PIG_STRIDE + NBAND + band];
        }
    }
    return vec2<f32>(k, s);
}

fn load_refl(c: u32) -> array<f32, 16> {
    var r: array<f32, 16>;
    for (var b = 0u; b < 8u; b++) {
        let v = unpack2x16float(sub_refl[c * 8u + b]);
        r[2u * b] = v.x;
        r[2u * b + 1u] = v.y;
    }
    return r;
}

fn store_refl(c: u32, r: array<f32, 16>) {
    for (var b = 0u; b < 8u; b++) {
        sub_refl[c * 8u + b] = pack2x16float(vec2<f32>(r[2u * b], r[2u * b + 1u]));
    }
}

fn spectrum_to_rgb(r: array<f32, 16>) -> vec3<f32> {
    var xyz = vec3<f32>(0.0);
    for (var b = 0u; b < NBAND; b++) {
        xyz += r[b] * vec3<f32>(cmf[b * 3u], cmf[b * 3u + 1u], cmf[b * 3u + 2u]);
    }
    let rgb = vec3<f32>(
        3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z,
        -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z,
        0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z
    );
    return max(rgb, vec3<f32>(0.0));
}
