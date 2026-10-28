const DRAINED_FILM: f32 = 1e-7;
const WAKE_BAND: u32 = 8u;
const PLUG_RADIUS: i32 = 3;
const YIELD_MENISCUS: f32 = 1e-3;
fn pressure_at(i: i32, j: i32) -> vec2<f32> {
    return pres[idx(refl_i(i), refl_j(j))];
}

fn pressure_wet(i: i32, j: i32, fallback: vec2<f32>) -> vec2<f32> {
    let c = idx(refl_i(i), refl_j(j));
    let p = pres[c];
    if h[c] > 0.0 { return p; }
    return vec2<f32>(fallback.x, p.y);
}

@compute @workgroup_size(16, 16)
fn laplacian(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let i = i32(id.x);
    let j = i32(id.y);
    let c = idx(id.x, id.y);
    let hv = h[c];
    let s0 = hv + m[c];
    var hold = 0.0;
    var pl = 0.0;
    if hv > 0.0 { hold = hold_level(nose_dist(c)); }
    let u0 = s0 - hold;
    let ax = usurf(i - 1, j) + usurf(i + 1, j) + usurf(i, j - 1) + usurf(i, j + 1);
    let dg = usurf(i - 1, j - 1) + usurf(i + 1, j - 1) + usurf(i - 1, j + 1) + usurf(i + 1, j + 1);
    let l = (W_AXIS * ax + W_DIAG_LAP * dg - (4.0 * W_AXIS + 4.0 * W_DIAG_LAP) * u0) * P.inv_dx * P.inv_dx;
    if hv > 0.0 {
        let d = nose_dist(c);
        if d <= 3.0 {
            let n = fnorm[c];
            if dot(n, n) > 0.5 {
                pl = line_pressure(c) * exp(-(d - 1.0) * 0.7);
            }
        }
    }
    pres[c] = vec2<f32>(-P.phys.y * l + disjoining(hv), P.phys.x * P.g.z * (s0 - hold) + pl);
}

fn yield_cap(gc: vec2<f32>, ty: f32) -> vec2<f32> {
    let gcl = length(gc);
    let gc_max = P.phys.y / (YIELD_MENISCUS * YIELD_MENISCUS);
    if ty <= 0.0 || gcl <= gc_max { return gc; }
    return gc * (gc_max / gcl);
}

fn plug_mobility(ca: u32, cb: u32, gtilt: vec2<f32>) -> f32 {
    let gb = length(gtilt);
    if gb < 1e-6 { return 0.0; }
    let up = select(cb, ca, gtilt.x >= 0.0);
    return tmp[up] * h[up] / gb;
}

@compute @workgroup_size(16, 16)
fn plug(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    tmp[c] = 0.0;
    if h[c] <= 0.0 { return; }
    let gb = P.phys.x * length(P.g.xy);
    if gb < 1e-6 { return; }
    var sum = 0.0;
    var mass = 0.0;
    var yielding = false;
    for (var dy = -PLUG_RADIUS; dy <= PLUG_RADIUS; dy++) {
        for (var dxi = -PLUG_RADIUS; dxi <= PLUG_RADIUS; dxi++) {
            let x = i32(id.x) + dxi;
            let y = i32(id.y) + dy;
            if x < 0 || y < 0 || x >= i32(P.size.x) || y >= i32(P.size.y) { continue; }
            let n = idx(u32(x), u32(y));
            let hn = h[n];
            if hn <= 0.0 { continue; }
            let r = rheo[n];
            if r.y > 0.0 { yielding = true; }
            let q = hb_flux(gb, hn, max(r.x, 1e-4), r.y).x;
            sum += q;
            mass += hn;
        }
    }
    if !yielding || mass <= 0.0 { return; }
    tmp[c] = sum / mass;
}

fn yield_depth(ca: u32, cb: u32, hf: f32, ty: f32, g: vec2<f32>) -> f32 {
    if ty <= 0.0 { return hf; }
    return select(h[cb], h[ca], g.x >= 0.0);
}

fn hb_flux(gmag: f32, hf: f32, mu: f32, ty: f32) -> vec3<f32> {
    let n = P.phys.z;
    if gmag < 1e-9 || hf <= 0.0 { return vec3<f32>(0.0); }
    let y = max(hf - ty / gmag, 0.0);
    if y <= 0.0 { return vec3<f32>(0.0); }
    var up = 0.0;
    if abs(n - 1.0) < 1e-4 {
        up = 0.5 * gmag / mu * y * y;
    } else {
        up = n / (n + 1.0) * pow(gmag / mu, 1.0 / n) * pow(y, (n + 1.0) / n);
    }
    let q = up * (hf - n * y / (2.0 * n + 1.0));
    return vec3<f32>(q, y, up);
}

fn band_flux(up: f32, y: f32, za: f32, zb: f32) -> f32 {
    let n = P.phys.z;
    var ta = 0.0;
    var tb = 0.0;
    if y > 0.0 {
        let ra = max(y - za, 0.0) / y;
        let rb = max(y - zb, 0.0) / y;
        if abs(n - 1.0) < 1e-4 {
            ta = ra * ra * ra;
            tb = rb * rb * rb;
        } else {
            let e = (2.0 * n + 1.0) / n;
            ta = pow(ra, e);
            tb = pow(rb, e);
        }
    }
    return up * ((zb - za) - n / (2.0 * n + 1.0) * y * (ta - tb));
}

struct FaceFlux {
    q: f32,
    y: f32,
    up: f32,
}

fn feeders(c: u32) -> u32 {
    let i = i32(c % P.size.x);
    let j = i32(c / P.size.x);
    let hc = h[c];
    let thr = 2.0 * hc + P.ein.w;
    var n = 0u;
    if i > 0 && h[idx(u32(i - 1), u32(j))] > thr { n++; }
    if i + 1 < i32(P.size.x) && h[idx(u32(i + 1), u32(j))] > thr { n++; }
    if j > 0 && h[idx(u32(i), u32(j - 1))] > thr { n++; }
    if j + 1 < i32(P.size.y) && h[idx(u32(i), u32(j + 1))] > thr { n++; }
    return n;
}

fn usurf(i: i32, j: i32) -> f32 {
    let c = idx(refl_i(i), refl_j(j));
    var hold = 0.0;
    if h[c] > 0.0 { hold = hold_level(nose_dist(c)); }
    return h[c] + m[c] - hold;
}

fn hold_level(d: f32) -> f32 {
    return P.front.x * (1.0 - exp(-(max(d, 1.0) - 0.5) * P.front.y));
}

const COX_LOG: f32 = 132.0;

fn front_normal(c: u32) -> vec2<f32> {
    let i = i32(c % P.size.x);
    let j = i32(c / P.size.x);
    var d = vec2<f32>(0.0);
    for (var dj = -2; dj <= 2; dj++) {
        for (var di = -2; di <= 2; di++) {
            if di == 0 && dj == 0 { continue; }
            let x = i + di;
            let y = j + dj;
            if x < 0 || y < 0 || x >= i32(P.size.x) || y >= i32(P.size.y) { continue; }
            if di * di + dj * dj > 5 { continue; }
            if h[idx(u32(x), u32(y))] <= P.ein.w {
                let o = vec2<f32>(f32(di), f32(dj));
                d += o / dot(o, o);
            }
        }
    }
    let l = length(d);
    if l < 1e-6 { return vec2<f32>(0.0); }
    return d / l;
}

fn wet_gain(n: vec2<f32>) -> f32 {
    let s2 = 0.70710678;
    var w = W_AXIS * (max(n.x, 0.0) + max(-n.x, 0.0) + max(n.y, 0.0) + max(-n.y, 0.0));
    w += W_DIAG_FLUX * (max(s2 * (n.x + n.y), 0.0) + max(-s2 * (n.x + n.y), 0.0) + max(s2 * (n.x - n.y), 0.0) + max(-s2 * (n.x - n.y), 0.0));
    let m = max(max(abs(n.x), abs(n.y)), 1e-3);
    return 1.0 / (m * max(w, 1e-3));
}

fn line_pressure(c: u32) -> f32 {
    let i = i32(c % P.size.x);
    let j = i32(c / P.size.x);
    if i < 1 || j < 1 || i + 1 >= i32(P.size.x) || j + 1 >= i32(P.size.y) { return 0.0; }
    let nl = fnorm[idx(u32(i - 1), u32(j))];
    let nr = fnorm[idx(u32(i + 1), u32(j))];
    let nd = fnorm[idx(u32(i), u32(j - 1))];
    let nu = fnorm[idx(u32(i), u32(j + 1))];
    if dot(nl, nl) < 0.5 || dot(nr, nr) < 0.5 || dot(nd, nd) < 0.5 || dot(nu, nu) < 0.5 { return 0.0; }
    let curv = (nr.x - nl.x + nu.y - nd.y) * 0.5 * P.inv_dx;
    return P.phys.y * curv * P.front2.z;
}

fn front_shape(c: u32, n: vec2<f32>) -> f32 {
    let i = i32(c % P.size.x);
    let j = i32(c / P.size.x);
    var w = 0.0;
    var w0 = 0.0;
    for (var dj = -3; dj <= 3; dj++) {
        for (var di = -3; di <= 3; di++) {
            if di == 0 && dj == 0 { continue; }
            if di * di + dj * dj > 9 { continue; }
            let o = vec2<f32>(f32(di), f32(dj));
            if dot(o, n) < -0.5 { w0 += 1.0; }
            let x = i + di;
            let y = j + dj;
            if x < 0 || y < 0 || x >= i32(P.size.x) || y >= i32(P.size.y) { continue; }
            if h[idx(u32(x), u32(y))] > P.ein.w { w += 1.0; }
        }
    }
    if w0 < 1.0 || (u32(P.front2.w) & 2u) != 0u { return 1.0; }
    return pow(clamp(w / w0, 0.0, 2.5), P.front2.y);
}

fn wet_threshold(c: u32) -> f32 {
    let ty = rheo[c].y;
    let ycap = ty / (P.phys.x * P.g.z * max(h[c], 1e-4)) * P.dx;
    return max(hold_level(1.0), ycap);
}

fn front_dist(c: u32) -> f32 {
    if h[c] <= 0.0 { return 0.0; }
    return nose_dist(c);
}

fn cox_speed(excess: f32, mu: f32) -> f32 {
    if excess <= 0.0 { return 0.0; }
    let theta = atan(P.ein.x);
    let span = max(P.front.x - hold_level(1.0), 1e-6);
    let th = theta + (3.14159265 - theta) * clamp(excess / span, 0.0, 1.0);
    let adv = min(theta + P.front2.x, 3.1);
    if th <= adv { return 0.0; }
    return P.phys.y * (th * th * th - adv * adv * adv) / (COX_LOG * max(mu, 1e-4));
}

fn cell_excess(c: u32) -> f32 {
    var e = h[c] - hold_level(front_dist(c));
    let o = unpack_off(delta[c]);
    let ol = length(vec2<f32>(f32(o.x), f32(o.y)));
    if ol > 0.5 && ol < 60.0 {
        let dir = vec2<f32>(f32(o.x), f32(o.y)) * (-1.0 / ol);
        let bias = vec2<f32>(0.5 * sign(f32(-o.x)), 0.5 * sign(f32(-o.y)));
        for (var k = 0u; k < 3u; k++) {
            let reach = f32(4u << k);
            let step = vec2<i32>(dir * reach + bias);
            let ri = i32(c % P.size.x) + step.x;
            let rj = i32(c / P.size.x) + step.y;
            if ri < 0 || rj < 0 || ri >= i32(P.size.x) || rj >= i32(P.size.y) { break; }
            let cr = idx(u32(ri), u32(rj));
            if h[cr] <= 0.0 { break; }
            e = max(e, h[cr] - hold_level(front_dist(cr)));
        }
    }
    return e;
}

fn rim_translation(ca: u32, cb: u32, hf: f32, face_dir: vec2<f32>) -> vec2<f32> {
    if (u32(P.front2.w) & 1u) != 0u { return vec2<f32>(0.0); }
    let da = front_dist(ca);
    let db = front_dist(cb);
    if max(da, db) > 3.0 / P.front.y { return vec2<f32>(0.0); }
    var n = vec2<f32>(0.0);
    if da > db && h[ca] > 0.0 { n = fnorm[ca]; }
    else if db > da && h[cb] > 0.0 { n = fnorm[cb]; }
    else {
        if h[ca] > 0.0 { n += fnorm[ca]; }
        if h[cb] > 0.0 { n += fnorm[cb]; }
    }
    let nl = length(n);
    if nl < 1e-6 { return vec2<f32>(0.0); }
    n = n / nl;
    let s = dot(n, face_dir);
    if abs(s) < 1e-4 { return vec2<f32>(0.0); }
    var src = ca;
    var dst = cb;
    if s < 0.0 { src = cb; dst = ca; }
    if h[src] <= 0.0 { return vec2<f32>(0.0); }
    let e = cell_excess(src);
    if e <= 0.0 { return vec2<f32>(0.0); }
    var e_out = -hold_level(1.0);
    if h[dst] > 0.0 { e_out = cell_excess(dst); }
    let room = clamp(1.0 - max(e_out, 0.0) / e, 0.0, 1.0);
    let u = cox_speed(e, rheo[src].x);
    return vec2<f32>(u * hf * room * mix(1.0, abs(s), P.front2.x), sign(s));
}

fn face(ca: u32, cb: u32, gcap: vec2<f32>, ghyd: vec2<f32>, gtilt: vec2<f32>, diag: bool, face_dir: vec2<f32>) -> FaceFlux {
    var out: FaceFlux;
    out.q = 0.0;
    out.y = 0.0;
    out.up = 0.0;
    let ha = h[ca];
    let hb = h[cb];
    let hf = 0.5 * (ha + hb);
    if hf <= 0.0 { return out; }
    var shape = 1.0;
    var wetting = false;
    var proj = 1.0;
    if min(ha, hb) <= 0.0 {
        let wet = select(cb, ca, ha > 0.0);
        let dry = select(ca, cb, ha > 0.0);
        if h[wet] <= wet_threshold(wet) { return out; }
        wetting = true;
        var n = fnorm[wet];
        if dot(n, n) < 0.5 { n = front_normal(wet); }
        let toward = select(-1.0, 1.0, ha > 0.0);
        proj = max(dot(n, face_dir * toward), 0.0);
        if proj <= 0.0 { return out; }
        shape = front_shape(dry, n) * wet_gain(n);
        if shape <= 0.0 { return out; }
    }
    let ra = rheo[ca];
    let rb = rheo[cb];
    let wa = ha / (ha + hb);
    let wb = 1.0 - wa;
    let mu = exp(wa * log(max(ra.x, 1e-4)) + wb * log(max(rb.x, 1e-4)));
    let ty = wa * ra.y + wb * rb.y;
    let thin = min(ha, hb) / hf;
    let fcap = thin * thin * thin;
    let g = yield_cap(gcap * fcap, ty) + ghyd + gtilt;
    let gmag = length(g);
    let qy = hb_flux(gmag, yield_depth(ca, cb, hf, ty, g), mu, ty);
    let meff = select(0.0, qy.x / max(gmag, 1e-9), qy.x > 0.0);
    var mtilt = meff;
    if ty > 0.0 {
        mtilt = plug_mobility(ca, cb, gtilt);
    }
    if meff <= 0.0 && mtilt <= 0.0 { return out; }
    let fcap2 = pow(thin, P.front3.w);
    let qvec = mtilt * gtilt + min(meff, P.misc.w) * ghyd + min(meff, P.phys.w) * (gcap * fcap2);
    var qn = qvec.x;
    let tr = rim_translation(ca, cb, hf, face_dir);
    if wetting {
        let wet = select(cb, ca, ha > 0.0);
        let want = tr.x * 2.0 * h[wet] / max(hf, 1e-9);
        qn = tr.y * max(qn * tr.y * proj, want) * shape;
        let vcap = 2.0;
        if abs(qn) > vcap * hf { qn = sign(qn) * vcap * hf; }
    } else {
        if tr.x > 0.0 {
            qn = tr.y * max(qn * tr.y, tr.x);
        }
        let vcap = 2.0;
        if abs(qn) > vcap * hf { qn = sign(qn) * vcap * hf; }
        let lo = min(ha, hb);
        let hi = max(ha, hb);
        if lo < 0.5 * hi {
            if diag {
                qn = 0.0;
            } else {
                var tgt = cb;
                if hb > ha { tgt = ca; }
                let n = feeders(tgt);
                qn = qn / pow(f32(max(n, 1u)), 0.7);
            }
        }
    }
    out.q = qn;
    out.y = qy.y;
    out.up = qy.z;
    return out;
}


fn face_pair(id: vec3<u32>) -> array<FaceFlux, 4> {
    let i = i32(id.x);
    let j = i32(id.y);
    let c = idx(id.x, id.y);
    var fx: FaceFlux;
    var fy: FaceFlux;
    var fd: FaceFlux;
    var fe: FaceFlux;
    fx.q = 0.0; fx.y = 0.0; fx.up = 0.0;
    fy.q = 0.0; fy.y = 0.0; fy.up = 0.0;
    fd.q = 0.0; fd.y = 0.0; fd.up = 0.0;
    fe.q = 0.0; fe.y = 0.0; fe.up = 0.0;
    let p00 = pressure_at(i, j);
    let gxy = P.phys.x * P.g.xy;
    let s2 = 0.70710678;
    let wa = W_AXIS;
    let wd = W_DIAG_FLUX;
    if id.x + 1u < P.size.x && cell_active(id.x + 1u, id.y) {
        let cb = idx(id.x + 1u, id.y);
        let p10 = pressure_at(i + 1, j);
        let gm = -(p10 - p00) * P.inv_dx;
        let pt = pressure_wet(i, j + 1, p00) + pressure_wet(i + 1, j + 1, p10) - pressure_wet(i, j - 1, p00) - pressure_wet(i + 1, j - 1, p10);
        let gt = -pt * 0.25 * P.inv_dx;
        fx = face(c, cb, vec2<f32>(gm.x, gt.x), vec2<f32>(gm.y, gt.y), gxy, false, vec2<f32>(1.0, 0.0));
        fx.q *= wa;
    }
    if id.y + 1u < P.size.y && cell_active(id.x, id.y + 1u) {
        let cb = idx(id.x, id.y + 1u);
        let p01 = pressure_at(i, j + 1);
        let gm = -(p01 - p00) * P.inv_dx;
        let pt = pressure_wet(i + 1, j, p00) + pressure_wet(i + 1, j + 1, p01) - pressure_wet(i - 1, j, p00) - pressure_wet(i - 1, j + 1, p01);
        let gt = -pt * 0.25 * P.inv_dx;
        fy = face(c, cb, vec2<f32>(gm.x, gt.x), vec2<f32>(gm.y, gt.y), gxy.yx, false, vec2<f32>(0.0, 1.0));
        fy.q *= wa;
    }
    if id.x + 1u < P.size.x && id.y + 1u < P.size.y && cell_active(id.x + 1u, id.y + 1u) {
        let cb = idx(id.x + 1u, id.y + 1u);
        let p11 = pressure_at(i + 1, j + 1);
        let ga = -(p11 - p00) * P.inv_dx * s2;
        let pa = pressure_wet(i + 1, j, p00);
        let pb = pressure_wet(i, j + 1, p11);
        let gp = -(pa - pb) * P.inv_dx * s2;
        let gt = vec2<f32>(gxy.x + gxy.y, gxy.x - gxy.y) * s2;
        let hs = P.ein.w;
        let open = min(h[c], h[cb]) > 0.0 || h[idx(id.x + 1u, id.y)] > hs || h[idx(id.x, id.y + 1u)] > hs;
        if open {
            fd = face(c, cb, vec2<f32>(ga.x, gp.x), vec2<f32>(ga.y, gp.y), gt, true, vec2<f32>(s2, s2));
            fd.q *= wd;
        }
    }
    if id.x + 1u < P.size.x && id.y > 0u && cell_active(id.x + 1u, id.y - 1u) {
        let cb = idx(id.x + 1u, id.y - 1u);
        let p1m = pressure_at(i + 1, j - 1);
        let ga = -(p1m - p00) * P.inv_dx * s2;
        let pa = pressure_wet(i + 1, j, p00);
        let pb = pressure_wet(i, j - 1, p1m);
        let gp = -(pa - pb) * P.inv_dx * s2;
        let gt = vec2<f32>(gxy.x - gxy.y, gxy.x + gxy.y) * s2;
        let hs = P.ein.w;
        let open = min(h[c], h[cb]) > 0.0 || h[idx(id.x + 1u, id.y)] > hs || h[idx(id.x, id.y - 1u)] > hs;
        if open {
            fe = face(c, cb, vec2<f32>(ga.x, gp.x), vec2<f32>(ga.y, gp.y), gt, true, vec2<f32>(s2, -s2));
            fe.q *= wd;
        }
    }
    var o: array<FaceFlux, 4>;
    o[0] = fx;
    o[1] = fy;
    o[2] = fd;
    o[3] = fe;
    return o;
}

@compute @workgroup_size(16, 16)
fn fluxes(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    let ff = face_pair(id);
    flux[c] = vec4<f32>(ff[0].q, ff[1].q, ff[2].q, ff[3].q);
}

fn band_split(cu: u32, hf: f32, y: f32, up: f32, total: f32) -> vec4<f32> {
    let sh = strata_h[cu];
    let hu = sh.x + sh.y + sh.z + sh.w;
    if hu <= 0.0 { return vec4<f32>(total, 0.0, 0.0, 0.0); }
    var frac = sh / hu;
    if y > 0.0 && up > 0.0 {
        let k = hf / hu;
        let z1 = sh.x * k;
        let z2 = z1 + sh.y * k;
        let z3 = z2 + sh.z * k;
        let z4 = z3 + sh.w * k;
        let b = vec4<f32>(band_flux(up, y, 0.0, z1), band_flux(up, y, z1, z2), band_flux(up, y, z2, z3), band_flux(up, y, z3, z4));
        let bs = b.x + b.y + b.z + b.w;
        if bs > 0.0 { frac = b / bs; }
    }
    return frac * total;
}

@compute @workgroup_size(16, 16)
fn strata_split(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    let ff = face_pair(id);
    let a = acc[c];
    var sx = vec4<f32>(0.0);
    var sy = vec4<f32>(0.0);
    var sd = vec4<f32>(0.0);
    var se = vec4<f32>(0.0);
    if a.x != 0.0 && id.x + 1u < P.size.x {
        let cb = idx(id.x + 1u, id.y);
        var cu = c;
        if a.x < 0.0 { cu = cb; }
        sx = band_split(cu, 0.5 * (h[c] + h[cb]), ff[0].y, ff[0].up, a.x);
    }
    if a.y != 0.0 && id.y + 1u < P.size.y {
        let cb = idx(id.x, id.y + 1u);
        var cu = c;
        if a.y < 0.0 { cu = cb; }
        sy = band_split(cu, 0.5 * (h[c] + h[cb]), ff[1].y, ff[1].up, a.y);
    }
    if a.z != 0.0 && id.x + 1u < P.size.x && id.y + 1u < P.size.y {
        let cb = idx(id.x + 1u, id.y + 1u);
        var cu = c;
        if a.z < 0.0 { cu = cb; }
        sd = band_split(cu, 0.5 * (h[c] + h[cb]), ff[2].y, ff[2].up, a.z);
    }
    if a.w != 0.0 && id.x + 1u < P.size.x && id.y > 0u {
        let cb = idx(id.x + 1u, id.y - 1u);
        var cu = c;
        if a.w < 0.0 { cu = cb; }
        se = band_split(cu, 0.5 * (h[c] + h[cb]), ff[3].y, ff[3].up, a.w);
    }
    acc_s[4u * c] = sx;
    acc_s[4u * c + 1u] = sy;
    acc_s[4u * c + 2u] = sd;
    acc_s[4u * c + 3u] = se;
}

fn nb_ok(i: i32, j: i32) -> bool {
    if i < 0 || j < 0 || i >= i32(P.size.x) || j >= i32(P.size.y) { return false; }
    return cell_active(u32(i), u32(j));
}

@compute @workgroup_size(16, 16)
fn limiter(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    let i = i32(id.x);
    let j = i32(id.y);
    let f = flux[c];
    var out = max(f.x, 0.0) + max(f.y, 0.0) + max(f.z, 0.0) + max(f.w, 0.0);
    if nb_ok(i - 1, j) { out += max(-flux[idx(id.x - 1u, id.y)].x, 0.0); }
    if nb_ok(i, j - 1) { out += max(-flux[idx(id.x, id.y - 1u)].y, 0.0); }
    if nb_ok(i - 1, j - 1) { out += max(-flux[idx(id.x - 1u, id.y - 1u)].z, 0.0); }
    if nb_ok(i - 1, j + 1) { out += max(-flux[idx(id.x - 1u, id.y + 1u)].w, 0.0); }
    var a = 1.0;
    if out > 0.0 { a = min(1.0, 0.5 * h[c] * P.dx / (P.dt * out)); }
    alpha[c] = a;
}

struct Inflow {
    best: f32,
    src: u32,
    div: f32,
    accv: f32,
}

fn own_face(c: u32, q: f32, cb: u32, has: bool, ac: f32, st: ptr<function, Inflow>) {
    var ab = ac;
    if q < 0.0 && has {
        ab = alpha[cb];
        if -q * ab > (*st).best { (*st).best = -q * ab; (*st).src = cb; }
    }
    (*st).div += q * ab;
    (*st).accv = q * ab;
}

fn nb_face(c: u32, q: f32, cn: u32, ac: f32, st: ptr<function, Inflow>) {
    var an = alpha[cn];
    if q < 0.0 { an = ac; }
    if q * an > (*st).best { (*st).best = q * an; (*st).src = cn; }
    (*st).div -= q * an;
}

fn slope_cap(c: u32) -> f32 {
    let ty = rheo[c].y;
    let hb = max(h[c], 1e-4);
    return max(P.ein.x, ty / (P.phys.x * P.g.z * hb));
}

fn pack_off(dx: i32, dy: i32) -> f32 {
    let ux = u32(clamp(dx + 128, 0, 255));
    let uy = u32(clamp(dy + 128, 0, 255));
    return bitcast<f32>((ux << 8u) | uy | 0x10000u);
}

fn unpack_off(v: f32) -> vec2<i32> {
    let bits = bitcast<u32>(v);
    if (bits & 0x10000u) == 0u { return vec2<i32>(64, 64); }
    return vec2<i32>(i32((bits >> 8u) & 0xffu) - 128, i32(bits & 0xffu) - 128);
}

fn nose_dist(c: u32) -> f32 {
    let o = unpack_off(delta[c]);
    return length(vec2<f32>(f32(o.x), f32(o.y)));
}

fn nose_cap(c: u32) -> f32 {
    return slope_cap(c) * P.dx * max(nose_dist(c), 1.0);
}

fn shed(c: u32) -> f32 {
    let hc = h[c];
    let hmin = 2.0 * P.ein.y;
    if hc <= hmin { return 0.0; }
    return max(hc - nose_cap(c), 0.0);
}

fn hat(i: i32, j: i32) -> f32 {
    if i < 0 || j < 0 || i >= i32(P.size.x) || j >= i32(P.size.y) { return 0.0; }
    if !cell_active(u32(i), u32(j)) { return 0.0; }
    return h[idx(u32(i), u32(j))];
}

fn okf(i: i32, j: i32) -> f32 {
    if i < 0 || j < 0 || i >= i32(P.size.x) || j >= i32(P.size.y) { return 0.0; }
    if !cell_active(u32(i), u32(j)) { return 0.0; }
    return 1.0;
}

fn outward(i: i32, j: i32) -> vec2<f32> {
    var d = vec2<f32>(0.0);
    for (var dj = -2; dj <= 2; dj++) {
        for (var di = -2; di <= 2; di++) {
            if di * di + dj * dj <= 5 && hat(i + di, j + dj) <= 0.0 { d += vec2<f32>(f32(di), f32(dj)); }
        }
    }
    return d;
}

struct Shed {
    amount: f32,
    i0: i32,
    j0: i32,
    w: vec4<f32>,
}

fn shed_of(i: i32, j: i32) -> Shed {
    var o: Shed;
    o.amount = 0.0;
    o.i0 = i;
    o.j0 = j;
    o.w = vec4<f32>(0.0);
    if i < 0 || j < 0 || i >= i32(P.size.x) || j >= i32(P.size.y) { return o; }
    if !cell_active(u32(i), u32(j)) { return o; }
    let c = idx(u32(i), u32(j));
    let e = shed(c);
    if e <= 0.0 { return o; }
    let hc = h[c];
    var dir = outward(i, j);
    if length(dir) < 1e-6 {
        dir = -vec2<f32>(0.5 * (hat(i + 1, j) - hat(i - 1, j)), 0.5 * (hat(i, j + 1) - hat(i, j - 1)));
    }
    let gm = length(dir);
    if gm < 1e-9 { return o; }
    dir = dir / gm;
    let px = f32(i) + dir.x;
    let py = f32(j) + dir.y;
    let fx0 = floor(px);
    let fy0 = floor(py);
    let tx = px - fx0;
    let ty = py - fy0;
    let i0 = i32(fx0);
    let j0 = i32(fy0);
    var w = vec4<f32>((1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty);
    w = w * vec4<f32>(okf(i0, j0), okf(i0 + 1, j0), okf(i0, j0 + 1), okf(i0 + 1, j0 + 1));
    let ws = w.x + w.y + w.z + w.w;
    if ws <= 1e-6 { return o; }
    w = w / ws;
    let ht = w.x * hat(i0, j0) + w.y * hat(i0 + 1, j0) + w.z * hat(i0, j0 + 1) + w.w * hat(i0 + 1, j0 + 1);
    let amount = min(e, 0.5 * max(hc - ht, 0.0));
    if amount <= 0.0 { return o; }
    o.amount = amount;
    o.i0 = i0;
    o.j0 = j0;
    o.w = w;
    return o;
}

fn share_to(sd: Shed, i: i32, j: i32) -> f32 {
    if sd.amount <= 0.0 { return 0.0; }
    let di = i - sd.i0;
    let dj = j - sd.j0;
    if di == 0 && dj == 0 { return sd.amount * sd.w.x; }
    if di == 1 && dj == 0 { return sd.amount * sd.w.y; }
    if di == 0 && dj == 1 { return sd.amount * sd.w.z; }
    if di == 1 && dj == 1 { return sd.amount * sd.w.w; }
    return 0.0;
}

@compute @workgroup_size(16, 16)
fn nose(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    if h[c] <= 0.0 {
        delta[c] = bitcast<f32>(0u);
        fnorm[c] = vec2<f32>(0.0);
        return;
    }
    let i = i32(id.x);
    let j = i32(id.y);
    var best = vec2<i32>(64, 64);
    var bd = 1e9;
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            let x = i + di;
            let y = j + dj;
            if x < 0 || y < 0 || x >= i32(P.size.x) || y >= i32(P.size.y) { continue; }
            let n = idx(u32(x), u32(y));
            var cand = vec2<i32>(di, dj);
            if h[n] > 0.0 {
                if (bitcast<u32>(delta[n]) & 0x10000u) == 0u { continue; }
                let o = unpack_off(delta[n]);
                cand = vec2<i32>(o.x + di, o.y + dj);
                let tx = i + cand.x;
                let ty = j + cand.y;
                if tx < 0 || ty < 0 || tx >= i32(P.size.x) || ty >= i32(P.size.y) { continue; }
                if h[idx(u32(tx), u32(ty))] > 0.0 { continue; }
            } else if di == 0 && dj == 0 {
                continue;
            }
            let d = length(vec2<f32>(f32(cand.x), f32(cand.y)));
            if d < bd { bd = d; best = cand; }
        }
    }
    if bd > 1e8 {
        delta[c] = bitcast<f32>(0u);
        fnorm[c] = vec2<f32>(0.0);
        return;
    }
    delta[c] = pack_off(best.x, best.y);
    if bd <= 3.0 { fnorm[c] = front_normal(c); } else { fnorm[c] = vec2<f32>(0.0); }
}

fn store_shed(c: u32, s: Shed) {
    acc_s[4u * c] = vec4<f32>(s.amount, bitcast<f32>(s.i0), bitcast<f32>(s.j0), 0.0);
    acc_s[4u * c + 1u] = s.w;
}

fn load_shed(c: u32) -> Shed {
    let a = acc_s[4u * c];
    var s: Shed;
    s.amount = a.x;
    s.i0 = bitcast<i32>(a.y);
    s.j0 = bitcast<i32>(a.z);
    s.w = acc_s[4u * c + 1u];
    return s;
}

@compute @workgroup_size(16, 16)
fn shed_pass(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    store_shed(idx(id.x, id.y), shed_of(i32(id.x), i32(id.y)));
}

@compute @workgroup_size(16, 16)
fn slump(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    let i = i32(id.x);
    let j = i32(id.y);
    let hc = h[c];
    let mine = load_shed(c);
    let self_share = share_to(mine, i, j);
    var inn = 0.0;
    var best_in = 0.0;
    var src = c;
    var fx = 0.0;
    var fy = 0.0;
    var fd = 0.0;
    var fe = 0.0;
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            if di == 0 && dj == 0 { continue; }
            let x = i + di;
            let y = j + dj;
            if x < 0 || y < 0 || x >= i32(P.size.x) || y >= i32(P.size.y) { continue; }
            if !cell_active(u32(x), u32(y)) { continue; }
            let sn = load_shed(idx(u32(x), u32(y)));
            let give = share_to(sn, i, j);
            let take = share_to(mine, x, y);
            inn += give;
            if give > best_in { best_in = give; src = idx(u32(x), u32(y)); }
            let net = take - give;
            if di == 1 && dj == 0 { fx = net; }
            if di == 0 && dj == 1 { fy = net; }
            if di == 1 && dj == 1 { fd = net; }
            if di == 1 && dj == -1 { fe = net; }
        }
    }
    let outs = mine.amount - self_share;
    let hn2 = max(hc - outs + inn, 0.0);
    tmp[c] = hn2;
    acc[c] += vec4<f32>(fx, fy, fd, fe) * P.dx;
    if hc <= 0.0 && hn2 > 0.0 && src != c {
        rheo[c] = rheo[src];
        let ts = strata_h[src];
        var ks = 0u;
        for (var k = 0u; k < 4u; k++) { if ts[k] > 0.0 { ks = k; } }
        strata_h[c] = vec4<f32>(hn2, 0.0, 0.0, 0.0);
        strata_pig[c * 8u] = strata_pig[(src * 4u + ks) * 2u];
        strata_pig[c * 8u + 1u] = strata_pig[(src * 4u + ks) * 2u + 1u];
        for (var k = 1u; k < 4u; k++) {
            strata_pig[c * 8u + 2u * k] = vec4<u32>(0u);
            strata_pig[c * 8u + 2u * k + 1u] = vec4<u32>(0u);
        }
        strata_cure[c] = vec4<f32>(strata_cure[src][ks], 0.0, 0.0, 0.0);
    }
}

@compute @workgroup_size(16, 16)
fn apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    begin_motion(li);
    let id = tile_cell(wid, lid);
    var moved = false;
    if in_grid(id) { moved = apply_cell(id); }
    end_motion(moved, wid, li);
}

fn apply_cell(id: vec3<u32>) -> bool {
    let c = idx(id.x, id.y);
    let i = i32(id.x);
    let j = i32(id.y);
    let ac = alpha[c];
    var st: Inflow;
    st.best = 0.0;
    st.src = c;
    st.div = 0.0;
    st.accv = 0.0;
    let f = flux[c];
    let hx = id.x + 1u < P.size.x;
    let hy = id.y + 1u < P.size.y;
    let hm = id.y > 0u;
    var accv = vec4<f32>(0.0);
    own_face(c, f.x, select(c, idx(min(id.x + 1u, P.size.x - 1u), id.y), hx), hx, ac, &st);
    accv.x = st.accv;
    own_face(c, f.y, select(c, idx(id.x, min(id.y + 1u, P.size.y - 1u)), hy), hy, ac, &st);
    accv.y = st.accv;
    own_face(c, f.z, select(c, idx(min(id.x + 1u, P.size.x - 1u), min(id.y + 1u, P.size.y - 1u)), hx && hy), hx && hy, ac, &st);
    accv.z = st.accv;
    own_face(c, f.w, select(c, idx(min(id.x + 1u, P.size.x - 1u), select(0u, id.y - 1u, hm)), hx && hm), hx && hm, ac, &st);
    accv.w = st.accv;
    if nb_ok(i - 1, j) { let cn = idx(id.x - 1u, id.y); nb_face(c, flux[cn].x, cn, ac, &st); }
    if nb_ok(i, j - 1) { let cn = idx(id.x, id.y - 1u); nb_face(c, flux[cn].y, cn, ac, &st); }
    if nb_ok(i - 1, j - 1) { let cn = idx(id.x - 1u, id.y - 1u); nb_face(c, flux[cn].z, cn, ac, &st); }
    if nb_ok(i - 1, j + 1) { let cn = idx(id.x - 1u, id.y + 1u); nb_face(c, flux[cn].w, cn, ac, &st); }
    let src = st.src;
    let h0 = h[c];
    var hn = max(h0 - P.dt * st.div * P.inv_dx, 0.0);
    if hn < h0 && hn < DRAINED_FILM {
        hn = 0.0;
    }
    h[c] = hn;
    let moved = abs(hn - h0) > MOVE_EPS;
    if h0 <= 0.0 && src != c {
        rheo[c] = rheo[src];
        let ts = strata_h[src];
        var ks = 0u;
        for (var k = 0u; k < 4u; k++) { if ts[k] > 0.0 { ks = k; } }
        strata_h[c] = vec4<f32>(hn, 0.0, 0.0, 0.0);
        strata_pig[c * 8u] = strata_pig[(src * 4u + ks) * 2u];
        strata_pig[c * 8u + 1u] = strata_pig[(src * 4u + ks) * 2u + 1u];
        for (var k = 1u; k < 4u; k++) {
            strata_pig[c * 8u + 2u * k] = vec4<u32>(0u);
            strata_pig[c * 8u + 2u * k + 1u] = vec4<u32>(0u);
        }
        strata_cure[c] = vec4<f32>(strata_cure[src][ks], 0.0, 0.0, 0.0);
    }
    acc[c] += accv * P.dt;
    return moved;
}

@compute @workgroup_size(16, 16)
fn commit_tmp(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    begin_motion(li);
    let id = tile_cell(wid, lid);
    var moved = false;
    if in_grid(id) {
        let c = idx(id.x, id.y);
        moved = abs(tmp[c] - h[c]) > MOVE_EPS;
        h[c] = tmp[c];
    }
    end_motion(moved, wid, li);
}

@group(2) @binding(0) var<storage, read_write> stats_out: array<vec2<f32>>;
@group(2) @binding(1) var<storage, read_write> pool_out: array<f32>;

@group(2) @binding(2) var<storage, read_write> cg_m: array<vec4<f32>>;
@group(2) @binding(3) var<storage, read_write> cg_d: array<f32>;
@group(2) @binding(4) var<storage, read_write> cg_x: array<f32>;
@group(2) @binding(5) var<storage, read_write> cg_r: array<f32>;
@group(2) @binding(6) var<storage, read_write> cg_p: array<f32>;
@group(2) @binding(7) var<storage, read_write> cg_q: array<f32>;
@group(2) @binding(8) var<storage, read_write> cg_part: array<f32>;
@group(2) @binding(9) var<storage, read_write> cg_s: array<vec4<f32>>;
@group(2) @binding(10) var<storage, read_write> cg_part2: array<f32>;
@group(2) @binding(11) var<storage, read_write> h_frame: array<f32>;
@group(2) @binding(12) var<storage, read_write> fnorm: array<vec2<f32>>;
@group(2) @binding(13) var<storage, read_write> cg_t: array<f32>;
@group(2) @binding(14) var<storage, read_write> cg_t2: array<f32>;

fn face_mob(ca: u32, cb: u32, gcap: vec2<f32>, ghyd: vec2<f32>, gtilt: vec2<f32>, diag: bool) -> f32 {
    let ha = h[ca];
    let hb = h[cb];
    if min(ha, hb) <= 0.0 { return 0.0; }
    let hf = 0.5 * (ha + hb);
    let ra = rheo[ca];
    let rb = rheo[cb];
    let wa = ha / (ha + hb);
    let wb = 1.0 - wa;
    let mu = exp(wa * log(max(ra.x, 1e-4)) + wb * log(max(rb.x, 1e-4)));
    let ty = wa * ra.y + wb * rb.y;
    let thin = min(ha, hb) / hf;
    let fcap = thin * thin * thin;
    let g = yield_cap(gcap * fcap, ty) + ghyd + gtilt;
    let gmag = length(g);
    let qy = hb_flux(gmag, yield_depth(ca, cb, hf, ty, g), mu, ty);
    if qy.x <= 0.0 { return 0.0; }
    var mob = min(max(qy.x / gmag - P.misc.w, 0.0), P.front.z);
    if min(ha, hb) < 0.5 * max(ha, hb) {
        if diag { return 0.0; }
        var tgt = cb;
        if hb > ha { tgt = ca; }
        mob = mob / pow(f32(max(feeders(tgt), 1u)), 0.7);
    }
    return mob;
}

fn kappa() -> f32 {
    return P.front.w * P.phys.x * P.g.z * P.inv_dx * P.inv_dx;
}

fn hold_u(c: u32) -> f32 {
    var hold = 0.0;
    if h[c] > 0.0 { hold = hold_level(nose_dist(c)); }
    return m[c] - hold;
}

@compute @workgroup_size(16, 16)
fn cg_setup(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let i = i32(id.x);
    let j = i32(id.y);
    let c = idx(id.x, id.y);
    var mob = vec4<f32>(0.0);
    let p00 = pressure_at(i, j);
    let gxy = P.phys.x * P.g.xy;
    let s2 = 0.70710678;
    if id.x + 1u < P.size.x && cell_active(id.x + 1u, id.y) {
        let cb = idx(id.x + 1u, id.y);
        let p10 = pressure_at(i + 1, j);
        let gm = -(p10 - p00) * P.inv_dx;
        let pt = pressure_wet(i, j + 1, p00) + pressure_wet(i + 1, j + 1, p10) - pressure_wet(i, j - 1, p00) - pressure_wet(i + 1, j - 1, p10);
        let gt = -pt * 0.25 * P.inv_dx;
        mob.x = face_mob(c, cb, vec2<f32>(gm.x, gt.x), vec2<f32>(gm.y, gt.y), gxy, false);
    }
    if id.y + 1u < P.size.y && cell_active(id.x, id.y + 1u) {
        let cb = idx(id.x, id.y + 1u);
        let p01 = pressure_at(i, j + 1);
        let gm = -(p01 - p00) * P.inv_dx;
        let pt = pressure_wet(i + 1, j, p00) + pressure_wet(i + 1, j + 1, p01) - pressure_wet(i - 1, j, p00) - pressure_wet(i - 1, j + 1, p01);
        let gt = -pt * 0.25 * P.inv_dx;
        mob.y = face_mob(c, cb, vec2<f32>(gm.x, gt.x), vec2<f32>(gm.y, gt.y), gxy.yx, false);
    }
    if id.x + 1u < P.size.x && id.y + 1u < P.size.y && cell_active(id.x + 1u, id.y + 1u) {
        let cb = idx(id.x + 1u, id.y + 1u);
        let p11 = pressure_at(i + 1, j + 1);
        let ga = -(p11 - p00) * P.inv_dx * s2;
        let pa = pressure_wet(i + 1, j, p00);
        let pb = pressure_wet(i, j + 1, p11);
        let gp = -(pa - pb) * P.inv_dx * s2;
        let gt = vec2<f32>(gxy.x + gxy.y, gxy.x - gxy.y) * s2;
        mob.z = face_mob(c, cb, vec2<f32>(ga.x, gp.x), vec2<f32>(ga.y, gp.y), gt, true);
    }
    if id.x + 1u < P.size.x && id.y > 0u && cell_active(id.x + 1u, id.y - 1u) {
        let cb = idx(id.x + 1u, id.y - 1u);
        let p1m = pressure_at(i + 1, j - 1);
        let ga = -(p1m - p00) * P.inv_dx * s2;
        let pa = pressure_wet(i + 1, j, p00);
        let pb = pressure_wet(i, j - 1, p1m);
        let gp = -(pa - pb) * P.inv_dx * s2;
        let gt = vec2<f32>(gxy.x - gxy.y, gxy.x + gxy.y) * s2;
        mob.w = face_mob(c, cb, vec2<f32>(ga.x, gp.x), vec2<f32>(ga.y, gp.y), gt, true);
    }
    cg_m[c] = mob;
}

struct Nb {
    c: u32,
    m: f32,
}

fn cg_nb(k: u32, i: u32, j: u32) -> Nb {
    var o: Nb;
    o.c = 0u;
    o.m = 0.0;
    let c = idx(i, j);
    switch k {
        case 0u: { if i + 1u < P.size.x { o.c = idx(i + 1u, j); o.m = W_AXIS * cg_m[c].x; } }
        case 1u: { if j + 1u < P.size.y { o.c = idx(i, j + 1u); o.m = W_AXIS * cg_m[c].y; } }
        case 2u: { if i + 1u < P.size.x && j + 1u < P.size.y { o.c = idx(i + 1u, j + 1u); o.m = W_DIAG_LAP * cg_m[c].z; } }
        case 3u: { if i + 1u < P.size.x && j > 0u { o.c = idx(i + 1u, j - 1u); o.m = W_DIAG_LAP * cg_m[c].w; } }
        case 4u: { if i > 0u { o.c = idx(i - 1u, j); o.m = W_AXIS * cg_m[o.c].x; } }
        case 5u: { if j > 0u { o.c = idx(i, j - 1u); o.m = W_AXIS * cg_m[o.c].y; } }
        case 6u: { if i > 0u && j > 0u { o.c = idx(i - 1u, j - 1u); o.m = W_DIAG_LAP * cg_m[o.c].z; } }
        default: { if i > 0u && j + 1u < P.size.y { o.c = idx(i - 1u, j + 1u); o.m = W_DIAG_LAP * cg_m[o.c].w; } }
    }
    return o;
}

fn lap_src(c: u32, which: u32) -> f32 {
    switch which {
        case 0u: { return cg_p[c]; }
        case 1u: { return h[c]; }
        case 2u: { return cg_t2[c]; }
        case 3u: { return cg_t[c]; }
        case 4u: { return cg_x[c] + hold_u(c); }
        default: { return hold_u(c); }
    }
}

fn lap_w(i: u32, j: u32, c: u32, which: u32) -> f32 {
    let ii = i32(i);
    let jj = i32(j);
    let vc = lap_src(c, which);
    var acc_v = 0.0;
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            if di == 0 && dj == 0 { continue; }
            let x = ii + di;
            let y = jj + dj;
            if x < 0 || y < 0 || x >= i32(P.size.x) || y >= i32(P.size.y) { continue; }
            if !cell_active(u32(x), u32(y)) { continue; }
            let n = idx(u32(x), u32(y));
            if h[n] <= 0.0 { continue; }
            var w = W_AXIS;
            if di != 0 && dj != 0 { w = W_DIAG_LAP; }
            acc_v += w * (lap_src(n, which) - vc);
        }
    }
    return acc_v;
}

fn lap_diag(i: u32, j: u32, c: u32) -> f32 {
    let ii = i32(i);
    let jj = i32(j);
    var s1 = 0.0;
    var s2 = 0.0;
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            if di == 0 && dj == 0 { continue; }
            let x = ii + di;
            let y = jj + dj;
            if x < 0 || y < 0 || x >= i32(P.size.x) || y >= i32(P.size.y) { continue; }
            if !cell_active(u32(x), u32(y)) { continue; }
            if h[idx(u32(x), u32(y))] <= 0.0 { continue; }
            var w = W_AXIS;
            if di != 0 && dj != 0 { w = W_DIAG_LAP; }
            s1 += w;
            s2 += w * w;
        }
    }
    return s1 * s1 + s2;
}

@compute @workgroup_size(16, 16)
fn cg_lap_p(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    cg_t[c] = lap_w(id.x, id.y, c, 0u);
}

@compute @workgroup_size(16, 16)
fn cg_lap_h(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    cg_t[c] = lap_w(id.x, id.y, c, 1u);
    cg_t2[c] = lap_w(id.x, id.y, c, 5u);
}

@compute @workgroup_size(16, 16)
fn cg_lap_u(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    cg_t[c] = lap_w(id.x, id.y, c, 4u);
}

fn apply_a(i: u32, j: u32, c: u32) -> f32 {
    let k = kappa();
    let pc = cg_p[c];
    var acc_v = pc;
    for (var f = 0u; f < 8u; f++) {
        let nb = cg_nb(f, i, j);
        if nb.m > 0.0 { acc_v += k * nb.m * (pc - cg_p[nb.c]); }
    }
    if h[c] > 0.0 { acc_v += P.front3.x * lap_w(i, j, c, 3u); }
    return acc_v;
}

var<workgroup> wg_sum: array<f32, 256>;
var<workgroup> wg_sum2: array<f32, 256>;

fn wg_reduce(li: u32, v: f32, slot: u32) {
    wg_sum[li] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if li < s { wg_sum[li] += wg_sum[li + s]; }
        workgroupBarrier();
    }
    if li == 0u { cg_part[slot] = wg_sum[0]; }
}

@compute @workgroup_size(16, 16)
fn cg_init(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let id = tile_cell(wid, lid);
    var v = 0.0;
    if in_grid(id) {
        let c = idx(id.x, id.y);
        let k = kappa();
        let hc = h[c];
        let uc = hold_u(c);
        var diag = 1.0;
        var b = hc;
        var ax = hc;
        for (var f = 0u; f < 8u; f++) {
            let nb = cg_nb(f, id.x, id.y);
            if nb.m > 0.0 {
                diag += k * nb.m;
                b += k * nb.m * (hold_u(nb.c) - uc);
                ax += k * nb.m * (hc - h[nb.c]);
            }
        }
        if hc > 0.0 {
            let cc = P.front3.x;
            diag += cc * lap_diag(id.x, id.y, c);
            b -= cc * lap_w(id.x, id.y, c, 2u);
            ax += cc * lap_w(id.x, id.y, c, 3u);
        }
        let r = b - ax;
        cg_d[c] = diag;
        cg_x[c] = hc;
        cg_r[c] = r;
        cg_p[c] = r / diag;
        v = r * r / diag;
    }
    wg_reduce(li, v, wid.x);
}

@compute @workgroup_size(16, 16)
fn cg_iter_a(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let id = tile_cell(wid, lid);
    var v = 0.0;
    if in_grid(id) {
        let c = idx(id.x, id.y);
        let q = apply_a(id.x, id.y, c);
        cg_q[c] = q;
        v = cg_p[c] * q;
    }
    wg_reduce(li, v, wid.x);
}

@compute @workgroup_size(16, 16)
fn cg_iter_b(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let id = tile_cell(wid, lid);
    var v = 0.0;
    if in_grid(id) {
        let c = idx(id.x, id.y);
        let alpha = cg_s[0].z;
        let r = cg_r[c] - alpha * cg_q[c];
        cg_x[c] = cg_x[c] + alpha * cg_p[c];
        cg_r[c] = r;
        v = r * r / cg_d[c];
    }
    wg_reduce(li, v, wid.x);
}

@compute @workgroup_size(16, 16)
fn cg_iter_c(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    let beta = cg_s[0].w;
    cg_p[c] = cg_r[c] / cg_d[c] + beta * cg_p[c];
}

fn total_parts(li: u32) -> f32 {
    let n = atomicLoad(&active_count[0]);
    var v = 0.0;
    for (var t = li; t < n; t += 256u) { v += cg_part[t]; }
    wg_sum[li] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if li < s { wg_sum[li] += wg_sum[li + s]; }
        workgroupBarrier();
    }
    return wg_sum[0];
}

@compute @workgroup_size(256)
fn cg_reduce_init(@builtin(local_invocation_index) li: u32) {
    let rz = total_parts(li);
    if li == 0u {
        cg_s[0] = vec4<f32>(rz, 0.0, 0.0, 0.0);
        cg_s[1] = vec4<f32>(rz, 0.0, 0.0, 0.0);
    }
}

@compute @workgroup_size(256)
fn cg_reduce_mass(@builtin(local_invocation_index) li: u32) {
    let n = atomicLoad(&active_count[0]);
    var d = 0.0;
    var tot = 0.0;
    for (var t = li; t < n; t += 256u) { d += cg_part[t]; tot += cg_part2[t]; }
    wg_sum[li] = d;
    wg_sum2[li] = tot;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if li < s { wg_sum[li] += wg_sum[li + s]; wg_sum2[li] += wg_sum2[li + s]; }
        workgroupBarrier();
    }
    if li == 0u {
        let s1 = cg_s[1];
        cg_s[1] = vec4<f32>(s1.x, wg_sum[0], wg_sum2[0], cg_s[0].x);
    }
}

@compute @workgroup_size(16, 16)
fn cg_rescale(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    let s1 = cg_s[1];
    if s1.z <= 0.0 { return; }
    let f = 1.0 - s1.y / s1.z;
    h[c] = max(h[c] * f, 0.0);
}

@compute @workgroup_size(256)
fn cg_reduce_a(@builtin(local_invocation_index) li: u32) {
    let pq = total_parts(li);
    if li == 0u {
        let s = cg_s[0];
        var alpha = 0.0;
        if pq > 0.0 { alpha = s.x / pq; }
        cg_s[0] = vec4<f32>(s.x, pq, alpha, 0.0);
    }
}

@compute @workgroup_size(256)
fn cg_reduce_b(@builtin(local_invocation_index) li: u32) {
    let rz = total_parts(li);
    if li == 0u {
        let s = cg_s[0];
        var beta = 0.0;
        if s.x > 0.0 { beta = rz / s.x; }
        cg_s[0] = vec4<f32>(rz, s.y, s.z, beta);
    }
}

fn pot(c: u32) -> f32 {
    return cg_x[c] + hold_u(c);
}

@compute @workgroup_size(16, 16)
fn cg_finish(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    begin_motion(li);
    let id = tile_cell(wid, lid);
    var moved = false;
    var dv = 0.0;
    var tv = 0.0;
    if in_grid(id) {
        let c = idx(id.x, id.y);
        let h0 = h[c];
        let hn = max(cg_x[c], 0.0);
        let rg = P.phys.x * P.g.z * P.inv_dx;
        let pc = pot(c);
        var f = vec4<f32>(0.0);
        let mob = cg_m[c];
        if mob.x > 0.0 { f.x = W_AXIS * mob.x * rg * (pc - pot(idx(id.x + 1u, id.y))); }
        if mob.y > 0.0 { f.y = W_AXIS * mob.y * rg * (pc - pot(idx(id.x, id.y + 1u))); }
        if mob.z > 0.0 { f.z = W_DIAG_FLUX * mob.z * rg * (pc - pot(idx(id.x + 1u, id.y + 1u))) * 0.70710678; }
        if mob.w > 0.0 { f.w = W_DIAG_FLUX * mob.w * rg * (pc - pot(idx(id.x + 1u, id.y - 1u))) * 0.70710678; }
        if h0 > 0.0 {
            let kc = P.front3.y;
            let tc = cg_t[c];
            if id.x + 1u < P.size.x { let n = idx(id.x + 1u, id.y); if h[n] > 0.0 && cell_active(id.x + 1u, id.y) { f.x += W_AXIS * kc * (cg_t[n] - tc); } }
            if id.y + 1u < P.size.y { let n = idx(id.x, id.y + 1u); if h[n] > 0.0 && cell_active(id.x, id.y + 1u) { f.y += W_AXIS * kc * (cg_t[n] - tc); } }
            if id.x + 1u < P.size.x && id.y + 1u < P.size.y { let n = idx(id.x + 1u, id.y + 1u); if h[n] > 0.0 && cell_active(id.x + 1u, id.y + 1u) { f.z += W_DIAG_FLUX * kc * (cg_t[n] - tc) * 0.70710678; } }
            if id.x + 1u < P.size.x && id.y > 0u { let n = idx(id.x + 1u, id.y - 1u); if h[n] > 0.0 && cell_active(id.x + 1u, id.y - 1u) { f.w += W_DIAG_FLUX * kc * (cg_t[n] - tc) * 0.70710678; } }
        }
        acc[c] += f * P.front.w;
        h[c] = hn;
        moved = abs(hn - h0) > MOVE_EPS;
        dv = hn - h0;
        tv = hn;
    }
    wg_sum2[li] = tv;
    wg_reduce(li, dv, wid.x);
    if li == 0u {
        var t = 0.0;
        for (var k = 0u; k < 256u; k++) { t += wg_sum2[k]; }
        cg_part2[wid.x] = t;
    }
    end_motion(moved, wid, li);
}

@compute @workgroup_size(16, 16)
fn pool8(@builtin(global_invocation_id) id: vec3<u32>) {
    let pw = (P.size.x + 3u) / 4u;
    let ph = (P.size.y + 3u) / 4u;
    if id.x >= pw || id.y >= ph { return; }
    var mx = 0.0;
    var ms = 0.0;
    for (var dy = 0u; dy < 4u; dy++) {
        for (var dxi = 0u; dxi < 4u; dxi++) {
            let i = id.x * 4u + dxi;
            let j = id.y * 4u + dy;
            if i < P.size.x && j < P.size.y {
                let c = idx(i, j);
                mx = max(mx, h[c] + m[c]);
                ms = max(ms, m[c]);
            }
        }
    }
    pool_out[id.y * pw + id.x] = mx;
    pool_out[pw * ph + id.y * pw + id.x] = ms;
}


var<workgroup> wg_hmax: array<f32, 256>;
var<workgroup> wg_mumin: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn reduce_tiles(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(local_invocation_id) lid: vec3<u32>) {
    var hm = 0.0;
    var mm = 1e9;
    for (var dy = 0u; dy < 4u; dy++) {
        for (var dxi = 0u; dxi < 4u; dxi++) {
            let i = wid.x * TILE + lid.x * 4u + dxi;
            let j = wid.y * TILE + lid.y * 4u + dy;
            if i < P.size.x && j < P.size.y {
                let c = idx(i, j);
                let v = h[c];
                if v > 4.0 * P.ein.w {
                    hm = max(hm, v);
                    mm = min(mm, rheo[c].x);
                }
            }
        }
    }
    wg_hmax[li] = hm;
    wg_mumin[li] = mm;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if li < s {
            wg_hmax[li] = max(wg_hmax[li], wg_hmax[li + s]);
            wg_mumin[li] = min(wg_mumin[li], wg_mumin[li + s]);
        }
        workgroupBarrier();
    }
    if li == 0u {
        stats_out[wid.x + wid.y * P.tiles.x] = vec2<f32>(wg_hmax[0], wg_mumin[0]);
    }
}

var<workgroup> wg_any: atomic<u32>;

@compute @workgroup_size(16, 16)
fn mark_tiles(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if li == 0u { atomicStore(&wg_any, 0u); }
    workgroupBarrier();
    let i = wid.x * 16u + lid.x;
    let j = wid.y * 16u + lid.y;
    if i < P.size.x && j < P.size.y {
        let c = idx(i, j);
        if h[c] > 0.0 {
            var bits = 1u;
            if lid.x < WAKE_BAND { bits |= 2u; }
            if lid.x >= 16u - WAKE_BAND { bits |= 4u; }
            if lid.y < WAKE_BAND { bits |= 8u; }
            if lid.y >= 16u - WAKE_BAND { bits |= 16u; }
            if abs(h[c] - h_frame[c]) > FRAME_EPS { bits |= MOVED; }
            atomicOr(&wg_any, bits);
        }
    }
    workgroupBarrier();
    if li == 0u {
        let t = wid.x + wid.y * tiles16().x;
        var flags = atomicLoad(&wg_any);
        if tile_moved[t] != 0u { flags |= MOVED; }
        tile_moved[t] = 0u;
        tile_flags[t] = flags;
    }
}

@compute @workgroup_size(256)
fn compact_tiles(@builtin(global_invocation_id) id: vec3<u32>) {
    let tw = tiles16();
    let nt = tw.x * tw.y;
    let t = id.x;
    if t >= nt { return; }
    let tx = i32(t % tw.x);
    let ty = i32(t / tw.x);
    var any = tile_flags[t] & MOVED;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dxi = -1; dxi <= 1; dxi++) {
            if dxi == 0 && dy == 0 { continue; }
            let x = tx + dxi;
            let y = ty + dy;
            if x >= 0 && y >= 0 && x < i32(tw.x) && y < i32(tw.y) {
                let fl = tile_flags[u32(x) + u32(y) * tw.x];
                var need = true;
                if dxi == 1 && (fl & 2u) == 0u { need = false; }
                if dxi == -1 && (fl & 4u) == 0u { need = false; }
                if dy == 1 && (fl & 8u) == 0u { need = false; }
                if dy == -1 && (fl & 16u) == 0u { need = false; }
                if need { any |= fl & MOVED; }
            }
        }
    }
    if any != 0u {
        let slot = atomicAdd(&active_count[0], 1u);
        tile_list[slot] = t;
        tile_flags[t] = tile_flags[t] | 32u;
    } else {
        tile_flags[t] = tile_flags[t] & 31u;
    }
}

@compute @workgroup_size(16, 16)
fn clear_acc(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    acc[c] = vec4<f32>(0.0);
    h_frame[c] = h[c];
}
