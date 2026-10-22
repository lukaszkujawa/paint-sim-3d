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
    let l = (surf(i - 1, j) + surf(i + 1, j) + surf(i, j - 1) + surf(i, j + 1) - 4.0 * s0) * P.inv_dx * P.inv_dx;
    pres[c] = vec2<f32>(-P.phys.y * l + disjoining(hv), P.phys.x * P.g.z * s0);
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

fn face(ca: u32, cb: u32, gcap: vec2<f32>, ghyd: vec2<f32>, gtilt: vec2<f32>, diag: bool) -> FaceFlux {
    var out: FaceFlux;
    out.q = 0.0;
    out.y = 0.0;
    out.up = 0.0;
    let ha = h[ca];
    let hb = h[cb];
    let hf = 0.5 * (ha + hb);
    if hf <= 0.0 { return out; }
    if min(ha, hb) <= 0.0 {
        let wet = select(cb, ca, ha > 0.0);
        if h[wet] <= nose_cap(wet) { return out; }
    }
    let ra = rheo[ca];
    let rb = rheo[cb];
    let wa = ha / (ha + hb);
    let wb = 1.0 - wa;
    let mu = exp(wa * log(max(ra.x, 1e-4)) + wb * log(max(rb.x, 1e-4)));
    let ty = wa * ra.y + wb * rb.y;
    let thin = min(ha, hb) / hf;
    let fcap = thin * thin * thin;
    let g = gcap * fcap + ghyd + gtilt;
    let gmag = length(g);
    let qy = hb_flux(gmag, hf, mu, ty);
    if qy.x <= 0.0 { return out; }
    let meff = qy.x / gmag;
    let qvec = meff * gtilt + min(meff, P.misc.w) * ghyd + min(meff, P.phys.w) * (gcap * fcap);
    var qn = qvec.x;
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
    out.q = qn;
    out.y = qy.y;
    out.up = qy.z;
    return out;
}

const FRONT_DIAG: f32 = 0.5;

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
    let wa = 2.0 / 3.0;
    let wd = s2 / 3.0;
    if id.x + 1u < P.size.x {
        let cb = idx(id.x + 1u, id.y);
        let p10 = pressure_at(i + 1, j);
        let gm = -(p10 - p00) * P.inv_dx;
        let pt = pressure_wet(i, j + 1, p00) + pressure_wet(i + 1, j + 1, p10) - pressure_wet(i, j - 1, p00) - pressure_wet(i + 1, j - 1, p10);
        let gt = -pt * 0.25 * P.inv_dx;
        fx = face(c, cb, vec2<f32>(gm.x, gt.x), vec2<f32>(gm.y, gt.y), gxy, false);
        fx.q *= wa;
    }
    if id.y + 1u < P.size.y {
        let cb = idx(id.x, id.y + 1u);
        let p01 = pressure_at(i, j + 1);
        let gm = -(p01 - p00) * P.inv_dx;
        let pt = pressure_wet(i + 1, j, p00) + pressure_wet(i + 1, j + 1, p01) - pressure_wet(i - 1, j, p00) - pressure_wet(i - 1, j + 1, p01);
        let gt = -pt * 0.25 * P.inv_dx;
        fy = face(c, cb, vec2<f32>(gm.x, gt.x), vec2<f32>(gm.y, gt.y), gxy.yx, false);
        fy.q *= wa;
    }
    if id.x + 1u < P.size.x && id.y + 1u < P.size.y {
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
            fd = face(c, cb, vec2<f32>(ga.x, gp.x), vec2<f32>(ga.y, gp.y), gt, true);
            fd.q *= wd;
        }
    }
    if id.x + 1u < P.size.x && id.y > 0u {
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
            fe = face(c, cb, vec2<f32>(ga.x, gp.x), vec2<f32>(ga.y, gp.y), gt, true);
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
        return;
    }
    let i = i32(id.x);
    let j = i32(id.y);
    var best = vec2<i32>(64, 64);
    var bd = 1e9;
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            if di == 0 && dj == 0 { continue; }
            let x = i + di;
            let y = j + dj;
            if x < 0 || y < 0 || x >= i32(P.size.x) || y >= i32(P.size.y) { continue; }
            let n = idx(u32(x), u32(y));
            var cand = vec2<i32>(di, dj);
            if h[n] > 0.0 {
                let o = unpack_off(delta[n]);
                cand = vec2<i32>(o.x + di, o.y + dj);
            }
            let d = length(vec2<f32>(f32(cand.x), f32(cand.y)));
            if d < bd { bd = d; best = cand; }
        }
    }
    delta[c] = pack_off(best.x, best.y);
}

@compute @workgroup_size(16, 16)
fn slump(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    let i = i32(id.x);
    let j = i32(id.y);
    let hc = h[c];
    let mine = shed_of(i, j);
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
            let sn = shed_of(x, y);
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
fn apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
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
    let hn = max(h0 - P.dt * st.div * P.inv_dx, 0.0);
    h[c] = hn;
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
}

@compute @workgroup_size(16, 16)
fn commit_tmp(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = tile_cell(wid, lid);
    if !in_grid(id) { return; }
    let c = idx(id.x, id.y);
    h[c] = tmp[c];
}

@group(2) @binding(0) var<storage, read_write> stats_out: array<vec2<f32>>;
@group(2) @binding(1) var<storage, read_write> pool_out: array<f32>;

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
        if h[idx(i, j)] > 0.0 {
            var bits = 1u;
            if lid.x < 2u { bits |= 2u; }
            if lid.x > 13u { bits |= 4u; }
            if lid.y < 2u { bits |= 8u; }
            if lid.y > 13u { bits |= 16u; }
            atomicOr(&wg_any, bits);
        }
    }
    workgroupBarrier();
    if li == 0u {
        tile_flags[wid.x + wid.y * tiles16().x] = atomicLoad(&wg_any);
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
    var any = tile_flags[t] & 1u;
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
                if need { any |= fl & 1u; }
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
}
