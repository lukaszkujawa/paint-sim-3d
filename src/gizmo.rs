use crate::tools::{push_tube, Tool, ToolKind, TubeVertex};
use glam::{DVec2, DVec3};
use std::f64::consts::TAU;

const ACCENT: [f32; 3] = [0.85, 0.22, 0.02];
const GUIDE: [f32; 3] = [0.35, 0.36, 0.4];
const STEEL: [f32; 3] = [0.28, 0.3, 0.33];
const WOOD: [f32; 3] = [0.32, 0.17, 0.07];
const DARK: [f32; 3] = [0.012, 0.012, 0.014];
const PLASTIC: [f32; 3] = [0.55, 0.55, 0.52];
const TIN: [f32; 3] = [0.3, 0.31, 0.33];
const MARK_PX: f64 = 1.3;
const OUTLINE: f64 = 1.8;
const SURFACE_LIFT: f64 = 6e-4;
const SEGS: u32 = 20;
const KNIFE_TILT_DEG: f64 = 25.0;
const KNIFE_BLADE_LEN: f64 = 0.07;
const STICK_LEN: f64 = 0.32;
const STICK_COAT: f64 = 0.07;
const BOTTLE_R: f64 = 0.028;
const BOTTLE_H: f64 = 0.13;
const NOZZLE_H: f64 = 0.03;
const CAN_R: f64 = 0.075;
const CAN_H: f64 = 0.13;
const CAN_TIP_DEG: f64 = 55.0;

pub struct GizmoCtx {
    pub ground: Option<DVec3>,
    pub knife_dir: DVec2,
    pub paint: [f32; 3],
    pub px: f64,
    pub knife_gap: f64,
    pub sponge_radius: f64,
    pub knife_fill: f32,
    pub knife_paint: [f32; 3],
    pub pour_radius: f64,
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn v3(p: DVec3) -> [f32; 3] {
    [p.x as f32, p.y as f32, p.z as f32]
}

fn frame_of(axis: DVec3) -> (DVec3, DVec3) {
    let a = axis.normalize_or(DVec3::Z);
    let helper = if a.z.abs() < 0.9 { DVec3::Z } else { DVec3::X };
    let u = a.cross(helper).normalize();
    (u, a.cross(u))
}

pub fn push_frustum(verts: &mut Vec<TubeVertex>, idx: &mut Vec<u32>, a: DVec3, b: DVec3, ra: f64, rb: f64, col: [f32; 3], caps: bool) {
    let axis = b - a;
    let len = axis.length();
    if len <= 0.0 {
        return;
    }
    let (u, w) = frame_of(axis);
    let slope = (ra - rb) / len;
    let base = verts.len() as u32;
    for (c, r) in [(a, ra), (b, rb)] {
        for s in 0..SEGS {
            let t = s as f64 / SEGS as f64 * TAU;
            let radial = u * t.cos() + w * t.sin();
            let n = (radial + axis / len * slope).normalize();
            verts.push(TubeVertex { pos: v3(c + radial * r), nrm: v3(n), col });
        }
    }
    for s in 0..SEGS {
        let s1 = (s + 1) % SEGS;
        idx.extend_from_slice(&[base + s, base + SEGS + s, base + s1, base + s1, base + SEGS + s, base + SEGS + s1]);
    }
    if !caps {
        return;
    }
    for (c, r, n) in [(a, ra, -axis / len), (b, rb, axis / len)] {
        if r <= 0.0 {
            continue;
        }
        let centre = verts.len() as u32;
        verts.push(TubeVertex { pos: v3(c), nrm: v3(n), col });
        for s in 0..SEGS {
            let t = s as f64 / SEGS as f64 * TAU;
            verts.push(TubeVertex { pos: v3(c + (u * t.cos() + w * t.sin()) * r), nrm: v3(n), col });
        }
        for s in 0..SEGS {
            let s1 = (s + 1) % SEGS;
            idx.extend_from_slice(&[centre, centre + 1 + s, centre + 1 + s1]);
        }
    }
}

pub fn push_box(verts: &mut Vec<TubeVertex>, idx: &mut Vec<u32>, c: DVec3, hu: DVec3, hv: DVec3, hw: DVec3, col: [f32; 3]) {
    for (n, a, b) in [(hu, hv, hw), (-hu, hw, hv), (hv, hw, hu), (-hv, hu, hw), (hw, hu, hv), (-hw, hv, hu)] {
        let nn = n.normalize_or(DVec3::Z);
        let base = verts.len() as u32;
        for (sa, sb) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            verts.push(TubeVertex { pos: v3(c + n + a * sa + b * sb), nrm: v3(nn), col });
        }
        idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

fn push_mark(verts: &mut Vec<TubeVertex>, idx: &mut Vec<u32>, pts: &[DVec3], thick: f64, col: [f32; 3]) {
    let outer = thick * OUTLINE;
    let low: Vec<DVec3> = pts.iter().map(|p| DVec3::new(p.x, p.y, SURFACE_LIFT + outer)).collect();
    let high: Vec<DVec3> = pts.iter().map(|p| DVec3::new(p.x, p.y, SURFACE_LIFT + 2.0 * outer)).collect();
    push_tube(verts, idx, &low, outer, DARK);
    push_tube(verts, idx, &high, thick, col);
}

fn push_ring(verts: &mut Vec<TubeVertex>, idx: &mut Vec<u32>, c: DVec3, r: f64, thick: f64, col: [f32; 3]) {
    let n = 64;
    let pts: Vec<DVec3> = (0..=n).map(|i| c + DVec3::new((i as f64 / n as f64 * TAU).cos() * r, (i as f64 / n as f64 * TAU).sin() * r, 0.0)).collect();
    push_mark(verts, idx, &pts, thick, col);
}

fn push_dashes(verts: &mut Vec<TubeVertex>, idx: &mut Vec<u32>, a: DVec3, b: DVec3, dash: f64, thick: f64, col: [f32; 3]) {
    let len = (b - a).length();
    if len <= 0.0 {
        return;
    }
    let n = ((len / (2.0 * dash)).floor() as usize).clamp(1, 200);
    for i in 0..n {
        let t0 = (2 * i) as f64 * dash / len;
        let t1 = ((2 * i + 1) as f64 * dash / len).min(1.0);
        if t0 >= 1.0 {
            break;
        }
        push_tube(verts, idx, &[a.lerp(b, t0), a.lerp(b, t1)], thick, col);
    }
}

fn landing(verts: &mut Vec<TubeVertex>, idx: &mut Vec<u32>, tip: DVec3, spread: f64, active: bool, cx: &GizmoCtx) {
    let foot = DVec3::new(tip.x, tip.y, SURFACE_LIFT);
    let line = (cx.px * MARK_PX).max(3e-4);
    push_dashes(verts, idx, tip, foot, (tip.z * 0.04).clamp(0.002, 0.012), line, GUIDE);
    let col = if active { cx.paint } else { ACCENT };
    push_ring(verts, idx, foot, spread.max(cx.px * 6.0), line, col);
    push_mark(verts, idx, &[foot - DVec3::X * line * 3.0, foot + DVec3::X * line * 3.0], line, col);
    push_mark(verts, idx, &[foot - DVec3::Y * line * 3.0, foot + DVec3::Y * line * 3.0], line, col);
}

pub fn tool_gizmo(verts: &mut Vec<TubeVertex>, idx: &mut Vec<u32>, tool: &Tool, cx: &GizmoCtx) {
    let line = (cx.px * MARK_PX).max(3e-4);
    match tool.kind {
        ToolKind::Stick => {
            let tip = tool.pos;
            let lean = DVec3::new(-0.25, 0.35, 1.0).normalize();
            let top = tip + lean * STICK_LEN;
            let r = 0.5 * tool.stick_diameter;
            let fill = if tool.infinite { 1.0 } else { (tool.stick_load / tool.stick_capacity.max(1e-12)).clamp(0.0, 1.0) };
            let coat = STICK_COAT * fill;
            if coat > 1e-4 {
                push_frustum(verts, idx, tip, tip + lean * coat, r * 1.12, r * 1.06, cx.paint, false);
                let drop = tip - DVec3::Z * r * 0.6;
                push_frustum(verts, idx, tip, drop, r * 1.0, r * 0.25, cx.paint, true);
            }
            push_frustum(verts, idx, tip + lean * coat, top, r, r, WOOD, true);
            if coat <= 1e-4 {
                push_frustum(verts, idx, tip - lean * r * 0.3, tip, r * 0.7, r, WOOD, true);
            }
            landing(verts, idx, tip, (tool.stick_diameter * 0.6).max(0.004), tool.flowing, cx);
        }
        ToolKind::Bottle => {
            let tip = tool.pos;
            let lean = DVec3::new(-0.2, 0.25, 1.0).normalize();
            let shoulder = tip + lean * NOZZLE_H;
            push_frustum(verts, idx, tip, shoulder, tool.bottle_radius.max(5e-4) * 1.4, 0.007, PLASTIC, true);
            let body0 = shoulder + lean * 0.012;
            push_frustum(verts, idx, shoulder, body0, 0.007, BOTTLE_R, PLASTIC, false);
            push_frustum(verts, idx, body0, body0 + lean * BOTTLE_H, BOTTLE_R, BOTTLE_R, mix(cx.paint, [1.0, 1.0, 1.0], 0.25), true);
            landing(verts, idx, tip, (tool.bottle_radius * 3.0).max(0.004), tool.flowing, cx);
        }
        ToolKind::Bucket => {
            let lip = tool.pos;
            let tip = CAN_TIP_DEG.to_radians();
            let back = DVec3::new(-0.6, 0.8, 0.0).normalize();
            let axis = (back * tip.sin() + DVec3::Z * tip.cos()).normalize();
            let (u, _) = frame_of(axis);
            let down = (DVec3::Z - axis * axis.z).normalize_or(u);
            let mouth = lip + down * CAN_R;
            let bottom = mouth + axis * CAN_H;
            push_frustum(verts, idx, mouth, bottom, CAN_R, CAN_R * 0.92, TIN, false);
            push_frustum(verts, idx, bottom, bottom + axis * 0.002, CAN_R * 0.92, 0.0, TIN, false);
            push_frustum(verts, idx, mouth + axis * 0.01, mouth + axis * 0.012, CAN_R * 0.97, 0.0, cx.paint, false);
            let rim: Vec<DVec3> = (0..=40).map(|i| { let t = i as f64 / 40.0 * TAU; let (a, b) = frame_of(axis); mouth + (a * t.cos() + b * t.sin()) * CAN_R }).collect();
            push_tube(verts, idx, &rim, 0.002, STEEL);
            landing(verts, idx, lip, (tool.bucket_radius * 1.5).max(0.006), tool.flowing, cx);
        }
        ToolKind::Pour => {
            if let Some(g) = cx.ground {
                let c = DVec3::new(g.x, g.y, SURFACE_LIFT);
                push_ring(verts, idx, c, cx.pour_radius.max(line * 3.0), line, if tool.flowing { cx.paint } else { ACCENT });
                push_mark(verts, idx, &[c - DVec3::X * line * 3.0, c + DVec3::X * line * 3.0], line, ACCENT);
                push_mark(verts, idx, &[c - DVec3::Y * line * 3.0, c + DVec3::Y * line * 3.0], line, ACCENT);
            }
        }
        ToolKind::Knife => {
            let Some(g) = cx.ground else { return };
            let d = DVec3::new(cx.knife_dir.x, cx.knife_dir.y, 0.0).normalize_or(DVec3::X);
            let blade = DVec3::new(-d.y, d.x, 0.0);
            let hw = 0.5 * tool.knife_width;
            let tilt = KNIFE_TILT_DEG.to_radians();
            let back = -d * tilt.cos() + DVec3::Z * tilt.sin();
            let edge = DVec3::new(g.x, g.y, cx.knife_gap);
            let normal = blade.cross(back).normalize();
            let thick = 4e-4;
            let centre = edge + back * (0.5 * KNIFE_BLADE_LEN) + normal * thick;
            push_box(verts, idx, centre, blade * hw, back * (0.5 * KNIFE_BLADE_LEN), normal * thick, STEEL);
            let heel = edge + back * KNIFE_BLADE_LEN;
            let tip_l = edge - blade * hw;
            let tip_r = edge + blade * hw;
            let edge_col = if cx.knife_fill > 0.0 { mix(STEEL, cx.knife_paint, (0.4 + 0.6 * cx.knife_fill).min(1.0)) } else { STEEL };
            push_tube(verts, idx, &[tip_l, tip_r], 7e-4, edge_col);
            let neck = [heel, heel + DVec3::Z * 0.025 - d * 0.01, heel + DVec3::Z * 0.045 - d * 0.05];
            push_tube(verts, idx, &neck, 0.0025, STEEL);
            push_frustum(verts, idx, neck[2], neck[2] + (DVec3::Z * 0.06 - d * 0.1), 0.006, 0.008, WOOD, true);
            if !tool.flowing {
                let foot_l = DVec3::new(tip_l.x, tip_l.y, SURFACE_LIFT);
                let foot_r = DVec3::new(tip_r.x, tip_r.y, SURFACE_LIFT);
                push_mark(verts, idx, &[foot_l, foot_r], line * 0.7, ACCENT);
            }
        }
        ToolKind::Sponge => {
            if let (Some(g), false) = (cx.ground, tool.flowing) {
                push_ring(verts, idx, DVec3::new(g.x, g.y, SURFACE_LIFT), cx.sponge_radius, line * 0.7, ACCENT);
            }
        }
        ToolKind::Brush | ToolKind::Roller => {}
    }
}
