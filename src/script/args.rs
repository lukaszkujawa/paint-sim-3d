
#[derive(Clone)]
pub(crate) struct AutoScript {
    pub(crate) frames: u64,
    pub(crate) shot: String,
    pub(crate) beauty: String,
    pub(crate) view: String,
    pub(crate) tilt: f32,
    pub(crate) cam: String,
    pub(crate) script: String,
    pub(crate) debug: bool,
    pub(crate) cure: f32,
    pub(crate) pace: bool,
    pub(crate) sigma: f32,
    pub(crate) angle: f32,
    pub(crate) yield_pa: f32,
    pub(crate) mu: f64,
    pub(crate) pig: i32,
    pub(crate) layer: String,
    pub(crate) seed: u64,
    pub(crate) dry: u32,
    pub(crate) export: String,
    pub(crate) stop: u64,
    pub(crate) rounds: u32,
    pub(crate) snap_save: String,
    pub(crate) snap_load: String,
    pub(crate) camat: String,
    pub(crate) vres: [u32; 2],
    pub(crate) spp: u32,
    pub(crate) light: Vec<f32>,
    pub(crate) aces: bool,
    pub(crate) agx: bool,
    pub(crate) show_ui: bool,
}

pub(crate) fn parse_auto() -> Option<AutoScript> {
    parse_args(&std::env::args().collect::<Vec<_>>())
}

fn parse_args(args: &[String]) -> Option<AutoScript> {
    let mut frames = None;
    let mut shot = "auto.png".to_string();
    let mut beauty = String::new();
    let mut view = String::new();
    let mut tilt = 0.0f32;
    let mut cam = String::new();
    let mut script = String::new();
    let mut debug = false;
    let mut cure = -1.0f32;
    let mut pace = false;
    let mut sigma = -1.0f32;
    let mut yield_pa = -1.0f32;
    let mut pig = -1i32;
    let mut layer = String::new();
    let mut seed: u64 = 7;
    let mut dry: u32 = 0;
    let mut export = String::new();
    let mut mu = -1.0f64;
    let mut angle = -1.0f32;
    let mut stop = 0u64;
    let mut rounds = 0u32;
    let mut snap_save = String::new();
    let mut snap_load = String::new();
    let mut camat = String::new();
    let mut vres = [3072u32, 2048u32];
    let mut spp = 64u32;
    let mut light = Vec::new();
    let mut aces = false;
    let mut agx = false;
    let mut show_ui = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--auto" => { frames = Some(args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(200)); i += 1; }
            "--shot" => { shot = args.get(i + 1).cloned().unwrap_or(shot); i += 1; }
            "--beauty" => { beauty = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--view" => { view = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--tilt" => { tilt = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.0); i += 1; }
            "--cam" => { cam = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--script" => { script = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--debug" => { debug = true; }
            "--pace" => { pace = true; }
            "--sigma" => { sigma = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(-1.0); i += 1; }
            "--pig" => { pig = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(-1); i += 1; }
            "--layer" => { layer = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--seed" => { seed = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(7); i += 1; }
            "--dry" => { dry = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0); i += 1; }
            "--export" => { export = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--yield" => { yield_pa = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(-1.0); i += 1; }
            "--mu" => { mu = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(-1.0); i += 1; }
            "--angle" => { angle = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(-1.0); i += 1; }
            "--cure" => { cure = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.0); i += 1; }
            "--stop" => { stop = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0); i += 1; }
            "--aces" => { aces = true; }
            "--agx" => { agx = true; }
            "--showui" => { show_ui = true; }
            "--light" => { light = args.get(i + 1).map(|s| s.split(',').filter_map(|t| t.parse().ok()).collect()).unwrap_or_default(); i += 1; }
            "--rounds" => { rounds = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0); i += 1; }
            "--snap-save" => { snap_save = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--snap-load" => { snap_load = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--camat" => { camat = args.get(i + 1).cloned().unwrap_or_default(); i += 1; }
            "--spp" => { spp = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(64); i += 1; }
            "--vres" => {
                let v: Vec<u32> = args.get(i + 1).map(|s| s.split('x').filter_map(|t| t.parse().ok()).collect()).unwrap_or_default();
                if v.len() == 2 { vres = [v[0], v[1]]; }
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    frames.map(|frames| AutoScript { frames, shot, beauty, view, tilt, cam, script, debug, cure, pace, sigma, angle, yield_pa, mu, pig, layer, seed, dry, export, stop, rounds, snap_save, snap_load, camat, vres, spp, light, aces, agx, show_ui })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(line: &str) -> Option<AutoScript> {
        let args: Vec<String> = std::iter::once("canvaz").chain(line.split_whitespace()).map(String::from).collect();
        parse_args(&args)
    }

    #[test]
    fn interactive_without_auto() {
        assert!(parsed("--script brush --agx").is_none());
    }

    #[test]
    fn reads_options() {
        let a = parsed("--auto 5 --script brush --spp 3 --agx --showui --shot x.png --beauty y.png").unwrap();
        assert_eq!((a.frames, a.script.as_str(), a.spp, a.agx, a.show_ui, a.shot.as_str(), a.beauty.as_str()), (5, "brush", 3, true, true, "x.png", "y.png"));
    }

    #[test]
    fn defaults_stay_off() {
        let a = parsed("--auto 5").unwrap();
        assert!(!a.agx && !a.aces && !a.show_ui && a.beauty.is_empty());
    }
}
