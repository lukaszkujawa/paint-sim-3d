use crate::film::FilmSettings;
use crate::pollock;

pub(crate) fn pollock_stats(seeds: u64) {
    let settings = FilmSettings::default();
    let dt = 1.0 / 60.0;
    let mut ratios = Vec::new();
    let mut rings = vec![0.0f32; 8];
    for seed in 0..seeds {
        let mut p = pollock::Painter::new(seed, settings.width_m as f64, settings.height_m as f64);
        while !p.done {
            p.step(dt);
        }
        let (edge, inner) = p.ink_balance();
        ratios.push(edge / inner.max(1e-6));
        for (k, v) in p.ink_rings().iter().enumerate() {
            rings[k] += v / seeds as f32;
        }
    }
    let mean = ratios.iter().sum::<f32>() / ratios.len().max(1) as f32;
    let min = ratios.iter().cloned().fold(f32::MAX, f32::min);
    let max = ratios.iter().cloned().fold(0.0f32, f32::max);
    println!("edge/interior ink over {} seeds: mean {:.2} min {:.2} max {:.2}", seeds, mean, min, max);
    let profile: Vec<String> = rings.iter().map(|v| format!("{:.2}", v)).collect();
    println!("ink density by ring, edge to centre: {}", profile.join(" "));
}
