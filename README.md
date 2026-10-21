# paint-sim

Real-time, physically simulated oil/enamel paint on a canvas, built for Pollock-style pouring, dripping, throwing and flicking, with palette-knife scraping, subtractive pigment mixing and layered curing. Rust + wgpu (WGSL compute), tuned for an RTX 5090.

## Run

```
cargo run --release
cargo run --release -- --res 1536        (smaller grid, faster)
```

Scripted test scenes (render N frames, save a PNG, print a mass report, exit):

```
cargo run --release -- --auto 400 --script jet    --cam top   --shot jet.png
cargo run --release -- --auto 250 --script coil   --cam close --shot coil.png
cargo run --release -- --auto 200 --script bucket --cam top   --shot bucket.png
cargo run --release -- --auto 120 --script flick  --cam top   --shot flick.png
cargo run --release -- --auto 178 --script knife  --cam close --shot knife.png
cargo run --release -- --auto 320 --script brush  --cam blob  --shot brush.png
cargo run --release -- --auto 13000 --script pollock --cam top --shot pollock.png --beauty pollock_print.png
cargo run --release -- --auto 600 --script pollock --layer "black lasso" --cam top --shot lasso.png
cargo run --release -- --auto 800 --script lines --cam top --beauty lines.png
cargo run --release -- --auto 330 --script layers --cure 0.6 --cam blob --shot layers.png
cargo run --release -- --auto 400 --tilt 60 --shot tilt.png
```

## Controls

| Input | Action |
| --- | --- |
| 1 / 2 / 3 / 4 / 5 / 6 | stick, squeeze bottle, bucket, direct pour, palette knife, bristle brush |
| beauty render (top / view) | offline print-quality export: 6144x4096 (two pixels per film cell), 16 jittered samples per pixel, soft sun from a jittered light cone, no UI or tools; "top" is an orthographic view of the whole canvas, "view" uses the current camera. Paint edges are reconstructed below the cell size from the thickness field (cubic B-spline iso-contour at a smooth fraction of the local interior thickness, between the paint colour and what lies under it); in this mode the ray hits and normals use a Gaussian-reconstructed heightfield (sigma one cell) capped by a meniscus profile (H tanh(d tan(theta) / min(H, 2 l_c)) from the sub-cell edge), so rims read as a smooth bubble edge; the sun is integrated over a 0.1 rad cone with 36 samples per pixel. Before rendering, the paint relief is upsampled to a 2x finer grid (cubic B-spline) and relaxed by a short mean-curvature flow restricted to paint thicker than about 1 mm (src/render/refine.rs, shaders/refine.wgsl), the same idea as a subdivision pass at render time: it costs about two seconds and only smooths cell-scale steps, it cannot change the planform the solver produced. Saved as beauty_*.png next to the binary (also `--beauty file.png` on scripted runs) |
| Pollock: paint (experimental) | hands the tools to an automatic painter (src/pollock.rs): three bucket floods, a drying pause (the cure clock runs fast between "sessions" so later lines sit on dry paint instead of bleeding into wet washes), then about twenty layers of pours, drips, air shapes and spatter over roughly seven minutes of painting time (the time-scale slider runs the physics up to 4x faster per frame). Hand speed per gesture is drawn relative to the coiling-to-straight transition U* = 3.04 Q*^1/2 H*^3/4 of Palacios et al. (most strokes 1.2-2x above it, a few deliberately below it to loop), with Ornstein-Uhlenbeck tremor in speed and height, slowing in tight turns (the 2/3 power law), lasso sweeps and figure-8s at 0.2-0.4 m that lay circles and arcs from the air, whip snaps that accelerate to 3-6 m/s and stop dead so the thread flies off and breaks into drops, and thin-paint flings that splash on impact; stop it at any time and keep painting by hand. `--layer name` starts a scripted run at a given layer |
| raw canvas (on clear) | unprimed cotton-duck colour (reflectance 0.50-0.66) and a 25 um weave relief (period 1.4 mm) on the next clear; the weave gives spreading paint an irregular contact line instead of grid-aligned polygons |
| brush: LMB | press the bristles onto the canvas (wheel sets the press depth); release to lift. R dips the brush in the current mix. The brush scrapes wet paint into its load, blends it with what it carries, lays it back down along the stroke and pushes the surplus ahead |
| Left mouse | apply the tool (pour, squeeze, scrape) |
| Mouse wheel | tool height above the canvas (right button held: zoom) |
| + / - | zoom |
| Right drag | orbit camera |
| Middle drag | pan |
| R | dip the stick (reload) |
| Space | pause |
| F12 | screenshot |

The hand speed is read from the cursor: slow passes coil, fast passes draw straight lines and eventually break the thread into dashes; a sharp flick throws paint ahead of the tool. Paint mix sliders set the pigment volume fractions, thinner lowers viscosity, body scales the yield stress (0 = enamel, 1 = tube oil). Tilt the canvas to make thick paint run.

## Physics

Airborne paint is a discrete viscous thread (Lagrangian rod, implicit viscous stretching and bending with an objective bending rate, sloped-area surface tension, air drag, adaptive sub-stepping, volume-conserving split/merge, pinch-off into drops). Threads freeze where they touch the surface and are laid into the film as tapered bands whose radius and thickness are continuous from one thread edge to the next (one capsule per edge gave every line a 2 mm ribbing); drops land with a viscous-regime spreading footprint. The surface a thread lands on is a 4x max-pooled copy of the film plus the exact parabolic profiles of the capsules laid in the last few frames (hashed on the pooled grid, absolute top heights so piles stack immediately); the earlier bounding-box "recent" map inflated the surface ahead of the contact point and made dragged threads land in frame-locked pulses, the bead chains that looked like a coiled telephone cord.

Paint on the canvas is a Herschel–Bulkley thin film (yield stress, consistency, flow index) on a 0.39 mm grid: capillary, hydrostatic, tilt and disjoining-pressure driving terms, explicit flux-form update with per-face mobility caps tied to the sub-step, a per-cell outflow limiter, adaptive sub-steps, and an active-tile list so cost scales with the painted area. Fluxes run on an 8-neighbour stencil (axis faces weighted 2/3, diagonal faces 1/6, the isotropic nine-point form), so puddles and drops spread as circles instead of the octagons a four-neighbour scheme produces; at the wetting front (a cell holding less than half of its neighbour) only axis faces carry paint and the flux into the thin cell is divided by the number of feeding neighbours to the power 0.7, which removes the direction dependence of the front speed (a front facing a diagonal is fed by two cells, one facing an axis by one), so large pours spread as smooth rounded shapes instead of regular octagons. The wetting front of thick paint is a dam-break on this grid (one-cell cliffs, polygonal outlines), so a slump pass follows each sub-step: cells taller than the capillary-held height 2 l_c sin(theta/2) shed, conservatively and carrying their pigment, whatever exceeds a maximum front slope (tan of the contact angle, or the yield-stress slope tau_y / rho g h for body paint) one step along the local downhill gradient (bilinear split, so the direction is not tied to the grid), with the allowed height growing with the Euclidean distance to the dry edge (nearest-dry-cell offsets propagated every sub-step). Puddles then come out round with a rounded shoulder instead of an octagon with a cliff; a chamfer distance gave octagonal piles and neighbour-based shedding gave square ones. Pigment travels in up to four material strata per cell (fresh paint rides on older paint, plug flow carries them together), with a back-fill so advancing fronts inherit composition. Curing raises yield stress and viscosity and commits the bottom stratum into the substrate relief and reflectance.

Colour is computed by 16-band Kubelka–Munk compositing of each stratum over the substrate, with a Saunderson surface correction (wet paint darker and more saturated), then CIE XYZ to sRGB. The surface is shaded with a GGX clear coat, heightfield normals, horizon shadows and a sky/ground environment. Pigment absorption and scattering tables live in `assets/pigments.ron`, seeded from literature reflectance curves, and can be edited.

## Layout

```
shaders/   common, film (flux solver, tiles, stats), strata, deposit, knife, brush, cure, km, shade, render, tube
src/film   GPU film state, parameters, per-frame encoding, readbacks
src/rod    viscous thread model and banded solver (unit tests: cargo test)
src/tools  tools, coarse contact surface, airborne update, tube meshes
src/color  pigments, Kubelka–Munk tables, colour matching
src/render canvas and tube render pipelines, camera
```
