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
cargo run --release -- --auto 400 --script jet    --cam macro --view jet_macro.png      (--view: perspective beauty render from the chosen camera)
cargo run --release -- --auto 250 --script coil   --cam close --shot coil.png
cargo run --release -- --auto 200 --script bucket --cam top   --shot bucket.png
cargo run --release -- --auto 120 --script flick  --cam top   --shot flick.png
cargo run --release -- --auto 178 --script knife  --cam close --shot knife.png
cargo run --release -- --auto 320 --script brush  --cam blob  --shot brush.png
cargo run --release -- --auto 7000 --script pollock --seed 18 --cam top --shot pollock.png --beauty pollock_print.png
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
| Pollock: paint (experimental) | hands the tools to an automatic painter (src/pollock.rs) that plans a different session every time. It first draws a palette from one of six families (autumn rhythm, blue poles, convergence, earth, monochrome, lavender mist), jittering every mix, then picks how to prepare the ground: raw canvas, bucket floods, swept pours from the pour tool, pools spread with the palette knife, or pools run downhill by tilting the canvas for a few seconds. After a drying pause (the cure clock runs fast between sessions so later lines sit on dry paint) it lays a skeleton in the main colour, three to six middle layers of pours, drips, lassos, snaps, flings and dots in the second and accent colours, one to three accent layers of drops and spatter, and usually a final tracery, with drying pauses between thick layers. The plan is printed at the start and shown in the panel. Hand speed per gesture is drawn relative to the coiling-to-straight transition U* = 3.04 Q*^1/2 H*^3/4 of Palacios et al. (most strokes 1.2-2x above it, a few deliberately below it to loop), with Ornstein-Uhlenbeck tremor in speed and height, slowing in tight turns, lasso sweeps and figure-8s that lay circles and arcs from the air, whip snaps that accelerate to 3-6 m/s and stop dead so the thread flies off and breaks into drops, and thin-paint flings that splash on impact; stop it at any time and keep painting by hand. `--seed N` picks the plan for a scripted run and `--layer name` starts at a given layer |
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

Airborne paint is a discrete viscous thread (Lagrangian rod, implicit viscous stretching and bending with an objective bending rate, sloped-area surface tension, air drag, adaptive sub-stepping, volume-conserving split/merge, pinch-off into drops). Threads freeze where they touch the surface, and only there. The nozzle carries a jitter of 0.3% of its radius per sub-step; without any perturbation a perfectly straight falling thread never buckles, with it the thread coils by itself (about 1.3 mm radius for the bottle) and the moving-nozzle regimes come out as in Palacios et al.: coils and meanders when the hand is slower than U* = 3.04 Q*^1/2 H*^3/4 (about 0.8 m/s for the bottle at 10 cm), straight lines above it (`--script lines` lays seven lines from 0.3 to 2.0 m/s).

The frozen prefix is laid into the film as a continuous ribbon (src/rod/ribbon.rs): the landed polyline, its volume and its impact speed are resampled by arc length with a step equal to the local footprint radius, with mitred joints between segments and per-segment heights computed on the deposit lattice so GPU mass equals CPU mass. When the thread stacks on itself faster than three thread cross-sections per unit length (a held nozzle, a coil, a very slow drag) the ribbon emits round piles instead, sized from the pending volume, so a held pour grows a mound continuously; previously the ribbon waited for arc length that never came and dumped the whole pour as one disc when the thread detached. Drops land with a viscous-regime spreading footprint. The surface a thread lands on is a 4x max-pooled copy of the film plus the exact parabolic profiles of the capsules laid in the last few frames.

Paint on the canvas is a Herschel–Bulkley thin film (yield stress, consistency, flow index) on a 0.39 mm grid with gravity, tilt, capillary and disjoining-pressure terms, pigment strata, an active-tile list (tiles sleep when nothing in them moved more than 2 um in a frame) and an explicit flux-form update with a per-cell outflow limiter. Flux between an active cell and a sleeping cell is zero in both directions, which is what keeps mass exact at tile borders (an earlier version applied the active side only and created or destroyed up to 5% of a tilted pour).

The contact line is modelled explicitly instead of through the biharmonic term. Every wet cell carries its Euclidean distance d to the nearest dry cell (a validated one-step propagation; the old propagation collapsed to zero wherever two cells pointed at each other and imprinted that pattern on the paint). The hydrostatic pressure is taken from the excess over the equilibrium meniscus profile, hold(d) = H_c (1 - exp(-(d - 1/2) dx / l_c)) with H_c = 2 l_c sin(theta/2) the capillary puddle height, so a puddle at rest is exactly a flat pancake with a meniscus rim and nothing drives flow in that state; a 7 mL enamel puddle settles to 1.87 mm. The contact line advances with a Cox–Voinov law, U = sigma (theta_app^3 - theta_adv^3) / (132 mu), where the apparent angle grows with the excess of the rim over the meniscus (sampled 4, 8 and 16 cells inside) and theta_adv is the equilibrium angle plus a hysteresis (15 deg by default) that pins small excesses instead of letting thin films creep. Rim cells carry the translation flux U h (n . e) of the moving profile, where n is a smooth front normal stored per cell; the flux that wets a dry cell is the same translation projected on the face and normalised by the front orientation (so axis-aligned and diagonal fronts advance at the same speed), multiplied by a front-shape factor (the wet count in a radius-3 disc over the count a straight front with that normal would have, to the power 1.5, so tips stall and notches fill). Measured on a settled puddle the radius varies by about 2% with direction. The old slump pass (cone-cap shedding) is off by default.

The capped explicit capillary term still runs between wet cells, on the deviation from the meniscus profile (so it does not fight the equilibrium), scaled by (thin/thick)^2.5 for unequal pairs; it is what selects the finger wavelength of paint running down a tilted canvas (without it the fingers are two cells wide and perfectly regular). Mobility beyond what the explicit sub-step can carry (thick pours of fluid paint) is handled by an implicit hydrostatic step: a Jacobi-preconditioned conjugate-gradient solve per frame over the active tiles for the mobility excess over the explicit cap, with the mobility bounded so that kappa M <= 40 per implicit step (up to four steps per frame) and a rescale that removes the drift of a truncated solve; the UI shows the residual and drift. Pigment travels in up to four material strata per cell with a back-fill so advancing fronts inherit composition. Curing raises yield stress and viscosity and commits the bottom stratum into the substrate relief and reflectance. The palette knife pushes its scrapings as one bow-wave ribbon ahead of the blade (width from the scraped volume) rather than a wall of cell-sized capsules.

Scripted scenes for checking this: `hold` (bottle held still, pours in pulses), `settle` (pour 5 s then rest), `pour` (direct pour tool: hold, slow drag, fast drag; add `--tilt 45` for drips), `lines`, `coil`, `bucket --stop N` (stop the flow at frame N), `--mu`, `--yield`, `--tilt`. `PAINT_DUMP=prefix` writes the height, relief and distance rasters at the shot; environment toggles `PAINT_NO_HOLD`, `PAINT_NO_TRANSL`, `PAINT_NO_SHAPE`, `PAINT_NO_CAP`, `PAINT_NO_IMPLICIT`, `PAINT_FLAT_SURFACE`, `PAINT_NO_RECENT`, `PAINT_HYST`, `PAINT_FCAP`, `PAINT_KSHAPE` switch the individual mechanisms off or retune them for ablation runs.

Colour is computed by 16-band Kubelkaâ€“Munk compositing of each stratum over the substrate, with a Saunderson surface correction (wet paint darker and more saturated), then CIE XYZ to sRGB. The surface is shaded with GGX (height-correlated Smith visibility, a second broad lobe that grows with curing, roughness-blurred environment with horizon occlusion), heightfield normals, horizon shadows and a sky/ground environment. Roughness varies over the surface: wet paint is glossy, cured paint and cured paint committed into the substrate are semi-matte, bare canvas is matte, and films thinner than the weave go matte. A procedural detail layer (shaders/render.wgsl) adds microgeometry below the cell size at render time, as true displacement in the ray hit and in the normal: bristle grooves across brushed paint and finer scratches on knifed paint, whose depth is limited by the local thickness and by the yield stress needed to hold the groove against capillary levelling, and a labyrinthine wrinkle on thick curing films. Each deposit pass records how and in which direction a cell was last laid (thread, drop, pour, brush, knife); the shade pass writes that, the cure and the yield stress to a detail texture the renderer samples bilinearly. Detail that falls below the pixel footprint fades out, so it costs nothing in wide views. Pigment absorption and scattering tables live in `assets/pigments.ron`, seeded from literature reflectance curves, and can be edited.

## Layout

```
shaders/   common, film (flux solver, tiles, stats), strata, deposit, knife, brush, cure, km, shade, render, tube
src/film   GPU film state, parameters, per-frame encoding, readbacks
src/rod    viscous thread model and banded solver (unit tests: cargo test)
src/tools  tools, coarse contact surface, airborne update, tube meshes
src/color  pigments, Kubelkaâ€“Munk tables, colour matching
src/render canvas and tube render pipelines, camera
```
