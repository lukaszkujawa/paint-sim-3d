# Continuous ribbons, multiscale surface detail, PBR review

Date: 2026-10-24

## Findings

Baseline renders (`shots/before/`) of the `coil` and `lines` scenes show two artefacts:

1. Sub-millimetre ribbing along every landed thread. `Deposit::band` turns each
   frozen rod edge into a butt-jointed band. Consecutive edges carry volumes that
   differ by 2-8x (edge volumes halve at every remesh split, landed lengths do
   not scale with them), so the height per unit length jumps at every joint, and
   kinks between segments leave wedge gaps on the outside and double coverage on
   the inside.
2. Beads every few millimetres to centimetres. The landed edge data itself is
   periodic. Instrumenting the rod showed the cause: the chain-freeze rule
   pinned a vertex as soon as it came within 2.4 radii of its pinned neighbour,
   up to three times that above the surface, so the thread stood on its landing
   point as a frozen column that was emitted as a pile, after which the nozzle
   tore the thread above into a hair until it landed again. Removing airborne
   pinning (a vertex freezes only on contact with the film or with paint laid
   earlier) gives continuous laying with volume per unit length varying by at
   most about 2x.

## Part 1: ribbon deposition

- `Rod` owns a `Ribbon` (`src/rod/ribbon.rs`). `emit_front` appends the landed
  vertex (xy, edge volume, impact speed) instead of emitting one landing per
  edge. Fragments created by `split_at` take the ribbon with them (the lower
  fragment keeps the landed prefix).
- `Ribbon::drain(step, flush)` resamples the landed polyline by arc length with
  a step equal to the local footprint radius (never below one film cell), so
  joints never tilt by more than the segment length allows. Volume per unit length comes from the smoothed
  cumulative volume `V_w(s) = mean of V over [s - w, s + w]`, so every emitted
  segment carries `V_w(s1) - V_w(s0)` and the sum telescopes to the landed total
  exactly. The window half-width `w` is the local footprint half-width
  `R = f * sqrt(lambda / pi)` (the scale over which an impacting thread
  redistributes itself), never smaller than one step. Segments are emitted only
  once the data they depend on is complete (`s1 + w <= s_max`) or on flush.
- Footprint radius and impact speed are interpolated per node; height follows
  from mass conservation (parabolic cross-section, `h = k R (1 - x^2)`).
- `Deposit` becomes a general ribbon segment: `p0, p1, n0, n1, r0, r1, h0, vol,
  kind, pig`. `n0`/`n1` are the joint bisector directions; `deposit.wgsl` bounds
  each segment by the bisector half-planes (mitre joints) so neighbours tile the
  ribbon with no gaps or overlaps, and uses the perpendicular distance to the
  segment line inside that region. `kind` holds cap bits (round cap at start /
  end) used for ribbon ends and drops. Volume is stored explicitly so the mass
  report does not re-derive it from the geometry.

## Part 2: procedural surface detail

- A per-cell tag buffer (deposition method + lay direction) is written by the
  deposit pass; `shade.wgsl` writes a second texture with lay direction, method
  and top-stratum cure, and the roughness now varies with cure and thinner
  content.
- `render.wgsl` adds a procedural displacement `d(p)` evaluated in world space:
  bristle grooves across the stroke direction for brushed paint (depth limited by
  the local thickness), faint longitudinal striations for laid threads, and
  curing-skin wrinkles whose amplitude grows with cure and thickness. The
  displacement enters the ray-hit refinement and the normal (true displacement
  near the hit, bump-mapped normal), so silhouettes and shading agree at any zoom.

## Part 3: shading

- GGX with height-correlated Smith visibility, two lobes (clear coat on wet paint,
  broader pigment lobe on dry paint), roughness-dependent environment blur,
  spatially varying roughness from the detail field, and specular occlusion from
  the horizon term.

## Validation

`coil`, `lines`, `jet`, `knife`, `layers`, `brush` at identical cameras in
`shots/before` and `shots/after`: `--beauty` top view, `--view` perspective
close-up, `--cam macro`. Mass report error is 0.00% for thread scenes (was
0.04%); the brush scene reports about -4% before and after (its reservoir
accounting, untouched).
