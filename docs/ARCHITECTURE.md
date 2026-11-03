# Canvaz code layout

`src/main.rs` only declares modules and starts the event loop.

| Module | Responsibility |
|---|---|
| `app/core.rs` | `App` state and start-up |
| `app/frame.rs` | The per-frame sequence (`step`), timing, cursor tracking, render passes |
| `app/events.rs` | Window, mouse, pen and keyboard events |
| `app/tool_frame.rs` | Per-frame work for brush, roller, sponge and rubber |
| `app/knife.rs` | Palette knife load, per-frame work and scraped paint |
| `app/brushes.rs` | Brush building and pressure, pen input, the Pollock painter hookup |
| `app/canvas.rs` | Creating and rebuilding the canvas |
| `app/capture.rs` | Screenshots, GPU readback, snapshots, Blender export |
| `app/pointer.rs` | Cursor to canvas mapping, panel hit test, eyedropper sampling |
| `app/stats.rs` | Diagnostic statistics |
| `script/` | Scripted test runs: `args.rs` parses options, `scenes/` holds one function per scene, `input.rs` drives the cursor, `finish.rs` writes results at the end of a run |
| `evolve.rs`, `evolve/` | Painting from a photo: `colour`, `filters`, `focus`, `photo`, `plan` and `driver` are shared, `sketch` and `evolver` are the planner |
| `film/` | GPU paint simulation: `mod.rs` owns the buffers and the frame, `deposit` paint shapes, `settings` parameters, `substrate` canvas, `journal` undo, `readback` CPU copies |
| `panels.rs`, `panels/` | Interface: `panels.rs` is the frame and menus, the rest is one file per inspector section |
| `pollock.rs`, `pollock/` | Pollock painter: `colour` mixing, `palette` schemes, `layers` plans |
| `brush`, `roller`, `sponge`, `tools`, `rod` | Tool physics |
| `color`, `render`, `gpu`, `gizmo`, `session` | Pigments, rendering, GPU helpers, tool meshes, saving |

## The photo painter

**Paint** (`evolver.rs`, `sketch.rs`) searches for strokes that reduce colour error and rolls back any that do not help. It uses the tool and size you select.

## Testing

Run these after changing anything:

```
cargo test --release
python tools/regress.py capture --exe target/release/canvaz.exe --out OUT --photo PHOTO
python tools/regress.py compare REFERENCE OUT
python tools/gallery.py
```

- `regress.py` runs every scene through the real program and compares images pixel by pixel. Capture the reference twice to learn each scene's noise: the photo scenes are deliberately random, so judge those with the gallery.
- `gallery.py` paints a fixed set of photos and writes one side-by-side sheet with colour error.
- Scripted runs open a window, so avoid clicking in it. `--showui` keeps the panels in the screenshot.
