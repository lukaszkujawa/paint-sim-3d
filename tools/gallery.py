import argparse
import os
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

DEFAULT_PHOTOS = [
    "ref.png",
    "Qwen_image_2.1_00015.png",
    "1.jpg",
    "Qwen_image_2.1_00011.png",
    "2.jpg",
    "rainy-urban-evening-stockcake.jpg",
]

MODES = {
    "classic": ["--tool", "brush", "--flat", "--size", "150", "--then", "flat70@200,flat20@600,round8@1200"],
}

FRAMES = {"classic": 30000}

CELL = (640, 427)


def lab(rgb):
    c = rgb / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    x = c @ m.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(x > 0.008856, np.cbrt(x), 7.787 * x + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def fitted(image, size):
    w, h = size
    ratio = w / h
    if image.width / image.height > ratio:
        cw = int(image.height * ratio)
        x = (image.width - cw) // 2
        image = image.crop((x, 0, x + cw, image.height))
    else:
        ch = int(image.width / ratio)
        y = (image.height - ch) // 2
        image = image.crop((0, y, image.width, y + ch))
    return image.resize(size, Image.BOX)


def paint(args, photo, mode, out):
    name = os.path.splitext(os.path.basename(photo))[0]
    beauty = os.path.join(out, f"{name}_{mode}.png")
    if os.path.exists(beauty) and not args.force:
        return beauty
    command = [args.exe, "--auto", str(args.frames or FRAMES[mode]), "--script", "evolve", "--target", photo, "--seed", str(args.seed), "--res", str(args.res), "--spp", str(args.spp), "--beauty", beauty, "--shot", os.path.join(out, f"{name}_{mode}_live.png")] + MODES[mode]
    started = time.time()
    with open(os.path.join(out, f"{name}_{mode}.log"), "w") as log:
        env = dict(os.environ, RUST_LOG="info")
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    print(f"{name} {mode}: {time.time() - started:.0f} s", flush=True)
    return beauty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="*")
    parser.add_argument("--exe", default=os.path.join("target", "release", "canvaz.exe"))
    parser.add_argument("--out", default=os.path.join("target", "gallery"))
    parser.add_argument("--modes", default="classic")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--res", type=int, default=3072)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    photos = args.photos or [os.path.join(downloads, p) for p in DEFAULT_PHOTOS]
    photos = [p for p in photos if os.path.exists(p)]
    modes = args.modes.split(",")
    os.makedirs(args.out, exist_ok=True)
    w, h = CELL
    sheet = Image.new("RGB", (w * (len(modes) + 1), (h + 24) * len(photos)), (24, 24, 26))
    draw = ImageDraw.Draw(sheet)
    for row, photo in enumerate(photos):
        reference = fitted(Image.open(photo).convert("RGB"), CELL)
        target = lab(np.asarray(reference, float))
        top = row * (h + 24)
        sheet.paste(reference, (0, top + 24))
        draw.text((6, top + 6), os.path.basename(photo), fill=(220, 220, 220))
        for col, mode in enumerate(modes):
            path = paint(args, photo, mode, args.out)
            if not os.path.exists(path):
                draw.text(((col + 1) * w + 6, top + 6), f"{mode}: no result", fill=(240, 120, 120))
                print(f"{os.path.basename(photo)} {mode}: no result", flush=True)
                continue
            shown = fitted(Image.open(path).convert("RGB"), CELL)
            painted = lab(np.asarray(shown, float))
            error = float(np.sqrt(((painted - target) ** 2).sum(-1)).mean())
            sheet.paste(shown, ((col + 1) * w, top + 24))
            draw.text(((col + 1) * w + 6, top + 6), f"{mode}  dE {error:.1f}", fill=(220, 220, 220))
            print(f"{os.path.basename(photo)} {mode}: dE {error:.1f}", flush=True)
    result = os.path.join(args.out, "gallery.png")
    sheet.save(result)
    print(result)


if __name__ == "__main__":
    main()
