import argparse
import json
import os
import re
import subprocess
import time

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SCENES = {
    "jet": ["--script", "jet", "--auto", "300"],
    "coil": ["--script", "coil", "--auto", "300"],
    "bucket": ["--script", "bucket", "--auto", "300"],
    "flick": ["--script", "flick", "--auto", "300"],
    "lines": ["--script", "lines", "--auto", "300"],
    "layers": ["--script", "layers", "--auto", "400"],
    "knife": ["--script", "knife", "--auto", "400"],
    "knifeart": ["--script", "knifeart", "--auto", "400"],
    "brush": ["--script", "brush", "--auto", "400"],
    "brushcurve": ["--script", "brushcurve", "--auto", "400"],
    "brushlab": ["--script", "brushlab", "--auto", "600"],
    "brushpress": ["--script", "brushpress", "--auto", "400"],
    "brushsizes": ["--script", "brushsizes", "--auto", "400"],
    "roller": ["--script", "roller", "--auto", "400"],
    "sponge": ["--script", "sponge", "--auto", "400"],
    "stick": ["--script", "stick", "--auto", "400"],
    "bottle": ["--script", "bottle", "--auto", "300"],
    "pour": ["--script", "pour", "--auto", "300"],
    "mixing": ["--script", "mixing", "--auto", "400"],
    "settle": ["--script", "settle", "--auto", "300"],
    "tiltoil": ["--script", "tiltoil", "--auto", "400"],
    "physics": ["--script", "physics", "--auto", "400"],
    "pollock": ["--script", "pollock", "--auto", "500"],
    "undotest2": ["--script", "undotest2", "--auto", "300"],
    "undosave": ["--script", "undosave", "--auto", "300"],
    "undoclear": ["--script", "undoclear", "--auto", "300"],
    "undorubber": ["--script", "undorubber", "--auto", "300"],
    "eyedrop": ["--script", "eyedrop", "--auto", "560", "--showui"],
    "beauty": ["--script", "brush", "--auto", "200", "--spp", "2", "--beauty", "BEAUTY"],
    "agx": ["--script", "brush", "--auto", "200", "--agx", "--spp", "2", "--beauty", "BEAUTY"],
}

PHOTO_SCENES = {
    "evolve": ["--script", "evolve", "--tool", "brush", "--flat", "--size", "150", "--seed", "7", "--auto", "2500", "--then", "flat70@60"],
}


def capture(args):
    os.makedirs(args.out, exist_ok=True)
    exe = os.path.abspath(args.exe)
    scenes = dict(SCENES)
    if args.photo:
        for name, command in PHOTO_SCENES.items():
            scenes[name] = command + ["--target", os.path.abspath(args.photo)]
    only = set(args.only.split(",")) if args.only else None
    summary = {}
    for name, command in scenes.items():
        if only and name not in only:
            continue
        shot = os.path.abspath(os.path.join(args.out, f"{name}.png"))
        beauty = os.path.abspath(os.path.join(args.out, f"{name}_beauty.png"))
        full = [exe, "--res", str(args.res), "--shot", shot] + [beauty if c == "BEAUTY" else c for c in command]
        started = time.time()
        try:
            run = subprocess.run(full, cwd=args.out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, RUST_LOG="warn"), timeout=args.timeout)
            text = run.stdout.decode("utf-8", "replace")
            code = run.returncode
        except subprocess.TimeoutExpired:
            text = ""
            code = -1
        with open(os.path.join(args.out, f"{name}.log"), "w", encoding="utf-8") as log:
            log.write(text)
        mass = re.findall(r"err ([0-9.]+)%", text)
        summary[name] = {"code": code, "seconds": round(time.time() - started, 1), "panic": "panicked" in text, "shot": os.path.exists(shot), "mass_err": float(mass[-1]) if mass else None}
        print(name, summary[name], flush=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as out:
        json.dump(summary, out, indent=1)


def difference(a, b):
    x = Image.open(a).convert("RGB")
    y = Image.open(b).convert("RGB")
    if x.size != y.size:
        return None
    d = np.abs(np.asarray(x, np.int16) - np.asarray(y, np.int16))
    return float(d.mean()), int(d.max()), float((d.max(-1) > 8).mean() * 100)


def compare(args):
    left = json.load(open(os.path.join(args.a, "summary.json"), encoding="utf-8"))
    right = json.load(open(os.path.join(args.b, "summary.json"), encoding="utf-8"))
    bad = 0
    print(f"{'scene':12} {'mean':>7} {'max':>4} {'%>8':>6}  status")
    for name in sorted(set(left) & set(right)):
        rows = []
        for suffix in ["", "_beauty"]:
            pa, pb = os.path.join(args.a, name + suffix + ".png"), os.path.join(args.b, name + suffix + ".png")
            if os.path.exists(pa) and os.path.exists(pb):
                rows.append(difference(pa, pb))
        failed = right[name]["panic"] or right[name]["code"] != left[name]["code"] or not right[name]["shot"]
        worst = max([r for r in rows if r], default=None, key=lambda r: r[0])
        if rows and any(r is None for r in rows):
            failed = True
        status = "FAILED" if failed else "same" if worst and worst[1] == 0 else "differs" if worst else "no image"
        bad += failed
        if worst:
            print(f"{name:12} {worst[0]:7.3f} {worst[1]:4d} {worst[2]:6.2f}  {status}")
        else:
            print(f"{name:12} {'':7} {'':4} {'':6}  {status}")
    print("failures", bad)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("capture")
    run.add_argument("--exe", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--res", type=int, default=1536)
    run.add_argument("--timeout", type=int, default=240)
    run.add_argument("--photo")
    run.add_argument("--only")
    run.set_defaults(func=capture)
    cmp = sub.add_parser("compare")
    cmp.add_argument("a")
    cmp.add_argument("b")
    cmp.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
