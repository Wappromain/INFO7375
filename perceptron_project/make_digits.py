"""
make_digits.py
Generates 20 x 20 grayscale "handwritten" digit images.

Each digit is defined as a set of pen strokes (polylines and arcs) in a unit
square. Every image gets its own random variation: wobble on each stroke point,
slant, rotation, size, position, and pen thickness. Strokes are drawn on a large
canvas and then shrunk to 20 x 20, which gives soft anti aliased edges like a
scanned handwritten digit.

Output layout (the loader in perceptron/data.py reads this layout, so you can
swap in your own images by following the same structure):

    data/train/<digit>/<digit>_<k>.png      labeled training images
    data/test/test_<n>.png                  unlabeled test images
    data/test_answer_key.csv                answers, kept separate from the images

Usage:
    python make_digits.py                 # 10 train per digit, 3 test per digit
    python make_digits.py --train 10 --test 3 --seed 7
"""
import argparse
import csv
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BIG = 100          # drawing canvas size in pixels
OUT = 20           # final image size


def arc(cx, cy, rx, ry, start_deg, end_deg, n=24):
    """Points along an ellipse arc. Angles in degrees, 0 = right, 90 = down."""
    pts = []
    for i in range(n + 1):
        t = math.radians(start_deg + (end_deg - start_deg) * i / n)
        pts.append((cx + rx * math.cos(t), cy + ry * math.sin(t)))
    return pts


def digit_strokes(d, rng):
    """Return a list of strokes (each a list of (x, y) in 0..1) for digit d.
    A few digits have alternative styles, chosen at random."""
    r = rng.random
    if d == 0:
        return [arc(0.5, 0.5, 0.26 + 0.04 * r(), 0.38, -90, 270 + 10 * r())]
    if d == 1:
        strokes = [[(0.52, 0.12), (0.52, 0.88)]]
        if r() < 0.6:                                   # flag at the top
            strokes.append([(0.35, 0.28), (0.52, 0.12)])
        if r() < 0.3:                                   # base serif
            strokes.append([(0.36, 0.88), (0.68, 0.88)])
        return strokes
    if d == 2:
        return [arc(0.5, 0.33, 0.24, 0.2, 200, 360) +
                [(0.70, 0.45), (0.28, 0.86), (0.76, 0.86)]]
    if d == 3:
        return [arc(0.48, 0.31, 0.22, 0.18, 200, 450),
                arc(0.48, 0.67, 0.25, 0.2, 270, 520)]
    if d == 4:
        if r() < 0.5:                                   # closed style
            return [[(0.62, 0.12), (0.24, 0.62), (0.80, 0.62)],
                    [(0.62, 0.12), (0.62, 0.88)]]
        return [[(0.36, 0.12), (0.28, 0.58), (0.78, 0.58)],   # open style
                [(0.64, 0.30), (0.64, 0.88)]]
    if d == 5:
        return [[(0.72, 0.14), (0.34, 0.14), (0.31, 0.46)],
                [(0.31, 0.46)] + arc(0.48, 0.64, 0.24, 0.22, 230, 500)]
    if d == 6:
        return [[(0.66, 0.12), (0.42, 0.34), (0.30, 0.62)],
                arc(0.5, 0.66, 0.21, 0.21, 180, 540)]
    if d == 7:
        strokes = [[(0.26, 0.15), (0.76, 0.15), (0.44, 0.88)]]
        if r() < 0.3:                                   # European crossbar
            strokes.append([(0.42, 0.50), (0.70, 0.50)])
        return strokes
    if d == 8:
        return [arc(0.5, 0.31, 0.19, 0.17, 0, 360),
                arc(0.5, 0.67, 0.23, 0.2, 0, 360)]
    if d == 9:
        tail = [(0.70, 0.36), (0.68, 0.62), (0.56, 0.88)] if r() < 0.5 \
            else [(0.70, 0.36), (0.70, 0.88)]
        return [arc(0.5, 0.35, 0.2, 0.2, 0, 360), tail]
    raise ValueError(d)


def render(d, rng):
    strokes = digit_strokes(d, rng)

    # Global style for this "writer"
    slant = rng.uniform(-0.25, 0.25)          # shear
    angle = math.radians(rng.uniform(-12, 12))
    sx = rng.uniform(0.75, 1.0)
    sy = rng.uniform(0.85, 1.05)
    dx = rng.uniform(-0.06, 0.06)
    dy = rng.uniform(-0.05, 0.05)
    width = rng.randint(7, 13)
    wobble = rng.uniform(0.008, 0.025)

    def transform(p):
        x, y = p[0] - 0.5, p[1] - 0.5
        x += rng.gauss(0, wobble)
        y += rng.gauss(0, wobble)
        x = x * sx + slant * (-y)             # shear: top leans right
        y = y * sy
        xr = x * math.cos(angle) - y * math.sin(angle)
        yr = x * math.sin(angle) + y * math.cos(angle)
        return ((xr + 0.5 + dx) * BIG, (yr + 0.5 + dy) * BIG)

    img = Image.new("L", (BIG, BIG), 0)
    draw = ImageDraw.Draw(img)
    for s in strokes:
        pts = [transform(p) for p in s]
        draw.line(pts, fill=255, width=width, joint="curve")
        for (x, y) in (pts[0], pts[-1]):      # round pen ends
            h = width / 2
            draw.ellipse([x - h, y - h, x + h, y + h], fill=255)

    img = img.filter(ImageFilter.GaussianBlur(1.2))
    img = img.resize((OUT, OUT), Image.LANCZOS)

    # light scanner noise
    arr = np.asarray(img, dtype=np.float32)
    arr += np.random.default_rng(rng.randint(0, 10**9)).normal(0, 6, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=10, help="training images per digit")
    ap.add_argument("--test", type=int, default=3, help="test images per digit")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    for d in range(10):
        folder = os.path.join(args.out, "train", str(d))
        os.makedirs(folder, exist_ok=True)
        for k in range(args.train):
            render(d, rng).save(os.path.join(folder, f"{d}_{k}.png"))

    # Test images: shuffled, unlabeled file names; answers in a separate file
    test_dir = os.path.join(args.out, "test")
    os.makedirs(test_dir, exist_ok=True)
    labels = [d for d in range(10) for _ in range(args.test)]
    rng.shuffle(labels)
    with open(os.path.join(args.out, "test_answer_key.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label"])
        for n, d in enumerate(labels):
            name = f"test_{n:02d}.png"
            render(d, rng).save(os.path.join(test_dir, name))
            w.writerow([name, d])

    print(f"Wrote {10 * args.train} training and {len(labels)} test images to {args.out}/")


if __name__ == "__main__":
    main()
