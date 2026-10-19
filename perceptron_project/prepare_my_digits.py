"""
prepare_my_digits.py
Turns your own handwritten digits into 20 x 20 training or test images.

The network expects what MNIST style data looks like: a light digit on a dark
background, cropped to the ink, and centered with a small margin. Photos of
paper are the opposite (dark ink on white), unevenly lit, and off center, so
this script fixes all of that for you.

------------------------------------------------------------------------------
OPTION A: one photo of a grid sheet (fastest)
------------------------------------------------------------------------------
On paper, draw a grid of 10 rows x 10 columns (or print grid_template.png).
Row 1 = ten different 0s, row 2 = ten 1s, ... row 10 = ten 9s.
Use a dark pen or marker. Photograph it straight on, cropped to the grid edges.

    python prepare_my_digits.py --sheet my_sheet.jpg --out my_data/train

For unlabeled test digits, add --test. Set --rows and --cols to match YOUR grid,
or every image will be several digits squashed together (output is one flat
folder named mytest_00.png, mytest_01.png, ... in reading order):

    python prepare_my_digits.py --sheet my_test_sheet.jpg --rows 10 --test --out my_data/test

------------------------------------------------------------------------------
OPTION B: separate image files (for example drawn in Paint)
------------------------------------------------------------------------------
Put them in folders, one per digit, any size, any colors:
    raw/0/a.png, raw/0/b.png, ..., raw/9/j.png

    python prepare_my_digits.py --folder raw --out my_data/train

A flat folder of images also works (useful for test images, keeps file names):
    python prepare_my_digits.py --folder raw_test --out my_data/test

------------------------------------------------------------------------------
Then train on your own images:
    python train.py --train-dir my_data/train --test-dir my_data/test --answer-key ""
A contact sheet (preview.png) is written next to the output so you can check
the crops before training.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageFilter, ImageOps

OUT = 20      # final image size
BOX = 16      # the digit's longest side inside the 20 x 20 frame (MNIST uses 20 in 28)
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")


def remove_grid_lines(a, frac=0.92):
    """Blank any row or column that is 'on' across most of the cell. Grid lines
    span the whole cell; pen strokes almost never do."""
    on = a > 25
    for axis in (1, 0):                       # axis 1: rows, axis 0: columns
        full = np.nonzero(on.mean(axis=axis) > frac)[0]
        for i in full:
            lo, hi = max(i - 2, 0), i + 3
            if axis == 1:
                a[lo:hi, :] = 0
            else:
                a[:, lo:hi] = 0
    return a


def components(mask):
    """Connected groups of ink pixels (8 neighbors). Returns a list of
    (size, touches_border, pixel_indices)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack, pix, border = [(y0, x0)], [], False
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            if y in (0, h - 1) or x in (0, w - 1):
                border = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        out.append((len(pix), border, tuple(np.array(pix).T)))
    return out


def normalize_digit(img, margin_frac=0.0):
    """Any image of one digit -> 20 x 20 uint8 array, white digit on black."""
    g = ImageOps.exif_transpose(img).convert("L")
    w, h = g.size
    if margin_frac:                                   # skip the cell border itself
        mx, my = int(w * margin_frac), int(h * margin_frac)
        g = g.crop((mx, my, w - mx, h - my))
    a = np.asarray(g, dtype=np.float64)

    # Make ink bright and background dark, whatever the original colors were
    if np.median(a) > 127:
        a = 255.0 - a

    # Remove uneven lighting: subtract a heavily blurred copy (the background)
    bg = np.asarray(Image.fromarray(a.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(max(a.shape) / 8)), dtype=np.float64)
    a = np.clip(a - bg, 0, None)
    a = remove_grid_lines(a)

    # Stretch contrast and zero out faint noise
    top = np.percentile(a, 99.5)
    if top <= 0:
        return np.zeros((OUT, OUT), np.uint8), False      # empty cell
    a = np.clip(a / top, 0, 1)
    a[a < 0.35] = 0

    # Drop specks, and small bits of a neighbor's digit poking in at the edge
    mask = a > 0.5
    if mask.sum() < 5:
        return np.zeros((OUT, OUT), np.uint8), False
    comps = components(mask)
    biggest = max(c[0] for c in comps)
    for size, border, idx in comps:
        if size < 0.02 * biggest or (border and size < 0.15 * biggest):
            a[idx] = 0
            mask[idx] = False

    # Crop to the ink
    ys, xs = np.nonzero(mask)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Thicken thin pen strokes so they are about 2 pixels wide after shrinking.
    # Stroke width is estimated as 2 * ink area / ink boundary length.
    padded = np.pad(mask, 1)
    inner = (padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    boundary = max(1, int((mask & ~inner).sum()))
    stroke = 2 * mask.sum() / boundary
    target = max(a.shape) * 2.2 / BOX
    k = int(round(target - stroke))
    if k >= 2:
        k |= 1                                        # filter size must be odd
        a = np.pad(a, k)
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(
            ImageFilter.MaxFilter(k)), dtype=np.float64) / 255
        ys, xs = np.nonzero(a > 0.5)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Scale so the longest side is BOX pixels, keeping the aspect ratio
    hh, ww = a.shape
    s = BOX / max(hh, ww)
    nw, nh = max(1, round(ww * s)), max(1, round(hh * s))
    small = Image.fromarray((a * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    small = np.asarray(small, dtype=np.float64)

    # Place it so its center of mass sits in the middle of the 20 x 20 frame
    canvas = np.zeros((OUT, OUT))
    total = small.sum()
    cy = (small.sum(axis=1) @ np.arange(nh)) / total
    cx = (small.sum(axis=0) @ np.arange(nw)) / total
    top_y = int(round(OUT / 2 - cy - 0.5))
    left_x = int(round(OUT / 2 - cx - 0.5))
    top_y = min(max(top_y, 0), OUT - nh)
    left_x = min(max(left_x, 0), OUT - nw)
    canvas[top_y:top_y + nh, left_x:left_x + nw] = small
    return np.clip(canvas, 0, 255).astype(np.uint8), True


def split_sheet(path, rows, cols):
    sheet = ImageOps.exif_transpose(Image.open(path)).convert("L")
    W, H = sheet.size
    for r in range(rows):
        for c in range(cols):
            box = (round(c * W / cols), round(r * H / rows),
                   round((c + 1) * W / cols), round((r + 1) * H / rows))
            yield r, c, sheet.crop(box)


def save_preview(paths, out_png, cols=10):
    if not paths:
        return
    rows = int(np.ceil(len(paths) / cols))
    sheet = Image.new("L", (cols * 22, rows * 22), 60)
    for i, p in enumerate(paths):
        sheet.paste(Image.open(p), ((i % cols) * 22 + 1, (i // cols) * 22 + 1))
    sheet.resize((sheet.width * 4, sheet.height * 4), Image.NEAREST).save(out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sheet", help="photo or scan of a grid of digits")
    src.add_argument("--folder", help="folder of separate digit images")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--test", action="store_true",
                    help="sheet mode: write unlabeled test images instead of one folder per row")
    ap.add_argument("--margin", type=float, default=0.01,
                    help="fraction of each grid cell edge to ignore")
    args = ap.parse_args()

    written, empty = [], []

    if args.sheet:
        for r, c, cell in split_sheet(args.sheet, args.rows, args.cols):
            arr, ok = normalize_digit(cell, args.margin)
            if not ok:
                empty.append(f"row {r + 1}, column {c + 1}")
                continue
            if args.test:
                folder, name = args.out, f"mytest_{r * args.cols + c:02d}.png"
            else:
                folder, name = os.path.join(args.out, str(r)), f"{r}_{c}.png"
            os.makedirs(folder, exist_ok=True)
            p = os.path.join(folder, name)
            Image.fromarray(arr, "L").save(p)
            written.append(p)
    else:
        for root, _, files in sorted(os.walk(args.folder)):
            rel = os.path.relpath(root, args.folder)
            for f in sorted(files):
                if not f.lower().endswith(IMG_EXT):
                    continue
                arr, ok = normalize_digit(Image.open(os.path.join(root, f)))
                if not ok:
                    empty.append(os.path.join(rel, f))
                    continue
                folder = os.path.join(args.out, rel) if rel != "." else args.out
                os.makedirs(folder, exist_ok=True)
                p = os.path.join(folder, os.path.splitext(f)[0] + ".png")
                Image.fromarray(arr, "L").save(p)
                written.append(p)

    preview = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                           os.path.basename(os.path.normpath(args.out)) + "_preview.png")
    save_preview(written, preview, cols=args.cols if args.sheet else 10)
    print(f"Wrote {len(written)} images to {args.out}")
    print(f"Check the crops here: {preview}")
    if empty:
        print("No digit found in: " + "; ".join(empty))


if __name__ == "__main__":
    main()
