"""
Builds the apple vs orange training set from images.

I turn every image into one row of five numbers (the inputs to my network) plus a label:
    hue        circular mean hue of the fruit pixels, in degrees (red ~0, orange ~30, yellow ~55, green ~90)
    saturation mean colour intensity, 0 to 255
    brightness mean of the HSV value channel, 0 to 255
    roundness  4*pi*area / perimeter^2 of the fruit outline (1.0 means a perfect circle)
    texture    variance of the Laplacian inside the fruit (higher means rougher, more dimpled skin)
    label      apple = 0, orange = 1

Augmentation (training and validation only)
    I keep each Fruits 360 image once as it is, then add augmented copies (2 per image for training,
    1 for validation) with random brightness, contrast, sharpness, saturation, a small hue shift, and a
    slight stretch. I do this so the two classes overlap and the network has to learn more than one
    simple rule. It also brings Fruits 360 closer to the look of brighter, sharper stock photos.

Sources
    Fruits 360 (100x100 version): I take the training set from its Training folder and the
        validation set from its Test folder, so no source image appears in both.
    Stock photos: a small real world test set that I never train on.

Usage:  python build_dataset.py <fruits-360-100x100 folder> <stock photo folder> <output folder>
"""
import os, sys, random
import cv2
import numpy as np
import pandas as pd

F360 = sys.argv[1]          # where I cloned the fruits-360-100x100 repository
PHOTOS = sys.argv[2]        # folder holding the stock photos for the real world test
OUT = sys.argv[3]           # where I write the CSV files
SEED = 42                   # I fix the seed so anyone rerunning this gets the same rows
random.seed(SEED)

APPLES = ["Apple Braeburn 1", "Apple Golden 1", "Apple Golden 2", "Apple Granny Smith 1",
          "Apple Pink Lady 1", "Apple Red 1", "Apple Red Delicious 1", "Apple Red Yellow 1"]
# I only use "Orange 1". The other orange folders in Fruits 360 (Orange 2 to 5) are dried or
# spoiled fruit, which would teach the network the wrong idea of what an orange looks like.
ORANGES = ["Orange 1"]


def fruit_mask(bgr, drop_green=False):
    """I separate the fruit from its white background and return a filled mask of the largest blob."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # A pixel counts as fruit if it is coloured or dark. White background and grey shadows are neither.
    m = (s >= 40) | (v < 120)
    if drop_green:
        # Some stock photos include a leaf. I remove green pixels so the leaf does not pull the hue.
        m &= ~((h >= 35) & (h <= 85) & (s > 60))
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # I take the convex hull so shiny highlights and stem notches do not leave holes in the mask.
    c = cv2.convexHull(max(cnts, key=cv2.contourArea))
    out = np.zeros_like(m)
    cv2.drawContours(out, [c], -1, 255, -1)
    return out


def normalise(bgr, mask, size=100):
    """I crop to the fruit and scale it onto a size x size canvas, so a 612 pixel stock photo and a
    100 pixel Fruits 360 image are measured at the same scale (this matters most for texture)."""
    x, y, w, h = cv2.boundingRect(mask)
    crop, mcrop = bgr[y:y + h, x:x + w], mask[y:y + h, x:x + w]
    k = (size - 4) / max(w, h)
    nw, nh = max(1, round(w * k)), max(1, round(h * k))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    mcrop = cv2.resize(mcrop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    img = np.full((size, size, 3), 255, np.uint8); m = np.zeros((size, size), np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    img[oy:oy + nh, ox:ox + nw] = crop; m[oy:oy + nh, ox:ox + nw] = mcrop
    return img, m


def augment(img, m, rng):
    """I randomly change only the fruit pixels: contrast, brightness, sharpness, hue, and saturation.
    I picked ranges wide enough to cover both dim Fruits 360 shots and bright, crisp stock photos."""
    out = img.astype(np.float32)
    inside = m > 0
    mean = out[inside].mean(axis=0)
    contrast = rng.uniform(0.8, 1.4)
    out[inside] = (out[inside] - mean) * contrast + mean            # contrast around the fruit's own mean colour
    out[inside] *= rng.uniform(0.85, 1.7)                           # brightness
    out = np.clip(out, 0, 255).astype(np.uint8)
    amount = rng.uniform(0, 4)                                      # sharpness through an unsharp mask (0 leaves it unchanged)
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    sharp = cv2.addWeighted(out, 1 + amount, blur, -amount, 0)
    if rng.random() < 0.25:                                         # a quarter of the time I soften the image instead
        sharp = cv2.GaussianBlur(out, (3, 3), 0)
    hsv = cv2.cvtColor(sharp, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.integers(-3, 4)) % 180        # hue shift of up to 6 degrees
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.65, 1.15), 0, 255)  # saturation, like different camera settings
    sharp = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    out[inside] = sharp[inside]                                     # I leave the background white
    return out


def features(bgr, drop_green=False, rng=None):
    """I return the five features for one image. If rng is given, I augment the image first."""
    if rng is not None:
        # I stretch the image by up to 12% on each axis, like fruit seen from a slight angle.
        # Without this, the network learned that "very round" means orange, because Fruits 360
        # only has one orange and it is almost perfectly round.
        sx, sy = rng.uniform(0.88, 1.12, size=2)
        bgr = cv2.resize(bgr, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
    mask = fruit_mask(bgr, drop_green)
    img, m = normalise(bgr, mask)
    if rng is not None:
        img = augment(img, m, rng)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    inside = m > 0
    # Hue is an angle, so I average it on the circle. A plain average of 2 and 178 (both red in
    # OpenCV's 0 to 179 scale) would wrongly give 90, which is green.
    ang = hsv[..., 0][inside].astype(float) * 2 * np.pi / 180
    hue = np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
    sat = hsv[..., 1][inside].mean()
    val = hsv[..., 2][inside].mean()
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    roundness = 4 * np.pi * cv2.contourArea(c) / cv2.arcLength(c, True) ** 2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inner = cv2.erode(m, np.ones((7, 7), np.uint8)) > 0              # I shrink the mask so the outline edge does not count as texture
    texture = cv2.Laplacian(gray, cv2.CV_64F)[inner].var()
    return [round(hue, 2), round(sat, 2), round(val, 2), round(roundness, 4), round(texture, 2)]


def from_fruits360(split, per_apple_variety, copies=0, rng=None):
    """I return one clean row per image, plus `copies` augmented rows per image."""
    rows = []
    for variety in APPLES + ORANGES:
        label = 1 if variety in ORANGES else 0
        files = sorted(os.listdir(f"{F360}/{split}/{variety}"))
        if label == 0:
            # I sample the same number from each apple variety so the apple class is not
            # dominated by one colour, and so apples and oranges end up balanced.
            files = random.sample(files, per_apple_variety)
        for f in files:
            img = cv2.imread(f"{F360}/{split}/{variety}/{f}")
            rows.append([f"{variety}/{f}", variety] + features(img) + [label])
            for k in range(copies):
                rows.append([f"{variety}/{f}#aug{k + 1}", variety] + features(img, rng=rng) + [label])
    return rows


COLS = ["source", "variety", "hue", "saturation", "brightness", "roundness", "texture", "label"]
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(SEED)
# Training: 479 oranges plus 60 of each of the 8 apple varieties (480). Each image appears once
# clean and twice augmented, which gives about 2,900 balanced rows.
train = pd.DataFrame(from_fruits360("Training", 60, copies=2, rng=rng), columns=COLS)
# Validation: 160 oranges plus 20 of each apple variety (160), clean and augmented once. This tests
# the model under the same widened conditions without sharing any source image with training.
val = pd.DataFrame(from_fruits360("Test", 20, copies=1, rng=rng), columns=COLS)
train.sample(frac=1, random_state=SEED).to_csv(f"{OUT}/train.csv", index=False)
val.sample(frac=1, random_state=SEED).to_csv(f"{OUT}/validation.csv", index=False)

# Real world test set from stock photos. I left out the two crate photos (many fruits and no
# plain background), the cut slice, and the whole plus half photo (the whole orange is partly
# hidden behind the cut half). I only keep whole fruit so the network cannot learn "cut means orange".
photos = sorted(os.listdir(PHOTOS))
# position in the sorted folder: (label, has_leaf)
keep = {0: (0, False), 1: (0, False), 2: (0, True), 4: (0, True), 5: (1, True), 9: (1, False)}
rows, counts = [], {0: 0, 1: 0}
for i, (label, leaf) in keep.items():
    img = cv2.imread(f"{PHOTOS}/{photos[i]}")
    kind = "apple" if label == 0 else "orange"
    counts[label] += 1
    rows.append([f"stock_{kind}_{counts[label]}", f"Stock photo {kind}"] + features(img, drop_green=leaf) + [label])
pd.DataFrame(rows, columns=COLS).to_csv(f"{OUT}/realworld_test.csv", index=False)

print("train", train.shape, train.label.value_counts().to_dict())
print("validation", val.shape, val.label.value_counts().to_dict())
print(pd.DataFrame(rows, columns=COLS).drop(columns="source").to_string(index=False))
