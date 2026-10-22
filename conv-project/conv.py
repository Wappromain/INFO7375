#!/usr/bin/env python3
"""Depthwise and pointwise convolution.

Takes an input image and a kernel, and produces the convolved image. The
--mode flag selects the operation:

  depthwise  Each input channel is convolved with its own 2-D spatial filter.
             Channels are never mixed.
               image  : (H, W, C)          (a 2-D (H, W) image is treated as C=1)
               kernel : (kH, kW)           same filter applied to every channel
                        (kH, kW, C)        one filter per channel
                        (kH, kW, C, M)     M filters per channel (channel multiplier)
               output : (H', W', C*M)

  pointwise  A 1x1 convolution: every pixel's channel vector is linearly
             mapped to a new channel vector. Pixels are never mixed.
               image  : (H, W, C_in)
               kernel : (C_in, C_out)      or (C_in,) for a single output channel
               output : (H', W', C_out)

Like deep-learning frameworks, the sliding-window operation is
cross-correlation; pass --flip to get mathematical convolution (kernel
rotated 180 degrees).

Supported file formats
  input/kernel : .npy, .json (nested lists), .txt/.csv (2-D numbers),
                 and for the image also .png/.jpg/.bmp/... (needs Pillow).
                 An RGBA image whose alpha is fully opaque is loaded as RGB.
  output       : .npy, .json, .txt/.csv (2-D or single-channel only),
                 or an image file (values clipped to 0..255; 1, 3 or 4 channels)
                 If --output is omitted the result is printed.

Examples
  python conv.py --mode depthwise --image photo.png --kernel blur3x3.json -o out.png
  python conv.py --mode pointwise --image photo.png --kernel rgb2gray.json -o gray.png
  python conv.py --mode depthwise --image img.npy --kernel k.npy --stride 2 --padding same
"""

import argparse
import json
import os
import sys

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------- #
# Core operations
# --------------------------------------------------------------------------- #

def _as_hwc(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3:
        raise ValueError(f"image must be (H, W) or (H, W, C), got shape {image.shape}")
    return image


def _pad_amounts(size, k, stride, padding):
    """Return (before, after) padding for one spatial axis."""
    if padding == "valid":
        return 0, 0
    if padding == "same":
        # TensorFlow-style SAME: output size = ceil(size / stride).
        out = -(-size // stride)
        total = max((out - 1) * stride + k - size, 0)
        return total // 2, total - total // 2
    p = int(padding)
    if p < 0:
        raise ValueError("padding must be non-negative")
    return p, p


def depthwise_conv(image, kernel, stride=1, padding="valid", bias=None, flip=False):
    """Depthwise 2-D convolution. Returns an (H', W', C*M) array."""
    x = _as_hwc(image)
    H, W, C = x.shape

    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim == 2:                     # one filter shared by all channels
        k = np.repeat(k[:, :, None], C, axis=2)
    if k.ndim == 3:                     # one filter per channel
        k = k[:, :, :, None]
    if k.ndim != 4:
        raise ValueError(f"depthwise kernel must be 2-D, 3-D or 4-D, got shape {k.shape}")
    kH, kW, kC, M = k.shape
    if kC != C:
        raise ValueError(f"kernel has {kC} channels but image has {C}")
    if flip:
        k = k[::-1, ::-1]

    (pt, pb), (pl, pr) = (_pad_amounts(H, kH, stride, padding),
                          _pad_amounts(W, kW, stride, padding))
    x = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))
    if x.shape[0] < kH or x.shape[1] < kW:
        raise ValueError("kernel is larger than the (padded) image")

    # windows: (H_out, W_out, C, kH, kW)
    windows = np.lib.stride_tricks.sliding_window_view(x, (kH, kW), axis=(0, 1))
    windows = windows[::stride, ::stride]
    out = np.einsum("hwcij,ijcm->hwcm", windows, k)
    out = out.reshape(out.shape[0], out.shape[1], C * M)

    if bias is not None:
        out = out + np.broadcast_to(np.asarray(bias, dtype=np.float64), (C * M,))
    return out


def pointwise_conv(image, kernel, stride=1, bias=None):
    """Pointwise (1x1) convolution. Returns an (H', W', C_out) array."""
    x = _as_hwc(image)[::stride, ::stride]
    C = x.shape[2]

    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim == 1:
        k = k[:, None]
    if k.ndim == 4 and k.shape[:2] == (1, 1):   # accept (1, 1, C_in, C_out) too
        k = k[0, 0]
    if k.ndim != 2:
        raise ValueError(f"pointwise kernel must be (C_in, C_out) or (C_in,), got shape {k.shape}")
    if k.shape[0] != C:
        raise ValueError(f"kernel expects {k.shape[0]} input channels but image has {C}")

    out = x @ k
    if bias is not None:
        out = out + np.broadcast_to(np.asarray(bias, dtype=np.float64), (k.shape[1],))
    return out


def convolve(image, kernel, mode, stride=1, padding="valid", bias=None, flip=False):
    """Dispatch on the mode flag: 'depthwise' or 'pointwise'."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if mode == "depthwise":
        return depthwise_conv(image, kernel, stride, padding, bias, flip)
    if mode == "pointwise":
        return pointwise_conv(image, kernel, stride, bias)
    raise ValueError(f"unknown mode {mode!r}; use 'depthwise' or 'pointwise'")


# --------------------------------------------------------------------------- #
# File I/O
# --------------------------------------------------------------------------- #

def load_array(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".json":
        with open(path) as f:
            return np.array(json.load(f), dtype=np.float64)
    if ext in (".txt", ".csv"):
        return np.loadtxt(path, delimiter="," if ext == ".csv" else None, ndmin=2)
    if ext in IMAGE_EXTS:
        from PIL import Image
        with Image.open(path) as img:
            if img.mode not in ("L", "RGB", "RGBA"):
                img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
            arr = np.asarray(img, dtype=np.float64)
        # A fully opaque alpha channel carries no information, so drop it and
        # treat the image as RGB. Real transparency is kept as a 4th channel.
        if arr.ndim == 3 and arr.shape[2] == 4 and np.all(arr[:, :, 3] == 255):
            arr = arr[:, :, :3]
        return arr
    raise ValueError(f"unsupported file type: {path}")


def save_array(path, arr):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, arr)
    elif ext == ".json":
        with open(path, "w") as f:
            json.dump(arr.tolist(), f)
    elif ext in (".txt", ".csv"):
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            raise ValueError("text output only supports single-channel results; use .npy or .json")
        np.savetxt(path, arr, delimiter="," if ext == ".csv" else " ", fmt="%.6g")
    elif ext in IMAGE_EXTS:
        from PIL import Image
        if arr.shape[2] not in (1, 3, 4):
            raise ValueError(f"cannot save {arr.shape[2]}-channel result as an image; use .npy or .json")
        img = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        Image.fromarray(img[:, :, 0] if img.shape[2] == 1 else img).save(path)
    else:
        raise ValueError(f"unsupported output type: {path}")


def print_result(arr):
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(f"output shape: {arr.shape}")
    for c in range(arr.shape[2]):
        print(f"\nchannel {c}:")
        print(arr[:, :, c])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_padding(value):
    if value in ("valid", "same"):
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("padding must be 'valid', 'same' or an integer")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Depthwise / pointwise convolution of an image with a kernel.",
        epilog="See the module docstring for shapes and formats.")
    parser.add_argument("--mode", "-m", required=True, choices=["depthwise", "pointwise"],
                        help="which convolution to perform")
    parser.add_argument("--image", "-i", required=True, help="input image file")
    parser.add_argument("--kernel", "-k", required=True, help="kernel (filter) file")
    parser.add_argument("--output", "-o", help="output file (printed if omitted)")
    parser.add_argument("--stride", "-s", type=int, default=1, help="stride (default 1)")
    parser.add_argument("--padding", "-p", type=parse_padding, default="valid",
                        help="'valid' (default), 'same', or an integer (depthwise only)")
    parser.add_argument("--bias", "-b", type=float, nargs="+",
                        help="bias: one value, or one per output channel")
    parser.add_argument("--flip", action="store_true",
                        help="flip the kernel for true convolution (depthwise only)")
    args = parser.parse_args(argv)

    try:
        image = load_array(args.image)
        kernel = load_array(args.kernel)
        out = convolve(image, kernel, args.mode, args.stride, args.padding, args.bias, args.flip)
        if args.output:
            save_array(args.output, out)
            print(f"{args.mode}: {np.shape(image)} * {kernel.shape} -> {out.shape}, saved to {args.output}")
        else:
            print_result(out)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
