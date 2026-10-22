# Depthwise and Pointwise Convolution

A NumPy implementation of the two convolutions that make up a depthwise separable convolution (the building block of MobileNet and Xception). The program takes an image and a kernel, and a `--mode` flag selects which convolution to apply.

## Depthwise vs. pointwise

| | Depthwise | Pointwise |
|---|---|---|
| What it mixes | Pixels within each channel | Channels within each pixel |
| Kernel shape | `(kH, kW)`, `(kH, kW, C)` or `(kH, kW, C, M)` | `(C_in, C_out)` or `(C_in,)` |
| Output channels | `C × M` (same as input when `M = 1`) | `C_out` |
| Typical use | Spatial filtering: blur, sharpen, edges | Changing channel count, recombining features |

**Depthwise:** each input channel is convolved with its own 2D filter, and channels never interact. A `(kH, kW)` kernel is shared by every channel; `(kH, kW, C)` gives one filter per channel; `(kH, kW, C, M)` gives `M` filters per channel (the channel multiplier).

**Pointwise:** a 1×1 convolution. Every pixel's channel vector is multiplied by a `(C_in, C_out)` matrix, so channels are combined but neighbouring pixels never interact.

Like PyTorch and TensorFlow, the sliding window computes cross correlation (the kernel is not flipped). Pass `--flip` for mathematical convolution.

## Requirements

* Python 3.8+
* NumPy 1.20 or newer (for `sliding_window_view`)
* Pillow (only needed for reading or writing image files)

```
pip install numpy pillow
```

## Usage

```
python conv.py --mode {depthwise,pointwise} --image IMAGE --kernel KERNEL [options]
```

| Option | Description |
|---|---|
| `-m, --mode` | `depthwise` or `pointwise` (required) |
| `-i, --image` | Input image: `.png`, `.jpg`, etc., or an array in `.npy`, `.json`, `.txt`, `.csv` |
| `-k, --kernel` | Kernel file: `.npy`, `.json`, `.txt`, `.csv` |
| `-o, --output` | Output file (image, `.npy`, `.json`, `.txt`, `.csv`). Printed if omitted |
| `-s, --stride` | Stride, default 1 |
| `-p, --padding` | `valid` (default), `same`, or an integer. Depthwise only |
| `-b, --bias` | One bias value, or one per output channel |
| `--flip` | Flip the kernel for true convolution. Depthwise only |

Image files load as `(H, W, C)` arrays. Grayscale images have 1 channel, RGB images 3. An RGBA image whose alpha channel is fully opaque is loaded as RGB; one with real transparency keeps its alpha as a 4th channel. When saving an image, values are clipped to 0..255.

## Examples

Using the sample `image.png` and the kernels in `kernels/`:

```
# Depthwise
python conv.py -m depthwise -i image.png -k kernels/blur3x3.json -o outputs/blur.png
python conv.py -m depthwise -i image.png -k kernels/sharpen3x3.json -p same -o outputs/sharpen.png
python conv.py -m depthwise -i image.png -k kernels/blur3x3.json -s 2 -p same -o outputs/blur_stride2.png
python conv.py -m depthwise -i image.png -k kernels/sobel_x.json -p same -o outputs/sobel_x.png

# Pointwise
python conv.py -m pointwise -i image.png -k kernels/rgb2gray.json -o outputs/gray.png
python conv.py -m pointwise -i image.png -k kernels/swap_rb.json -o outputs/swap_rb.png
```

| Command | Shape change |
|---|---|
| Depthwise blur, valid padding | `(550, 550, 3) → (548, 548, 3)` |
| Depthwise sharpen, same padding | `(550, 550, 3) → (550, 550, 3)` |
| Depthwise blur, stride 2 | `(550, 550, 3) → (275, 275, 3)` |
| Pointwise grayscale | `(550, 550, 3) → (550, 550, 1)` |
| Pointwise red/blue swap | `(550, 550, 3) → (550, 550, 3)` |

Results are in `outputs/`.

Small arrays work too, and the result is printed when `-o` is omitted:

```
echo '[[1,2],[3,4]]' > tiny.json
echo '[[1,0],[0,1]]' > k.json
python conv.py -m depthwise -i tiny.json -k k.json
# output shape: (1, 1, 1)  ->  1*1 + 4*1 = 5
```

## Kernel files

A kernel is a nested JSON list (or `.npy`, `.txt`, `.csv`).

| File | Mode | Shape | Effect |
|---|---|---|---|
| `blur3x3.json` | depthwise | `(3, 3)` | Box blur |
| `sharpen3x3.json` | depthwise | `(3, 3)` | Sharpen |
| `sobel_x.json` | depthwise | `(3, 3)` | Horizontal gradient (vertical edges) |
| `rgb2gray.json` | pointwise | `(3, 1)` | RGB to grayscale (luma weights) |
| `swap_rb.json` | pointwise | `(3, 3)` | Swap red and blue channels |

Edge kernels such as Sobel produce negative values, which are clipped to 0 when saved as an image. Save to `.npy` to keep the raw values.

## Using it from Python

```python
from conv import convolve, depthwise_conv, pointwise_conv

out = convolve(image, kernel, mode="depthwise", stride=1, padding="same")
out = convolve(image, kernel, mode="pointwise")

# Depthwise separable convolution: depthwise, then pointwise
features = pointwise_conv(depthwise_conv(image, spatial_kernel, padding="same"), channel_kernel)
```

## Tests

`test_conv.py` checks the vectorized implementation against plain loop versions across strides, padding modes, channel multipliers, shared kernels, flipping and bias, and checks the alpha channel handling.

```
python test_conv.py
```

It also runs under pytest: `python -m pytest test_conv.py`.

## Project layout

```
conv.py          implementation and command line interface
test_conv.py     tests
image.png        sample input image
kernels/         example kernel files
outputs/         results of the example commands
```
