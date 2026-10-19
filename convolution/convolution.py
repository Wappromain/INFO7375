"""
2D convolution with no padding and stride 1 (valid convolution).

Input image: 6 x 6
Filter:      3 x 3
Output size: (6 - 3 + 1) x (6 - 3 + 1) = 4 x 4

Note: like most deep learning libraries, this computes cross correlation
(the filter is not flipped). Set flip_kernel=True for textbook convolution.
"""

import numpy as np


def convolve2d(image, kernel, flip_kernel=False):
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)

    if flip_kernel:
        kernel = kernel[::-1, ::-1]

    H, W = image.shape
    kh, kw = kernel.shape

    # No padding, stride 1: output shrinks by (filter size - 1) in each dimension
    out_h = H - kh + 1
    out_w = W - kw + 1
    output = np.zeros((out_h, out_w))

    # Slide the filter over every valid position
    for i in range(out_h):
        for j in range(out_w):
            total = 0.0
            for m in range(kh):
                for n in range(kw):
                    total += image[i + m, j + n] * kernel[m, n]
            output[i, j] = total

    return output


if __name__ == "__main__":
    # 6 x 6 example image (left half bright, right half dark: a vertical edge)
    image = np.array([
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
    ])

    # 3 x 3 vertical edge detection filter
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ])

    result = convolve2d(image, kernel)

    print("Image (6 x 6):\n", image)
    print("\nFilter (3 x 3):\n", kernel)
    print(f"\nOutput ({result.shape[0]} x {result.shape[1]}):\n", result)

    # Sanity check against SciPy if available
    try:
        from scipy.signal import correlate2d
        assert np.allclose(result, correlate2d(image, kernel, mode="valid"))
        print("\nMatches scipy.signal.correlate2d (valid mode).")
    except ImportError:
        pass
