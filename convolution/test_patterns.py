"""
Test convolve2d (no padding, stride 1) on hand-built 6 x 6 patterns.

Each pattern has an output you can predict before running it.
A 6 x 6 image with a 3 x 3 filter gives a 4 x 4 output.
Needs convolution.py in the same folder.
"""

import numpy as np
from convolution import convolve2d

# ---------- 6 x 6 test images ----------

# Bright left half, dark right half
vertical_edge = np.array([[10, 10, 10, 0, 0, 0]] * 6)

# Bright top half, dark bottom half (the transpose of the vertical edge)
horizontal_edge = vertical_edge.T

# Bright on and below the diagonal, dark above it
diagonal = np.tril(np.full((6, 6), 10))

# Alternating 0 and 255
checkerboard = (np.indices((6, 6)).sum(axis=0) % 2) * 255

images = {
    "Vertical edge": vertical_edge,
    "Horizontal edge": horizontal_edge,
    "Diagonal": diagonal,
    "Checkerboard": checkerboard,
}

# ---------- 3 x 3 filters ----------

vertical_filter = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
])

horizontal_filter = vertical_filter.T   # [[1,1,1],[0,0,0],[-1,-1,-1]]

filters = {
    "Vertical filter": vertical_filter,
    "Horizontal filter": horizontal_filter,
}

# ---------- Run every image through every filter ----------

for img_name, img in images.items():
    print("=" * 45)
    print(f"{img_name} image (6 x 6):\n{img}")
    for f_name, k in filters.items():
        out = convolve2d(img, k).astype(int)
        print(f"\n{f_name} output (4 x 4):\n{out}")
    print()

# ---------- Checks against values worked out by hand ----------

# Vertical edge: the vertical filter fires in the two middle columns, the horizontal filter sees nothing
assert (convolve2d(vertical_edge, vertical_filter) == np.array([[0, 30, 30, 0]] * 4)).all()
assert (convolve2d(vertical_edge, horizontal_filter) == 0).all()

# Horizontal edge: the mirror image of the case above
assert (convolve2d(horizontal_edge, horizontal_filter) == np.array([[0, 30, 30, 0]] * 4).T).all()
assert (convolve2d(horizontal_edge, vertical_filter) == 0).all()

# Diagonal: both filters respond only along the diagonal, with opposite signs
d_v = convolve2d(diagonal, vertical_filter)
d_h = convolve2d(diagonal, horizontal_filter)
assert (d_v == -d_h).all()

# Checkerboard: in every 3 x 3 patch the left and right columns are identical,
# so the +1 and -1 columns cancel and both edge filters output all zeros
assert (convolve2d(checkerboard, vertical_filter) == 0).all()
assert (convolve2d(checkerboard, horizontal_filter) == 0).all()

print("All hand-worked checks passed.")
