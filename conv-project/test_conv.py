"""Checks conv.py against straightforward loop implementations. Run: python test_conv.py"""

import numpy as np

from conv import convolve, _pad_amounts

rng = np.random.default_rng(0)


def naive_depthwise(x, k, stride, padding):
    H, W, C = x.shape
    kH, kW, _, M = k.shape
    (pt, pb), (pl, pr) = _pad_amounts(H, kH, stride, padding), _pad_amounts(W, kW, stride, padding)
    x = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))
    Ho = (x.shape[0] - kH) // stride + 1
    Wo = (x.shape[1] - kW) // stride + 1
    out = np.zeros((Ho, Wo, C * M))
    for i in range(Ho):
        for j in range(Wo):
            for c in range(C):
                for m in range(M):
                    patch = x[i * stride:i * stride + kH, j * stride:j * stride + kW, c]
                    out[i, j, c * M + m] = np.sum(patch * k[:, :, c, m])
    return out


def naive_pointwise(x, k, stride):
    x = x[::stride, ::stride]
    H, W, C = x.shape
    out = np.zeros((H, W, k.shape[1]))
    for i in range(H):
        for j in range(W):
            for o in range(k.shape[1]):
                out[i, j, o] = sum(x[i, j, c] * k[c, o] for c in range(C))
    return out


def test_depthwise():
    for stride in (1, 2, 3):
        for padding in ("valid", "same", 1):
            for M in (1, 2):
                x = rng.normal(size=(9, 11, 3))
                k = rng.normal(size=(3, 5, 3, M))
                got = convolve(x, k, "depthwise", stride, padding)
                np.testing.assert_allclose(got, naive_depthwise(x, k, stride, padding))
    # 'same' with stride 1 keeps the spatial size
    assert convolve(x, rng.normal(size=(3, 3)), "depthwise", 1, "same").shape == (9, 11, 3)


def test_depthwise_shared_kernel_and_flip():
    x = rng.normal(size=(6, 6, 2))
    k2 = rng.normal(size=(3, 3))
    shared = convolve(x, k2, "depthwise")
    per_channel = convolve(x, np.stack([k2, k2], axis=2), "depthwise")
    np.testing.assert_allclose(shared, per_channel)
    flipped = convolve(x, k2, "depthwise", flip=True)
    np.testing.assert_allclose(flipped, convolve(x, k2[::-1, ::-1], "depthwise"))


def test_pointwise():
    for stride in (1, 2):
        x = rng.normal(size=(7, 5, 4))
        k = rng.normal(size=(4, 6))
        got = convolve(x, k, "pointwise", stride)
        np.testing.assert_allclose(got, naive_pointwise(x, k, stride))


def test_channels_independent_vs_mixed():
    # Depthwise never mixes channels; pointwise never mixes pixels.
    x = np.zeros((5, 5, 2))
    x[2, 2, 0] = 1.0
    dw = convolve(x, np.ones((3, 3)), "depthwise", padding="same")
    assert dw[:, :, 1].sum() == 0 and dw[:, :, 0].sum() == 9
    pw = convolve(x, np.array([[1.0, 2.0], [3.0, 4.0]]), "pointwise")
    assert np.count_nonzero(pw.any(axis=2)) == 1
    np.testing.assert_allclose(pw[2, 2], [1.0, 2.0])


def test_bias():
    x = rng.normal(size=(4, 4, 3))
    k = rng.normal(size=(3, 2))
    out = convolve(x, k, "pointwise", bias=[1.0, -1.0])
    np.testing.assert_allclose(out - convolve(x, k, "pointwise"), np.broadcast_to([1.0, -1.0], out.shape))


def test_load_drops_opaque_alpha_only():
    import os
    import tempfile
    from PIL import Image
    from conv import load_array
    with tempfile.TemporaryDirectory() as d:
        rgba = np.full((4, 5, 4), 255, dtype=np.uint8)
        rgba[:, :, :3] = rng.integers(0, 256, size=(4, 5, 3))
        opaque = os.path.join(d, "opaque.png")
        Image.fromarray(rgba).save(opaque)
        loaded = load_array(opaque)
        assert loaded.shape == (4, 5, 3)
        np.testing.assert_array_equal(loaded, rgba[:, :, :3])

        rgba[0, 0, 3] = 100                      # real transparency: keep alpha
        transparent = os.path.join(d, "transparent.png")
        Image.fromarray(rgba).save(transparent)
        assert load_array(transparent).shape == (4, 5, 4)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
