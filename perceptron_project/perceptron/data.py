"""
Loading images into matrices.

To use a different image set, point the loaders at another folder with the
same layout. Images of any size are resized to image_size, converted to
grayscale, and scaled to 0..1.

    <train_dir>/<label>/*.png     labeled training images
    <test_dir>/*.png              unlabeled test images
"""
import os

import numpy as np
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


def load_image(path, image_size=(20, 20), invert=False):
    img = Image.open(path).convert("L").resize(image_size)
    x = np.asarray(img, dtype=np.float64) / 255.0
    if invert:                       # use for dark ink on a white background
        x = 1.0 - x
    return x.reshape(-1)             # 20 x 20 -> 400 values


def one_hot(labels, n_classes):
    Y = np.zeros((n_classes, len(labels)))
    Y[labels, np.arange(len(labels))] = 1.0
    return Y


def load_labeled(train_dir, image_size=(20, 20), invert=False):
    """Returns X (n_pixels, m), Y one hot (n_classes, m), labels, class names."""
    classes = sorted(d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d)))
    xs, labels = [], []
    for ci, c in enumerate(classes):
        folder = os.path.join(train_dir, c)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(IMG_EXT):
                xs.append(load_image(os.path.join(folder, f), image_size, invert))
                labels.append(ci)
    X = np.stack(xs, axis=1)
    labels = np.array(labels)
    return X, one_hot(labels, len(classes)), labels, classes


def load_unlabeled(test_dir, image_size=(20, 20), invert=False):
    """Returns X (n_pixels, m) and the file names in the same order."""
    files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith(IMG_EXT))
    X = np.stack([load_image(os.path.join(test_dir, f), image_size, invert)
                  for f in files], axis=1)
    return X, files


def augment(X, Y, copies=10, max_rotate=12, max_shift=1, image_size=(20, 20), seed=0):
    """Data augmentation: adds `copies` randomly rotated and shifted versions of
    every training image. The network sees more ways each digit can be written,
    which helps it generalize to new handwriting. Labels are copied unchanged."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    xs, ys = [X], [Y]
    for _ in range(copies):
        cols = []
        for i in range(X.shape[1]):
            im = Image.fromarray((X[:, i].reshape(image_size) * 255).astype(np.uint8))
            im = im.rotate(rng.uniform(-max_rotate, max_rotate), resample=Image.BILINEAR,
                           translate=(int(rng.integers(-max_shift, max_shift + 1)),
                                      int(rng.integers(-max_shift, max_shift + 1))))
            cols.append(np.asarray(im, dtype=np.float64).reshape(-1) / 255.0)
        xs.append(np.stack(cols, axis=1))
        ys.append(Y)
    return np.hstack(xs), np.hstack(ys)
