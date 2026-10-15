"""Three demos for the one hidden layer network.
Run from the project folder:  python demos/run_demos.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nn import (Momentum, OneHiddenLayerNetwork, SGD, Standardizer, accuracy,
                make_moons, one_hot, train_test_split)


def demo_xor():
    print("\n=== 1. XOR (not linearly separable, so a single perceptron fails) ===")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1.]])
    y = np.array([0, 1, 1, 0])
    net = OneHiddenLayerNetwork(2, 4, 1, "tanh", "sigmoid", optimizer=SGD(0.5), seed=3)
    print(net.summary())
    net.fit(X, y, epochs=3000, verbose=1000)
    for xi, p in zip(X, net.predict_proba(X).ravel()):
        print(f"  {xi.astype(int)} -> {p:.3f}  -> class {int(p >= 0.5)}")
    return net, X, y


def demo_moons():
    print("\n=== 2. Two moons, train/test split, mini batch momentum ===")
    X, y = make_moons(600, noise=0.2, seed=7)
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.25, seed=7)
    scaler = Standardizer().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    net = OneHiddenLayerNetwork(2, 16, 1, "tanh", "sigmoid", optimizer=Momentum(0.05), seed=7)
    print(net.summary())
    net.fit(Xtr_s, ytr, epochs=400, batch_size=32,
            validation_data=(Xte_s, yte), verbose=100, seed=7)
    print(f"  test accuracy: {accuracy(yte, net.predict(Xte_s)):.3f}")
    return net, scaler, X, y


def make_blobs(n_per=150, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 2.5], [-2.2, -1.3], [2.2, -1.3]])
    X = np.vstack([c + rng.normal(0, 0.9, (n_per, 2)) for c in centers])
    y = np.repeat(np.arange(3), n_per)
    return X, y


def demo_multiclass():
    print("\n=== 3. Three classes, ReLU hidden layer, softmax output ===")
    X, y = make_blobs(seed=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.25, seed=1)
    net = OneHiddenLayerNetwork(2, 10, 3, "relu", "softmax", optimizer=SGD(0.1), seed=1)
    print(net.summary())
    net.fit(Xtr, one_hot(ytr, 3), epochs=300, batch_size=16,
            validation_data=(Xte, one_hot(yte, 3)), verbose=100, seed=1)
    print(f"  test accuracy: {accuracy(yte, net.predict(Xte)):.3f}")
    return net, X, y


def plot_boundary(ax, predict, X, y, title, transform=None):
    pad = 0.6
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300),
                         np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if transform is not None:
        grid = transform(grid)
    zz = predict(grid).reshape(xx.shape)
    k = int(y.max()) + 1
    colors = ["#3b6ea8", "#d9822b", "#4a9a6a"][:k]
    cmap = matplotlib.colors.ListedColormap(colors)
    ax.contourf(xx, yy, zz, levels=np.arange(k + 1) - 0.5, cmap=cmap, alpha=0.22)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=14 if len(X) > 10 else 120,
               edgecolors="white", linewidths=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    xor_net, Xx, yx = demo_xor()
    moon_net, scaler, Xm, ym = demo_moons()
    blob_net, Xb, yb = demo_multiclass()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    plot_boundary(axes[0, 0], xor_net.predict, Xx, yx, "XOR, 4 tanh hidden units")
    plot_boundary(axes[0, 1], moon_net.predict, Xm, ym, "Two moons, 16 tanh hidden units",
                  transform=scaler.transform)
    plot_boundary(axes[0, 2], blob_net.predict, Xb, yb, "3 classes, 10 ReLU hidden units")

    for ax, net, title in [(axes[1, 0], xor_net, "XOR loss"),
                           (axes[1, 1], moon_net, "Moons loss"),
                           (axes[1, 2], blob_net, "3 class loss")]:
        ax.plot(net.history.loss, color="#3b6ea8", label="train")
        if net.history.val_loss:
            ax.plot(net.history.val_loss, color="#d9822b", label="test")
            ax.legend(frameon=False)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("One hidden layer network: decision boundaries and training curves", fontsize=13)
    fig.tight_layout()
    out = os.path.join(ROOT, "demo_results.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
