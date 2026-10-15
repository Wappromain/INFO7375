"""
demo.py
Checks that everything is correct, then shows the network learning.
  1. Activation derivatives vs numerical derivatives
  2. Backprop gradient check for each loss, plus L1, L2 and dropout
  3. XOR (a single perceptron cannot learn it)
  4. 3 class spiral (ReLU hidden layers, Softmax output)
  5. Regression of sin(x) (Linear output, MSE)
  6. Overfitting experiment: no regularization vs L2, L1, dropout
Run:  python demo.py
"""

import numpy as np

from activations import Linear, ReLU, Sigmoid, Tanh, Softmax
from neural_network import DeepNeuralNetwork

rng = np.random.default_rng(0)


def section(title):
    print("\n" + "=" * 62 + f"\n{title}\n" + "=" * 62)


# 1 ---------------------------------------------------------------------------
section("1. Activation derivatives (max error vs numerical)")
Z = rng.standard_normal((4, 5)) * 2
eps = 1e-6
for act in (Linear(), ReLU(), Sigmoid(), Tanh(), Softmax()):
    dA = rng.standard_normal(Z.shape)
    numeric = np.zeros_like(Z)
    for idx in np.ndindex(Z.shape):
        Zp, Zm = Z.copy(), Z.copy()
        Zp[idx] += eps
        Zm[idx] -= eps
        numeric[idx] = np.sum(dA * (act(Zp) - act(Zm))) / (2 * eps)
    print(f"{act.name:8s} {np.max(np.abs(act.backward(dA, Z) - numeric)):.2e}")

# 2 ---------------------------------------------------------------------------
section("2. Backprop gradient check (relative difference)")
X = rng.standard_normal((3, 8))
Yb = rng.integers(0, 2, (1, 8)).astype(float)
Yc = np.eye(3)[rng.integers(0, 3, 8)].T
Yr = rng.standard_normal((2, 8))
cases = [
    ("sigmoid + BCE",          [3, 5, 4, 1], ["relu", "tanh", "sigmoid"], "binary_cross_entropy", Yb, {}),
    ("softmax + CCE",          [3, 5, 4, 3], ["tanh", "sigmoid", "softmax"], "categorical_cross_entropy", Yc, {}),
    ("linear + MSE",           [3, 5, 4, 2], ["tanh", "relu", "linear"], "mse", Yr, {}),
    ("tanh + MSE",             [3, 5, 2],    ["relu", "tanh"], "mse", Yr, {}),
    ("BCE + L2",               [3, 5, 4, 1], ["relu", "tanh", "sigmoid"], "binary_cross_entropy", Yb, dict(l2=0.7)),
    ("BCE + L1",               [3, 5, 4, 1], ["relu", "tanh", "sigmoid"], "binary_cross_entropy", Yb, dict(l1=0.5)),
    ("CCE + dropout",          [3, 5, 4, 3], ["tanh", "relu", "softmax"], "categorical_cross_entropy", Yc, dict(dropout=0.4)),
    ("CCE + L1 + L2 + dropout", [3, 5, 4, 3], ["tanh", "relu", "softmax"], "categorical_cross_entropy", Yc, dict(l1=0.2, l2=0.3, dropout=0.25)),
]
for name, dims, acts, loss, Y, kw in cases:
    net = DeepNeuralNetwork(dims, acts, loss=loss, seed=1, **kw)
    diff = net.gradient_check(X, Y, with_dropout="dropout" in kw)
    print(f"{name:25s} {diff:.2e}  {'PASS' if diff < 1e-6 else 'FAIL'}")

# 3 ---------------------------------------------------------------------------
section("3. XOR with a 2-4-1 network")
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=float)
Y = np.array([[0, 1, 1, 0]], dtype=float)
net = DeepNeuralNetwork([2, 4, 1], ["tanh", "sigmoid"], seed=3)
hist = net.fit(X, Y, epochs=3000, learning_rate=0.5)["train"]
print(f"cost {hist[0]:.4f} -> {hist[-1]:.4f}   predictions {net.predict(X).ravel()}"
      f"   targets {Y.astype(int).ravel()}")

# 4 ---------------------------------------------------------------------------
section("4. Three class spiral, [2, 32, 32, 3] ReLU/ReLU/Softmax")
N, K = 100, 3
Xs, ys = [], []
for k in range(K):
    r = np.linspace(0.0, 1, N)
    t = np.linspace(k * 4, (k + 1) * 4, N) + rng.standard_normal(N) * 0.2
    Xs.append(np.c_[r * np.sin(t), r * np.cos(t)])
    ys.append(np.full(N, k))
X, y = np.vstack(Xs).T, np.concatenate(ys)
net = DeepNeuralNetwork([2, 32, 32, K], ["relu", "relu", "softmax"],
                        loss="categorical_cross_entropy", seed=0)
hist = net.fit(X, np.eye(K)[y].T, epochs=3000, learning_rate=0.5, batch_size=64)["train"]
print(f"cost {hist[0]:.4f} -> {hist[-1]:.4f}   training accuracy {net.accuracy(X, y):.1%}")

# 5 ---------------------------------------------------------------------------
section("5. Regression of sin(x), [1, 20, 20, 1] Tanh/Tanh/Linear + MSE")
x = np.linspace(-np.pi, np.pi, 200).reshape(1, -1)
net = DeepNeuralNetwork([1, 20, 20, 1], ["tanh", "tanh", "linear"], loss="mse", seed=0)
hist = net.fit(x, np.sin(x), epochs=4000, learning_rate=0.1)["train"]
print(f"cost {hist[0]:.4f} -> {hist[-1]:.6f}   max abs error "
      f"{np.max(np.abs(net.predict(x) - np.sin(x))):.3f}")


# 6 ---------------------------------------------------------------------------
def make_moons(n, noise):
    n1 = n // 2
    t1, t2 = rng.uniform(0, np.pi, n1), rng.uniform(0, np.pi, n - n1)
    X = np.vstack([np.c_[np.cos(t1), np.sin(t1)],
                   np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]])
    X += rng.standard_normal(X.shape) * noise
    return X.T, np.r_[np.zeros(n1), np.ones(n - n1)].reshape(1, -1)


section("6. Overfitting: 60 noisy points, [2, 100, 100, 1] network")
X_tr, Y_tr = make_moons(60, 0.35)
X_te, Y_te = make_moons(2000, 0.35)
print(f"{'setting':22s} {'train':>7s} {'test':>7s} {'gap':>7s} {'test loss':>10s} {'|w|<1e-3':>9s}")
for name, kw in [("none", {}), ("L2 = 1.0", dict(l2=1.0)), ("L1 = 0.3", dict(l1=0.3)),
                 ("dropout 0.5", dict(dropout=0.5)),
                 ("L2 0.3 + dropout 0.3", dict(l2=0.3, dropout=0.3))]:
    net = DeepNeuralNetwork([2, 100, 100, 1], ["relu", "relu", "sigmoid"], seed=1, **kw)
    hist = net.fit(X_tr, Y_tr, epochs=6000, learning_rate=0.3, X_val=X_te, Y_val=Y_te)
    tr, te = net.accuracy(X_tr, Y_tr), net.accuracy(X_te, Y_te)
    w = np.concatenate([W.ravel() for W in net.weights()])
    print(f"{name:22s} {tr:7.1%} {te:7.1%} {tr - te:7.1%} {hist['val'][-1]:10.4f} "
          f"{np.mean(np.abs(w) < 1e-3):9.1%}")
