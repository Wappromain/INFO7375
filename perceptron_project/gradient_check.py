"""
gradient_check.py
Verifies backpropagation by comparing its gradients with numerical ones:

    dL/dw  ~=  (L(w + eps) - L(w - eps)) / (2 * eps)

If backprop is correct, the relative difference is tiny (below about 1e-6).
Run this any time you add a new activation or loss function.
"""
import numpy as np

from perceptron import NeuralNetwork, ForwardProp, BackProp, get_loss, one_hot


def check(activation, loss_name, eps=1e-5):
    rng = np.random.default_rng(0)
    X = rng.random((6, 5))
    Y = one_hot(rng.integers(0, 3, 5), 3)
    net = NeuralNetwork(6, [4], 3, hidden_activation=activation, seed=1)
    loss_fn = get_loss(loss_name)
    fwd, bwd = ForwardProp(), BackProp()

    fwd.run(net, X)
    bwd.run(net, loss_fn, Y)

    worst = 0.0
    for layer in net.layers:
        for arr, grad in ((layer.params.W, layer.params.dW), (layer.params.b, layer.params.db)):
            num = np.zeros_like(arr)
            for idx in np.ndindex(arr.shape):
                old = arr[idx]
                arr[idx] = old + eps
                lp = loss_fn.value(fwd.run(net, X), Y)
                arr[idx] = old - eps
                lm = loss_fn.value(fwd.run(net, X), Y)
                arr[idx] = old
                num[idx] = (lp - lm) / (2 * eps)
            rel = np.linalg.norm(grad - num) / (np.linalg.norm(grad) + np.linalg.norm(num) + 1e-12)
            worst = max(worst, rel)
    status = "OK" if worst < 1e-6 else "MISMATCH"
    print(f"  hidden={activation:<10s} loss={loss_name:<4s} max relative diff {worst:.2e}  {status}")
    return worst


if __name__ == "__main__":
    print("Gradient check (backprop vs numerical):")
    for act in ["sigmoid", "tanh", "leaky_relu"]:
        for loss in ["mse", "bce"]:
            check(act, loss)
