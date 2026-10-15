"""
Loss functions.

Each loss has:
    value(A, Y)      a single number: how wrong the network is, averaged over samples
    gradient(A, Y)   dL/dA, the starting point of backpropagation

A and Y have shape (n_outputs, m). Y is one hot: for digit 3, the column is
[0,0,0,1,0,0,0,0,0,0].

Note on the errors canceling out: summing raw errors (y - a) lets positive and
negative errors cancel. Both losses below avoid that, MSE by squaring and
cross entropy by using logs of probabilities, which are never negative losses.
"""
import numpy as np

EPS = 1e-12


class LossFunction:
    name = "base"

    def value(self, A, Y):
        raise NotImplementedError

    def gradient(self, A, Y):
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """L = (1 / 2m) * sum (a - y)^2    (the 1/2 makes the gradient clean)"""
    name = "mse"

    def value(self, A, Y):
        m = Y.shape[1]
        return float(np.sum((A - Y) ** 2) / (2 * m))

    def gradient(self, A, Y):
        m = Y.shape[1]
        return (A - Y) / m


class BinaryCrossEntropy(LossFunction):
    """Cross entropy for sigmoid outputs, each output treated as its own yes/no.
    With sigmoid outputs, the backprop term simplifies to (A - Y) / m, which
    avoids the slow learning MSE suffers when a sigmoid is saturated."""
    name = "bce"

    def value(self, A, Y):
        m = Y.shape[1]
        A = np.clip(A, EPS, 1 - EPS)
        return float(-np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m)

    def gradient(self, A, Y):
        m = Y.shape[1]
        A = np.clip(A, EPS, 1 - EPS)
        return (A - Y) / (A * (1 - A)) / m


LOSSES = {cls.name: cls for cls in [MeanSquaredError, BinaryCrossEntropy]}


def get_loss(spec):
    if isinstance(spec, LossFunction):
        return spec
    try:
        return LOSSES[spec.lower()]()
    except KeyError:
        raise ValueError(f"Unknown loss '{spec}'. Options: {list(LOSSES)}")
