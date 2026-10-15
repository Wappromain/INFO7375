"""
regularization.py
L1 and L2 weight regularization. m is the number of examples in the batch.

Cost added to the loss:
    L2: (lambda2 / 2m) * sum over layers of ||W||^2
    L1: (lambda1 /  m) * sum over layers of |W|
Gradient added to each dW:
    L2: (lambda2 / m) * W          shrinks weights in proportion to size
    L1: (lambda1 / m) * sign(W)    constant pull toward zero -> sparse weights
Biases are not regularized.
"""

import numpy as np


class Regularization:
    def __init__(self, l1=0.0, l2=0.0):
        if l1 < 0 or l2 < 0:
            raise ValueError("Regularization strengths must be >= 0")
        self.l1 = l1
        self.l2 = l2

    @property
    def active(self):
        return self.l1 > 0 or self.l2 > 0

    def cost(self, weights, m):
        """Penalty for a list of weight matrices [W1, ..., WL]."""
        total = 0.0
        if self.l2 > 0:
            total += self.l2 / (2 * m) * sum(float(np.sum(W ** 2)) for W in weights)
        if self.l1 > 0:
            total += self.l1 / m * sum(float(np.sum(np.abs(W))) for W in weights)
        return total

    def gradient(self, W, m):
        """Term added to dW for one layer."""
        grad = np.zeros_like(W)
        if self.l2 > 0:
            grad += (self.l2 / m) * W
        if self.l1 > 0:
            grad += (self.l1 / m) * np.sign(W)     # subgradient, sign(0) = 0
        return grad

    def __repr__(self):
        return f"Regularization(l1={self.l1}, l2={self.l2})"
