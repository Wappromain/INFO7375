"""
GradDescent: moves every W and b a small step against its gradient.

    W <- W - learning_rate * dW
    b <- b - learning_rate * db

Optional momentum keeps a running "velocity" so updates build speed in
consistent directions and smooth out zig zags:

    v <- momentum * v - learning_rate * dW
    W <- W + v
"""
import numpy as np


class GradDescent:
    def __init__(self, learning_rate=0.5, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._velocity = {}

    def step(self, model):
        for i, layer in enumerate(model.layers):
            p = layer.params
            if self.momentum:
                vW, vb = self._velocity.get(i, (np.zeros_like(p.W), np.zeros_like(p.b)))
                vW = self.momentum * vW - self.learning_rate * p.dW
                vb = self.momentum * vb - self.learning_rate * p.db
                self._velocity[i] = (vW, vb)
                p.W += vW
                p.b += vb
            else:
                p.W -= self.learning_rate * p.dW
                p.b -= self.learning_rate * p.db

    def reset(self):
        self._velocity = {}
