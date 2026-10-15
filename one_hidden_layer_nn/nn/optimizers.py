"""Optimizers update parameters from gradients. Kept separate from the layer so
the update rule can change without touching the forward/backward math."""
from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    def __init__(self, learning_rate: float = 0.1):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, layers) -> None:
        ...


class SGD(Optimizer):
    """Plain gradient descent: w <- w - lr * dL/dw"""

    def step(self, layers):
        for layer in layers:
            for key, p in layer.params().items():
                p -= self.learning_rate * layer.grads()[key]


class Momentum(Optimizer):
    """SGD with momentum: a running velocity smooths noisy mini batch updates."""

    def __init__(self, learning_rate: float = 0.1, beta: float = 0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self._velocity = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            for key, p in layer.params().items():
                v = self._velocity.setdefault((i, key), np.zeros_like(p))
                v *= self.beta
                v -= self.learning_rate * layer.grads()[key]
                p += v
