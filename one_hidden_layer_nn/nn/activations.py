"""Activation functions.

Every activation is a small object with two jobs:
  forward(Z)       -> A, the layer output
  backward(dA, Z)  -> dZ, the gradient pushed back through the activation

The default backward multiplies by the elementwise derivative. Softmax
overrides backward because its outputs depend on each other (full Jacobian).
"""
from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    name = "activation"

    @abstractmethod
    def forward(self, Z: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def derivative(self, Z: np.ndarray) -> np.ndarray:
        """Elementwise derivative dA/dZ evaluated at Z."""

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return dA * self.derivative(Z)

    def __repr__(self):
        return f"{type(self).__name__}()"


class Identity(Activation):
    name = "identity"

    def forward(self, Z):
        return Z

    def derivative(self, Z):
        return np.ones_like(Z)


class Step(Activation):
    """Rosenblatt's hard threshold. Included for comparison only: its derivative
    is zero almost everywhere, so it cannot be trained with backpropagation."""
    name = "step"

    def forward(self, Z):
        return (Z > 0).astype(float)

    def derivative(self, Z):
        return np.zeros_like(Z)


class Sigmoid(Activation):
    name = "sigmoid"

    def forward(self, Z):
        # Numerically stable form for large negative and positive Z
        out = np.empty_like(Z, dtype=float)
        pos = Z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-Z[pos]))
        ez = np.exp(Z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def derivative(self, Z):
        s = self.forward(Z)
        return s * (1.0 - s)


class Tanh(Activation):
    name = "tanh"

    def forward(self, Z):
        return np.tanh(Z)

    def derivative(self, Z):
        return 1.0 - np.tanh(Z) ** 2


class ReLU(Activation):
    name = "relu"

    def forward(self, Z):
        return np.maximum(0.0, Z)

    def derivative(self, Z):
        return (Z > 0).astype(float)


class Softmax(Activation):
    name = "softmax"

    def forward(self, Z):
        shifted = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=1, keepdims=True)

    def derivative(self, Z):
        raise NotImplementedError("Softmax has a full Jacobian; use backward().")

    def backward(self, dA, Z):
        # Jacobian vector product per row: s * (dA - sum(dA * s))
        s = self.forward(Z)
        return s * (dA - (dA * s).sum(axis=1, keepdims=True))


_REGISTRY = {cls.name: cls for cls in (Identity, Step, Sigmoid, Tanh, ReLU, Softmax)}


def get_activation(spec) -> Activation:
    """Accept either an Activation instance or its name ("relu", "sigmoid", ...)."""
    if isinstance(spec, Activation):
        return spec
    try:
        return _REGISTRY[spec.lower()]()
    except KeyError:
        raise ValueError(f"Unknown activation '{spec}'. Options: {sorted(_REGISTRY)}")
