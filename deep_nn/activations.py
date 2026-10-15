"""
activations.py
Linear, ReLU, Sigmoid, Tanh and Softmax activation functions.

Shapes: Z and A are (n_units, m), one COLUMN per training example.

Each activation class provides:
    forward(Z)       -> A = g(Z)
    derivative(Z)    -> g'(Z), elementwise
    backward(dA, Z)  -> dZ = dA * g'(Z)   (Softmax overrides this)
"""

import numpy as np


class Activation:
    """Base class. Subclasses implement forward() and derivative()."""

    name = "activation"

    def forward(self, Z):
        raise NotImplementedError

    def derivative(self, Z):
        raise NotImplementedError

    def backward(self, dA, Z):
        # Chain rule for elementwise activations
        return dA * self.derivative(Z)

    def __call__(self, Z):
        return self.forward(Z)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Linear(Activation):
    """g(z) = z. Output layer for regression."""

    name = "linear"

    def forward(self, Z):
        return Z

    def derivative(self, Z):
        return np.ones_like(Z, dtype=float)


class ReLU(Activation):
    """g(z) = max(0, z). Default choice for hidden layers."""

    name = "relu"

    def forward(self, Z):
        return np.maximum(0, Z)

    def derivative(self, Z):
        return (Z > 0).astype(float)          # g'(0) taken as 0


class Sigmoid(Activation):
    """g(z) = 1 / (1 + e^-z). Output in (0, 1); binary classification."""

    name = "sigmoid"

    def forward(self, Z):
        # Stable form: never exponentiate a large positive number
        Z = np.asarray(Z, dtype=float)
        out = np.empty_like(Z)
        pos = Z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-Z[pos]))
        ez = np.exp(Z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def derivative(self, Z):
        s = self.forward(Z)
        return s * (1.0 - s)


class Tanh(Activation):
    """g(z) = tanh(z). Output in (-1, 1), zero centered."""

    name = "tanh"

    def forward(self, Z):
        return np.tanh(Z)

    def derivative(self, Z):
        return 1.0 - np.tanh(Z) ** 2


class Softmax(Activation):
    """g(z)_i = e^{z_i} / sum_j e^{z_j}, per column. Multiclass output.

    Not elementwise: each output depends on every input in its column,
    so the derivative is a Jacobian J = diag(s) - s s^T rather than a
    vector. backward() applies J without building it."""

    name = "softmax"

    def forward(self, Z):
        Z = np.asarray(Z, dtype=float)
        e = np.exp(Z - np.max(Z, axis=0, keepdims=True))   # avoids overflow
        return e / np.sum(e, axis=0, keepdims=True)

    def derivative(self, Z):
        raise NotImplementedError("Softmax has a Jacobian; use backward() or jacobian().")

    def jacobian(self, z):
        """Jacobian for a single example: J_ij = s_i (delta_ij - s_j)."""
        s = self.forward(np.asarray(z, dtype=float).reshape(-1, 1)).ravel()
        return np.diag(s) - np.outer(s, s)

    def backward(self, dA, Z):
        # Jacobian vector product for all columns: dZ = S * (dA - sum(dA * S))
        S = self.forward(Z)
        return S * (dA - np.sum(dA * S, axis=0, keepdims=True))


ACTIVATIONS = {cls.name: cls for cls in (Linear, ReLU, Sigmoid, Tanh, Softmax)}


def get_activation(act):
    """Accept an Activation instance or one of the names above."""
    if isinstance(act, Activation):
        return act
    try:
        return ACTIVATIONS[act.lower()]()
    except KeyError:
        raise ValueError(f"Unknown activation '{act}'. Choose from {list(ACTIVATIONS)}")
