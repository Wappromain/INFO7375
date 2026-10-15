"""
Activation functions.

Every activation is a class with two methods:
    forward(Z)      the function itself, a = f(z)
    derivative(Z)   f'(z), used by backpropagation

To add a new activation, subclass Activation, implement both methods, and
register it in ACTIVATIONS. Nothing else in the code needs to change.
"""
import numpy as np


class Activation:
    name = "base"

    def forward(self, Z):
        raise NotImplementedError

    def derivative(self, Z):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Sigmoid(Activation):
    """sigma(z) = 1 / (1 + e^-z); sigma'(z) = sigma(z) * (1 - sigma(z))"""
    name = "sigmoid"

    def forward(self, Z):
        Z = np.clip(Z, -500, 500)          # avoid overflow in exp
        return 1.0 / (1.0 + np.exp(-Z))

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


class LeakyReLU(Activation):
    name = "leaky_relu"

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, Z):
        return np.where(Z > 0, Z, self.alpha * Z)

    def derivative(self, Z):
        return np.where(Z > 0, 1.0, self.alpha)


class Step(Activation):
    """The original Rosenblatt / McCulloch-Pitts threshold. Its derivative is
    zero almost everywhere, so it cannot be trained with gradient descent.
    Included for comparison only."""
    name = "step"

    def forward(self, Z):
        return (Z > 0).astype(float)

    def derivative(self, Z):
        return np.zeros_like(Z)


ACTIVATIONS = {cls.name: cls for cls in [Sigmoid, Tanh, ReLU, LeakyReLU, Step]}


def get_activation(spec):
    """Accepts an Activation object or a name such as 'sigmoid'."""
    if isinstance(spec, Activation):
        return spec
    try:
        return ACTIVATIONS[spec.lower()]()
    except KeyError:
        raise ValueError(f"Unknown activation '{spec}'. Options: {list(ACTIVATIONS)}")
