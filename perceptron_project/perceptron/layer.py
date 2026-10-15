"""
Neuron and Layer.

A Layer owns one Parameters object (W, b) and one Activation. For speed it
computes all of its neurons at once with matrix math:

    Z = W @ A_prev + b        weighted sums for every neuron and every sample
    A = f(Z)                  activations

A Neuron is a lightweight view of one row of the layer: its own weight vector
w = W[j], its bias b[j], and the shared activation. It is there so the code
mirrors the biological picture (each neuron sums weighted inputs, adds a bias,
and fires through a threshold like function), and so you can inspect any single
neuron. Changing a Neuron's weights changes the layer, because it is a view.
"""
import numpy as np

from .activation import get_activation
from .parameters import Parameters


class Neuron:
    def __init__(self, layer, index):
        self.layer = layer
        self.index = index

    @property
    def weights(self):
        return self.layer.params.W[self.index]       # view, not a copy

    @property
    def bias(self):
        return self.layer.params.b[self.index, 0]

    def output(self, x):
        """Output of this single neuron for one input vector x of shape (n_in,)."""
        z = float(self.weights @ x + self.bias)
        return float(self.layer.activation.forward(np.array(z)))

    def __repr__(self):
        return f"Neuron({self.index} of {self.layer.name}, {self.weights.size} inputs)"


class Layer:
    def __init__(self, n_in, n_out, activation="sigmoid", init="xavier", name=None, rng=None):
        self.name = name or f"Layer({n_in}->{n_out})"
        self.params = Parameters(n_in, n_out, init=init, rng=rng)
        self.activation = get_activation(activation)
        # cache from the most recent forward pass, needed by backprop
        self.A_prev = None
        self.Z = None
        self.A = None

    @property
    def neurons(self):
        return [Neuron(self, j) for j in range(self.params.n_out)]

    def neuron(self, j):
        return Neuron(self, j)

    def set_activation(self, activation):
        self.activation = get_activation(activation)

    def __repr__(self):
        return f"{self.name}: {self.params}, activation={self.activation.name}"
