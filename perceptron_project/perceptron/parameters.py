"""
Parameters: the transmission (weight) matrix W and bias vector b of one layer,
plus the module that sets their initial values.

Shapes (column convention, one column per sample):
    W : (n_out, n_in)    row j holds the weights of neuron j
    b : (n_out, 1)

Initialization methods:
    "random"   small Gaussian values, scale 0.01 by default
    "xavier"   variance 1 / n_in, good default for sigmoid and tanh
    "he"       variance 2 / n_in, good default for ReLU
    "zeros"    all zeros. Fine for a single neuron, but in a hidden layer every
               neuron would get identical gradients and never specialize
               (the symmetry problem), so training stalls.
"""
import numpy as np


class Parameters:
    def __init__(self, n_in, n_out, init="xavier", scale=0.01, rng=None):
        self.n_in = n_in
        self.n_out = n_out
        self.init_method = init
        self.scale = scale
        self.rng = rng if rng is not None else np.random.default_rng()
        self.W = None
        self.b = None
        self.dW = None      # gradients, filled in by backprop
        self.db = None
        self.initialize()

    def initialize(self):
        shape = (self.n_out, self.n_in)
        m = self.init_method
        if m == "zeros":
            self.W = np.zeros(shape)
        elif m == "random":
            self.W = self.rng.normal(0.0, self.scale, shape)
        elif m == "xavier":
            self.W = self.rng.normal(0.0, np.sqrt(1.0 / self.n_in), shape)
        elif m == "he":
            self.W = self.rng.normal(0.0, np.sqrt(2.0 / self.n_in), shape)
        else:
            raise ValueError(f"Unknown init method '{m}'")
        self.b = np.zeros((self.n_out, 1))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def count(self):
        return self.W.size + self.b.size

    def as_dict(self):
        return {"W": self.W, "b": self.b}

    def __repr__(self):
        return f"Parameters(W{self.W.shape}, b{self.b.shape}, init='{self.init_method}')"
