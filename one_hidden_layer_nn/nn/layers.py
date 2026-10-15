"""Dense (fully connected) layer.

Shapes, with n = batch size:
  X  (n, n_in)      input
  W  (n_in, n_out)  one weight per (input feature, neuron) pair
  b  (1, n_out)     one bias per neuron (the adjustable threshold)
  Z = X @ W + b     (n, n_out) weighted sum, the "axon hillock"
  A = f(Z)          (n, n_out) activation, the "firing decision"
"""
import numpy as np

from .activations import Activation, ReLU, get_activation


class DenseLayer:
    def __init__(self, n_in: int, n_out: int, activation="sigmoid",
                 init: str = "auto", rng: np.random.Generator | None = None):
        if n_in < 1 or n_out < 1:
            raise ValueError("Layer sizes must be positive integers.")
        self.n_in = n_in
        self.n_out = n_out
        self.activation: Activation = get_activation(activation)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.W = self._init_weights(init)
        self.b = np.zeros((1, n_out))

        # Filled during forward/backward
        self._X = None
        self._Z = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def _init_weights(self, init: str) -> np.ndarray:
        if init == "auto":
            init = "he" if isinstance(self.activation, ReLU) else "xavier"
        if init == "xavier":
            # Glorot uniform: keeps signal variance stable for sigmoid/tanh
            limit = np.sqrt(6.0 / (self.n_in + self.n_out))
            return self.rng.uniform(-limit, limit, size=(self.n_in, self.n_out))
        if init == "he":
            return self.rng.normal(0.0, np.sqrt(2.0 / self.n_in), size=(self.n_in, self.n_out))
        if init == "zeros":
            # Fine for one neuron, but every hidden unit stays identical (symmetry problem)
            return np.zeros((self.n_in, self.n_out))
        raise ValueError(f"Unknown init '{init}'. Options: auto, xavier, he, zeros")

    # ---- core math -------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != self.n_in:
            raise ValueError(f"Expected {self.n_in} input features, got {X.shape[1]}.")
        self._X = X
        self._Z = X @ self.W + self.b
        return self.activation.forward(self._Z)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """Takes dL/dA for this layer's output, stores dW and db,
        returns dL/dX for the layer below."""
        if self._Z is None:
            raise RuntimeError("Call forward() before backward().")
        dZ = self.activation.backward(dA, self._Z)
        self.dW = self._X.T @ dZ
        self.db = dZ.sum(axis=0, keepdims=True)
        return dZ @ self.W.T

    # ---- parameter access for optimizers and persistence ------------------
    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}

    @property
    def n_params(self) -> int:
        return self.W.size + self.b.size

    def __repr__(self):
        return f"DenseLayer({self.n_in} -> {self.n_out}, {self.activation.name})"
