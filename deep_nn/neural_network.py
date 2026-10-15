"""
neural_network.py
A deep (multilayer) fully connected neural network:
class structure, forward propagation, loss (cost) functions, and
backpropagation. Uses activations.py, regularization.py and dropout.py.

Notation (layer l = 1..L, one column per example):
    A[0] = X                         shape (n_x, m)
    Z[l] = W[l] A[l-1] + b[l]        W[l]: (n_l, n_{l-1}), b[l]: (n_l, 1)
    A[l] = g[l](Z[l])                (then dropout on hidden layers, training only)
    J    = loss(A[L], Y) + regularization penalty

Backprop, from the output layer down:
    dZ[l]   = dA[l] * g'[l](Z[l])            (Softmax: Jacobian product)
    dW[l]   = dZ[l] A[l-1]^T + regularization gradient
    db[l]   = sum over examples of dZ[l]
    dA[l-1] = W[l]^T dZ[l], then the same dropout mask as the forward pass
The 1/m averaging is inside the loss gradient, so it is not repeated.
"""

import numpy as np

from activations import get_activation, Linear, Sigmoid, Softmax
from dropout import Dropout
from regularization import Regularization

EPS = 1e-12


class DeepNeuralNetwork:
    LOSSES = ("mse", "binary_cross_entropy", "categorical_cross_entropy")

    def __init__(self, layer_dims, activations, loss="binary_cross_entropy",
                 l1=0.0, l2=0.0, dropout=0.0, seed=None):
        """
        layer_dims : [n_x, n_1, ..., n_L], input size then units per layer
        activations: one per layer (length L), e.g. ["relu", "relu", "sigmoid"]
        loss       : "mse", "binary_cross_entropy" or "categorical_cross_entropy"
        l1, l2     : regularization strengths (0 disables)
        dropout    : drop rate applied to every hidden layer (0 disables)
        """
        if len(activations) != len(layer_dims) - 1:
            raise ValueError("Need exactly one activation per layer")
        if loss not in self.LOSSES:
            raise ValueError(f"loss must be one of {self.LOSSES}")
        self.layer_dims = list(layer_dims)
        self.L = len(layer_dims) - 1
        self.activations = [get_activation(a) for a in activations]
        self.loss = loss
        self.regularization = Regularization(l1, l2)
        self.dropout = Dropout(dropout, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.parameters = {}
        self.caches = []
        self.initialize_parameters()

    # ------------------------------------------------------------ parameters
    def initialize_parameters(self):
        """Small random weights break the symmetry between units in a layer.
        Scale: He sqrt(2/n_in) for ReLU layers, Xavier sqrt(1/n_in) otherwise.
        Biases start at zero."""
        for l in range(1, self.L + 1):
            n_in, n_out = self.layer_dims[l - 1], self.layer_dims[l]
            gain = 2.0 if self.activations[l - 1].name == "relu" else 1.0
            self.parameters[f"W{l}"] = self.rng.standard_normal((n_out, n_in)) * np.sqrt(gain / n_in)
            self.parameters[f"b{l}"] = np.zeros((n_out, 1))

    def weights(self):
        return [self.parameters[f"W{l}"] for l in range(1, self.L + 1)]

    # ---------------------------------------------------- forward propagation
    def forward(self, X, training=False, reuse_masks=False):
        """Returns AL. Caches (A_prev, Z, mask) per layer for backprop.
        Dropout applies to hidden layers only when training=True.
        reuse_masks keeps the previous masks (for gradient checking)."""
        old_masks = [c["mask"] for c in self.caches] if reuse_masks else None
        self.caches = []
        A = X
        for l in range(1, self.L + 1):
            W, b = self.parameters[f"W{l}"], self.parameters[f"b{l}"]
            Z = W @ A + b
            cache = {"A_prev": A, "Z": Z, "mask": None}
            A = self.activations[l - 1].forward(Z)
            if training and l < self.L and self.dropout.active:
                A, cache["mask"] = self.dropout.forward(
                    A, old_masks[l - 1] if reuse_masks else None)
            self.caches.append(cache)
        return A

    # -------------------------------------------------- loss (cost) functions
    def loss_value(self, AL, Y):
        """Average loss over the m examples, without regularization."""
        m = Y.shape[1]
        if self.loss == "mse":
            return float(np.sum((AL - Y) ** 2) / (2 * m))
        if self.loss == "binary_cross_entropy":
            A = np.clip(AL, EPS, 1 - EPS)
            return float(-np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m)
        A = np.clip(AL, EPS, 1.0)                     # categorical, Y one hot
        return float(-np.sum(Y * np.log(A)) / m)

    def loss_gradient(self, AL, Y):
        """dJ/dAL, with the 1/m averaging included."""
        m = Y.shape[1]
        if self.loss == "mse":
            return (AL - Y) / m
        if self.loss == "binary_cross_entropy":
            A = np.clip(AL, EPS, 1 - EPS)
            return (-(Y / A) + (1 - Y) / (1 - A)) / m
        A = np.clip(AL, EPS, 1.0)
        return -(Y / A) / m

    def compute_cost(self, AL, Y):
        """J = loss + regularization penalty."""
        return self.loss_value(AL, Y) + self.regularization.cost(self.weights(), Y.shape[1])

    # ------------------------------------------------------- backpropagation
    def _output_simplifies(self):
        """These pairs simplify to dZ[L] = (AL - Y) / m."""
        out = self.activations[-1]
        return ((self.loss == "binary_cross_entropy" and isinstance(out, Sigmoid)) or
                (self.loss == "categorical_cross_entropy" and isinstance(out, Softmax)) or
                (self.loss == "mse" and isinstance(out, Linear)))

    def backward(self, AL, Y):
        """Must follow forward() on the same batch. Returns dW1..dWL, db1..dbL."""
        grads = {}
        m = Y.shape[1]
        Y = Y.reshape(AL.shape)
        dA = self.loss_gradient(AL, Y)

        for l in reversed(range(1, self.L + 1)):
            cache = self.caches[l - 1]
            W = self.parameters[f"W{l}"]

            if l == self.L and self._output_simplifies():
                dZ = (AL - Y) / m
            else:
                dZ = self.activations[l - 1].backward(dA, cache["Z"])

            grads[f"dW{l}"] = dZ @ cache["A_prev"].T + self.regularization.gradient(W, m)
            grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True)

            dA = W.T @ dZ
            if l > 1:   # undo the dropout applied to A[l-1] in the forward pass
                dA = self.dropout.backward(dA, self.caches[l - 2]["mask"])
        return grads

    def update_parameters(self, grads, learning_rate):
        """Gradient descent: theta = theta - alpha * dtheta."""
        for l in range(1, self.L + 1):
            self.parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    # --------------------------------------------------------------- training
    def fit(self, X, Y, epochs=1000, learning_rate=0.1, batch_size=None,
            X_val=None, Y_val=None, print_every=0):
        """(Mini batch) gradient descent. Costs are recorded with dropout off.
        Returns {"train": [...]} plus "val" when validation data is given."""
        m = X.shape[1]
        history = {"train": []}
        if X_val is not None:
            history["val"] = []
        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(m) if batch_size else np.arange(m)
            step = batch_size or m
            for i in range(0, m, step):
                idx = order[i:i + step]
                Xb, Yb = X[:, idx], Y[:, idx]
                AL = self.forward(Xb, training=True)
                self.update_parameters(self.backward(AL, Yb), learning_rate)
            history["train"].append(self.compute_cost(self.forward(X), Y))
            if X_val is not None:
                history["val"].append(self.loss_value(self.forward(X_val), Y_val))
            if print_every and epoch % print_every == 0:
                print(f"epoch {epoch:5d}  cost {history['train'][-1]:.6f}")
        return history

    # ------------------------------------------------------------- prediction
    def predict_proba(self, X):
        return self.forward(X, training=False)

    def predict(self, X):
        AL = self.predict_proba(X)
        out = self.activations[-1]
        if isinstance(out, Softmax):
            return np.argmax(AL, axis=0)
        if isinstance(out, Sigmoid):
            return (AL > 0.5).astype(int)
        return AL

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X).ravel() == np.asarray(y).ravel()))

    # ------------------------------------------------------ gradient checking
    def gradient_check(self, X, Y, epsilon=1e-7, with_dropout=False):
        """Relative difference between backprop and centered finite
        differences; below ~1e-7 means backprop is correct. With dropout,
        one set of masks is sampled and reused for every evaluation."""
        grads = self.backward(self.forward(X, training=with_dropout), Y)

        def J():
            return self.compute_cost(
                self.forward(X, training=with_dropout, reuse_masks=with_dropout), Y)

        analytic, numeric = [], []
        for l in range(1, self.L + 1):
            for p in ("W", "b"):
                theta = self.parameters[f"{p}{l}"]
                for idx in np.ndindex(theta.shape):
                    old = theta[idx]
                    theta[idx] = old + epsilon
                    j_plus = J()
                    theta[idx] = old - epsilon
                    j_minus = J()
                    theta[idx] = old
                    numeric.append((j_plus - j_minus) / (2 * epsilon))
                    analytic.append(grads[f"d{p}{l}"][idx])
        a, n = np.array(analytic), np.array(numeric)
        return float(np.linalg.norm(a - n) / (np.linalg.norm(a) + np.linalg.norm(n) + 1e-12))

    def __repr__(self):
        acts = ", ".join(a.name for a in self.activations)
        return (f"DeepNeuralNetwork(dims={self.layer_dims}, activations=[{acts}], "
                f"loss={self.loss}, {self.regularization}, {self.dropout})")
