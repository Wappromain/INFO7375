"""
Softmax activation, standalone.

    softmax(z)_k = e^{z_k} / sum_j e^{z_j}

Used on the output layer for multiclass classification: it turns a column of
raw scores (logits) into a probability distribution over the classes.

Shape convention: Z is (n_classes, m), one column per example, so the sum is
taken down each column (axis=0). Pass axis=1 if your data is row per example.

Softmax is NOT elementwise. Every output depends on every input in its column,
so its derivative is a full Jacobian per example:

    J_ij = s_i (delta_ij - s_j)      i.e.  J = diag(s) - s s^T

That is why derivative() refuses to return a single elementwise array, and
backward() applies the Jacobian directly instead.
"""
import numpy as np


class Softmax:
    name = "softmax"

    def __init__(self, axis=0):
        self.axis = axis

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, Z):
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:                       # single example as a flat vector
            Z = Z.reshape(-1, 1) if self.axis == 0 else Z.reshape(1, -1)
            return self.forward(Z).ravel()
        # Subtracting the max avoids overflow in exp; the shift cancels out
        # because softmax(z + c) = softmax(z) for any constant c.
        shifted = Z - np.max(Z, axis=self.axis, keepdims=True)
        e = np.exp(shifted)
        return e / np.sum(e, axis=self.axis, keepdims=True)

    def __call__(self, Z):
        return self.forward(Z)

    # ------------------------------------------------------------------
    # Derivatives
    # ------------------------------------------------------------------
    def derivative(self, Z):
        raise NotImplementedError(
            "Softmax has a full Jacobian, not an elementwise derivative. "
            "Use backward(dA, Z), jacobian(z), or the softmax plus "
            "cross entropy shortcut dZ = A - Y.")

    def jacobian(self, z):
        """Jacobian for ONE example z (length n): J = diag(s) - s s^T."""
        s = self.forward(np.asarray(z, dtype=float).ravel())
        return np.diag(s) - np.outer(s, s)

    def backward(self, dA, Z):
        """Given dL/dA, return dL/dZ for every example at once.

        Per example, dZ = J^T dA = s * (dA - sum_k dA_k s_k). J is symmetric,
        so this is the Jacobian vector product without building J."""
        S = self.forward(Z)
        dA = np.asarray(dA, dtype=float)
        return S * (dA - np.sum(dA * S, axis=self.axis, keepdims=True))

    def __repr__(self):
        return f"Softmax(axis={self.axis})"
