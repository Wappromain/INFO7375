"""
dropout.py
Inverted dropout for hidden layer activations.

Training:
    D = (rand(shape of A) < keep_prob)     random 0/1 mask
    A = A * D / keep_prob                  dividing keeps E[A] unchanged
Backward (same mask as the forward pass):
    dA = dA * D / keep_prob
Inference:
    nothing is dropped and nothing is rescaled, because the rescaling
    already happened during training.
"""

import numpy as np


class Dropout:
    def __init__(self, rate=0.0, seed=None):
        """rate is the fraction of units to DROP; keep_prob = 1 - rate."""
        if not 0.0 <= rate < 1.0:
            raise ValueError("Dropout rate must be in [0, 1)")
        self.rate = rate
        self.keep_prob = 1.0 - rate
        self.rng = np.random.default_rng(seed)

    @property
    def active(self):
        return self.rate > 0

    def make_mask(self, shape):
        return (self.rng.random(shape) < self.keep_prob).astype(float)

    def forward(self, A, mask=None):
        """Returns (dropped A, mask). Pass a mask to reuse it instead of
        sampling a new one (used for gradient checking)."""
        if not self.active:
            return A, None
        if mask is None:
            mask = self.make_mask(A.shape)
        return A * mask / self.keep_prob, mask

    def backward(self, dA, mask):
        if mask is None:
            return dA
        return dA * mask / self.keep_prob

    def __repr__(self):
        return f"Dropout(rate={self.rate})"
