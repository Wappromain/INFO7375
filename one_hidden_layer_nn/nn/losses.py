"""Loss functions.

value(A, Y)    -> scalar loss averaged over the batch
gradient(A, Y) -> dL/dA, same shape as A (already divided by batch size)

Errors are squared or logged, never summed raw, so positive and negative
errors cannot cancel out.
"""
from abc import ABC, abstractmethod

import numpy as np

_EPS = 1e-12


class Loss(ABC):
    name = "loss"

    @abstractmethod
    def value(self, A: np.ndarray, Y: np.ndarray) -> float:
        ...

    @abstractmethod
    def gradient(self, A: np.ndarray, Y: np.ndarray) -> np.ndarray:
        ...

    def __repr__(self):
        return f"{type(self).__name__}()"


class MeanSquaredError(Loss):
    name = "mse"

    def value(self, A, Y):
        return float(np.mean(np.sum((A - Y) ** 2, axis=1)) / 2.0)

    def gradient(self, A, Y):
        return (A - Y) / A.shape[0]


class BinaryCrossEntropy(Loss):
    """Pairs with a sigmoid output layer."""
    name = "bce"

    def value(self, A, Y):
        A = np.clip(A, _EPS, 1 - _EPS)
        return float(-np.mean(np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A), axis=1)))

    def gradient(self, A, Y):
        A = np.clip(A, _EPS, 1 - _EPS)
        return (A - Y) / (A * (1 - A)) / A.shape[0]


class CategoricalCrossEntropy(Loss):
    """Pairs with a softmax output layer and one hot targets."""
    name = "cce"

    def value(self, A, Y):
        A = np.clip(A, _EPS, 1.0)
        return float(-np.mean(np.sum(Y * np.log(A), axis=1)))

    def gradient(self, A, Y):
        A = np.clip(A, _EPS, 1.0)
        return -(Y / A) / A.shape[0]


_REGISTRY = {cls.name: cls for cls in (MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy)}


def get_loss(spec) -> Loss:
    if isinstance(spec, Loss):
        return spec
    try:
        return _REGISTRY[spec.lower()]()
    except KeyError:
        raise ValueError(f"Unknown loss '{spec}'. Options: {sorted(_REGISTRY)}")
