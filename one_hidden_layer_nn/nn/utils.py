"""Data helpers so the project has no dependency beyond NumPy."""
import numpy as np


def one_hot(labels, n_classes: int | None = None) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).ravel()
    k = n_classes if n_classes is not None else labels.max() + 1
    out = np.zeros((labels.size, k))
    out[np.arange(labels.size), labels] = 1.0
    return out


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true).ravel() == np.asarray(y_pred).ravel()))


def train_test_split(X, Y, test_size: float = 0.2, seed: int | None = None):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(round(len(X) * (1 - test_size)))
    return X[idx[:cut]], X[idx[cut:]], Y[idx[:cut]], Y[idx[cut:]]


def make_moons(n: int = 400, noise: float = 0.15, seed: int | None = None):
    """Two interleaving half circles: a classic non linearly separable set."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = np.linspace(0, np.pi, n1)
    t2 = np.linspace(0, np.pi, n - n1)
    X = np.vstack([np.c_[np.cos(t1), np.sin(t1)],
                   np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]])
    X += rng.normal(0, noise, X.shape)
    y = np.r_[np.zeros(n1), np.ones(n - n1)].astype(int)
    return X, y


class Standardizer:
    """Scale features to zero mean, unit variance using training statistics only."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
