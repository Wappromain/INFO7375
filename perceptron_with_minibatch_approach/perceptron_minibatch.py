"""
Perceptron with input normalization and mini batch training.

Notation follows the course notes:
    W      weight vector (w1 ... wN), one weight per input feature
    b      bias (adjustable threshold, b = -theta)
    X(p)   input vector for sample p  (rows of X are samples)
    d(p)   desired (target) output for sample p
    y(p)   actual output the perceptron produces for sample p
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Input normalization
# ---------------------------------------------------------------------------

class Normalizer:
    """
    Rescales each input feature so no single feature dominates the weighted sum.

    method="zscore":  x' = (x - mean) / std        -> each feature has mean 0, std 1
    method="minmax":  x' = (x - min) / (max - min) -> each feature lies in [0, 1]

    Statistics are computed ONLY from the training data in fit(), then reused
    unchanged in transform() for validation/test data. Fitting on test data
    would leak information about the test set into training.
    """

    METHODS = ("zscore", "minmax")

    def __init__(self, method="zscore", eps=1e-12):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got {method!r}")
        self.method = method
        self.eps = eps
        self.fitted = False

    def fit(self, X):
        X = self._check(X)
        if self.method == "zscore":
            self.shift_ = X.mean(axis=0)
            scale = X.std(axis=0)
        else:  # minmax
            self.shift_ = X.min(axis=0)
            scale = X.max(axis=0) - X.min(axis=0)
        # A constant feature has zero spread. Dividing by zero would give NaN,
        # so use a scale of 1: the feature becomes all zeros (it carries no
        # information that could separate classes anyway).
        self.scale_ = np.where(scale < self.eps, 1.0, scale)
        self.n_features_ = X.shape[1]
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Normalizer must be fit on training data before transform")
        X = self._check(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"expected {self.n_features_} features, got {X.shape[1]}")
        return (X - self.shift_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Xn):
        if not self.fitted:
            raise RuntimeError("Normalizer is not fit")
        return np.asarray(Xn, dtype=float) * self.scale_ + self.shift_

    @staticmethod
    def _check(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D: (number of samples, number of features)")
        if X.shape[0] == 0:
            raise ValueError("X has no samples")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")
        return X


# ---------------------------------------------------------------------------
# 2. Mini batch splitting
# ---------------------------------------------------------------------------

def make_mini_batches(X, d, batch_size, rng=None, shuffle=True):
    """
    Split (X, d) into mini batches.

    Each epoch the samples are shuffled (so batches differ every epoch) and then
    cut into consecutive chunks of batch_size. The last batch is smaller if the
    sample count does not divide evenly. Every sample appears exactly once.

        batch_size = 1          -> online / stochastic (classic perceptron rule)
        batch_size = len(X)     -> full batch (one update per epoch)
        anything in between     -> mini batch
    """
    X = np.asarray(X, dtype=float)
    d = np.asarray(d, dtype=float)
    if len(X) != len(d):
        raise ValueError("X and d must have the same number of samples")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    P = len(X)
    order = np.arange(P)
    if shuffle:
        rng = rng if rng is not None else np.random.default_rng()
        rng.shuffle(order)

    batches = []
    for start in range(0, P, batch_size):
        idx = order[start:start + batch_size]
        batches.append((X[idx], d[idx]))
    return batches


# ---------------------------------------------------------------------------
# 3. Perceptron trained with mini batches
# ---------------------------------------------------------------------------

class Perceptron:
    def __init__(self, n_features, learning_rate=0.1, init="zeros", seed=None):
        self.lr = learning_rate
        self.rng = np.random.default_rng(seed)
        # Zero init is legitimate for a single perceptron (see notes: the
        # symmetry problem only matters when several neurons share a layer).
        if init == "zeros":
            self.W = np.zeros(n_features)
        elif init == "random":
            self.W = self.rng.normal(0.0, 0.01, n_features)
        else:
            raise ValueError("init must be 'zeros' or 'random'")
        self.b = 0.0
        self.history = []

    def net(self, X):
        return np.asarray(X, dtype=float) @ self.W + self.b

    def predict(self, X):
        """Step activation: y(p) = 1 if W.X(p) + b > 0 else 0."""
        return (self.net(X) > 0).astype(float)

    def errors(self, X, d):
        """Number of misclassified samples."""
        return int(np.sum(self.predict(X) != np.asarray(d)))

    def train(self, X, d, batch_size=1, epochs=100, shuffle=True, verbose=False):
        """
        Mini batch perceptron rule.

        For each batch B:
            y(p)     = step(W.X(p) + b)            for every p in B
            e(p)     = d(p) - y(p)                 (0 if correct, +1 or -1 if wrong)
            W       += lr * mean over B of e(p) X(p)
            b       += lr * mean over B of e(p)

        Averaging over the batch keeps the step size comparable no matter how
        large the batch is. With batch_size=1 this is exactly the classic
        online perceptron rule.

        Stops early when an epoch ends with zero training errors.
        Returns the number of epochs actually run.
        """
        X = np.asarray(X, dtype=float)
        d = np.asarray(d, dtype=float)
        self.history = []
        for epoch in range(1, epochs + 1):
            for Xb, db in make_mini_batches(X, d, batch_size, self.rng, shuffle):
                e = db - self.predict(Xb)
                self.W += self.lr * (e @ Xb) / len(Xb)
                self.b += self.lr * e.mean()
            n_err = self.errors(X, d)
            self.history.append(n_err)
            if verbose:
                print(f"epoch {epoch:4d}  errors {n_err}")
            if n_err == 0:
                return epoch
        return epochs


# ---------------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------------

def make_badly_scaled_data(n=200, seed=0):
    """Two linearly separable classes; feature 1 is in the tens of thousands,
    feature 2 is between 0 and 1. A useful stress test for normalization."""
    rng = np.random.default_rng(seed)
    income = rng.uniform(20_000, 80_000, n)
    ratio = rng.uniform(0, 1, n)
    d = ((income - 50_000) / 30_000 + (ratio - 0.5) * 2 > 0).astype(float)
    return np.column_stack([income, ratio]), d


if __name__ == "__main__":
    X, d = make_badly_scaled_data()
    X_train, X_test, d_train, d_test = X[:150], X[150:], d[:150], d[150:]

    print("Without normalization")
    raw = Perceptron(2, seed=1)
    ep = raw.train(X_train, d_train, batch_size=1, epochs=100)
    print(f"  epochs {ep}, train errors {raw.errors(X_train, d_train)}, "
          f"test accuracy {1 - raw.errors(X_test, d_test) / len(d_test):.0%}")

    norm = Normalizer("zscore").fit(X_train)      # fit on training data only
    Xn_train, Xn_test = norm.transform(X_train), norm.transform(X_test)

    print("\nWith z-score normalization, comparing batch sizes")
    for bs in (1, 8, 32, len(Xn_train)):
        p = Perceptron(2, seed=1)
        ep = p.train(Xn_train, d_train, batch_size=bs, epochs=200)
        acc = 1 - p.errors(Xn_test, d_test) / len(d_test)
        label = "full" if bs == len(Xn_train) else bs
        print(f"  batch {label!s:>4}: epochs {ep:3d}, "
              f"train errors {p.errors(Xn_train, d_train)}, test accuracy {acc:.0%}")
