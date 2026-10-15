"""Network classes.

NeuralNetwork            general container: a stack of DenseLayers, a loss,
                         an optimizer, and the training loop.
OneHiddenLayerNetwork    the assignment: input -> hidden -> output, built
                         from three sizes instead of a list of layers.
"""
from dataclasses import dataclass, field

import numpy as np

from .activations import Sigmoid, Softmax
from .layers import DenseLayer
from .losses import Loss, get_loss
from .optimizers import SGD, Optimizer


@dataclass
class History:
    loss: list = field(default_factory=list)
    accuracy: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_accuracy: list = field(default_factory=list)
    stopped_epoch: int | None = None


class NeuralNetwork:
    def __init__(self, layers: list[DenseLayer], loss="mse", optimizer: Optimizer | None = None):
        if not layers:
            raise ValueError("A network needs at least one layer.")
        for a, b in zip(layers, layers[1:]):
            if a.n_out != b.n_in:
                raise ValueError(f"Layer size mismatch: {a} feeds {b}.")
        self.layers = layers
        self.loss: Loss = get_loss(loss)
        self.optimizer = optimizer if optimizer is not None else SGD(0.1)
        self.history = History()

    # ---- forward / backward ---------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, A: np.ndarray, Y: np.ndarray) -> None:
        grad = self.loss.gradient(A, Y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_step(self, X: np.ndarray, Y: np.ndarray) -> float:
        """One forward pass, one backward pass, one parameter update."""
        A = self.forward(X)
        loss = self.loss.value(A, Y)
        self.backward(A, Y)
        self.optimizer.step(self.layers)
        return loss

    # ---- training loop ---------------------------------------------------
    def fit(self, X, Y, epochs: int = 1000, batch_size: int | None = None,
            validation_data=None, target_loss: float | None = None,
            shuffle: bool = True, verbose: int = 0, seed: int | None = None) -> History:
        """batch_size=None is full batch; batch_size=1 is the online (per sample)
        rule from the classic perceptron. target_loss stops training early."""
        X, Y = self._prepare(X, Y)
        n = X.shape[0]
        bs = n if batch_size is None else max(1, min(batch_size, n))
        rng = np.random.default_rng(seed)

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n) if shuffle else np.arange(n)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                self.train_step(X[idx], Y[idx])

            loss, acc = self.evaluate(X, Y)
            self.history.loss.append(loss)
            self.history.accuracy.append(acc)
            if validation_data is not None:
                vl, va = self.evaluate(*validation_data)
                self.history.val_loss.append(vl)
                self.history.val_accuracy.append(va)

            if verbose and (epoch == 1 or epoch % verbose == 0 or epoch == epochs):
                msg = f"epoch {epoch:5d}  loss {loss:.5f}  acc {acc:.3f}"
                if validation_data is not None:
                    msg += f"  val_loss {vl:.5f}  val_acc {va:.3f}"
                print(msg)

            if target_loss is not None and loss <= target_loss:
                self.history.stopped_epoch = epoch
                if verbose:
                    print(f"Reached target loss {target_loss} at epoch {epoch}.")
                break
        return self.history

    # ---- inference -------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        return self.forward(np.atleast_2d(np.asarray(X, dtype=float)))

    def predict(self, X) -> np.ndarray:
        """Class labels for classifiers, raw outputs otherwise."""
        A = self.predict_proba(X)
        out_act = self.layers[-1].activation
        if isinstance(out_act, Softmax):
            return A.argmax(axis=1)
        if isinstance(out_act, Sigmoid):
            return (A >= 0.5).astype(int).ravel() if A.shape[1] == 1 else (A >= 0.5).astype(int)
        return A

    def evaluate(self, X, Y) -> tuple[float, float]:
        X, Y = self._prepare(X, Y)
        A = self.forward(X)
        loss = self.loss.value(A, Y)
        if A.shape[1] == 1:
            acc = float(np.mean((A >= 0.5) == (Y >= 0.5)))
        else:
            acc = float(np.mean(A.argmax(axis=1) == Y.argmax(axis=1)))
        return loss, acc

    # ---- helpers ---------------------------------------------------------
    def _prepare(self, X, Y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but Y has {Y.shape[0]}.")
        if Y.shape[1] != self.layers[-1].n_out:
            raise ValueError(f"Y has {Y.shape[1]} columns but the output layer has "
                             f"{self.layers[-1].n_out} neurons. One hot encode class labels.")
        return X, Y

    @property
    def n_params(self) -> int:
        return sum(layer.n_params for layer in self.layers)

    def summary(self) -> str:
        lines = [f"{type(self).__name__}  (loss={self.loss.name}, "
                 f"optimizer={type(self.optimizer).__name__}, lr={self.optimizer.learning_rate})"]
        for i, layer in enumerate(self.layers):
            lines.append(f"  [{i}] {layer!r:40s} params={layer.n_params}")
        lines.append(f"  total params = {self.n_params}")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        arrays = {}
        for i, layer in enumerate(self.layers):
            arrays[f"W{i}"] = layer.W
            arrays[f"b{i}"] = layer.b
        np.savez(path, **arrays)

    def load(self, path: str) -> None:
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            if data[f"W{i}"].shape != layer.W.shape:
                raise ValueError(f"Saved weights for layer {i} do not match this architecture.")
            layer.W[...] = data[f"W{i}"]
            layer.b[...] = data[f"b{i}"]


class OneHiddenLayerNetwork(NeuralNetwork):
    """Input -> one hidden DenseLayer -> output DenseLayer."""

    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int,
                 hidden_activation="tanh", output_activation="sigmoid",
                 loss=None, optimizer: Optimizer | None = None,
                 init: str = "auto", seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.hidden = DenseLayer(n_inputs, n_hidden, hidden_activation, init, rng)
        self.output = DenseLayer(n_hidden, n_outputs, output_activation, init, rng)
        if loss is None:
            loss = {"sigmoid": "bce", "softmax": "cce"}.get(self.output.activation.name, "mse")
        super().__init__([self.hidden, self.output], loss, optimizer)
