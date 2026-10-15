"""
Training: ties everything together.

One epoch = one pass over the training set:
    forward -> loss -> backward -> gradient descent step
done in mini batches (batch_size=None means the whole set at once).

Stops when the epoch limit is reached or when the training loss drops below
target_loss, the practical "good enough" stopping rule.
"""
import numpy as np

from .loss import get_loss
from .optimizer import GradDescent
from .propagation import BackProp, ForwardProp


class Training:
    def __init__(self, model, loss="mse", optimizer=None, batch_size=None, seed=None):
        self.model = model
        self.loss_fn = get_loss(loss)
        self.optimizer = optimizer or GradDescent()
        self.batch_size = batch_size
        self.forward = ForwardProp()
        self.backward = BackProp()
        self.rng = np.random.default_rng(seed)
        self.history = {"loss": [], "accuracy": [], "val_accuracy": []}

    # ---- the core loop -------------------------------------------------
    def fit(self, X, Y, epochs=1000, target_loss=None, X_val=None, Y_val=None,
            print_every=100):
        m = X.shape[1]
        bs = self.batch_size or m
        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(m)
            for start in range(0, m, bs):
                idx = order[start:start + bs]
                Xb, Yb = X[:, idx], Y[:, idx]
                self.forward.run(self.model, Xb)
                self.backward.run(self.model, self.loss_fn, Yb)
                self.optimizer.step(self.model)

            loss = self.loss_fn.value(self.forward.run(self.model, X), Y)
            acc = self.accuracy(X, Y)
            self.history["loss"].append(loss)
            self.history["accuracy"].append(acc)
            if X_val is not None:
                self.history["val_accuracy"].append(self.accuracy(X_val, Y_val))

            if print_every and (epoch % print_every == 0 or epoch == 1):
                msg = f"epoch {epoch:5d}  loss {loss:.5f}  train acc {acc:6.1%}"
                if X_val is not None:
                    msg += f"  test acc {self.history['val_accuracy'][-1]:6.1%}"
                print(msg)

            if target_loss is not None and loss <= target_loss:
                print(f"Reached target loss {target_loss} at epoch {epoch} "
                      f"(loss {loss:.5f}, train acc {acc:.1%})")
                break
        return self.history

    # ---- using the trained network ------------------------------------
    def predict_proba(self, X):
        return self.forward.run(self.model, X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=0)

    def accuracy(self, X, Y):
        return float(np.mean(self.predict(X) == np.argmax(Y, axis=0)))
