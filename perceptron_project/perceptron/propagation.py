"""
ForwardProp and BackProp.

Forward path, for each layer l = 1..L:
    Z[l] = W[l] @ A[l-1] + b[l]
    A[l] = f(Z[l])                         A[0] = X, the input pixels

Backward path (chain rule), starting from the loss:
    dA[L] = dLoss/dA[L]                    from the LossFunction
    for l = L..1:
        dZ[l]   = dA[l] * f'(Z[l])         elementwise
        dW[l]   = dZ[l] @ A[l-1].T         (the 1/m is already inside dA[L])
        db[l]   = sum of dZ[l] over samples
        dA[l-1] = W[l].T @ dZ[l]           pass the error back to the previous layer

Because each layer asks its own Activation for f'(Z), any activation can be
swapped in without touching this file.
"""
import numpy as np


class ForwardProp:
    def run(self, model, X):
        A = X
        for layer in model.layers:
            layer.A_prev = A
            layer.Z = layer.params.W @ A + layer.params.b
            layer.A = layer.activation.forward(layer.Z)
            A = layer.A
        return A


class BackProp:
    def run(self, model, loss_fn, Y):
        A_out = model.layers[-1].A
        dA = loss_fn.gradient(A_out, Y)
        for layer in reversed(model.layers):
            dZ = dA * layer.activation.derivative(layer.Z)
            layer.params.dW = dZ @ layer.A_prev.T
            layer.params.db = np.sum(dZ, axis=1, keepdims=True)
            dA = layer.params.W.T @ dZ
        return dA   # gradient w.r.t. the input, rarely needed but handy for checks
