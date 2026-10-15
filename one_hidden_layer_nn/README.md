# One Hidden Layer Neural Network (Object Oriented, NumPy only)

A from scratch neural network with one hidden layer, written in Python with NumPy as the only dependency (matplotlib only for the demo plots).

## Quick start

```bash
python -m unittest discover -s tests -v     # 15 tests, includes gradient checks
python demos/run_demos.py                   # XOR, two moons, 3 class demo + plot
```

```python
from nn import OneHiddenLayerNetwork, SGD

net = OneHiddenLayerNetwork(n_inputs=2, n_hidden=4, n_outputs=1,
                            hidden_activation="tanh", output_activation="sigmoid",
                            optimizer=SGD(0.5), seed=3)
net.fit(X, y, epochs=3000)
net.predict(X)
```

## Class hierarchy

```
Activation (abstract)            Loss (abstract)              Optimizer (abstract)
 ├── Identity                     ├── MeanSquaredError         ├── SGD
 ├── Step                         ├── BinaryCrossEntropy       └── Momentum
 ├── Sigmoid                      └── CategoricalCrossEntropy
 ├── Tanh
 ├── ReLU
 └── Softmax  (overrides backward)

DenseLayer  ── has an ──> Activation

NeuralNetwork  ── has many ──> DenseLayer
               ── has one  ──> Loss
               ── has one  ──> Optimizer
 └── OneHiddenLayerNetwork   (builds input → hidden → output from three sizes)
```

| Class | Responsibility | Key methods |
|---|---|---|
| `Activation` | Nonlinearity applied to Z | `forward(Z)`, `derivative(Z)`, `backward(dA, Z)` |
| `Loss` | Measures error, gives dL/dA | `value(A, Y)`, `gradient(A, Y)` |
| `DenseLayer` | Holds W and b, computes Z = XW + b and A = f(Z), computes its own gradients | `forward(X)`, `backward(dA)`, `params()`, `grads()` |
| `Optimizer` | Turns gradients into parameter updates | `step(layers)` |
| `NeuralNetwork` | Chains layers, runs the training loop, prediction, evaluation, save/load | `fit`, `predict`, `predict_proba`, `evaluate`, `summary`, `save`, `load` |
| `OneHiddenLayerNetwork` | The assignment's architecture, picks a sensible default loss | constructor only |

## Design decisions (for discussion with the TA)

1. **Composition over inheritance for layers.** A layer *has* an activation rather than there being a `SigmoidLayer`, `ReLULayer`, and so on. This avoids a class for every combination and lets any activation be swapped in by name.
2. **Each layer owns its backward pass.** The network only passes the gradient down the stack: `grad = layer.backward(grad)`. Adding a layer type later does not touch the network class.
3. **Optimizer is separate from the layer.** The update rule (plain SGD, momentum) can change without editing any gradient math. The optimizer reads `params()` and `grads()` and updates in place.
4. **`OneHiddenLayerNetwork` subclasses a general `NeuralNetwork`.** The assignment asks for one hidden layer, but the base class works for any depth, so extending to two hidden layers is a constructor change, not a rewrite.
5. **Softmax overrides `backward`.** Every other activation is elementwise, so `dZ = dA * f'(Z)`. Softmax outputs depend on each other, so it uses the full Jacobian vector product instead. Keeping this inside the Softmax class means the layer never needs a special case.
6. **Strategy objects accept names or instances.** `"relu"` and `ReLU()` both work, via small registries (`get_activation`, `get_loss`).
7. **Batch size controls the training style.** `batch_size=1` is the classic online perceptron rule (update after each sample), `None` is full batch (update once per epoch), anything in between is mini batch.

## Shapes and math

With n samples, the hidden layer computes `Z1 = X W1 + b1` with `W1` shaped (n_inputs, n_hidden), then `A1 = f(Z1)`. The output layer computes `Z2 = A1 W2 + b2` and `A2 = g(Z2)`.

Backpropagation, per layer, given dL/dA from above:

```
dZ = activation.backward(dA, Z)
dW = Xᵀ dZ
db = sum of dZ over the batch
dX = dZ Wᵀ   (passed to the layer below)
```

## Verification

`tests/test_network.py` compares the analytic gradients against finite difference estimates for four activation/loss combinations. Relative error is below 1e-6 in all cases, so the backprop math is correct. Other tests cover shapes, numerical stability, input validation, save/load, and learning (XOR, two moons, early stopping).

One test demonstrates the **symmetry problem** from the course notes: with all zero hidden weights, every hidden unit receives the same gradient and stays identical after training.

## Demo results

| Demo | Architecture | Result |
|---|---|---|
| XOR | 2 → 4 tanh → 1 sigmoid | 100% (a single perceptron cannot do this) |
| Two moons | 2 → 16 tanh → 1 sigmoid, momentum, mini batch 32 | 96% test accuracy |
| Three classes | 2 → 10 ReLU → 3 softmax | 97% test accuracy |

In the three class run, test loss rises while training loss keeps falling. This is mild overfitting: accuracy holds, but the network becomes overconfident on a few overlapping points. Early stopping on validation loss or weight decay would address it.

## Possible next steps

- Weight decay (L2) and early stopping on validation loss
- Adam optimizer (one new `Optimizer` subclass)
- A `Callback` hook in `fit` for logging and early stopping
- Run on a real dataset (Iris or MNIST digits)

## Questions to bring to the TA

1. Should the loss be owned by the network (current design) or passed to `fit`?
2. Is a separate `Optimizer` hierarchy in scope, or should updates live in `DenseLayer.update(lr)` for simplicity?
3. Does the course expect the MSE convention with ½ (current) or without?
4. Should `OneHiddenLayerNetwork` subclass a general `NeuralNetwork`, or should the assignment be a single standalone class?
