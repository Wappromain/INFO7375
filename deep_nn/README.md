# Deep Neural Network from Scratch (NumPy)

A fully connected, multilayer neural network written in plain NumPy, with no deep learning framework. It implements forward propagation, three loss functions, backpropagation, five activation functions, L1/L2 regularization and dropout, and verifies every gradient numerically.

## Requirements

Python 3.8 or newer and NumPy.

```bash
pip install numpy
python demo.py
```

The demo takes about a minute.

## Files

| File | Contents |
|---|---|
| `activations.py` | `Linear`, `ReLU`, `Sigmoid`, `Tanh` and `Softmax` classes, each with `forward`, `derivative` and `backward` |
| `regularization.py` | `Regularization(l1, l2)`: the cost penalty and the gradient term added to each weight gradient |
| `dropout.py` | `Dropout(rate)`: inverted dropout (mask, rescale, and the matching backward step) |
| `neural_network.py` | `DeepNeuralNetwork`: class structure, forward propagation, loss functions, backpropagation, training and gradient checking |
| `demo.py` | Correctness checks and training experiments |

## Quick start

```python
import numpy as np
from neural_network import DeepNeuralNetwork

# Data is column oriented: X is (n_features, m), Y is (n_outputs, m)
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]], dtype=float)
Y = np.array([[0, 1, 1, 0]], dtype=float)          # XOR

net = DeepNeuralNetwork(
    layer_dims=[2, 4, 1],                 # input size, then units per layer
    activations=["tanh", "sigmoid"],      # one per layer
    loss="binary_cross_entropy",
    l1=0.0, l2=0.0, dropout=0.0,          # regularization (0 disables)
    seed=3,
)
history = net.fit(X, Y, epochs=3000, learning_rate=0.5)
print(net.predict(X))                      # [[0 1 1 0]]
```

### Constructor options

| Argument | Meaning |
|---|---|
| `layer_dims` | `[n_x, n_1, ..., n_L]`: input size followed by the number of units in each layer |
| `activations` | One name per layer: `"linear"`, `"relu"`, `"sigmoid"`, `"tanh"`, `"softmax"` |
| `loss` | `"mse"`, `"binary_cross_entropy"` or `"categorical_cross_entropy"` (Y one hot) |
| `l1`, `l2` | Regularization strengths (lambda) |
| `dropout` | Fraction of hidden units dropped during training |
| `seed` | Random seed for reproducible results |

### Main methods

| Method | Purpose |
|---|---|
| `forward(X, training=False)` | Forward propagation; returns the output layer activations |
| `compute_cost(AL, Y)` | Loss plus regularization penalty |
| `backward(AL, Y)` | Backpropagation; returns `dW1..dWL`, `db1..dbL` |
| `update_parameters(grads, learning_rate)` | One gradient descent step |
| `fit(X, Y, epochs, learning_rate, batch_size, X_val, Y_val)` | Full or mini batch training; returns cost history |
| `predict(X)`, `predict_proba(X)`, `accuracy(X, y)` | Inference |
| `gradient_check(X, Y, with_dropout=False)` | Compares backprop with numerical gradients |

## How it works

### Forward propagation

For each layer `l = 1..L`, with `A[0] = X`:

```
Z[l] = W[l] A[l-1] + b[l]
A[l] = g(Z[l])
```

During training, dropout is applied to the hidden layer outputs (never the output layer).

### Loss functions

| Loss | Formula | Pair with |
|---|---|---|
| Mean squared error | `J = (1/2m) * sum (A - Y)^2` | Linear output (regression) |
| Binary cross entropy | `J = -(1/m) * sum [y log a + (1 - y) log(1 - a)]` | Sigmoid output |
| Categorical cross entropy | `J = -(1/m) * sum y log a` | Softmax output |

The regularization penalty is added on top of the loss.

### Backpropagation

Working from the output layer back to the first:

```
dZ[l]   = dA[l] * g'(Z[l])
dW[l]   = dZ[l] A[l-1]^T  +  regularization gradient
db[l]   = sum of dZ[l] over the examples
dA[l-1] = W[l]^T dZ[l]     (then the same dropout mask used going forward)
```

For the three recommended output and loss pairs above, the output layer simplifies to `dZ[L] = (A[L] - Y) / m`, which the code uses because it is both simpler and more numerically stable.

### Activation functions

| Activation | g(z) | g'(z) | Typical use |
|---|---|---|---|
| Linear | z | 1 | Regression output |
| ReLU | max(0, z) | 1 if z > 0, else 0 | Hidden layers |
| Sigmoid | 1 / (1 + e^-z) | s(1 - s) | Binary output |
| Tanh | tanh(z) | 1 - tanh^2(z) | Hidden layers |
| Softmax | e^z_i / sum e^z_j | Jacobian `diag(s) - s s^T` | Multiclass output |

Softmax is not elementwise: each output depends on every input for that example, so its derivative is a matrix. Its `backward` applies that matrix directly without building it. Sigmoid and Softmax are written to avoid overflow on large inputs.

### Regularization

| Type | Added to cost | Added to dW | Effect |
|---|---|---|---|
| L2 | `(lambda/2m) * sum W^2` | `(lambda/m) * W` | Shrinks all weights ("weight decay") |
| L1 | `(lambda/m) * sum abs(W)` | `(lambda/m) * sign(W)` | Pushes many weights to zero (sparsity) |

Biases are not regularized.

### Dropout (inverted)

During training each hidden unit is kept with probability `keep_prob = 1 - rate`. Kept outputs are divided by `keep_prob` so the expected activation is unchanged, which means no adjustment is needed at prediction time. Backpropagation reuses the same mask.

### Weight initialization

Weights start small and random (He scaling for ReLU layers, Xavier scaling otherwise), and biases start at zero. Random starting weights matter in multilayer networks: if every unit in a layer started identical, they would receive identical gradients and never learn different features.

## What the demo checks

1. **Activation derivatives:** each `backward` matches a numerical derivative (error about 1e-10).
2. **Gradient check:** backpropagation matches finite differences (about 1e-9) for all three losses, and with L1, L2 and dropout enabled.
3. **XOR:** the classic problem a single perceptron cannot solve, learned by a 2-4-1 network.
4. **Three class spiral:** about 99% training accuracy with ReLU hidden layers and a Softmax output.
5. **sin(x) regression:** Linear output with MSE.
6. **Overfitting experiment:** a large network on 60 noisy points, comparing no regularization against L2, L1, dropout and L2 plus dropout on 2,000 held out points.

## Limitations

This is a learning implementation. It supports fully connected layers and plain (mini batch) gradient descent only, runs on the CPU, and favors readable code over speed.
