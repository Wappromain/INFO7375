# Perceptron with Input Normalization and Mini Batch Training

A single perceptron written from scratch in NumPy, extended with two features:

1. **Input normalization** so features on very different scales contribute fairly to the weighted sum.
2. **Mini batch training** so the batch size can range from online (one sample per update) to full batch (one update per epoch).

The code follows the notation from the course notes.

| Symbol | Meaning |
|---|---|
| `W` | weight vector (w1 ... wN), one weight per input feature |
| `b` | bias, the adjustable threshold (b = θ negated) |
| `X(p)` | input vector for sample p (each row of `X` is one sample) |
| `d(p)` | desired (target) output for sample p |
| `y(p)` | actual output the perceptron produces for sample p |

## Files

| File | Contents |
|---|---|
| `perceptron_minibatch.py` | `Normalizer`, `make_mini_batches`, `Perceptron`, and a demo |
| `test_perceptron_minibatch.py` | pytest suite (41 tests) |

## Requirements

Python 3.8 or later, plus NumPy and pytest.

```
pip install numpy pytest
```

## Usage

Run the demo:

```
python perceptron_minibatch.py
```

Run the tests:

```
pytest -v
```

Use it in your own code:

```python
from perceptron_minibatch import Normalizer, Perceptron

norm = Normalizer("zscore").fit(X_train)   # fit on training data only
Xn_train = norm.transform(X_train)
Xn_test = norm.transform(X_test)           # reuse training statistics

p = Perceptron(n_features=Xn_train.shape[1], learning_rate=0.1, seed=1)
epochs_run = p.train(Xn_train, d_train, batch_size=8, epochs=200)

print(p.errors(Xn_test, d_test), "test errors")
```

## How it works

### Input normalization

Two methods are available.

| Method | Formula | Result |
|---|---|---|
| `"zscore"` (default) | x' = (x minus mean) / std | each feature has mean 0 and std 1 |
| `"minmax"` | x' = (x minus min) / (max minus min) | each feature lies in [0, 1] |

Key design points:

* **Fit on training data only.** `fit()` stores the statistics from the training set. `transform()` applies those same statistics to validation and test data. Refitting on test data would leak information about the test set into training.
* **Constant features are safe.** A feature with zero spread would cause a divide by zero. Its scale is set to 1 instead, so the feature becomes all zeros rather than NaN.
* **Reversible.** `inverse_transform()` maps normalized values back to the original scale.
* **Input checks.** Raises an error for non 2D input, empty input, NaN or infinite values, a feature count that differs from training, or calling `transform()` before `fit()`.

### Mini batch training

Each epoch, `make_mini_batches()` shuffles the samples and splits them into consecutive chunks of `batch_size`. The final batch is smaller when the sample count does not divide evenly, and every sample appears exactly once per epoch.

For each batch B, `Perceptron.train()` applies:

```
y(p) = step(W · X(p) + b)          for every p in B
e(p) = d(p) − y(p)                 0 if correct, +1 or −1 if wrong
W   += lr × mean over B of e(p) X(p)
b   += lr × mean over B of e(p)
```

Averaging over the batch keeps the step size comparable regardless of batch size.

| `batch_size` | Behavior |
|---|---|
| `1` | online learning, exactly the classic perceptron rule |
| between 1 and the dataset size | mini batch |
| the dataset size or larger | full batch, one update per epoch |

Training stops early once an epoch ends with zero training errors, and `train()` returns the number of epochs actually run. Per epoch error counts are stored in `p.history`.

Weights start at zero by default. This is fine for a single perceptron; the symmetry problem only arises when several neurons in a layer start identical. Pass `init="random"` for small random weights instead.

## Demo results

The demo uses 200 samples with two features on very different scales: one in the tens of thousands and one between 0 and 1. It trains on 150 samples and tests on 50.

| Setup | Epochs to converge | Test accuracy |
|---|---|---|
| No normalization, batch 1 | did not converge in 100 | 54% |
| Z score, batch 1 | 10 | 98% |
| Z score, batch 8 | 4 | 100% |
| Z score, batch 32 | 12 | 100% |
| Z score, full batch | 3 | 98% |

Without normalization, the large feature dominates the weighted sum and training never settles. With normalization, every batch size converges. Epoch count does not fall steadily as batch size grows: larger batches make fewer updates per epoch, but each update is smoother.

## Tests

| Area | What is checked |
|---|---|
| Normalization | mean 0 and std 1 for z score; [0, 1] range for min max; a hand calculated z score; correlated features end up identical; constant features become zero; inverse round trip; test data uses training statistics; input is not modified; invalid input raises errors |
| Mini batches | correct batch sizes, including a partial last batch and a batch larger than the dataset; every sample used once; inputs stay paired with targets; shuffling is reproducible with a seed; order is kept when shuffling is off; invalid batch sizes and mismatched lengths raise errors |
| Training | batch size 1 matches the classic rule computed by hand; a batch update equals the average of individual updates; AND is learned at every batch size; XOR is not learned (not linearly separable); normalization lowers errors on badly scaled data |

## Limitations

* A single perceptron can only learn linearly separable patterns, so XOR remains out of reach. Multi layer networks with backpropagation are needed for that.
* The step activation has no useful derivative, so this update rule does not extend directly to hidden layers.
