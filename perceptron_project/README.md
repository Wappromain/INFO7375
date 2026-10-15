# Perceptron for Handwritten Digits (0 to 9)

An object oriented neural network written from scratch in NumPy, trained on
20 x 20 grayscale handwritten digit images.

## Quick start

```bash
pip install numpy pillow matplotlib
python make_digits.py        # creates data/ (100 labeled train + 30 unlabeled test images)
python gradient_check.py     # proves backprop matches numerical gradients
python train.py              # trains, tests, writes results/
```

Default result: 100% training accuracy, 96.7% (29 of 30) on unseen test images.

## Class design

| Class | File | Responsibility |
|---|---|---|
| `Activation` (+ `Sigmoid`, `Tanh`, `ReLU`, `LeakyReLU`, `Step`) | `activation.py` | `forward(Z)` and `derivative(Z)` |
| `Parameters` | `parameters.py` | Transmission matrix `W (n_out, n_in)`, bias `b (n_out, 1)`, their gradients, and the initialization module (`xavier`, `he`, `random`, `zeros`) |
| `Neuron` | `layer.py` | View of one row of a layer: its weights, bias, and output for one input |
| `Layer` | `layer.py` | One `Parameters` plus one `Activation`; caches `A_prev`, `Z`, `A` for backprop |
| `Model` / `NeuralNetwork` | `model.py` | Ordered stack of layers; `NeuralNetwork(400, [32], 10)` = one hidden layer |
| `LossFunction` (+ `MeanSquaredError`, `BinaryCrossEntropy`) | `loss.py` | `value(A, Y)` and `gradient(A, Y)` |
| `ForwardProp` | `propagation.py` | `Z = W @ A_prev + b`, `A = f(Z)` layer by layer |
| `BackProp` | `propagation.py` | Chain rule from the loss back to every `dW`, `db` |
| `GradDescent` | `optimizer.py` | `W -= lr * dW`, `b -= lr * db` (optional momentum) |
| `Training` | `training.py` | Epoch loop: forward, loss, backward, update; predict and accuracy |

Data convention: each image is flattened to a column of 400 values in 0..1, so
`X` has shape `(400, m)` and `Y` is one hot with shape `(10, m)`.

## Changing things

* **Image set**: `python train.py --train-dir path/train --test-dir path/test`.
  Training images go in one subfolder per label; test images in a flat folder.
  Images of any size are resized to `--size` (default 20). For dark ink on
  white paper, use `invert=True` in `perceptron/data.py` loaders.
* **Activation**: `--activation tanh` (hidden) and `--output-activation`.
  To add a new one, subclass `Activation`, implement `forward` and
  `derivative`, add it to `ACTIVATIONS`, then rerun `gradient_check.py`.
* **Architecture**: `--hidden 64` or `--hidden 64 32` for two hidden layers.
* **Training**: `--lr`, `--momentum`, `--batch`, `--epochs`, `--loss bce`,
  `--init he`, `--target-loss 0.01`.

## Problems encountered (for class discussion)

1. **Training stuck at 10% accuracy, loss frozen at 0.5.** First run used
   learning rate 2.0 with momentum 0.9. Momentum 0.9 makes the effective step
   about lr / (1 - 0.9) = 10x larger, so the first updates overshot and drove
   all ten sigmoid outputs to 0. Because 9 of every 10 targets are 0, "answer
   no to everything" is an easy trap; there the sigmoids are saturated,
   sigma'(z) is about 0, gradients vanish, and learning stops. Fix: plain
   gradient descent at lr 1.0 trains reliably across seeds.

   | loss | lr | momentum | train / test accuracy (3 seeds, 300 epochs) |
   |---|---|---|---|
   | MSE | 2.0 | 0.9 | 0.10/0.10, 0.20/0.20, 0.10/0.10 |
   | MSE | 0.5 | 0.9 | 0.50/0.50, 0.70/0.70, 0.54/0.50 |
   | MSE | 1.0 | 0.0 | 1.00/0.97, 1.00/0.97, 1.00/1.00 |
   | BCE | 1.0 | 0.0 | 1.00/1.00, 1.00/0.97, 1.00/1.00 |

2. **Step activation cannot learn in the hidden layer.** Its derivative is 0
   everywhere, so backprop passes nothing back. With `--activation step` the
   output layer still learns on top of frozen random hidden features
   (96% train, 63% test), which shows why Rosenblatt's step perceptron had to
   be replaced with smooth activations once networks gained hidden layers.

3. **Zero initialization.** `--init zeros` makes every hidden neuron identical,
   they receive identical gradients and never specialize (symmetry problem).
   Fine for a single neuron, broken for a hidden layer. Measured: after 300
   epochs all 32 hidden neurons are still one identical weight row, and
   accuracy is 20%.

4. **Small dataset.** 100 training images is tiny; the network reaches 100%
   training accuracy almost immediately, so test accuracy is the honest
   measure. The one miss is an open top 9 with a straight tail, read as a 4.

## Other activations (300 epochs, He init, lr 0.3)

| hidden activation | train | test |
|---|---|---|
| sigmoid (default, lr 1.0) | 100% | 96.7% |
| tanh | 100% | 96.7% |
| ReLU | 100% | 93.3% |
| Leaky ReLU | 100% | 93.3% |
| step | 96% | 63.3% |
