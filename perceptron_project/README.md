# Perceptron for Handwritten Digits (0 to 9)

An object oriented neural network written from scratch in NumPy, trained to
recognize 20 x 20 grayscale images of handwritten digits. It was trained and
tested on two datasets: a generated set of digits, and digits handwritten by
the author.

## Results at a glance

| Dataset | Training accuracy | Test accuracy |
|---|---|---|
| Generated digits (100 train, 30 test) | 100% | 96.7% (29 of 30) |
| My handwriting (100 train, 99 test) | 100% | 81.8% |
| My handwriting with data augmentation | 100% | 88.9% |

## Project files

| File | Purpose |
|---|---|
| `perceptron/` | The neural network package (classes described below) |
| `train.py` | Builds, initializes, trains and tests the network; writes plots and predictions |
| `gradient_check.py` | Verifies backpropagation against numerical gradients |
| `make_digits.py` | Generates the synthetic training and test digits in `data/` |
| `make_grid_template.py` | Writes `grid_template.png`, a printable 10 x 10 grid for handwriting |
| `prepare_my_digits.py` | Converts a photo of a filled grid (or separate image files) into 20 x 20 training or test images |
| `data/` | Generated digits: `train/<digit>/` labeled images, `test/` unlabeled images, and a separate answer key |
| `results/` | Output of the last `train.py` run on the generated digits |

## Quick start

```bash
pip install numpy pillow matplotlib     # on Windows without admin rights: python -m pip install --user ...
python gradient_check.py                # all six lines should end in OK
python train.py                         # trains on data/, writes results/
```

`data/` is already included. Run `python make_digits.py` only to regenerate it.

`train.py` prints the loss and accuracy every 100 epochs, then the final
training and test accuracy and any misclassified test images. It saves to the
output folder:

* `training_curves.png`: loss (log scale) and accuracy per epoch
* `test_predictions.png`: every test image with its predicted digit and confidence
* `test_predictions.csv`: the same as a table
* `weights.npz`: the trained W and b matrices

## Class design

| Class | File | Responsibility |
|---|---|---|
| `Activation` (`Sigmoid`, `Tanh`, `ReLU`, `LeakyReLU`, `Step`) | `activation.py` | `forward(Z)` and `derivative(Z)` |
| `Parameters` | `parameters.py` | Transmission matrix `W (n_out, n_in)`, bias `b (n_out, 1)`, their gradients, and the initialization module (`xavier`, `he`, `random`, `zeros`) |
| `Neuron` | `layer.py` | View of one row of a layer: its weights, bias, and output for one input |
| `Layer` | `layer.py` | One `Parameters` plus one `Activation`; caches `A_prev`, `Z`, `A` for backprop |
| `Model` / `NeuralNetwork` | `model.py` | Ordered stack of layers; `NeuralNetwork(400, [32], 10)` is one hidden layer |
| `LossFunction` (`MeanSquaredError`, `BinaryCrossEntropy`) | `loss.py` | `value(A, Y)` and `gradient(A, Y)` |
| `ForwardProp` | `propagation.py` | `Z = W @ A_prev + b`, `A = f(Z)` layer by layer |
| `BackProp` | `propagation.py` | Chain rule from the loss back to every `dW`, `db` |
| `GradDescent` | `optimizer.py` | `W -= lr * dW`, `b -= lr * db` (optional momentum) |
| `Training` | `training.py` | Epoch loop of forward, loss, backward, update; predict and accuracy |
| data helpers | `data.py` | Load image folders into matrices, one hot labels, data augmentation |

Data convention: each image is flattened to a column of 400 values between 0
and 1, so `X` has shape `(400, m)` and `Y` is one hot with shape `(10, m)`.

The default network is 400 inputs, 32 sigmoid hidden neurons and 10 sigmoid
outputs (13,162 trainable parameters), Xavier initialization, mean squared
error loss, gradient descent with learning rate 1.0, mini batches of 10.

## Training on your own handwriting

1. Print `grid_template.png` and write ten versions of each digit in its row
   (row 0 for zeros, row 1 for ones, and so on). A digital drawing on a
   10 x 10 grid works the same way.
2. Photograph or export the sheet cropped to the outer edge of the grid.
3. Convert it, then open `my_data/train_preview.png` to check every crop:
   ```bash
   python prepare_my_digits.py --sheet my_sheet.png --out my_data/train
   ```
4. Make a test sheet of mixed digits and convert it. **`--rows` and `--cols`
   must match the grid on that sheet.**
   ```bash
   python prepare_my_digits.py --sheet my_test.png --rows 10 --test --out my_data/test
   ```
5. Train and test:
   ```bash
   python train.py --train-dir my_data/train --test-dir my_data/test --augment 10 --epochs 60 --out my_results
   ```
   Test images have no labels, so check `my_results/test_predictions.png` by
   eye, or pass a CSV of `file,label` with `--answer-key` to get an accuracy.

For each grid cell, `prepare_my_digits.py` makes the ink white on black,
removes uneven lighting, detects and removes grid lines, drops specks and
fragments of neighboring digits, crops to the ink, thickens thin pen strokes to
about 2 pixels, and centers the digit by its center of mass in a 20 x 20 frame.
Separate image files (for example drawn in Paint) are supported with
`--folder`; see the script's header.

## Changing things

* **Image set**: `--train-dir` (one subfolder per label) and `--test-dir`
  (a flat folder). Images of any size are resized to `--size` (default 20).
* **Activation**: `--activation tanh` for the hidden layer, and
  `--output-activation`. To add a new one, subclass `Activation`, implement
  `forward` and `derivative`, add it to `ACTIVATIONS`, then rerun
  `gradient_check.py`.
* **Architecture**: `--hidden 64`, or `--hidden 64 32` for two hidden layers.
* **Training**: `--lr`, `--momentum`, `--batch`, `--epochs`, `--loss bce`,
  `--init he`, `--target-loss 0.01`.
* **Data augmentation**: `--augment 10` adds ten randomly rotated (up to 12
  degrees) and shifted (up to 1 pixel) copies of each training image. Each
  epoch then covers 11 times as many images, so fewer epochs are needed.

## Experiments on my handwriting

Test set: 99 handwritten digits from a separate test sheet (one ambiguous
digit excluded). Mean of 3 random seeds.

| Change | Test accuracy |
|---|---|
| Baseline (32 hidden, MSE) | 82.5% |
| 64 hidden neurons | 82.5% |
| Binary cross entropy loss | 82.5% |
| Data augmentation x10 | 86.5% |
| Augmentation, 64 hidden, cross entropy | 86.9% |

A wider layer and a different loss changed nothing; more varied training data
was what helped. A separate five fold check (train on 8 of each digit, test on
the remaining 2) gave 89% on unseen digits from the training sheet itself.

Common errors: fives read as 3 or 6, zeros with a tail at the top read as 6,
open top nines read as 4. Most of these are digits a person might also
hesitate over.

The gap between 96.7% on generated digits and 82% on real handwriting shows
that the network only knows the variety it was trained on.

## Problems encountered (for class discussion)

1. **Training stuck at 10% accuracy, loss frozen at 0.5.** Learning rate 2.0
   with momentum 0.9 made the effective step about lr / (1 minus 0.9), ten
   times larger, so the first updates overshot and drove all ten sigmoid
   outputs to 0. Because 9 of every 10 targets are 0, answering "no" to
   everything is an easy trap. There the sigmoids are saturated, their
   derivative is near 0, gradients vanish, and learning stops. Plain gradient
   descent at learning rate 1.0 trains reliably.

   | loss | lr | momentum | train / test accuracy (3 seeds, 300 epochs) |
   |---|---|---|---|
   | MSE | 2.0 | 0.9 | 0.10/0.10, 0.20/0.20, 0.10/0.10 |
   | MSE | 0.5 | 0.9 | 0.50/0.50, 0.70/0.70, 0.54/0.50 |
   | MSE | 1.0 | 0.0 | 1.00/0.97, 1.00/0.97, 1.00/1.00 |
   | BCE | 1.0 | 0.0 | 1.00/1.00, 1.00/0.97, 1.00/1.00 |

   Reproduce with `python train.py --lr 2.0 --momentum 0.9`.

2. **Step activation cannot learn in the hidden layer.** Its derivative is 0
   everywhere, so backprop passes nothing back. With `--activation step` only
   the output layer learns, on top of frozen random hidden features (96% train,
   63% test). This is why the step perceptron had to give way to smooth
   activations once networks gained hidden layers.

3. **Zero initialization.** With `--init zeros` every hidden neuron starts
   identical, receives identical gradients and never specializes (the symmetry
   problem). After 300 epochs all 32 hidden neurons were still one identical
   weight row, and accuracy was 20%.

4. **Every test prediction was "4".** The test sheet had 10 rows but was
   converted with `--rows 3`, so each image was about three digits squashed
   into one. The network never saw a real digit. Converting with `--rows 10`
   fixed it. Lesson: inspect the preprocessed inputs before blaming the model.

5. **Thin pen strokes nearly vanished.** Digits drawn with a fine pen were
   about 2 pixels wide in a 200 pixel cell, which became faint, broken lines at
   20 x 20. The converter now measures stroke width and thickens thin strokes.

6. **Digits touching the grid were clipped.** Trimming a fixed margin to hide
   grid lines cut off parts of digits that reached the cell edge (a 0 became a
   "C"). The converter now detects grid lines as rows or columns that are ink
   across at least 92% of the cell. An earlier 75% threshold also erased the
   long vertical stroke of one tall 1.

7. **Two bugs in `train.py`.** Test images named like the generated set's
   answer key could have been scored against the wrong answers; handwritten
   test images now use a `mytest_` prefix, and scoring is skipped if the key
   does not cover every test file. Without an answer key, a meaningless
   "test acc" value was printed; it is now hidden.

## Other activations (generated digits, 300 epochs, He init, lr 0.3)

| Hidden activation | Train | Test |
|---|---|---|
| sigmoid (default, Xavier init, lr 1.0) | 100% | 96.7% |
| tanh | 100% | 96.7% |
| ReLU | 100% | 93.3% |
| Leaky ReLU | 100% | 93.3% |
| step | 96% | 63.3% |
