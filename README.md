# INFO 7375: Neural Networks from Scratch

Three projects that build neural networks step by step in Python and NumPy, without a deep learning framework. Each project adds one major idea to the one before it, moving from a single layer of artificial neurons to a deep multilayer network with regularization.

## Projects

| Folder | Model | Key idea it adds |
|---|---|---|
| `perceptron_project/` | Single layer perceptron | Weighted sums, bias, activation, loss, and gradient based training, applied to handwritten digit recognition (0 to 9) |
| `one_hidden_layer_nn/` | Network with one hidden layer | Hidden units, nonlinear activations, and backpropagation through a hidden layer |
| `deep_nn/` | Deep (multilayer) network | Any number of layers, five activation functions, three loss functions, L1/L2 regularization and dropout |

Each folder has its own README with setup steps, usage and implementation details.

## Folder structure

```
INFO7375/
├── README.md                     (this file)
│
├── perceptron_project/
│   ├── README.md
│   ├── make_digits.py            builds the digit image dataset
│   ├── train.py                  trains the perceptron and writes results
│   ├── gradient_check.py         verifies gradients numerically
│   ├── perceptron/               the perceptron package
│   │   ├── activation.py   data.py        layer.py
│   │   ├── loss.py         model.py       optimizer.py
│   │   ├── parameters.py   propagation.py training.py
│   │   └── __init__.py
│   ├── data/
│   │   ├── train/0 ... train/9/  training images, one folder per digit
│   │   ├── test/                 30 test images (test_00.png ... test_29.png)
│   │   └── test_answer_key.csv
│   └── results/
│       ├── weights.npz           trained weights
│       ├── training_curves.png
│       ├── training_sheet.png
│       ├── test_predictions.png
│       └── test_predictions.csv
│
├── one_hidden_layer_nn/
│   ├── README.md
│   ├── demo_results.png
│   ├── nn/                       the network package
│   │   ├── activations.py  layers.py      losses.py
│   │   ├── network.py      optimizers.py  utils.py
│   │   └── __init__.py
│   ├── demos/run_demos.py
│   └── tests/test_network.py
│
└── deep_nn/
    ├── README.md
    ├── activations.py            Linear, ReLU, Sigmoid, Tanh, Softmax
    ├── regularization.py         L1 and L2 regularization
    ├── dropout.py                inverted dropout
    ├── neural_network.py         network class, forward propagation, loss, backpropagation
    └── demo.py                   gradient checks and training experiments
```

## Learning progression

### 1. Perceptron (`perceptron_project/`)

A perceptron computes a weighted sum of its inputs plus a bias and passes it through an activation function. Training adjusts the weights to reduce the error between predicted and correct outputs. This project applies a single layer of perceptrons to classify images of handwritten digits, with the dataset organized into one training folder per digit and a separate labeled test set.

Geometrically, each perceptron draws one straight line (or hyperplane) through the input space. **Limitation:** a single layer can only separate classes that are linearly separable. It cannot learn XOR, because no single straight line separates XOR's two classes (Minsky and Papert, 1969).

### 2. One hidden layer network (`one_hidden_layer_nn/`)

Adding a hidden layer with nonlinear activations lets the network combine several boundaries into one curved decision boundary, so it can solve problems like XOR. The hidden units have no target values of their own, so training requires backpropagation: the output error is passed backward through the network to compute a gradient for every weight.

**Limitation:** one hidden layer can in principle approximate many functions, but it may need a very large number of units, and the architecture is fixed at a single hidden layer.

### 3. Deep neural network (`deep_nn/`)

The deep network generalizes backpropagation to any number of layers through one loop that runs from the output layer back to the input. It also adds the tools needed to train larger networks well:

- Linear, ReLU, Sigmoid, Tanh and Softmax activations
- Mean squared error, binary cross entropy and categorical cross entropy losses
- He and Xavier weight initialization, so units in a layer start different and learn different features
- L1 and L2 regularization and inverted dropout to reduce overfitting
- Mini batch gradient descent
- Numerical gradient checking to confirm backpropagation is correct

## Requirements

Python 3.8 or newer and NumPy. The perceptron project also reads and writes images and plots, so it may need additional packages such as Pillow and Matplotlib; see its README.

```bash
pip install numpy
```

Run each project from inside its own folder, following that folder's README.
