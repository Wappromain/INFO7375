"""
A multilayer (deep) neural network for binary classification, which I wrote from scratch in NumPy
so every step of the forward pass and backpropagation is visible.

Architecture:  (10/ReLU) - (8/ReLU)^2 - (4/ReLU) - (1/Sigmoid)

    input X(p)  : 5 features (hue, saturation, brightness, roundness, texture)
    layer 1     : 10 neurons, ReLU
    layer 2     :  8 neurons, ReLU
    layer 3     :  8 neurons, ReLU
    layer 4     :  4 neurons, ReLU
    layer 5     :  1 neuron,  Sigmoid  -> y(p), the probability that the object is an orange

I follow the course notation: X(p) is the input vector of sample p, d(p) is the desired (target)
output, and y(p) is the actual output of the network.
"""
import numpy as np

# Each entry is (number of neurons, activation function). Changing this list changes the network.
LAYERS = [(10, "relu"), (8, "relu"), (8, "relu"), (4, "relu"), (1, "sigmoid")]


def relu(z):
    # ReLU passes positive values through and turns negative values into 0.
    return np.maximum(0.0, z)


def relu_derivative(z):
    # The slope of ReLU is 1 where the input was positive and 0 elsewhere.
    return (z > 0).astype(float)


def sigmoid(z):
    # Sigmoid squashes any number into the range 0 to 1, so I can read the output as a probability.
    # I clip z first so np.exp does not overflow on very large values.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class DeepNetwork:
    def __init__(self, n_inputs, layers=LAYERS, seed=0):
        rng = np.random.default_rng(seed)
        self.activations = [act for _, act in layers]
        self.W, self.b = [], []
        fan_in = n_inputs
        for n_neurons, act in layers:
            # I start with small random weights. If every weight started equal, every neuron in a
            # layer would get the same update and they would never learn different things.
            # For ReLU layers I use He initialisation (scale sqrt(2 / fan_in)); for the sigmoid
            # output I use Xavier (scale sqrt(1 / fan_in)). Both keep signals from shrinking or
            # exploding as they pass through the layers. Biases start at zero.
            scale = np.sqrt(2.0 / fan_in) if act == "relu" else np.sqrt(1.0 / fan_in)
            self.W.append(rng.normal(0.0, scale, size=(fan_in, n_neurons)))
            self.b.append(np.zeros(n_neurons))
            fan_in = n_neurons

    def count_parameters(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    # ---------- forward pass ----------
    def forward(self, X):
        """I push the inputs through every layer and return the output y along with a cache of each
        layer's weighted sum z and output a. Backpropagation needs that cache later."""
        a = X
        cache = [(None, a)]
        for W, b, act in zip(self.W, self.b, self.activations):
            z = a @ W + b                       # weighted sum of the inputs plus the bias
            a = relu(z) if act == "relu" else sigmoid(z)
            cache.append((z, a))
        return a.ravel(), cache

    def predict_proba(self, X):
        return self.forward(X)[0]

    def predict(self, X, threshold=0.5):
        # I call anything at or above 0.5 an orange (1) and anything below an apple (0).
        return (self.predict_proba(X) >= threshold).astype(int)

    # ---------- loss ----------
    @staticmethod
    def loss(y, d, eps=1e-12):
        """I measure error with binary cross entropy between the actual outputs y(p) and the desired
        outputs d(p). It punishes confident wrong answers much more than mean squared error does.
        I clip y so log(0) never happens."""
        y = np.clip(y, eps, 1 - eps)
        return -np.mean(d * np.log(y) + (1 - d) * np.log(1 - y))

    # ---------- backpropagation ----------
    def backward(self, cache, d):
        """I compute how much each weight and bias contributed to the error, starting at the output
        layer and working backwards one layer at a time."""
        m = d.shape[0]
        y = cache[-1][1]
        # With a sigmoid output and cross entropy loss, the derivative at the output neuron
        # simplifies neatly to (y - d). I divide by m because the loss is an average over the batch.
        delta = (y - d.reshape(-1, 1)) / m
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
        for layer in reversed(range(len(self.W))):
            a_prev = cache[layer][1]
            grads_W[layer] = a_prev.T @ delta       # weight gradient = input to this layer times its error
            grads_b[layer] = delta.sum(axis=0)      # bias gradient = the error itself
            if layer > 0:
                z_prev = cache[layer][0]
                # I send the error back through this layer's weights, then multiply by the ReLU
                # slope of the previous layer. Neurons that were switched off (z <= 0) get no blame.
                delta = (delta @ self.W[layer].T) * relu_derivative(z_prev)
        return grads_W, grads_b

    def update(self, grads_W, grads_b, learning_rate):
        # Gradient descent: I move every weight a small step against its gradient to lower the loss.
        for i in range(len(self.W)):
            self.W[i] -= learning_rate * grads_W[i]
            self.b[i] -= learning_rate * grads_b[i]

    def save(self, path, **extra):
        arrays = {f"W{i + 1}": w for i, w in enumerate(self.W)}
        arrays.update({f"b{i + 1}": b for i, b in enumerate(self.b)})
        np.savez(path, **arrays, **extra)

    def describe(self, n_inputs):
        lines, fan_in = [], n_inputs
        for i, (W, act) in enumerate(zip(self.W, self.activations), start=1):
            lines.append(f"layer {i}: {W.shape[1]:>2} neurons, {act:<7} weights {fan_in}x{W.shape[1]} + {W.shape[1]} biases")
            fan_in = W.shape[1]
        lines.append(f"total trainable parameters: {self.count_parameters()}")
        return "\n".join(lines)


if __name__ == "__main__":
    # Running this file on its own prints the architecture so you can confirm it matches the spec.
    net = DeepNetwork(n_inputs=5)
    print(net.describe(5))
