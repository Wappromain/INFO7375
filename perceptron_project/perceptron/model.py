"""
Model: an ordered stack of Layers.

NeuralNetwork(400, [32], 10) builds the one hidden layer network the
assignment asks for:

    input (400 pixels) -> hidden (32 neurons, sigmoid) -> output (10 neurons, sigmoid)

Pass more sizes in the hidden list, e.g. [64, 32], to go deeper later.
"""
import numpy as np

from .layer import Layer


class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        return self

    def initialize(self):
        """Reset every layer's W and b to fresh initial values."""
        for layer in self.layers:
            layer.params.initialize()

    def parameter_count(self):
        return sum(l.params.count() for l in self.layers)

    def summary(self):
        lines = [f"{self.__class__.__name__}"]
        for l in self.layers:
            lines.append(f"  {l}")
        lines.append(f"  total trainable parameters: {self.parameter_count()}")
        return "\n".join(lines)

    def save(self, path):
        arrays = {}
        for i, l in enumerate(self.layers):
            arrays[f"W{i}"] = l.params.W
            arrays[f"b{i}"] = l.params.b
        np.savez(path, **arrays)

    def load(self, path):
        data = np.load(path)
        for i, l in enumerate(self.layers):
            l.params.W = data[f"W{i}"]
            l.params.b = data[f"b{i}"]


class NeuralNetwork(Model):
    def __init__(self, n_inputs, hidden_sizes, n_outputs,
                 hidden_activation="sigmoid", output_activation="sigmoid",
                 init="xavier", seed=None):
        super().__init__()
        rng = np.random.default_rng(seed)
        sizes = [n_inputs] + list(hidden_sizes) + [n_outputs]
        for i in range(len(sizes) - 1):
            last = i == len(sizes) - 2
            self.add(Layer(
                sizes[i], sizes[i + 1],
                activation=output_activation if last else hidden_activation,
                init=init,
                name="output" if last else f"hidden{i + 1}",
                rng=rng,
            ))
