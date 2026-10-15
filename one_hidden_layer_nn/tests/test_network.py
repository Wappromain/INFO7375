"""Run with:  python -m unittest discover -s tests -v   (from the project folder)"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn import (DenseLayer, Momentum, OneHiddenLayerNetwork, SGD, Sigmoid, Softmax,
                make_moons, one_hot)


def numerical_gradients(net, X, Y, eps=1e-6):
    """Finite difference estimate of dL/dparam for every parameter."""
    grads = []
    for layer in net.layers:
        for p in (layer.W, layer.b):
            g = np.zeros_like(p)
            it = np.nditer(p, flags=["multi_index"])
            for _ in it:
                i = it.multi_index
                old = p[i]
                p[i] = old + eps
                lp = net.loss.value(net.forward(X), Y)
                p[i] = old - eps
                lm = net.loss.value(net.forward(X), Y)
                p[i] = old
                g[i] = (lp - lm) / (2 * eps)
            grads.append(g)
    return grads


def analytic_gradients(net, X, Y):
    A = net.forward(X)
    net.backward(A, Y)
    return [g for layer in net.layers for g in (layer.dW, layer.db)]


class TestGradients(unittest.TestCase):
    """If backprop math is wrong anywhere, these fail."""

    def _check(self, hidden, output, loss, Y):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6, 3))
        net = OneHiddenLayerNetwork(3, 4, Y.shape[1], hidden, output, loss=loss, seed=1)
        for a, n in zip(analytic_gradients(net, X, Y), numerical_gradients(net, X, Y)):
            rel = np.linalg.norm(a - n) / max(1e-12, np.linalg.norm(a) + np.linalg.norm(n))
            self.assertLess(rel, 1e-6, f"{hidden}/{output}/{loss} rel err {rel:.2e}")

    def test_tanh_sigmoid_bce(self):
        self._check("tanh", "sigmoid", "bce", np.array([[0], [1], [1], [0], [1], [0.]]))

    def test_sigmoid_sigmoid_mse(self):
        self._check("sigmoid", "sigmoid", "mse", np.random.default_rng(2).random((6, 2)))

    def test_relu_softmax_cce(self):
        self._check("relu", "softmax", "cce", one_hot([0, 1, 2, 1, 0, 2], 3))

    def test_tanh_identity_mse(self):
        self._check("tanh", "identity", "mse", np.random.default_rng(3).normal(size=(6, 1)))


class TestStructure(unittest.TestCase):
    def test_shapes(self):
        net = OneHiddenLayerNetwork(5, 8, 3, output_activation="softmax", seed=0)
        self.assertEqual(net.hidden.W.shape, (5, 8))
        self.assertEqual(net.output.W.shape, (8, 3))
        self.assertEqual(net.predict_proba(np.zeros((10, 5))).shape, (10, 3))
        self.assertEqual(net.n_params, 5 * 8 + 8 + 8 * 3 + 3)

    def test_default_loss_matches_output(self):
        self.assertEqual(OneHiddenLayerNetwork(2, 2, 1).loss.name, "bce")
        self.assertEqual(OneHiddenLayerNetwork(2, 2, 3, output_activation="softmax").loss.name, "cce")
        self.assertEqual(OneHiddenLayerNetwork(2, 2, 1, output_activation="identity").loss.name, "mse")

    def test_softmax_rows_sum_to_one(self):
        A = Softmax().forward(np.random.default_rng(0).normal(size=(4, 5)) * 50)
        np.testing.assert_allclose(A.sum(axis=1), 1.0)

    def test_sigmoid_is_stable(self):
        out = Sigmoid().forward(np.array([[-1000.0, 0.0, 1000.0]]))
        np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]])

    def test_bad_input_raises(self):
        net = OneHiddenLayerNetwork(2, 3, 1)
        with self.assertRaises(ValueError):
            net.forward(np.zeros((4, 3)))
        with self.assertRaises(ValueError):
            net.fit(np.zeros((4, 2)), np.zeros((5, 1)), epochs=1)

    def test_zero_init_symmetry_problem(self):
        """With identical starting weights, hidden units never become different."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1.]])
        Y = np.array([[0], [1], [1], [0.]])
        net = OneHiddenLayerNetwork(2, 4, 1, "tanh", init="zeros", optimizer=SGD(0.5))
        net.output.W[:] = 0.1  # break the trivial all zero case for the output only
        net.fit(X, Y, epochs=200)
        cols = net.hidden.W.T
        self.assertTrue(np.allclose(cols, cols[0]))

    def test_save_load_roundtrip(self):
        net = OneHiddenLayerNetwork(3, 4, 2, output_activation="softmax", seed=0)
        X = np.random.default_rng(0).normal(size=(5, 3))
        before = net.predict_proba(X)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.npz")
            net.save(path)
            other = OneHiddenLayerNetwork(3, 4, 2, output_activation="softmax", seed=99)
            other.load(path)
        np.testing.assert_allclose(other.predict_proba(X), before)


class TestLearning(unittest.TestCase):
    def test_learns_xor(self):
        """XOR: the problem a single perceptron cannot solve."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1.]])
        y = np.array([0, 1, 1, 0])
        net = OneHiddenLayerNetwork(2, 4, 1, "tanh", "sigmoid", optimizer=SGD(0.5), seed=3)
        net.fit(X, y, epochs=3000)
        np.testing.assert_array_equal(net.predict(X), y)

    def test_learns_moons_with_momentum(self):
        X, y = make_moons(300, noise=0.1, seed=0)
        net = OneHiddenLayerNetwork(2, 16, 1, "tanh", optimizer=Momentum(0.1), seed=0)
        net.fit(X, y, epochs=300, batch_size=32, seed=0)
        self.assertGreater(net.history.accuracy[-1], 0.95)

    def test_target_loss_stops_early(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1.]])
        y = np.array([0, 0, 0, 1])  # AND, linearly separable, easy
        net = OneHiddenLayerNetwork(2, 3, 1, optimizer=SGD(1.0), seed=0)
        h = net.fit(X, y, epochs=10000, target_loss=0.05)
        self.assertIsNotNone(h.stopped_epoch)
        self.assertLess(h.stopped_epoch, 10000)

    def test_loss_decreases(self):
        X, y = make_moons(200, seed=1)
        net = OneHiddenLayerNetwork(2, 8, 1, seed=1)
        h = net.fit(X, y, epochs=100)
        self.assertLess(h.loss[-1], h.loss[0])


if __name__ == "__main__":
    unittest.main()
