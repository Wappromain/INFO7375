"""Tests for normalization and mini batch training.  Run with:  pytest -v"""

import numpy as np
import pytest

from perceptron_minibatch import (
    Normalizer, make_mini_batches, Perceptron, make_badly_scaled_data)


# ---------------------------------------------------------------- normalization

@pytest.fixture
def X():
    return np.array([[1.0, 100.0, 5.0],
                     [2.0, 200.0, 5.0],
                     [3.0, 300.0, 5.0],
                     [4.0, 400.0, 5.0]])


def test_zscore_gives_mean_0_std_1(X):
    Xn = Normalizer("zscore").fit_transform(X[:, :2])
    np.testing.assert_allclose(Xn.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Xn.std(axis=0), 1, atol=1e-12)


def test_zscore_matches_hand_calculation():
    # mean 2.5, population std sqrt(1.25)
    Xn = Normalizer("zscore").fit_transform([[1.0], [2.0], [3.0], [4.0]])
    expected = (np.array([1, 2, 3, 4]) - 2.5) / np.sqrt(1.25)
    np.testing.assert_allclose(Xn[:, 0], expected)


def test_minmax_gives_range_0_to_1(X):
    Xn = Normalizer("minmax").fit_transform(X[:, :2])
    np.testing.assert_allclose(Xn.min(axis=0), 0)
    np.testing.assert_allclose(Xn.max(axis=0), 1)
    np.testing.assert_allclose(Xn[:, 0], [0, 1 / 3, 2 / 3, 1])


def test_features_end_up_on_same_scale(X):
    # Feature 2 is 100x feature 1 but perfectly correlated, so after
    # normalization both columns should be identical.
    for method in Normalizer.METHODS:
        Xn = Normalizer(method).fit_transform(X[:, :2])
        np.testing.assert_allclose(Xn[:, 0], Xn[:, 1])


@pytest.mark.parametrize("method", Normalizer.METHODS)
def test_constant_feature_becomes_zero_not_nan(X, method):
    Xn = Normalizer(method).fit_transform(X)
    assert np.all(np.isfinite(Xn))
    np.testing.assert_allclose(Xn[:, 2], 0)


@pytest.mark.parametrize("method", Normalizer.METHODS)
def test_inverse_transform_round_trip(X, method):
    norm = Normalizer(method).fit(X)
    np.testing.assert_allclose(norm.inverse_transform(norm.transform(X)), X)


def test_test_data_uses_training_statistics_only():
    train = np.array([[0.0], [10.0]])
    test = np.array([[20.0]])
    norm = Normalizer("minmax").fit(train)
    # Test value lies outside the training range, so it maps above 1.
    # If the normalizer had refit on test data, it would map to 0.
    np.testing.assert_allclose(norm.transform(test), [[2.0]])
    np.testing.assert_allclose(norm.shift_, [0.0])
    np.testing.assert_allclose(norm.scale_, [10.0])


def test_fit_does_not_modify_input(X):
    original = X.copy()
    Normalizer("zscore").fit_transform(X)
    np.testing.assert_array_equal(X, original)


def test_transform_before_fit_raises(X):
    with pytest.raises(RuntimeError):
        Normalizer().transform(X)


def test_wrong_feature_count_raises(X):
    norm = Normalizer().fit(X)
    with pytest.raises(ValueError):
        norm.transform(X[:, :2])


@pytest.mark.parametrize("bad", [
    np.array([1.0, 2.0, 3.0]),            # 1D
    np.empty((0, 2)),                     # no samples
    np.array([[1.0, np.nan], [2.0, 3.0]]),
    np.array([[1.0, np.inf], [2.0, 3.0]]),
])
def test_invalid_input_raises(bad):
    with pytest.raises(ValueError):
        Normalizer().fit(bad)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        Normalizer("l2")


# ---------------------------------------------------------------- mini batches

@pytest.fixture
def data():
    X = np.arange(20, dtype=float).reshape(10, 2)   # 10 samples, 2 features
    d = np.arange(10, dtype=float)                  # d(p) = p, to track samples
    return X, d


@pytest.mark.parametrize("bs, expected_sizes", [
    (1, [1] * 10),
    (3, [3, 3, 3, 1]),
    (5, [5, 5]),
    (10, [10]),
    (25, [10]),       # batch larger than dataset -> one full batch
])
def test_batch_sizes(data, bs, expected_sizes):
    X, d = data
    batches = make_mini_batches(X, d, bs, np.random.default_rng(0))
    assert [len(b[0]) for b in batches] == expected_sizes


@pytest.mark.parametrize("bs", [1, 3, 4, 10])
def test_every_sample_appears_exactly_once(data, bs):
    X, d = data
    batches = make_mini_batches(X, d, bs, np.random.default_rng(0))
    seen = np.concatenate([db for _, db in batches])
    assert sorted(seen) == list(range(10))


def test_inputs_and_targets_stay_paired(data):
    X, d = data
    for Xb, db in make_mini_batches(X, d, 3, np.random.default_rng(0)):
        for x, target in zip(Xb, db):
            np.testing.assert_array_equal(x, X[int(target)])


def test_shuffle_changes_order_but_is_reproducible(data):
    X, d = data
    a = make_mini_batches(X, d, 10, np.random.default_rng(42))[0][1]
    b = make_mini_batches(X, d, 10, np.random.default_rng(42))[0][1]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, d)


def test_no_shuffle_keeps_order(data):
    X, d = data
    batches = make_mini_batches(X, d, 4, shuffle=False)
    np.testing.assert_array_equal(np.concatenate([db for _, db in batches]), d)


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_invalid_batch_size_raises(data, bad):
    X, d = data
    with pytest.raises(ValueError):
        make_mini_batches(X, d, bad)


def test_mismatched_lengths_raise(data):
    X, d = data
    with pytest.raises(ValueError):
        make_mini_batches(X, d[:5], 2)


# ---------------------------------------------------------------- training

def test_batch_size_1_matches_classic_perceptron_rule():
    # One misclassified sample: x=[1, 2], d=1, start from zero weights.
    # Classic rule: W += lr * (d - y) * x = 0.5 * 1 * [1, 2]
    p = Perceptron(2, learning_rate=0.5)
    p.train(np.array([[1.0, 2.0]]), np.array([1.0]), batch_size=1, epochs=1)
    np.testing.assert_allclose(p.W, [0.5, 1.0])
    assert p.b == pytest.approx(0.5)


def test_batch_update_is_average_of_individual_updates():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = np.array([1.0, 1.0])            # both wrong at zero weights (y = 0)
    p = Perceptron(2, learning_rate=1.0)
    p.train(X, d, batch_size=2, epochs=1, shuffle=False)
    np.testing.assert_allclose(p.W, [0.5, 0.5])   # mean of [1,0] and [0,1]
    assert p.b == pytest.approx(1.0)


@pytest.mark.parametrize("bs", [1, 4, 16, 64])
def test_learns_AND_for_any_batch_size(bs):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 16, dtype=float)
    d = np.array([0, 0, 0, 1] * 16, dtype=float)
    p = Perceptron(2, learning_rate=0.1, seed=0)
    p.train(X, d, batch_size=bs, epochs=500)
    assert p.errors(X, d) == 0


def test_cannot_learn_XOR():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    d = np.array([0, 1, 1, 0], dtype=float)
    p = Perceptron(2, seed=0)
    p.train(X, d, batch_size=2, epochs=200)
    assert p.errors(X, d) > 0


def test_normalization_helps_on_badly_scaled_data():
    X, d = make_badly_scaled_data()
    Xn = Normalizer("zscore").fit_transform(X)
    raw = Perceptron(2, seed=1)
    raw.train(X, d, batch_size=8, epochs=100)
    norm = Perceptron(2, seed=1)
    norm.train(Xn, d, batch_size=8, epochs=100)
    assert norm.errors(Xn, d) < raw.errors(X, d)
