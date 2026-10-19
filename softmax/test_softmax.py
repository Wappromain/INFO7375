"""Tests for Softmax. Run with:  python test_softmax.py   (or pytest)"""
import numpy as np
from softmax import Softmax

rng = np.random.default_rng(0)
sm = Softmax()                         # column per example


def numerical_jacobian(f, z, eps=1e-6):
    n = len(z)
    J = np.zeros((n, n))
    for j in range(n):
        zp, zm = z.copy(), z.copy()
        zp[j] += eps
        zm[j] -= eps
        J[:, j] = (f(zp) - f(zm)) / (2 * eps)
    return J


def cross_entropy(A, Y):
    return -np.sum(Y * np.log(A)) / A.shape[1]


# ---------------------------------------------------------------- forward
def test_known_values():
    out = sm([1.0, 2.0, 3.0])
    e = np.exp([1.0, 2.0, 3.0])
    assert np.allclose(out, e / e.sum())
    assert np.allclose(out, [0.09003057, 0.24472847, 0.66524096])


def test_columns_sum_to_one_and_positive():
    Z = rng.normal(size=(5, 8))
    A = sm(Z)
    assert A.shape == Z.shape
    assert np.all(A > 0) and np.all(A < 1)
    assert np.allclose(A.sum(axis=0), 1.0)


def test_each_column_independent():
    Z = rng.normal(size=(4, 3))
    A = sm(Z)
    for k in range(3):
        assert np.allclose(A[:, k], sm(Z[:, k]))


def test_shift_invariance():
    Z = rng.normal(size=(4, 6))
    assert np.allclose(sm(Z), sm(Z + 123.4))


def test_large_inputs_no_overflow():
    with np.errstate(all="raise"):    # any overflow / nan becomes an error
        A = sm(np.array([[1000.0], [1001.0], [1002.0]]))
        B = sm(np.array([[-1000.0], [-1001.0], [-1002.0]]))
    assert np.allclose(A.ravel(), sm([0.0, 1.0, 2.0]))
    assert np.allclose(B.ravel(), sm([0.0, -1.0, -2.0]))
    assert not np.any(np.isnan(A)) and not np.any(np.isnan(B))


def test_equal_inputs_give_uniform():
    assert np.allclose(sm(np.full(4, 7.0)), 0.25)


def test_preserves_argmax():
    Z = rng.normal(size=(6, 10))
    assert np.array_equal(np.argmax(Z, axis=0), np.argmax(sm(Z), axis=0))


def test_two_classes_reduce_to_sigmoid():
    z = rng.normal(size=5)
    Z = np.vstack([z, np.zeros(5)])   # logits (z, 0)
    sigmoid = 1 / (1 + np.exp(-z))
    assert np.allclose(sm(Z)[0], sigmoid)


def test_row_axis_option():
    Z = rng.normal(size=(7, 3))       # rows are examples
    assert np.allclose(Softmax(axis=1)(Z), sm(Z.T).T)
    assert np.allclose(Softmax(axis=1)(Z).sum(axis=1), 1.0)


# ------------------------------------------------------------- derivatives
def test_derivative_raises():
    try:
        sm.derivative(np.zeros((3, 1)))
    except NotImplementedError:
        return
    raise AssertionError("derivative() should raise NotImplementedError")


def test_jacobian_matches_finite_differences():
    z = rng.normal(size=5)
    J = sm.jacobian(z)
    assert np.allclose(J, numerical_jacobian(sm.forward, z), atol=1e-7)


def test_jacobian_properties():
    z = rng.normal(size=5)
    J = sm.jacobian(z)
    s = sm(z)
    assert np.allclose(J, J.T)                    # symmetric
    assert np.allclose(J.sum(axis=0), 0.0)        # outputs always sum to 1
    assert np.allclose(np.diag(J), s * (1 - s))   # diagonal is s_k(1 - s_k)


def test_backward_matches_explicit_jacobian():
    Z = rng.normal(size=(4, 6))
    dA = rng.normal(size=(4, 6))
    dZ = sm.backward(dA, Z)
    for k in range(6):
        assert np.allclose(dZ[:, k], sm.jacobian(Z[:, k]).T @ dA[:, k])


def test_softmax_cross_entropy_shortcut():
    # With cross entropy loss, the chain rule collapses to dZ = (A - Y) / m
    n, m = 4, 6
    Z = rng.normal(size=(n, m))
    Y = np.eye(n)[:, rng.integers(0, n, size=m)]  # one hot columns
    A = sm(Z)
    dA = -Y / A / m                                # dL/dA for cross entropy
    assert np.allclose(sm.backward(dA, Z), (A - Y) / m)


def test_full_gradient_check_through_loss():
    n, m, eps = 3, 4, 1e-6
    Z = rng.normal(size=(n, m))
    Y = np.eye(n)[:, rng.integers(0, n, size=m)]
    A = sm(Z)
    analytic = sm.backward(-Y / A / m, Z)
    numeric = np.zeros_like(Z)
    for i in range(n):
        for j in range(m):
            Zp, Zm = Z.copy(), Z.copy()
            Zp[i, j] += eps
            Zm[i, j] -= eps
            numeric[i, j] = (cross_entropy(sm(Zp), Y) - cross_entropy(sm(Zm), Y)) / (2 * eps)
    rel_err = np.linalg.norm(analytic - numeric) / np.linalg.norm(analytic + numeric)
    assert rel_err < 1e-7, rel_err


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
