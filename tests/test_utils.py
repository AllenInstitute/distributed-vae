from expecttest import assert_expected_inline
import pytest
import numpy as np

from mmidas._utils import (
    compose,
    compute_confmat,
    confmat_normalize,
    confmat_mean,
    compute_confmat_naive,
    confmat_normalize_naive,
    ecdf,
    classify,
    time_function
)


def test_confusion_matrix():
    assert_expected_inline(
        str(compute_confmat(np.array([1, 0, 2, 3]), np.array([1, 0, 2, 3]))),
        """\
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]""",
    )
    assert_expected_inline(
        str(
            compute_confmat(np.array([1, 0, 2, 3, 0, 3]), np.array([1, 0, 2, 3, 1, 3]))
        ),
        """\
[[1. 1. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 2.]]""",
    )


def test_ecdf():
    assert_expected_inline(
        str(ecdf(np.array([1, 0, 2, 3]))),
        """\
[0.25 0.25 0.25 0.25]""",
    )
    assert_expected_inline(
        str(ecdf(np.array([1, 0, 2, 3, 0, 3]))),
        """[0.33333333 0.16666667 0.16666667 0.33333333]""",
    )


def test_confmat_normalize():
    assert_expected_inline(
        str(
            confmat_normalize(
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
            )
        ),
        """\
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]""",
    )
    assert_expected_inline(
        str(
            confmat_normalize(
                compute_confmat(
                    np.array([1, 0, 2, 3, 0, 3]), np.array([1, 0, 2, 3, 1, 3])
                )
            )
        ),
        """\
[[0.5 0.5 0.  0. ]
 [0.  0.5 0.  0. ]
 [0.  0.  1.  0. ]
 [0.  0.  0.  1. ]]""",
    )


def test_confmat_mean():
    assert_expected_inline(
        str(
            confmat_mean(
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            )
        ),
        """1.0""",
    )
    assert_expected_inline(
        str(
            confmat_mean(
                compute_confmat(
                    np.array([1, 0, 2, 3, 0, 3]), np.array([1, 0, 2, 3, 1, 3])
                )
            )
        ),
        """1.25""",
    )


def test_classify():
    x = np.array([[0.7, 0.2, 0.1], [0.4, 0.1, 0.5], [0.3, 0.6, 0.1]])
    assert_expected_inline(str(classify(x)), """[0 2 1]""")
    x = np.array([[0.7, 0.2, 0.1], [0.4, 0.1, 0.5], [0.3, 0.6, 0.1], [0.1, 0.1, 0.8]])
    assert_expected_inline(str(classify(x)), """[0 2 1 2]""")


def test_confmat_vectorize_correctness():
    K = 10
    labels1 = np.random.randint(0, K, 100)
    labels2 = np.random.randint(0, K, 100)

    f_naive = compose(confmat_normalize_naive, compute_confmat_naive)
    f_vec = compose(confmat_normalize, compute_confmat)

    matrix_naive = f_naive(labels1, labels2, K=K)
    matrix_vectorize = f_vec(labels1, labels2, K=K)
    assert np.allclose(matrix_vectorize, matrix_naive)


def test_confmat_vectorize_time():
    K = 92
    labels1 = np.random.randint(0, K, 5000)
    labels2 = np.random.randint(0, K, 5000)

    f_naive = compose(confmat_normalize_naive, compute_confmat_naive)
    f_vec = compose(confmat_normalize, compute_confmat)

    time_vec = time_function(f_vec, labels1, labels2, K=K)
    time_naive = time_function(f_naive, labels1, labels2, K=K)
    print()
    print(f"naive version took: {time_naive}s")
    print(f"vectorized version took: {time_vec}s")
    print(f"speedup: {100 * (time_naive - time_vec) / time_naive:.2f}%")