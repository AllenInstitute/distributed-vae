from expecttest import assert_expected_inline
import pytest 
import numpy as np

from mmidas._utils import mk_confmat, ecdf, confmat_normalize, confmat_mean


def test_confusion_matrix():
    assert_expected_inline(str(mk_confmat(np.array([1, 0, 2, 3]), np.array([1, 0, 2, 3]))), """\
[[1 0 0 0]
 [0 1 0 0]
 [0 0 1 0]
 [0 0 0 1]]""")
    assert_expected_inline(str(mk_confmat(np.array([1, 0, 2, 3, 0, 3]), np.array([1, 0, 2, 3, 1, 3]))), """\
[[1 1 0 0]
 [0 1 0 0]
 [0 0 1 0]
 [0 0 0 2]]""")
    

def test_ecdf():
    assert_expected_inline(str(ecdf(np.array([1, 0, 2, 3]))), """\
[0.25 0.25 0.25 0.25]""")
    assert_expected_inline(str(ecdf(np.array([1, 0, 2, 3, 0, 3]))), """[0.33333333 0.16666667 0.16666667 0.33333333]""")


def test_confmat_normalize():
    assert_expected_inline(str(confmat_normalize(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))), """\
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]""")
    assert_expected_inline(str(confmat_normalize(mk_confmat(np.array([1, 0, 2, 3, 0, 3]), np.array([1, 0, 2, 3, 1, 3])))), """\
[[0.5 0.5 0.  0. ]
 [0.  0.5 0.  0. ]
 [0.  0.  1.  0. ]
 [0.  0.  0.  1. ]]""")
    

def test_confmat_mean():
    assert_expected_inline(str(confmat_mean(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))), """1.0""")
    assert_expected_inline(str(confmat_mean(mk_confmat(np.array([1, 0, 2, 3, 0, 3]), np.array([1, 0, 2, 3, 1, 3])))), """1.25""")