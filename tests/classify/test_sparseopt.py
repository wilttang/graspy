import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from sklearn.exceptions import NotFittedError

from graspy.classify.sparse_opt import SparseOpt


def test_input():
    x = [1]
    y = [1]
    D = [1]
    with pytest.raises(ValueError):
        sparse_opt = SparseOpt(x, y, D)
        sparse_opt._admm(0, 0, opt_type="0")