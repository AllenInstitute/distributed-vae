from expecttest import assert_expected_inline
import pytest 
import torch as th

from mmidas.evals import compute_consensus

def test_compute_consensus():
    # consensus = th.tensor([])

    assert_expected_inline(compute_consensus)