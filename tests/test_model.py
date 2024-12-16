from expecttest import assert_expected_inline

from mmidas.model import (
    is_normalized
)

def test_is_normalized():
    assert is_normalized([1, 0, 0, 0])