import pytest 
from expecttest import assert_expected_inline

def test_saving():
    assert_expected_inline("TODO", """TODO""")