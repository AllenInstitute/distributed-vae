import pytest 
from expecttest import assert_expected_inline

def test_foo():
    assert_expected_inline("foo", """foo""")