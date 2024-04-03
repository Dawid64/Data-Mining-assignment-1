
import pytest
from preprocessing.benchmarks._benchmark_abc import BenchmarkABC

@pytest.mark.parametrize('test123', [1, 2, 3, 4])
def test_abc1(test123):
    """ test pytest parametrize """
    assert isinstance(test123, int)

def test_abc():
    """ Testing abstract method TypeError """
    class TestBenchmarkClass(BenchmarkABC):
        """ Class for testing """
        def __init__(self) -> None:
            super().__init__()
    try:
        TestBenchmarkClass()
    except TypeError as e:
        assert 'evaluate' in str(e)
    else:
        raise AssertionError("TestBenchmarkClass should not be instantiable")
