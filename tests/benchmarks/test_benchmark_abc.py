from preprocessing.benchmarks._benchmark_abc import BenchmarkABC


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
