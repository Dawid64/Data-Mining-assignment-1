from preprocessing.selector import Selector, VARSelector
import pandas as pd


def test_selector():
    """ Testing common selector """
    selector = Selector()
    data_frame = pd.DataFrame(data={'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = selector.select(data_frame)
    assert isinstance(result, pd.DataFrame)
    assert data_frame.equals(result)


def test_var_selector():
    """ Testing VARSelector """
    selector = VARSelector()
    data = {'a': [1, 2, 3, 4], 'b': [5, 5, 5, 5],
            'c': [100000, 100001, 100002, 100001], 'd': [13, 14, 15, 16]}
    data_frame = pd.DataFrame(data=data.copy())
    result = selector.select(data_frame)
    assert isinstance(result, pd.DataFrame)
    expected_result = data_frame.drop(columns=['b'])
    assert result.equals(expected_result)
