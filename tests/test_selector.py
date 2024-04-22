from preprocessing.selector import Selector
import pandas as pd

def test_selector():
    """Testing default selector"""
    data = {'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8],
            'c': [9, 10, 11, 12], 'd': [13, 14, 15, 16]}
    data_frame = pd.DataFrame(data=data)
    selector = Selector()
    assert (selector.select(data_frame) == data_frame).all().all()
