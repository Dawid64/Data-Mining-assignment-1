from preprocessing.extractor import Extractor, PCAExtractor
import pandas as pd


def test_extractor():
    """ Testing common extractor """
    extractor = Extractor()
    data_frame = pd.DataFrame(data={'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert (extractor.extract(data_frame) == data_frame).all().all()


def test_pca_extractor():
    """ Testing PCAExtractor """
    number = 2
    extractor = PCAExtractor(num_components=number)
    data = {'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8],
            'c': [9, 10, 11, 12], 'd': [13, 14, 15, 16]}
    expected_result = pd.DataFrame(data={'PC1': [3, 1, -1, -3],
                                         'PC2': [0.0, 0.0, 0.0, 0.0]})
    data_frame = pd.DataFrame(data=data)
    result = extractor.extract(data_frame)
    assert (result.round(10) == expected_result).all().all()