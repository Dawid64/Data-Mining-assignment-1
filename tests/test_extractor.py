from preprocessing.extractor import Extractor, PCAExtractor
import pandas as pd


def test_extractor():
    """ Testing base extractor """
    extractor = Extractor()
    data_frame = pd.DataFrame(data={'a': [1, 2, 3], 'b': [4, 5, 6]})
    output = extractor.extract(data_frame)
    assert isinstance(output, pd.DataFrame)
    assert data_frame.equals(extractor.extract(data_frame))


def test_pca_extractor():
    """ Testing output of PCAExtractor 
    Test covers:
    - constructor parameter setting
    - output data type
    - calculations
    """
    number = 2
    extractor = PCAExtractor(num_components=number)
    assert extractor.num_components == number
    assert extractor.target == 'target'
    data = {'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8],
            'c': [9, 10, 11, 12], 'd': [13, 14, 15, 16], 'target': [1, 2, 3, 4]}
    expected_result = pd.DataFrame(data={'PC1': [3, 1, -1, -3],
                                         'PC2': [0.0, 0.0, 0.0, 0.0],
                                         'target': [1, 2, 3, 4]})
    data_frame = pd.DataFrame(data=data)
    result = extractor.extract(data_frame)
    assert isinstance(result, pd.DataFrame)
    assert (expected_result == result.round(10)).all().all()
