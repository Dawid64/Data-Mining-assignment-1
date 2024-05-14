import pandas as pd

from preprocessing.preprocessing import Preprocessing
from preprocessing.selector import Selector, VARSelector
from preprocessing.extractor import Extractor, PCAExtractor


def test_preprocessing():
    """ Testing Preprocessing """
    data = {'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8],
            'c': [9, 10, 11, 12], 'd': [13, 14, None, 16],
            'e': ['A', 'B', 'A', 'B'], 'target': [0, 0, 1, 1]}
    data_frame = pd.DataFrame(data=data)
    preprocessing = Preprocessing(
        dataset=data_frame, target='target', one_hot_threshold=0.9)
    assert preprocessing.target == 'target'
    assert preprocessing.one_hot_threshold == 3.6
    assert isinstance(preprocessing.selector, Selector)
    assert isinstance(preprocessing.extractor, Extractor)
    preprocessing._encode_dataset()
    new_data = {'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8],
                'c': [9, 10, 11, 12], 'd': [13, 14, None, 16],
                'e_A': [True, False, True, False],
                'e_B': [False, True, False, True], 'target': [0, 0, 1, 1]}
    assert pd.DataFrame(new_data).equals(preprocessing.dataset)
    preprocessing._na_handling()
    assert None not in preprocessing.dataset['d']


def test_preprocessing_apply():
    train_data = {'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8],
                  'c': [1, 1, 1, 1], 'd': [13, 14, None, 16],
                  'e': ['A', 'B', 'A', 'B'], 'target': [0, 0, 1, 1]}

    test_data = {'a': [1, 1, 1, 1, 1, 1], 'b': [5, 6, None, None, 2, 3],
                 'c': [9, 10, 11, 12, 0, 0], 'd': [13, 14, 2, None, 12, 5],
                 'e': ['B', 'B', 'B', 'B', 'D', 'E']}

    train_data_frame = pd.DataFrame(data=train_data)
    preprocessing = Preprocessing(
        dataset=train_data_frame, target='target', one_hot_threshold=0.9,
        selector=VARSelector(), extractor=PCAExtractor(num_components=1, target='target'))
    train_result = preprocessing.preprocess()
    assert isinstance(train_result, pd.DataFrame)
    assert 'target' in train_result.columns
    train_result.drop(columns=['target'], inplace=True)

    test_data_frame = pd.DataFrame(data=test_data)
    test_result = preprocessing.apply_preprocessing(test_data_frame)
    assert isinstance(test_result, pd.DataFrame)
    assert train_result.columns.equals(test_result.columns)
    assert train_result.shape[1] == test_result.shape[1]
