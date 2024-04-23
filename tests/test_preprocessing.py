import pandas as pd

from preprocessing.preprocessing import Preprocessing, PreprocessingABC
from preprocessing.selector import VARSelector, Selector
from preprocessing.extractor import PCAExtractor, Extractor


def test_preprocessing_abc():
    """ Testing PreprocessingABC """
    class PreprocessingABCImpl(PreprocessingABC):
        def preprocess(self) -> pd.DataFrame:
            pass

        def _split_features(self):
            pass

        def _encode_dataset(self):
            pass

        def _na_handling(self):
            pass

    preprocessing_abc = PreprocessingABCImpl()
    assert preprocessing_abc is not None
    assert isinstance(preprocessing_abc, PreprocessingABC)


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
