"""
Module with main preprocessing class.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .selector import Selector
from .extractor import Extractor


class Preprocessing:
    """
    A class for performing data preprocessing tasks.

    ### Parameters:
    - dataset (pd.DataFrame):
        The input dataset to be preprocessed.
    - path (str):
        The path to the dataset file. Either `dataset` or `path` must be provided.
    - target (str):
        The name of the target variable in the dataset.
    - selector (Selector):
        An instance of the Selector class for feature selection.
    - extractor (Extractor):
        An instance of the Extractor class for feature extraction.
    - one_hot_threshold (float):
        The threshold for one-hot encoding. Default is 0.9.

    ### Methods:
    - preprocess():
        Perform the complete preprocessing pipeline.
    """

    def __init__(self, dataset: pd.DataFrame = None, path: str = None, target: str = 'target',
                 selector: Selector = None, extractor: Extractor = None, one_hot_threshold: float = 0.9) -> None:
        if path is not None:
            self.dataset = self._load_dataset(path)
        elif dataset is not None:
            self.dataset = dataset
        else:
            raise ValueError(
                "You need to provide either dataset or path to dataset")
        self.target = target
        self.selector = Selector() if selector is None else selector
        self.extractor = Extractor() if extractor is None else extractor
        self.one_hot_threshold = one_hot_threshold * self.dataset.shape[0]

    def preprocess(self) -> pd.DataFrame:
        """
        Main method for performing the complete preprocessing pipeline.

        ### !!! preprocess works on the dataset inplace !!!

        ### Returns:
            pd.DataFrame: The preprocessed dataset.
        """
        self._na_handling()
        self._split_features()
        self._select(inplace=True)
        # self._scale(inplace=True)
        self._encode_dataset()
        self._extract(inplace=True)

        return self.dataset

    def _scale(self, inplace: bool = False) -> pd.DataFrame:
        num_columns = self.dataset.select_dtypes(include=['number'])
        dataset_scaled = StandardScaler().fit_transform(num_columns.to_numpy())
        scaled_dataframe = pd.DataFrame(
            dataset_scaled, columns=num_columns.columns)
        new_dataset = self.dataset.copy()
        new_dataset[num_columns.columns] = scaled_dataframe[num_columns.columns]
        if inplace:
            self.dataset = new_dataset
            return self.dataset
        return new_dataset

    def _select(self, inplace: bool = False) -> pd.DataFrame:
        if inplace:
            dataset = self.selector.select(self.dataset)
            return dataset
        return self.selector.select(self.dataset)

    def _extract(self, inplace: bool = False) -> pd.DataFrame:
        if inplace:
            dataset = self.extractor.extract(self.dataset)
            return dataset
        return self.extractor.extract(self.dataset)

    def _apply_selection(self, inplace: bool = False) -> pd.DataFrame:
        if inplace:
            dataset = self.selector.apply(self.dataset)
            return dataset
        return self.selector.apply(self.dataset)

    def _apply_extraction(self, inplace: bool = False) -> pd.DataFrame:
        if inplace:
            dataset = self.extractor.apply(self.dataset)
            return dataset
        return self.extractor.apply(self.dataset)

    def _load_dataset(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def _encode_dataset(self):
        types_to_encode = ['category', 'string', 'object']
        new_dataset = self.dataset.copy()
        for column in self.dataset.columns:
            if False is self.dataset[column].unique()[0] or True is self.dataset[column].unique()[0]:
                new_dataset[column] = new_dataset[column].astype('bool')
            if new_dataset[column].dtype not in types_to_encode:
                continue
            if self.dataset[column].nunique() < self.one_hot_threshold:
                new_dataset = pd.get_dummies(
                    new_dataset, columns=[column], dtype='bool')
            else:
                new_dataset.drop(column, axis=1, inplace=True)
        if self.target in new_dataset.columns:
            target = new_dataset.pop(self.target)
            self.encoded_columns = new_dataset.columns
            new_dataset[self.target] = target
        self.dataset = new_dataset

    def _apply_encode_dataset(self):
        self._encode_dataset()
        for column in self.dataset.columns:
            if column not in self.encoded_columns:
                self.dataset.drop(column, axis=1, inplace=True)
        for column in self.encoded_columns:
            if column not in self.dataset.columns:
                self.dataset[column] = False
        self.dataset = self.dataset.reindex(columns=self.encoded_columns)

    def _na_handling(self):
        self.dataset.dropna(inplace=True, axis=0)
        return
        cat = self.dataset.iloc[:, :-1].select_dtypes(include=['category',
                                                               'object', 'bool'])
        numbers = self.dataset.iloc[:, :-1].select_dtypes(include=['number'])
        numbers.fillna(0, inplace=True)

        modes = cat.mode().iloc[0]
        pd.set_option('future.no_silent_downcasting', True)
        cat.fillna({col: modes[col] for col in cat.columns}, inplace=True)
        pd.set_option('future.no_silent_downcasting', False)
        self.dataset = pd.concat(
            [cat, numbers, self.dataset[self.target]], axis=1)

    def _split_features(self):
        pass

    def _adjust_features(self):
        pass

    def _addjust_features(self):
        pass

    def apply_preprocessing(self, dataset):
        """
        Apply the preprocessing pipeline to a test dataset.

        ### Parameters:
        - dataset (pd.DataFrame):
            The test dataset to be preprocessed.

        ### Returns:
            pd.DataFrame: The preprocessed dataset.
        """
        self.dataset = dataset
        self._na_handling()
        self._split_features()
        self._addjust_features()
        self._apply_selection(inplace=True)
        # self._scale(inplace=True)
        self._apply_encode_dataset()
        self._apply_extraction(inplace=True)
        return self.dataset


class SpaceShipPreprocessing(Preprocessing):
    """
    Specific preprocessing class for the Spaceship-titanic dataset.

    Works the same as Preprocessing class with addition of splitting few specific features into few different features.
    """

    def _split_features(self):
        self.dataset[['GroupID', 'NumInGroup']
                     ] = self.dataset['PassengerId'].str.split('_', expand=True)
        self.dataset.drop(['PassengerId'], axis=1, inplace=True)
        self.dataset['GroupID'] = self.dataset['GroupID'].astype('category')
        self.dataset[['Deck', 'CabinNumber', "Side"]
                     ] = self.dataset['Cabin'].str.split('/', expand=True)
        self.dataset.drop(['Cabin'], axis=1, inplace=True)
        self.dataset['Deck'] = self.dataset['Deck'].astype('category')
        self.dataset['CabinNumber'] = self.dataset['CabinNumber'].astype(
            'float64')
        self.dataset['Side'] = self.dataset['Side'].astype('category')
