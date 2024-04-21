from .selector import Selector
from .extractor import Extractor
from .cabin_processing import cabin_to_side
import pandas as pd
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod


class PreprocessingABC(ABC):
    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _split_features(self):
        pass

    @abstractmethod
    def _encode_dataset(self):
        pass

    def _prepare_dataset(self):
        self._split_features()
        self._encode_dataset()
        self._na_handling()

    @abstractmethod
    def _na_handling(self):
        pass


class Preprocessing(PreprocessingABC):

    def __init__(self, dataset: pd.DataFrame = None, path: str = None, target: str = None,
                 selector: Selector = None, extractor: Extractor = None, one_hot_threshold: float = 0.1) -> None:
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
        self._prepare_dataset()

    def preprocess(self) -> pd.DataFrame:
        new_dataset = self.dataset.dropna()
        new_dataset = cabin_to_side(new_dataset)
        new_dataset = self.selector.select(new_dataset, 0.02)
        
        num_columns = new_dataset.select_dtypes(include=['number'])
        dataset_scaled = StandardScaler().fit_transform(num_columns.to_numpy())
        scaled_dataframe = pd.DataFrame(
            dataset_scaled, columns=num_columns.columns)
        final_dataset = self.dataset.copy()
        final_dataset[num_columns.columns] = scaled_dataframe[num_columns.columns]
    
    def select(self) -> pd.DataFrame:
        return self.selector.select(self.dataset)

    def extract(self) -> pd.DataFrame:
        return self.extractor.extract(self.dataset)

    def _load_dataset(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def _split_features(self):
        self.dataset[['GroupID', 'NumInGroup']
                     ] = self.dataset['PassengerId'].str.split('_', expand=True)
        self.dataset.drop(['PassengerId'], axis=1, inplace=True)
        self.dataset['GroupID'] = self.dataset['GroupID'].astype('category')
        self.dataset['GroupID'] = self.dataset['GroupID'].astype('category')
        self.dataset[['Deck', 'CabinNumber', "Side"]
                     ] = self.dataset['Cabin'].str.split('/', expand=True)
        self.dataset.drop(['Cabin'], axis=1, inplace=True)
        self.dataset['Deck'] = self.dataset['Deck'].astype('category')
        self.dataset['CabinNumber'] = self.dataset['CabinNumber'].astype(
            'category')
        self.dataset['Side'] = self.dataset['Side'].astype('category')
        self.dataset = self.dataset

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
        target = new_dataset.pop(self.target)
        new_dataset[self.target] = target
        self.dataset = new_dataset

    def _na_handling(self):
        self.dataset.dropna(inplace=True)
