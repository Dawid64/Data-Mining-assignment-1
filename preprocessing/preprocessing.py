from .selector import Selector
from .extractor import Extractor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod


class PreprocessingABC(ABC):
    @abstractmethod
    def preprocess(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def _encode_dataset(self):
        pass

    def _prepare_dataset(self):
        self._encode_dataset()
        self._na_handling()

    @abstractmethod
    def _na_handling(self):
        pass


class Preproccessing(PreprocessingABC):

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

    def preprocess(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset_scaled = StandardScaler().fit_transform(dataset.data)
        scaled_dataframe = pd.DataFrame(
            dataset_scaled, columns=dataset.feature_names)
        new_dataset = self.selector.select(scaled_dataframe)
        return new_dataset

    def select(self) -> pd.DataFrame:
        return self.selector.select(self.dataset)

    def extract(self) -> pd.DataFrame:
        return self.extractor.extract(self.dataset)

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
                new_dataset = pd.get_dummies(new_dataset, columns=[column])
            else:
                new_dataset.drop(column, axis=1, inplace=True)
        self.dataset = new_dataset

    def _na_handling(self):
        self.dataset.dropna(inplace=True)
