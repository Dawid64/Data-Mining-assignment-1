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
    def __init__(self, dataset: pd.DataFrame = None, path: str = None, target: str = None, selector: Selector = None, extractor: Extractor = None) -> None:
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
        new_dataset = self.dataset.copy()
        for column in self.dataset.columns:
            if self.dataset[column].dtype == 'string' or self.dataset[column].dtype == 'object':
                new_dataset[column] = pd.Categorical(self.dataset[column])
                new_dataset[column] = new_dataset[column].cat.codes
        new_dataset = pd.get_dummies(self.dataset, drop_first=True)
        self.dataset = pd.get_dummies(self.dataset, drop_first=True)

    def _na_handling(self):
        self.dataset.dropna(inplace=True)
