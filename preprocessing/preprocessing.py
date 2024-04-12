from .selector import Selector
from .extractor import Extractor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod


class PreprocessingABC(ABC):
    @abstractmethod
    def preprocess(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass
