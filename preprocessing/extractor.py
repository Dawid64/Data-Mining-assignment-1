from abc import ABC
import pandas as pd


class Extractor(ABC):
    def extract(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset
