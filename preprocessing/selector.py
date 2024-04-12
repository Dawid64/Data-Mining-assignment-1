from abc import ABC
import pandas as pd


class Selector(ABC):
    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset
