""" Class BenchmarkABC """
from abc import ABC, abstractmethod
import pandas as pd

class BenchmarkABC(ABC):
    """Abstract class for dataset preprocessing benchmarking.
    """

    def __init__(self) -> None:
        self.test = 123
        super().__init__()

    @abstractmethod
    def evaluate(self, dataset: pd.DataFrame) -> float:
        """This method checks the quality of given dataset using current benchmark.
        If provided dataset is more suitable for this benchmark method should return higher value.

        Args:
            dataset (pd.DataFrame): dataframe that we want to evaluate

        Returns:
            float: _description_
        """
