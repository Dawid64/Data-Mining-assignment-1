"""
Module for feature selection.
"""
import pandas as pd
from numpy import float64
from sklearn.preprocessing import MinMaxScaler


class Selector:
    """
    The base class for feature selection.

    This class provides an interface-like structure for feature selection algorithms.
    Subclasses should implement the `select` method to perform the actual feature selection.

    ### Attributes:
        None

    ### Methods:
        select: Perform feature selection on the given dataset.

    """

    def __init__(self) -> None:
        self.removed_columns = set()

    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature selection on the given dataset.

        ### Args:
            dataset (pd.DataFrame): The input dataset.

        ### Returns:
            pd.DataFrame: The dataset after feature selection.

        """
        return dataset

    def apply(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the feature selection on the given dataset.

        ### Args:
            dataset (pd.DataFrame): The input dataset.

        ### Returns:
            pd.DataFrame: The dataset after feature selection.

        """

        return dataset.drop(columns=list(self.removed_columns))


class VARSelector(Selector):
    """
    VarSelector is a class for feature selection based on variance.

    ### Parameters:
        var_treshold : float, default=0.0017:
            The threshold value for variance. Features with variances below this threshold will be dropped. Default is 0.0017.
    ### Methods:
        select(dataset: pd.DataFrame) -> pd.DataFrame:
            TODO

    """

    def __init__(self, var_treshold: float = 0.0017) -> None:
        self.var_treshold = var_treshold
        self.removed_columns = set()

    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature selection on the given dataset by focusing at variance.

        ### Args:
            dataset (pd.DataFrame): The input dataset.

        ### Returns:
            pd.DataFrame: The dataset after feature selection.

        """
        new_dataset = self._drop_unique(dataset)
        new_dataset = self._drop_low_var(new_dataset, self.var_treshold)
        self.removed_columns = set(dataset.columns).difference(
            set(new_dataset.columns))
        return new_dataset

    def _drop_unique(self, dataset: pd.DataFrame) -> pd.DataFrame:
        new_dataset = dataset.copy()
        threshold = len(new_dataset) * 0.01

        for column in new_dataset.columns:
            if new_dataset[column].dtype != float64:
                freq_most_common = new_dataset[column].value_counts().max()

                if freq_most_common < threshold:
                    new_dataset.drop(column, axis=1, inplace=True)

        return new_dataset

    def _drop_low_var(self, dataset: pd.DataFrame, threshold: float) -> pd.DataFrame:
        data_num = dataset.select_dtypes(include=['number'])
        dataset_scaled = MinMaxScaler().fit_transform(data_num.to_numpy())
        scaled_dataframe = pd.DataFrame(
            dataset_scaled, columns=data_num.columns)

        variances = scaled_dataframe.var()

        low_variance_columns = variances[variances < threshold].index

        dataset_filtered = dataset.drop(columns=low_variance_columns)
        return dataset_filtered
