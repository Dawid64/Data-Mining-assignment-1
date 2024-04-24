"""
This module contains classes for feature extraction using Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA).

Classes:
- Extractor: The base class for feature extraction.
- PCAExtractor: Performs PCA dimensionality reduction on a dataset.
- LDAExtractor: Performs LDA dimensionality reduction on a dataset.
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class Extractor:
    """
    The base class for feature extraction.

    This class provides an interface-like structure for feature extraction algorithms.
    Subclasses should implement the `extract` method to perform the actual feature extraction.

    ## Parameters:
        None

    ## Methods:
        extract: Perform feature extraction on the given dataset.

    """

    def extract(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature extraction on the given dataset.

        ### Args:
            dataset (pd.DataFrame): The input dataset.

        ### Returns:
            pd.DataFrame: The dataset after feature extraction.

        """
        return dataset


class PCAExtractor(Extractor):
    """
    PCAExtractor is a class that performs Principal Component Analysis (PCA) dimensionality reduction on a dataset.

    ## Parameters:
        num_components (int): The number of components to keep after dimensionality reduction. Default is 2.
        target (str): The name of the target variable in the dataset. Default is 'target'.

    ## Methods:
        extract(dataset: pd.DataFrame) -> pd.DataFrame:
            Applies PCA dimensionality reduction on the input dataset and returns the transformed dataset.

    """

    def __init__(self, num_components: int = 2, target: str = 'target') -> None:
        self.num_components = num_components
        self.target = target

    def extract(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Applies PCA dimensionality reduction on the input dataset and returns the transformed dataset.

        ### Args:
            dataset (pd.DataFrame): The input dataset to perform PCA on.

        ### Returns:
            pd.DataFrame: The transformed dataset after PCA dimensionality reduction.
        """
        pca = PCA(n_components=self.num_components)
        X_pca = pca.fit_transform(dataset)
        frame = pd.DataFrame(data=X_pca, columns=[
                             f'PC{i}' for i in range(1, self.num_components + 1)])
        if self.target in dataset.columns:
            frame[self.target] = dataset[self.target].astype('float64')
        return frame


class LDAExtractor(Extractor):
    """
    LDAExtractor is a class that performs Linear Discriminant Analysis (LDA) dimensionality reduction on a dataset.

    ## Parameters:
        num_components (int): The number of components to keep after dimensionality reduction. Default is 2.
        target (str): The name of the target variable in the dataset. Default is 'target'.

    ## Methods:
        extract(dataset: pd.DataFrame) -> pd.DataFrame:
            Applies LDA dimensionality reduction on the input dataset and returns the transformed dataset.

    """

    def __init__(self, num_components: int = 2, target: str = 'target') -> None:
        self.num_components = num_components
        self.target = target

    def extract(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Applies LDA dimensionality reduction on the input dataset and returns the transformed dataset.

        ### Args:
            dataset (pd.DataFrame): The input dataset to perform LDA on.

        ### Returns:
            pd.DataFrame: The transformed dataset after LDA dimensionality reduction.
        """
        lda = LDA(n_components=self.num_components)
        X_lda = lda.fit_transform(dataset.drop(
            columns=[self.target]), dataset[self.target])
        frame = pd.DataFrame(data=X_lda, columns=[
                             f'PC{i}' for i in range(1, self.num_components + 1)])
        if self.target in dataset.columns:
            frame[self.target] = dataset[self.target].astype('float64')
        return frame
