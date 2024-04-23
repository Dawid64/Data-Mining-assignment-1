from abc import ABC
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class Extractor(ABC):
    def extract(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset


class PCAExtractor(Extractor):

    def __init__(self, num_components: int = 2, target: str = 'target') -> None:
        self.num_components = num_components
        self.target = target

    def extract(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pca = PCA(n_components=self.num_components)
        X_pca = pca.fit_transform(dataset)
        frame = pd.DataFrame(data=X_pca, columns=[
                             f'PC{i}' for i in range(1, self.num_components + 1)])
        if self.target in dataset.columns:
            frame[self.target] = dataset[self.target].astype('float64')
        return frame


class LDAExtractor(Extractor):

    def __init__(self, num_components: int = 2, target: str = 'target') -> None:
        self.num_components = num_components
        self.target = target

    def extract(self, dataset: pd.DataFrame) -> pd.DataFrame:
        lda = LDA(n_components=self.num_components)
        X_lda = lda.fit_transform(dataset.drop(
            columns=[self.target]), dataset[self.target])
        frame = pd.DataFrame(data=X_lda, columns=[
                             f'PC{i}' for i in range(1, self.num_components + 1)])
        if self.target in dataset.columns:
            frame[self.target] = dataset[self.target].astype('float64')
        return frame
