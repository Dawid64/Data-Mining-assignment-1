from abc import ABC
import pandas as pd
from sklearn.decomposition import PCA


class Extractor(ABC):
    def extract(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset


class PCAExtractor(Extractor):

    def __init__(self, num_components: int = 2) -> None:
        self.num_components = num_components

    def extract(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pca = PCA(n_components=self.num_components)
        X_pca = pca.fit_transform(dataset)
        frame = pd.DataFrame(data=X_pca, columns=[
                             f'PC{i}' for i in range(1, self.num_components + 1)])
        return frame
