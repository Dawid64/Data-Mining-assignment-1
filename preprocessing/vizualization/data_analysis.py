import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def covariance_heatmap(dataset: pd.DataFrame, save: str = None) -> None:
    columns = dataset.select_dtypes(include=['number', 'bool']).dropna()
    X = StandardScaler().fit_transform(columns)
    n = len(columns)
    C = np.dot(X.T, X)/(n-1)
    plt.figure(figsize=(4, 4))
    sns.heatmap(C, xticklabels=columns.columns,
                yticklabels=columns.columns, cmap='magma')
    if save:
        plt.savefig(save)
    else:
        plt.show()


def pair_plot(dataset: pd.DataFrame, target: str = 'target', save: str = None) -> None:
    if target not in dataset:
        target = None
    sns.pairplot(dataset, hue=target)
    if save:
        plt.savefig(save)
    else:
        plt.show()
