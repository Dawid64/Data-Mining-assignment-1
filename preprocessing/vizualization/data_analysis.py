import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def covariance_heatmap(data: pd.DataFrame) -> None:
    columns = data.select_dtypes(include=['number', 'bool']).dropna()
    X = StandardScaler().fit_transform(columns)
    n = len(columns)
    C = np.dot(X.T, X)/(n-1)
    plt.figure(figsize=(4, 4))
    sns.heatmap(C, xticklabels=columns.columns,
                yticklabels=columns.columns, cmap='magma')
    plt.show()
