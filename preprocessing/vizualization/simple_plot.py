import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simple_plot(data1: pd.Series, data2: pd.Series, y: pd.Series):
    y = y.astype(int)
    colrs = np.array(['red', 'green', 'blue', 'pink', 'yellow',
                      'black', 'purple', 'orange', 'brown', 'gray'])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(
        f'Comparison between {data1.name} and {data2.name}', fontsize=20)
    ax.set_xlabel(data1.name, fontsize=15)
    ax.set_ylabel(data2.name, fontsize=15)
    ax.grid()
    ax.scatter(data1, data2, c=colrs[y], s=20)
    plt.show()
