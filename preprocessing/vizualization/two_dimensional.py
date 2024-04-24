import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simple_plot(data1: pd.Series, data2: pd.Series, y: pd.Series, save: str = None) -> None:
    """
    Generate a scatter plot comparing two data series.

    ### Parameters:
    - data1 (pd.Series): First data series to be plotted.
    - data2 (pd.Series): Second data series to be plotted.
    - y (pd.Series): Categorical labels for coloring the data points.
    - save (str, optional): File path to save the plot. If not provided, the plot will be displayed.

    ### Returns:
    None
    """
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
    if save:
        plt.savefig(save)
    else:
        plt.show()


def compare_tape_graphs(tape1: list[float], tape2: list[float], save: str = None) -> None:
    """
    Compare two tapes by plotting their accuracy over epochs.

    ### Parameters:
    tape1 (list[float]): The accuracy values for tape 1.
    tape2 (list[float]): The accuracy values for tape 2.
    save (str, optional): The file path to save the plot. If not provided, the plot will be displayed.

    ### Returns:
    None
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title('Comparison between two tapes', fontsize=20)
    ax.set_xlabel('No. Epoch', fontsize=15)
    ax.set_ylabel('Accuracy', fontsize=15)
    ax.grid()
    ax.plot(tape1, label='Tape 1')
    ax.plot(tape2, label='Tape 2')
    ax.legend()
    if save:
        plt.savefig(save)
    else:
        plt.show()
