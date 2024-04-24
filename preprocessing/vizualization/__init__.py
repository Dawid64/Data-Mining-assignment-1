"""

The :mod:`preprocessing.vizualization` subpackage includes several visualization functions for analyzing and visualizing data.

Currently implemented visualization functions:
- covariance_heatmap: Generates a heatmap to visualize the covariance between variables in a dataset.
- pair_plot: Generates a pair plot to visualize the relationships between variables in a dataset.
- simple_plot: Generates a simple plot to visualize a relationship between 2 variables in a dataset.
- compare_tape_graphs: Generates a plot to comparing the performance of two models over time.

## Usage:
>>> from preprocessing.vizualization import covariance_heatmap, pair_plot, simple_plot
>>> dataset = pd.read_csv('your_dataset.csv')
>>> covariance_heatmap(dataset)
>>> pair_plot(dataset)
>>> simple_plot(dataset['variable_name'], dataset['variable2_name'], dataset['target'])
"""


from .data_analysis import covariance_heatmap, pair_plot
from .two_dimensional import simple_plot, compare_tape_graphs
