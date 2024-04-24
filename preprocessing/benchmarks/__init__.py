"""
The :mod:`preprocessing.benchmarks` subpackage includes several benchmarks for testing either
the data preprocessing gives promissing results or not.

Curently implemented benchmarks:
- ClassifierBenchmark - Simple classification algorithm.
- DNNBenchmark - Deep Neural Network algorithm.
- SOMBenchmark - Self-Organizing tree algorithm.

## Usage:
>>> from preprocessing.benchmarks import DNNBenchmark
>>> dataset = pd.read_csv('your_dataset.csv')
>>> dnn = DNNBenchmark()
>>> dnn.evaluate(dataset, target='Your Target')
"""
from .classifier import ClassifierBenchmark
from .dnn import DNNBenchmark
from .som import SOMBenchmark
