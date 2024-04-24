"""
The :mod:`preprocessing` package includes tools for basic data preprocessing including data standarization, feature selection and feature extraction.
As well as subpackages for benchmarking the quality of dataset and vizualization of the data.

## It includes:
- feature extraction:
    - PCAExtractor
    - LDAExtractor
- feature selection:
    - VARSelector
- preprocessing:
    - Preprocessing
    - SpaceShipPreprocessing
- benchmarks (subpackage for benchmarking)
- vizualization (subpackage for vizualizations)

## Usage:
>>> import pandas as pd
>>> import preprocessing as pr
>>> dataset = pd.read_csv('Your_dataset.csv')
>>> selector = pr.VARSelector()
>>> extractor = pr.PCAExtractor(num_components=8, target='Your_Target')
>>> preprocessing = pr.Preprocessing(dataset, target='Your_Target',
>>>                                  selector=selector, extractor=extractor)
>>> new_dataset = preprocessing.preprocess()
>>> dnn = pr.benchmarks.DNNBenchmark()
>>> dnn.evaluate(new_dataset, target='Your_Target')
"""
from . import vizualization, benchmarks
from .extractor import PCAExtractor, LDAExtractor
from .selector import VARSelector
from .preprocessing import Preprocessing, SpaceShipPreprocessing
