import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from ._benchmark_abc import BenchmarkABC


class ClassifierBenchmark(BenchmarkABC):
    """
    Classifier benchmarking class.
    Splits class into train and test data, trains classifier on train data and evaluates it on test data.
    """

    def __init__(self, printing: bool = True) -> None:
        self.printing = printing

    def evaluate(self, dataset: pd.DataFrame, target: str = 'target') -> float:

        train_size = int(len(dataset) * 0.7)
        train_data = dataset.sample(n=train_size, random_state=82)
        test_data = dataset.drop(train_data.index)
        classifier = SVC()
        y = train_data[target].copy()
        y = y.to_numpy()
        X = train_data.copy().drop(target, axis=1)
        X = X.select_dtypes(include=['number', 'bool'])
        X = X.fillna(0)
        X = X.to_numpy()
        classifier.fit(X, y)

        y_test = test_data[target].copy()
        y_test = y_test.to_numpy()
        X_test = test_data.copy().drop([target], axis=1)
        X_test = X_test.select_dtypes(include=['number', 'bool'])
        X_test = X_test.fillna(0)
        X_test = X_test.to_numpy()

        accuracy = classifier.score(X_test, y_test)

        if self.printing:
            predicts = classifier.predict(X_test)
            print(classification_report(y_test, predicts))
        return accuracy
