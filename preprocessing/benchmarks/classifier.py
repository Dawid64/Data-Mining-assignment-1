from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
from ._benchmark_abc import BenchmarkABC
from preprocessing import *


class Classifier(BenchmarkABC):
    def __init__(self, printing: bool = True) -> None:
        self.printing = printing

    def evaluate(self, dataset: pd.DataFrame, target: str = 'target') -> float:

        train_size = int(len(dataset) * 0.7)
        train_data = dataset.sample(n=train_size, random_state=82)
        a = dataset.sample(n=train_size, random_state=82)
        assert (train_data == a).all().all()
        test_data = dataset.drop(train_data.index)

        classifier = SVC()
        # pd.options.mode.copy_on_write = True

        X = train_data.copy().drop(target, axis=1)
        X = X.select_dtypes(include='number')
        X = X.fillna(0)
        print(X.head())
        X = X.to_numpy()
        y = train_data[target]

        y = y.to_numpy()
        classifier.fit(X, y)

        X_test = test_data.copy().drop([target], axis=1)
        X_test = X_test.select_dtypes(include='number')
        X_test = X_test.fillna(0)
        X_test = X_test.to_numpy()
        y_test = test_data[target]
        y_test = y_test.to_numpy()

        predicts = classifier.predict(X_test)

        if self.printing:
            print(classification_report(y_test, predicts))
        return 0.0


if __name__ == '__main__':
    benchmark = Classifier()
    spaceship = pd.read_csv('spaceship-titanic/train.csv')

    # The simplest preprocessing just to make the algorithm work
    spaceship.dropna(inplace=True)
    spaceship['CryoSleep'] = spaceship['CryoSleep'].astype(int)
    spaceship['VIP'] = spaceship['VIP'].astype(int)
    spaceship['Transported'] = spaceship['Transported'].astype(int)

    print(spaceship.head())

    print(
        f"Before normalization: {benchmark.evaluate(spaceship, 'Transported')}")

    preprocess = Preprocessing(spaceship, target='Transported',
                               extractor=PCAExtractor(2))
    new_data = preprocess.preprocess()
    new_data['Transported'] = new_data['Transported'].astype(int)
    print(
        f"Before normalization: {benchmark.evaluate(new_data, 'Transported')}")
