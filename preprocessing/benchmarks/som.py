import random

import pandas as pd
import simpsom as sps
from _benchmark_abc import BenchmarkABC
from sklearn_som.som import SOM


class SOMBenchmark(BenchmarkABC):
    def __init__(self) -> None:
        self.test = 123
        super().__init__()

    def evaluate(self, dataset: pd.DataFrame) -> float:
        train, test = dataset.iloc[:int(dataset.shape[0] * 4 / 5), :], dataset.iloc[int(dataset.shape[0] * 4 / 5):, :]
        train_x, _ = train.iloc[:, :-1].to_numpy(), train.iloc[:, -1].to_numpy()
        test_x, test_y = test.iloc[:, :-1].to_numpy(), test.iloc[:, -1].to_numpy()
        som = SOM(m=2, n=1, dim=8)
        som.fit(train_x)
        preds = som.predict(test_x)
        return sum(abs(preds-test_y))/len(preds)


if __name__ == '__main__':
    benchmark = SOMBenchmark()
    spaceship = pd.read_csv('spaceship-titanic/train.csv')
    spaceship.drop(['PassengerId', 'HomePlanet', 'Cabin', 'Destination', 'Name'], axis=1,
                   inplace=True)
    spaceship.dropna(inplace=True)
    spaceship['CryoSleep'] = spaceship['CryoSleep'].astype(int)
    spaceship['VIP'] = spaceship['VIP'].astype(int)
    spaceship['Transported'] = spaceship['Transported'].astype(int)
    print(benchmark.evaluate(spaceship))
