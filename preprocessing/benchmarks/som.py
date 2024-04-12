import pandas as pd
import simpsom as sps
from _benchmark_abc import BenchmarkABC
from sklearn_som.som import SOM


class SOMBenchmark(BenchmarkABC):
    def __init__(self) -> None:
        self.test = 123
        super().__init__()

    def evaluate(self, dataset: pd.DataFrame) -> float:
        net = sps.SOMNet(10, 10, dataset, init='random', GPU=True)
        net.train()
        net.plot_convergence(show=True, print_out=True)
        # som = SOM(m=10, n=10, dim=13)
        # som.fit(dataset)
        # reds = som.predict(dataset)
        return 1.0


if __name__ == '__main__':
    benchmark = SOMBenchmark()
    spaceship = pd.read_csv('spaceship-titanic/train.csv')
    spaceship_x = spaceship.iloc[:, :-1]
    spaceship_x.drop(['PassengerId', 'HomePlanet', 'Cabin', 'Destination', 'Name'], axis=1, inplace=True)
    spaceship_x.dropna(inplace=True)
    spaceship_y = spaceship.iloc[:, -1]
    benchmark.evaluate(spaceship_x)
