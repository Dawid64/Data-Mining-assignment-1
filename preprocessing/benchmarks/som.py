import pandas as pd
import simpsom as sps
from _benchmark_abc import BenchmarkABC
from sklearn_som.som import SOM


class SOMBenchmark(BenchmarkABC):
    def __init__(self) -> None:
        self.test = 123
        super().__init__()

    def evaluate(self, dataset: pd.DataFrame) -> float:
        dataset_x = dataset.iloc[:, :-1]
        print(dataset_x)
        # labels = dataset.iloc[:, -1]
        # net = sps.SOMNet(10, 10, dataset_x, init='random', PBC=True, GPU=False)
        # net.train(epochs=100)
        # net.plot_convergence(show=True, print_out=True)
        # net.save('file', out_path='./')
        # position_node0 = net.node_list[0].pos
        # weights_node0 = net.node_list[0].weights
        # net.nodes_graph(colnum=0, out_path='./')
        # net.diff_graph(out_path='./')
        # net.project(dataset_x, out_path='./')
        # net.cluster(dataset_x)
        som = SOM(m=10, n=10, dim=13)
        som.fit(dataset)
        pred = som.predict(dataset_x)
        return 1.0


if __name__ == '__main__':
    benchmark = SOMBenchmark()
    spaceship = pd.read_csv('spaceship-titanic/train.csv')
    spaceship = spaceship.iloc[:, :-1]
    spaceship.drop(['PassengerId', 'HomePlanet', 'Cabin', 'Destination', 'Name', 'CryoSleep', 'VIP'], axis=1,
                   inplace=True)
    spaceship.dropna(inplace=True)
    benchmark.evaluate(spaceship)
