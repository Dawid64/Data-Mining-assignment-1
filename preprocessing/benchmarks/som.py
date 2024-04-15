import numpy as np
import pandas as pd
from numpy.ma.core import ceil
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.preprocessing import MinMaxScaler
from _benchmark_abc import BenchmarkABC


class SOMBenchmark(BenchmarkABC):
    def __init__(self, num_rows=10, num_cols=10, max_m_distance=4, max_learning_rate=0.5,
                 max_steps=int(7.5 * 10e3)) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.max_m_distance = max_m_distance
        self.max_learning_rate = max_learning_rate
        self.max_steps = max_steps
        super().__init__()

    # Euclidean distance
    @staticmethod
    def e_distance(x, y):
        return distance.euclidean(x, y)

    # Manhattan distance
    @staticmethod
    def m_distance(x, y):
        return distance.cityblock(x, y)

    # Best Matching Unit search
    def winning_neuron(self, data, t, som):
        winner = [0, 0]
        shortest_distance = np.sqrt(data.shape[1])  # initialise with max distance
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                distance = self.e_distance(som[row][col], data[t])
                if distance < shortest_distance:
                    shortest_distance = distance
                    winner = [row, col]
        return winner

    # Learning rate and neighbourhood range calculation
    def decay(self, step):
        coefficient = 1.0 - (np.float64(step) / self.max_steps)
        learning_rate = coefficient * self.max_learning_rate
        neighbourhood_range = ceil(coefficient * self.max_m_distance)
        return learning_rate, neighbourhood_range

    def predict(self, label_map, test_x, som):
        # test data
        # using the trained som, search the winning node of corresponding to the test data
        # get the label of the winning node
        winner_labels = []
        for t in range(test_x.shape[0]):
            winner = self.winning_neuron(test_x, t, som)
            row = winner[0]
            col = winner[1]
            predicted = label_map[row][col]
            winner_labels.append(predicted)
        return winner_labels

    def construct_map(self, map):
        # construct label map by choosing the most popular label among collected ones
        label_map = np.zeros(shape=(self.num_rows, self.num_cols), dtype=np.int64)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                label_list = map[row][col]
                if len(label_list) == 0:
                    label = 2
                else:
                    label = max(label_list, key=label_list.count)
                label_map[row][col] = label
        title = ('Iteration ' + str(self.max_steps))
        cmap = colors.ListedColormap(['tab:green', 'tab:red', 'tab:orange'])
        plt.imshow(label_map, cmap=cmap)
        plt.colorbar()
        plt.title(title)
        plt.show()
        return label_map

    def collect_labels(self, train_x, train_y, som):
        # collecting labels
        label_data = train_y
        map = np.empty(shape=(self.num_rows, self.num_cols), dtype=object)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                map[row][col] = []  # empty list to store the label
        for t in range(train_x.shape[0]):
            winner = self.winning_neuron(train_x, t, som)
            map[winner[0]][winner[1]].append(label_data[t])  # label of winning neuron
        return map

    def train(self, train_x, som):
        # start training iterations
        for step in range(self.max_steps):
            learning_rate, neighbourhood_range = self.decay(step)
            t = np.random.randint(0, high=train_x.shape[0])  # random index of training data
            winner = self.winning_neuron(train_x, t, som)
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    if self.m_distance([row, col], winner) <= neighbourhood_range:
                        som[row][col] += learning_rate * (
                                train_x[t] - som[row][col])  # update neighbour's weight

    def evaluate(self, dataset: pd.DataFrame) -> float:
        data_x = dataset.iloc[:, :-1].to_numpy()
        data_y = dataset.iloc[:, -1].to_numpy()
        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
        num_dims = train_x.shape[1]  # number of dimensions in the input data
        som = np.random.random_sample(size=(self.num_rows, self.num_cols, num_dims))
        self.train(train_x, som)
        map = self.collect_labels(train_x, train_y, som)
        label_map = self.construct_map(map)
        winner_labels = self.predict(label_map, test_x, som)
        return accuracy_score(test_y, np.array(winner_labels))


if __name__ == '__main__':
    benchmark = SOMBenchmark()
    spaceship = pd.read_csv('../../spaceship-titanic/train.csv')

    # The simplest preprocessing just to make the algorithm work
    spaceship.drop(['PassengerId', 'HomePlanet', 'Cabin', 'Destination', 'Name'], axis=1, inplace=True)
    spaceship.dropna(inplace=True)
    spaceship['CryoSleep'] = spaceship['CryoSleep'].astype(int)
    spaceship['VIP'] = spaceship['VIP'].astype(int)
    spaceship['Transported'] = spaceship['Transported'].astype(int)

    print(f"Before normalization: {benchmark.evaluate(spaceship)}")
    scaler = MinMaxScaler()
    spaceship = pd.DataFrame(scaler.fit_transform(spaceship))
    print(f"After normalization: {benchmark.evaluate(spaceship)}")
