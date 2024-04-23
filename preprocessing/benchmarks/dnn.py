import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
from sklearn import datasets, preprocessing
from sklearn.utils import Bunch
from ._benchmark_abc import BenchmarkABC


def _check_possibilities(x: list) -> int:
    s = set()
    for i in x:
        if i[1] in s:
            continue
        s.add(i[1])
    return len(s)


class DNNBenchmark(BenchmarkABC):
    """
    This benchmark contains small instance of Deep Neural Network.

    Neural Network contains linear hidden layer with 512 features (number of features can
    be changed in constructor) and ReLU normalization function.
    ## Parameters
    hidden_features : int, default : 512
        Number of features in linear hidden layer.
    """

    def __init__(self, hidden_features: int = 512) -> None:
        self._hidden_features = hidden_features
        self.tape_graph = []
        super().__init__()

    def evaluate(self, dataset: pd.DataFrame, target: str = 'target') -> float:
        dataset = dataset.select_dtypes(include=['number', 'bool']).dropna()
        params = dataset.drop([target], axis=1).to_numpy(dtype=np.float64)
        x = torch.tensor(params, dtype=torch.float)
        y = torch.tensor([int(v) for v in dataset[target]])
        dataset = data.TensorDataset(x, y)
        training, validation, test = data.random_split(dataset, [0.8, 0.15, 0.05],
                                                       generator=torch.Generator().manual_seed(42))
        model = self._create_model(
            len(training[0][0]), _check_possibilities(training))
        self._train_classifier(model, training, validation)
        logits = model(test[:][0])
        test_accuracy = self._compute_acc(logits, test[:][1]).item()
        return test_accuracy

    def _create_model(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, self._hidden_features),
            nn.ReLU(),
            nn.Linear(self._hidden_features, self._hidden_features),
            nn.ReLU(),
            nn.Linear(self._hidden_features, output_size))

    def _compute_acc(self, logits, expected):
        pred = logits.argmax(dim=1)
        return (pred == expected).type(torch.float).mean()

    def _train_classifier(self,
                          model: nn.Module,
                          training: data.Dataset,
                          validation: data.Dataset,
                          no_improvement: int = 20,
                          batch_size: int = 128,
                          max_epochs: int = 1000):
        opt = optim.Adam(model.parameters())
        best_acc = 0
        improvement_check = 0
        tape_graph = []
        for epoch in range(max_epochs):
            model.train()
            for X_batch, y_batch in data.DataLoader(training, batch_size=batch_size, shuffle=True):
                opt.zero_grad()
                logits = model(X_batch)
                loss = torch.nn.CrossEntropyLoss()(logits, y_batch)
                loss.backward()
                opt.step()
            model.eval()
            logits = model(validation[:][0])
            acc = self._compute_acc(logits, validation[:][1]).item()
            tape_graph.append(acc)
            if acc > best_acc:
                best_acc = acc
                improvement_check = 0
                continue
            improvement_check += 1
            if improvement_check >= no_improvement:
                break
            if epoch > max_epochs:
                break
        self.tape_graph = tape_graph


def main():
    import matplotlib.pyplot as plt
    from time import perf_counter
    org_dataset = pd.read_csv('spaceship-titanic/train.csv')[:1500]
    dataset = org_dataset.copy()
    dnn = DNNBenchmark(512)
    start = perf_counter()
    print('Before normalization:', dnn.evaluate(
        dataset, 'Transported'), 'Time:', perf_counter() - start)
    tape_graph = dnn.tape_graph
    plt.plot(tape_graph)
    from preprocessing.preprocessing import Preprocessing
    from preprocessing.selector import VARSelector
    dataset = org_dataset.copy()
    preprocessing = Preprocessing(
        dataset, target='Transported',  one_hot_threshold=0.1)
    start = perf_counter()
    print('After normalization', dnn.evaluate(
        preprocessing.preprocess(), 'Transported'), 'Time:', perf_counter() - start)

    tape_graph = dnn.tape_graph
    plt.plot(tape_graph)
    dataset = org_dataset.copy()
    new_data = pd.DataFrame()
    new_data['a'] = dataset['Transported']
    new_data['Transported'] = dataset['Transported']

    preprocessing = Preprocessing(
        new_data, target='Transported', selector=VARSelector(),  one_hot_threshold=0.1)
    start = perf_counter()
    print('After normalization with VARSelector', dnn.evaluate(
        new_data, 'Transported'), 'Time:', perf_counter() - start)

    tape_graph = dnn.tape_graph
    plt.plot(tape_graph)
    plt.legend(['Before preprocessing',
               'After preprocessing',
                'After preprocessing with VARSelection'])
    plt.title('DNN Benchmark accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('dnn.png')


if __name__ == '__main__':
    main()
