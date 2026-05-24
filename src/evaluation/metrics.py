import numpy as np
from typing import List, Tuple

Dataset = List[Tuple[List[float], List[float]]]


class Metrics:
    @staticmethod
    def mse(y_pred: List[float], y_true: List[float]) -> float:
        error = 0.0

        for yp, yt in zip(y_pred, y_true):
            error += (yt - yp) ** 2

        return error / len(y_pred)


    @staticmethod
    def accuracy(y_pred: List[float], y_true: List[float]) -> float:
        pred_class = int(np.argmax(y_pred))
        true_class = int(np.argmax(y_true))

        if (pred_class == true_class):
            return 1.0
        return 0.0

    @staticmethod
    def batch_accuracy(dataset: Dataset, model) -> float:
        correct = 0

        for x, y in dataset:
            out = model.forward(x)

            if np.argmax(out) == np.argmax(y):
                correct += 1

        return correct / len(dataset)