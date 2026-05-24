import numpy as np
from typing import List, Tuple

Dataset = List[Tuple[List[float], List[float]]]


class ConfusionMatrix:
    @staticmethod
    def compute(dataset: Dataset, model, num_classes: int):
        matrix = np.zeros((num_classes, num_classes), dtype=int)

        for x, y in dataset:
            pred = int(np.argmax(model.forward(x)))
            true = int(np.argmax(y))

            matrix[true][pred] += 1

        return matrix