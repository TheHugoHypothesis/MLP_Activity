"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

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