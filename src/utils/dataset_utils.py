import random
import numpy as np
from typing import List, Tuple

Dataset = List[Tuple[List[float], List[float]]]

"""
Classe que lida com pré-processamento do dataset. 
Pode ser usada para implementar K-Fold depois
"""
class DatasetUtils:
    @staticmethod
    def stratified_split(
        dataset: Dataset,
        p_train: float = 0.7,
        p_val: float = 0.15
    ):
        """
        Split estratificado manual (sem defaultdict).
        Mantém proporção de classes em treino/val/teste.
        """

        #Primeiro agrupamos os dados de acordo com a classe (letra) deles
        classes = {}

        for x, y in dataset:
            class_id = int(np.argmax(y))

            if class_id not in classes:
                classes[class_id] = []

            classes[class_id].append((x, y))

        train_set = []
        val_set = []
        test_set = []

        #então separamos proporcionalmente a quantidade de cada letra que vai para cada conjunto
        for class_id in classes:
            samples = classes[class_id]
            random.shuffle(samples)

            n_total = len(samples)
            n_train = int(n_total * p_train)
            n_val = int(n_total * (p_train + p_val))

            train_set.extend(samples[:n_train])
            val_set.extend(samples[n_train:n_val])
            test_set.extend(samples[n_val:])

        #embaralhar resultado final
        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)

        return train_set, val_set, test_set