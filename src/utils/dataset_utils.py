"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

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
    
    @staticmethod
    def stratified_k_fold_split(
        dataset: Dataset,
        k: int
    ):
        """
        Método de apoio para o K-Fold estratificado.
        Retorna uma lista de tuplas (train_set, val_set) para cada fold.
        """
                #Primeiro agrupamos os dados de acordo com a classe (letra) deles
        classes = {}

        for x, y in dataset:
            class_id = int(np.argmax(y))

            if class_id not in classes:
                classes[class_id] = []

            classes[class_id].append((x, y))

        # Particiona cada classe em k pedaços
        class_folds = {}
        for cid, samples in classes.items():
            random.shuffle(samples)

            n = len(samples)
            base = n // k
            remainder = n % k

            folds_for_class = []
            idx = 0
            # Se houver sobra, distribui 1 unidade extra para os primeiros folds
            for i in range(k):
                size = base + (1 if i < remainder else 0)
                folds_for_class.append(samples[idx: idx + size])
                idx += size
            class_folds[cid] = folds_for_class

        # Constrói os k folds combinando partes de cada classe
        folds = []
        for i in range(k):
            val = []
            train = []
            for cid in class_folds:
                for j, part in enumerate(class_folds[cid]):
                    if j == i:
                        val.extend(part)
                    else:
                        train.extend(part)
            random.shuffle(train)
            random.shuffle(val)
            folds.append((train, val))
        return folds