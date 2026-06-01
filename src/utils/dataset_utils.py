"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

import random
from typing import List, Tuple

Dataset = List[Tuple[List[float], List[float]]]

"""
Classe que lida com pré-processamento do dataset. 
"""
class DatasetUtils:
    @staticmethod
    def stratified_split(
        dataset: Dataset,
        p_train: float = 0.7,
        p_val: float = 0.15
    ):
        """
        Split estratificado manual (holdout)
        """

        #Primeiro agrupamos os dados de acordo com a classe (letra) deles
        classes = {}

        for x, y in dataset:
            class_id = y.index(max(y))

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
            class_id = y.index(max(y))

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
    
    @staticmethod
    def merge_classes(dataset: Dataset, merge_groups: List[List[str]]) -> Tuple[Dataset, int]:
        """
        Mescla grupos de caracteres em classes únicas.
        Ex: merge_groups = [["D", "O"], ["I", "J"]]
        Ajusta automaticamente os One-Hot targets de 26 elementos para o novo número de classes.
        """
        #converte letras para indices
        groups_idx = []
        for group in merge_groups:
            groups_idx.append([ord(char.upper()) - 65 for char in group])
        
        old_to_new = {}
        current_new_idx = 0
        mapped_old_indices = set()

        #mapeamento de classes unificadas
        for group in groups_idx:
            for old_idx in group:
                old_to_new[old_idx] = current_new_idx
                mapped_old_indices.add(old_idx)
            current_new_idx += 1

        #mapeia classes restantes
        for old_idx in range(26):
            if old_idx not in mapped_old_indices:
                old_to_new[old_idx] = current_new_idx
                current_new_idx += 1
        
        new_num_classes = current_new_idx
        modified_dataset = []
        
        #reconstroi vetores do dataset p nova dimensão reduzida
        for x, y in dataset:
            old_class_idx = y.index(max(y))
            new_class_idx = old_to_new[old_class_idx]

            new_y = [0.0] * new_num_classes
            new_y[new_class_idx] = 1.0
            modified_dataset.append((x, new_y))
        return modified_dataset, new_num_classes

    @staticmethod
    def random_split(
        dataset: Dataset,
        p_train: float = 0.7,
        p_val: float = 0.15
    ):
        """Split puramente aleatório, ignorando a estratificação de classes."""
        samples = list(dataset)
        random.shuffle(samples)

        n_total = len(samples)
        n_train = int(n_total * p_train)
        n_val = int(n_total * (p_train + p_val))

        train_set = samples[:n_train]
        val_set = samples[n_train:n_val]
        test_set = samples[n_val:]

        return train_set, val_set, test_set
    
    @staticmethod
    def random_k_fold_split(dataset: Dataset, k: int) -> List[Tuple[Dataset, Dataset]]:
        """Gera folds de K-Fold de forma puramente aleatória, sem estratificação."""
        samples = list(dataset)
        random.shuffle(samples)

        n = len(samples)
        base = n // k
        remainder = n % k

        slices = []
        idx = 0
        for i in range(k):
            size = base + (1 if i < remainder else 0)
            slices.append(samples[idx: idx + size])
            idx += size
        
        folds = []
        for i in range(k):
            val = slices[i]
            train = []
            for j in range(k):
                if j != i:
                    train.extend(slices[j])
            random.shuffle(train)
            random.shuffle(val)
            folds.append((train, val))
        
        return folds