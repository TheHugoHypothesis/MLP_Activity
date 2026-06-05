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


""" Classe que faz carregamento de dataset
Aqui só foi feito para se adequar ao carregamento do dataset de Alfabeto
Mas em teoria qualquer outro carregamento de dataset pode ser feito aqui.
"""
class DataLoader:
    @staticmethod
    def load_from_npy(
        x_path: str,
        y_path: str,
        normalize_x: str = None,
        convert_to_one_hot: bool = False
    ) -> List[Tuple[List[float], List[float]]]:
        x_raw = np.load(x_path)
        y_raw = np.load(y_path)

        x_flat = x_raw.reshape(len(x_raw), -1).astype(np.float32)

        if normalize_x == "minmax":
            x_min = x_flat.min(axis=0)
            x_max = x_flat.max(axis=0)
            range_val = np.where(x_max - x_min == 0, 1.0, x_max - x_min)
            x_flat = (x_flat - x_min) / range_val
        elif normalize_x == "standard":
            mean = x_flat.mean(axis=0)
            std = x_flat.std(axis=0)
            std = np.where(std == 0, 1.0, std)
            x_flat = (x_flat - mean) / std

        if convert_to_one_hot or len(y_raw.shape) == 1 or (len(y_raw.shape) == 2 and y_raw.shape[1] == 1):
            y_flat = y_raw.squeeze().astype(int)
            unique_labels = np.unique(y_flat)
            n_classes = len(unique_labels)
            label_to_idx = {val: idx for idx, val in enumerate(sorted(unique_labels))}
            y_one_hot = np.zeros((len(y_flat), n_classes))
            for i, val in enumerate(y_flat):
                y_one_hot[i, label_to_idx[val]] = 1.0
            y_raw = y_one_hot

        dataset = []
        for i in range(len(x_flat)):
            dataset.append((x_flat[i].tolist(), y_raw[i].tolist()))

        return dataset

    @staticmethod
    def load_character_from_alphabet(
        x_path: str,
        y_path: str
    ) -> List[Tuple[List[float], List[float]]]:
        return DataLoader.load_from_npy(x_path, y_path)