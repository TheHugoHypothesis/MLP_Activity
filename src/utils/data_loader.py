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
    def load_character_from_alphabet(
        x_path: str,
        y_path: str
    ) -> List[Tuple[List[float]]]:
        x_raw = np.load(x_path)
        y_raw = np.load(y_path)

        x_flat = x_raw.reshape(len(x_raw), -1)

        dataset = []
        for i in range(len(x_flat)):
            dataset.append((x_flat[i].tolist(), y_raw[i].tolist()))

        return dataset