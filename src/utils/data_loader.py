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


class DataLoader:
    """
    Classe responsável pelo carregamento, pipeline de entrada e pré-processamento de datasets.
    
    Centraliza as operações de leitura de arquivos binários do NumPy (.npy), além de 
    aplicar transformações numéricas, como o achatamento de matrizes (flattening), 
    normalização de escala e codificação One-Hot.
    """

    @staticmethod
    def minmax_normalize(x: np.ndarray) -> np.ndarray:
        """Aplica a normalização MinMax para reescalar os dados entre [0, 1]."""
        # Obtém o menor valor de cada coluna (axis=0 = operação por coluna)
        x_min = x.min(axis=0)
        # Obtém o maior valor de cada coluna
        x_max = x.max(axis=0)

        # Calcula a amplitude (max - min) de cada coluna.
        # Se a amplitude for 0 (coluna com valores constantes),
        # substitui por 1 para evitar divisão por zero.
        range_val = np.where(x_max - x_min == 0, 1.0, x_max - x_min)

        # # Aplica a fórmula da normalização Min-Max: (valor - mínimo da coluna) / amplitude da coluna
        return (x - x_min) / range_val
    
    @staticmethod
    def standard_normalize(x: np.ndarray) -> np.ndarray:
        """Aplica a padronização estatística (Z-score: média 0, desvio padrão 1)."""
        # Calcula a média de cada coluna (feature)
        mean = x.mean(axis=0)

        # Calcula o desvio padrão de cada coluna
        std = x.std(axis=0)

        # Se alguma coluna possuir desvio padrão igual a zero
        # (todos os valores são iguais), substitui por 1 para
        # evitar divisão por zero na normalização.
        std = np.where(std == 0, 1.0, std)

        # Aplica a fórmula do Z-score:
        # (valor - média) / desvio padrão
        return (x - mean) / std
    
    @staticmethod
    def to_one_hot(y: np.ndarray) -> np.ndarray:
        """Converte um vetor de rótulos/classes discretas para codificação One-Hot Encoding."""
        #Remove dimensões de tamanho 1 e converte os valores para inteiro
        y_flat = y.squeeze().astype(int)
        #Obtém todas as classes distintas presentes no vetor
        unique_labels = np.unique(y_flat)
        n_classes = len(unique_labels)
        
        #Cria um mapeamento:
        # classe original -> índice da coluna no vetor One-Hot
        label_to_idx = {val: idx for idx, val in enumerate(sorted(unique_labels))}
        
        # Cria uma matriz preenchida com zeros
        y_one_hot = np.zeros((len(y_flat), n_classes))

        # Para cada amostra, coloca 1 na coluna correspondente à classe
        for i, val in enumerate(y_flat):
            y_one_hot[i, label_to_idx[val]] = 1.0
            
        return y_one_hot
    
    @staticmethod
    def load_from_npy(
        x_path: str,
        y_path: str,
        normalize_x: str = None,
        convert_to_one_hot: bool = False
    ) -> List[Tuple[List[float], List[float]]]:
        """
        Carrega dados e rótulos de arquivos .npy, aplicando normalizações e formatações opcionais.
        
        Este método lê as matrizes originais, achata as dimensões de entrada para um formato 
        vetorial bidimensional, realiza o reescalonamento dos atributos numéricos (MinMax ou 
        Standardization) para padronizar a magnitude e; opcionalmente, 
        transforma os alvos discretos em vetores binários (One-Hot Encoding).
        """

        #Carrega a matriz de atributos (features) do arquivo .npy
        x_raw = np.load(x_path)
        #Carrega os rótulos/classes do arquivo .npy
        y_raw = np.load(y_path)

        ## Reorganiza os dados para uma matriz bidimensional: (n_amostras, n_atributos)
        x_processed = x_raw.reshape(len(x_raw), -1).astype(np.float32)

        if normalize_x == "minmax":
            x_processed = DataLoader.minmax_normalize(x_processed)
        elif normalize_x == "standard":
            x_processed = DataLoader.standard_normalize(x_processed)

        # Converte os rótulos para One-Hot Encoding quando:
        # - o usuário solicitar explicitamente
        # - y for um vetor unidimensional (shape = (N,))
        # - y for uma matriz coluna (shape = (N,1))
        if convert_to_one_hot or len(y_raw.shape) == 1 or (len(y_raw.shape) == 2 and y_raw.shape[1] == 1):
            y_raw = DataLoader.to_one_hot(y_raw)

        dataset = []
        for i in range(len(x_processed)):
            #tolist() converte arrays NumPy para listas Python
            dataset.append((x_processed[i].tolist(), y_raw[i].tolist()))

    @staticmethod
    def load_character_from_alphabet(
        x_path: str,
        y_path: str
    ) -> List[Tuple[List[float], List[float]]]:
        """
        Função especializada para o carregamento do dataset de caracteres do Alfabeto
        para o Dataset CARACTERES_COMPLETO. Nesse caso não usa nenhum tipo de normalização
        ou one-hot-encoding.
        """
        return DataLoader.load_from_npy(x_path, y_path)