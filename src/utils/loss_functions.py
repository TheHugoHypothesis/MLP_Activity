from abc import ABC, abstractmethod
from typing import List
import math

"""
Classe abstrata que condensa funções de erro, tendo:
- compute(...): calcula o erro total, dado duas listas de valores (preditos e reais)
- derivative(...): calcula a derivada do erro para cada par, dado duas listas de valores (preditos e reais)
"""
class LossFunction(ABC):
    @abstractmethod
    def compute(
        self,
        y_pred: List[float],
        y_real: List[float]
    ) -> float:
        pass

    @abstractmethod
    def derivative(
        self,
        y_pred: List[float],
        y_real: List[float]
    ) -> List[float]:
        pass

class MSE(LossFunction):
    def compute(
        self,
        y_pred: List[float], 
        y_real: List[float]
    ) -> float:
        total_error: float = 0.0
        
        #A função zip() pega par-a-par de cada lista como elementos de `prediction` e `real`
        for prediction, real in zip(y_pred, y_real):
            total_error += (real - prediction) ** 2

        #Normaliza o erro total
        return total_error / len(y_pred)

    def derivative(
        self,
        y_pred: List[float],
        y_real: List[float]
    ) -> List[float]:
        error_list: List[float] = []
        list_size = len(y_pred)

        for prediction, real in zip(y_pred, y_real):
            error_list.append((2.0 / list_size) * (prediction - real))

        return error_list
