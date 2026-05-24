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

class MAE(LossFunction):
    def compute(self, y_pred: List[float], y_real: List[float]) -> float:
        total_error = sum(abs(yp - yr) for yp, yr in zip(y_pred, y_real))
        return total_error / len(y_pred)
    
    def derivative(self, y_pred: List[float], y_real: List[float]) -> List[float]:
        n = len(y_pred)
        error_list = []
        for yp, yr in zip(y_pred, y_real):
            # Derivada do valor absoluto
            if yp > yr:
                error_list.append(1.0 / n)
            elif yp < yr:
                error_list.append(-1.0 / n)
            else:
                # Derivada indefinida no ponto 0, aproximada para 0
                error_list.append(0.0) 
        return error_list

class CategoricalCrossEntropy(LossFunction):
    def compute(self, y_pred: List[float], y_real: List[float]) -> float:
        epsilon = 1e-15 # Pequena constante para estabilidade numérica
        loss = 0.0
        for yp, yr in zip(y_pred, y_real):
            # Clipando y_pred para o intervalo [epsilon, 1 - epsilon]
            yp_clipped = max(epsilon, min(1.0 - epsilon, yp))
            loss -= yr * math.log(yp_clipped)
        return loss
    def derivative(self, y_pred: List[float], y_real: List[float]) -> List[float]:
        epsilon = 1e-15
        error_list = []
        for yp, yr in zip(y_pred, y_real):
            yp_clipped = max(epsilon, min(1.0 - epsilon, yp))
            # Derivada em relação a yp: - yr / yp
            error_list.append(-yr / yp_clipped)
        return error_list