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

class SoftmaxCrossEntropy(LossFunction):
    def _softmax(self, logits: List[float]) -> List[float]:
        # Estabilidade numérica: subtrai o máximo para evitar overflow do exp()
        max_val = max(logits)
        exp_shifted = [math.exp(li - max_val) for li in logits]
        sum_exp = sum(exp_shifted)
        return [e / sum_exp for e in exp_shifted]
    def compute(self, y_pred: List[float], y_real: List[float]) -> float:
        # y_pred aqui são os logits brutos da camada Linear
        probs = self._softmax(y_pred)
        
        # Cross-Entropy com proteção para log(0)
        epsilon = 1e-15
        loss = 0.0
        for p, yr in zip(probs, y_real):
            p_clipped = max(epsilon, min(1.0 - epsilon, p))
            loss -= yr * math.log(p_clipped)
        return loss
    def derivative(self, y_pred: List[float], y_real: List[float]) -> List[float]:
        # y_pred são os logits brutos da camada Linear
        probs = self._softmax(y_pred)
        
        # A derivada fundida de Softmax + Cross-Entropy em relação aos LOGITS é: (probs - y_real)
        return [p - yr for p, yr in zip(probs, y_real)]