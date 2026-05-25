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
    SOFT_MAX_EPSILON = 1e-15

    def _softmax(self, logits: List[float]) -> List[float]:
        #estabilização numérica do softmax
        #(evita overflow no exp), isso é feito pelas operações
        #de fazer shift em relação ao máximo da lista por cada elemento (l - max_logit)
        #e tirar exp(...), seguido da soma dos elementos dessa lista.
        max_logit = max(logits)

        exp_values = []
        for l in logits:
            exp_values.append(math.exp(l - max_logit))
        sum_exp = sum(exp_values)

        #calcula agora as probabilidades por normalizar os exponenciais anteriores
        #pela soma deles (transforma em distribuiçaão de probabilidade)
        probabilities = []
        for v in exp_values:
            probabilities.append(v / sum_exp)

        return probabilities

    def compute(self, y_pred: List[float], y_real: List[float]) -> float:
        #`y_pred` são os logits que é saída bruta da rede (antes do softmax)
        #nao representam probabilidade, i.e, são o campo local induzido da última camada
        #da rede
        probs = self._softmax(y_pred)

        loss = 0.0

        for predicted, target in zip(probs, y_real):

            # evita log(0)
            clipped = predicted
            if clipped < epsilon:
                clipped = epsilon
            elif clipped > 1.0 - epsilon:
                clipped = 1.0 - epsilon

            loss -= target * math.log(clipped)

        return loss

    def derivative(self, y_pred: List[float], y_real: List[float]) -> List[float]:
        # gradiente da softmax + cross entropy (em relação aos logits)

        probs = self._softmax(y_pred)

        gradients = []
        for predicted, target in zip(probs, y_real):
            gradients.append(predicted - target)

        return gradients