"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from abc import ABC, abstractmethod
from typing import List
import math

"""
Classe abstrata que condensa funções de erro, tendo:
- compute(...): calcula o erro total, dado duas listas de valores (preditos e reais). Retonra um
número real, que é o erro total
- derivative(...): calcula a derivada do erro para cada par, dado duas listas de valores (preditos e reais)
retorna o gradiente da função de erro em relação às predições, informando a direção em que os pesos devem ser atualizados
durante o backpropagation.
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
        
        #A função zip() pega par-a-par de cada lista como elementos de prediction e real
        for prediction, real in zip(y_pred, y_real):
            total_error += (real - prediction) ** 2

        #Normaliza o erro total
        return total_error / len(y_pred)

    #Calcula a derivada do MSE, 2/n * (y_previsto - y_real) em relação a cada saida da rede
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
        #Normaliza o erro total
        return total_error / len(y_pred)

    #Calcula a derivada da função módulo em relação a x normalizada.
    #basicamente, se yp (o predito) for maior que yr (o real), retorna 1/n
    #se o yp < yr, retorna -1/n 
    #se for igual retorna 0.
    #basicamente, o MAE não aumenta proporcional ao erro, mas somente indica a direção da correção
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

"""
Função de perda p/ problemas de classificação multiclasse. Implementa 2 operações
(i) Softmax: converte os logits (saída bruta da última camada da rede) em distribuição de probabilidade pela fórmula
P_i = exp(z_i) / Σ exp(z_j)
- em que z_i é o logit da classe i e P_i é a probabilidade associada a classe i

(ii) Cross Entropy: mede o quão distante a distribuição prevista está da distribuição alvo
L = -Σ y_i log(P_i)
- em que y_i é o valor esperado da classe i (tirado do dataset Y)
- P_i é a probabilidade prevista para a classe i no Softmax

Contudo, é muito comum combinar as duas fórmulas em uma só, porque a derivada em relação aos
logits se simplifica
dL/d(z_i) = P_i - y_i
que é uma subtração simples.

"""
class SoftmaxCrossEntropy(LossFunction):
    CROSS_ENTROPY_EPSILON = 1e-15

    def __init__(self):
        self._last_y_pred = None
        self._last_probs = None

    #Método que converte os logits em probabilidade
    #Recebe a saída bruta da última camada da rede e aplica a função Softmax para produzir uma distribuição de probabilidade
    #cuja soma dos elementos é igual a 1.
    # essa implementação usa estabilização numérica: exp(logit - max_logit) para evitar overflow
    # quando os logits são muito altos (números nativos Python tem dificuldade para números muito grandes)
    def _softmax(self, logits: List[float]) -> List[float]:
        #se for o mesmo vetor de entradas da ultima camada, retorna a probabilidade já calculada
        if self._last_y_pred is logits:
            return self._last_probs

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

        self._last_y_pred = logits
        self._last_probs = probabilities

        return probabilities

    # Calcula a perda Cross Entropy.
    def compute(self, y_pred: List[float], y_real: List[float]) -> float:
        #`y_pred` são os logits que é saída bruta da rede (antes do softmax)
        #nao representam probabilidade, i.e, são o campo local induzido da última camada
        #da rede
        probs = self._softmax(y_pred)

        loss = 0.0

        for predicted, target in zip(probs, y_real):
            
            # Clipping é feito, uma vez que a probabilidade acumulada pode ficar
            # muito próximo de 0 e o interpretador fazer clipping automático para 0,
            # devido a limitações de ponto flutante.
            # Ou seja, poderia ocorrer log(0), o que causa erro numérico. Assim, se o erro computado de predição
            # e alvo for muito pequeno, estabelece um erro mínimo de CROSS_ENTROPY_EPSILON
            clipped = max(predicted, self.CROSS_ENTROPY_EPSILON)
            loss -= target * math.log(clipped)

        return loss

    # Calcula o gradiente da derivada composta softmax+crossentropy
    def derivative(self, y_pred: List[float], y_real: List[float]) -> List[float]:
        probs = self._softmax(y_pred)
        
        # Calcula o gradiente par-a-par e retorna em forma de lista.
        gradients = []
        for predicted, target in zip(probs, y_real):
            gradients.append(predicted - target)

        return gradients