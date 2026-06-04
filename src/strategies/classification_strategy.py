"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from abc import ABC, abstractmethod
import random
from typing import List

""" Classe abstrata que determina uma saída específica numérica
dada uma lista de neuronios. Ou seja, define como a saída da rede é convertida 
em uma classe prevista.
"""
class ClassificationStrategy(ABC):
    @abstractmethod
    def predict_class(self, output: List[float]) -> int:
        pass

#Estratégia mais simples: escolhe o neurônio de maior ativação; em caso de empate escolhe-se a primeira ocorrência
class ArgMaxClassification(ClassificationStrategy):
    def predict_class(self, output: List[float]) -> int:
        return output.index(max(output))

#Como o ArgmaxClassification, mas em caso de empate escolhe aleatoriamente uma classificação ao invés de ser a primeira da lista
#pode ser considerada uma estratégia mais justa
class ArgMaxRandomAtTie(ClassificationStrategy):
    def predict_class(self, output: List[float]) -> int:
        max_value = max(output)
        classes = [
            index
            for index, value in enumerate(output)
            if value == max_value
        ]

        return random.choice(classes)

#estratégia usando um threshould
#em geral, usado para quando a rede tem apenas 1 neuronio de saída
class ThresholdClassification(ClassificationStrategy):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict_class(self, output: List[float]) -> int:
        if output[0] >= self.threshold:
            return 1
        else:
            return 0