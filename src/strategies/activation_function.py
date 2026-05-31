"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

import math
from abc import ABC, abstractmethod

"""
Módulo de agregação de:
    (1) funções de ativação;
    (2) derivadas de funções de ativação. 
O parâmetro 'vk' corresponde ao campo local induzido de um neurônio k.
"""

"""
Usa o módulo ABC para definir classe base abstrata (ABC)
de funções de ativação, contendo uma função de ativação (activate)
e a derivada (derivative).
"""
class ActivationFunction(ABC):
    """ Função de ativação """
    @abstractmethod
    def activate(self, vk: float) -> float:
        pass

    """ Derivada da função de ativação """
    @abstractmethod
    def derivative(self, vk: float) -> float:
        pass

class RELU(ActivationFunction):
    def activate(self, vk: float) -> float:
        return max(0.0, vk)
    
    def derivative(self, vk: float) -> float:
        #Usado para tirar indefinição no ponto 0
        if (vk == 0): 
            return 0.0
        
        #Função degrau corresponde à derivada da RELU
        if (vk > 0):
            return 1.0
        return 0.0

class LeakyRELU(ActivationFunction):
    LEAKY_RELU_ALPHA: float = 0.01

    def activate(self, vk: float) -> float:
        if vk > 0.0:
            return vk
        return self.LEAKY_RELU_ALPHA * vk
    
    def derivative(self, vk: float) -> float:
        if (vk > 0.0):
            return 1.0
        return self.LEAKY_RELU_ALPHA

class Sigmoid(ActivationFunction):
    def activate(self, vk: float) -> float:
        # Essa é uma implementação de sigmoid numericamente
        # estável, ou seja, evita overflow se 'vk' for
        # muito negativo.
        if vk >= 0.0:
            return 1.0 / (1.0 + math.exp(-vk))
        
        exp_vk = math.exp(vk)
        return exp_vk / (1.0 + exp_vk)

    def derivative(self, vk: float) -> float:
        s = self.activate(vk)
        return s * (1.0 - s)

class Linear(ActivationFunction):
    def activate(self, vk: float) -> float:
        return vk
    
    def derivative(self, vk: float) -> float:
        return 1.0
