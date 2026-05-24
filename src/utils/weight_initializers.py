from abc import ABC, abstractmethod
from typing import List
import random
import math

"""
Módulo que congrega diversas funções 
de inicialização de pesos na rede.
"""

class WeightInitializer(ABC):
    @abstractmethod
    def initialize(
        self,
        n_in: int,
        n_out: int
    ) -> List[float]:
        pass

# Inicialização uniforme
class UniformInitializer(WeightInitializer):
    def __init__(
        self,
        low_bound: float = -0.1,
        high_bound: float = 0.1
    ):
        self.low_bound = low_bound
        self.high_bound = high_bound
    
    def initialize(self, n_in: int, n_out: int) -> List[float]:
        weights: List[float] = []
        for weight in range(n_in):
            weights.append(random.uniform(self.low_bound, self.high_bound))
        return weights

# Inicialização usando distribuição normal
# com uma média especificada e um desvio-padrão
class NormalInitializer(WeightInitializer):
    def __init__(
        self,
        average: float = 0.0, 
        deviation: float = 1.0
    ):
        self.average = average
        self.deviation = deviation

    def initialize(self, n_in: int, n_out: int) -> List[float]:
        weights: List[float] = []
        for weight in range(n_in):
            weights.append(random.gauss(self.average, self.deviation))
        return weights

# Inicializiação Xavier-Glorot normalizada
class XavierGlorotInitializer(WeightInitializer):
    def initialize(self, n_in: int, n_out: int) -> List[float]:
        weights: List[float] = []

        deviation = math.sqrt(2.0 / (n_in + n_out))

        for weight in range(n_in):
            weights.append(random.gauss(0.0, deviation))
        return weights

# Inicialização He (usada em geral para ReLU)
class HeInitializer(WeightInitializer):
    def initialize(self, n_in: int, n_out: int) -> List[float]:
        weights: List[float] = []
        deviation = math.sqrt(2.0 / n_in)
        for weight in range(n_in):
            weights.append(random.gauss(0.0, deviation))
        return weights