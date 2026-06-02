"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from typing import List

""" Abstração para um conjunto de pesos numéricos 
(self.weight) e sua variação no caso desse projeto,
serve para encapsular os parâmetros do Neuron
Assim o Backpropagation e o Otimizador não precisam ter
contexto um do outro, trabalhando por meio dessa classe mais generica
"""
class Parameter:
    def __init__(
        self,
        weights: List[float],
        bias: float = 0.0,
        use_numpy: bool = False
    ):
        self.use_numpy = use_numpy
        if use_numpy:
            import numpy as np
            self._weights = np.array(weights, dtype=np.float64)
            self.weights_gradient = np.zeros_like(self._weights)
        else:
            self._weights = weights
            self.weights_gradient = [0.0 for _ in weights]

        self._bias = bias
        self.bias_gradient = 0.0

    @property
    def weights(self) -> List[float]:
        return self._weights

    @weights.setter
    def weights(self, new_weights: List[float]):
        if len(new_weights) != len(self._weights):
            raise ValueError(f"A qtde. de pesos informada é inválida.")
        if getattr(self, "use_numpy", False):
            import numpy as np
            self._weights = np.array(new_weights, dtype=np.float64)
        else:
            self._weights = new_weights

    @property
    def bias(self) -> float:
        return self._bias
    
    @bias.setter
    def bias(self, new_bias: float):
        self._bias = new_bias