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
            #essa instrução converte as listas de valores em objeto Python em um array Numpy com
            #elementos do tipo float64 (que são tipos nativos C de números reais) e guarda no objeto
            #self._weights
            self._weights = np.array(weights, dtype=np.float64)
            #essa instrução cria um novo array com a mesma dimensão e formato do self._weights criado acima
            #preenchido inicialmente com zeros (por isso uso de np.zeros_like). É o vetor usado para
            #armazenar os gradientes calculados.
            self.weights_gradient = np.zeros_like(self._weights)
        else:
            self._weights = weights
            self.weights_gradient = [0.0 for _ in weights]

        self._bias = bias
        self.bias_gradient = 0.0

    """Uso de getter e setters p/ os pesos e o viés
    isso traz duas vantagens:
    permite identificar se pesos de tamanhos incorretos estão sendo atribuidos;
    evita exposição direta das variaveis, então se for paralelizado no futuro evita 
    concorrência de threads
    Também já lida com as arrays numpy diretamente, sem que quem usa essa clase precise
    se preocupar com o tipo de rede.
    """
    @property
    def weights(self) -> List[float]:
        return self._weights

    @weights.setter
    def weights(self, new_weights: List[float]):
        if len(new_weights) != len(self._weights):
            raise ValueError(f"A qtde. de pesos informada é inválida.")
        if self.use_numpy:
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