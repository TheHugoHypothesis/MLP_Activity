"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from typing import List
from src.strategies.activation_function import ActivationFunction
from src.utils.linear_algebra import scalar_product_optimized as scalar_product
from src.core.parameter import Parameter

"""
Classe que representa um neurônio dentro da rede.
- self.weight_list e bias: corresponde à lista de pesos associada àquela instância, abstraídos pela classe Parameter;
- self.activation: corresponde à classe de função de ativação usada;
- self.last_entry: corresponde aos últimos inputs (entradas) recebidas;
- self.last_local_induced_field`: corresponde ao último campo local induzido calculado;
- self.output corresponde à última saída (yk) calculada;
- self.delta_k corresponde ao campo local induzido do neurônio da frente;
- self.use_numpy flag que indica se devem ser usadas arrays numpy

OBS: self.delta_k é passado durante o treino para o neurônio pela
classe Trainer.
"""
class PerceptronNeuron:
    def __init__(
        self, 
        weight_list: List[float], 
        activation: ActivationFunction,
        bias: float = 0,
        use_numpy: bool = False
    ):
        self.parameter = Parameter(weight_list, bias, use_numpy=use_numpy)
        self.activation = activation
        self.use_numpy = use_numpy

        self.last_entry: List[float] = []
        self.last_local_induced_field: float = 0.0
        self.output: float = 0.0
        self.delta_k: float = 0.0
    
    #Método que gera uma saída dado a lista de entrada do neurônio
    def feedforward(self, entry_list: List[float]) -> float:
        self.last_entry = entry_list
        if self.use_numpy:
            #aqui fazemos lazy import, i.e, só importamos a biblioteca se formos usar a versão com numpy
            #isso evita carregamento desnecessário se formos usar a versão sem numpy
            #e permite essa parte do código rodar sem que tenha a biblioteca instalada no sistema
            import numpy as np

            #essa linha utiliza a função np.dot(...) que equivale a um produto escalar entre duas listas numpy
            self.last_local_induced_field = np.dot(entry_list, self.weights) + self.bias
        else:
            self.last_local_induced_field = scalar_product(entry_list, self.weights) + self.bias
        self.output = self.activation.activate(self.last_local_induced_field)
        return self.output
    
    """Definições de getter e setter para os pesos e bias
    essas definições explícitas com @property e setter são úteis p/ type hinting
    ( o que impede tipos errados serem atribuidos) e facilitar acesso, as variáveis, permitindo
    fazer neuron.weights, ao invés de neuron.parameter.weights (por exemplo),
    o que juntando com os getter e setter de Parameter() melhora a arquitetura
    """
    @property
    def weights(self) -> List[float]:
        return self.parameter.weights
    
    @weights.setter
    def weights(self, new_weights: List[float]):
        self.parameter.weights = new_weights
    
    @property
    def bias(self) -> float:
        return self.parameter.bias
    
    @bias.setter
    def bias(self, new_bias: float):
        self.parameter.bias = new_bias