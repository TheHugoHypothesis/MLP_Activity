from typing import List
from src.utils.activation_function import ActivationFunction
from src.utils.linear_algebra import scalar_product

"""
Classe que representa um neurônio dentro da rede.
- `self.weight_list`: corresponde à lista de pesos associada àquela instância;
- `self.bias`: corresponde ao viés daquele neurônio;
- `self.activation`: corresponde à classe de função de ativação usada;
- `self.last_entry`: corresponde aos últimos inputs (entradas) recebidas;
- `self.last_local_induced_field`: corresponde ao último campo local induzido calculado;
- `self.output`: corresponde à última saída (yk) calculada;
- `self.delta_k`: corresponde ao campo local induzido do neurônio da frente;

OBS: `self.delta_k` é passado durante o treino para o neurônio pela
classe PerceptronLayer.
"""
class PerceptronNeuron:
    def __init__(
        self, 
        weight_list: List[float], 
        activation: ActivationFunction,
        bias: float = 0
    ):
        self.weight_list = weight_list
        self.bias = bias
        self.activation = activation

        self.last_entry: List[float] = []
        self.last_local_induced_field: float = 0.0
        self.output: float = 0.0
        self.delta_k: float = 0.0
    
    #Método que gera uma saída dado a lista de entrada do neurônio
    def feedforward(self, entry_list: List[float]) -> float:
        self.last_entry = entry_list
        self.last_local_induced_field = scalar_product(entry_list, self.weight_list) + self.bias
        self.output = self.activation.activate(self.last_local_induced_field)
        return self.output