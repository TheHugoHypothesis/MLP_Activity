
from typing import List
from dataclasses import dataclass

from src.utils.linear_algebra import scalar_product
from src.utils.activation_function import ActivationFunction
from src.utils.weight_initializers import WeightInitializer
from src.core.neuron import PerceptronNeuron

"""
Classe que representa uma camada com pesos dentro da rede.
"""
class PerceptronLayer:
    def __init__(
        self,
        number_of_neurons: int,
        number_of_inputs: int,
        activation: ActivationFunction,
        weight_initializer: WeightInitializer
    ):
        self.neurons = []

        for i in range(number_of_neurons):
            weights = weight_initializer.initialize(
                n_in=number_of_inputs,
                n_out=number_of_neurons
            )
            self.neurons.append(
                PerceptronNeuron(
                    weights,
                    activation
                )
            )

""" 
Classe de dados que representa a configuração de uma camada
- `n_neurons`: representa o número de neurônios daquela camada;
- `activation`: representa a função de ativação usada na camada;
- `initializer`: representa o inicializador (abstrato) de pesos usado naquela camada.
"""
@dataclass
class LayerConfig:
    n_neurons: int
    activation: ActivationFunction
    initializer: WeightInitializer 