"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from typing import List
from dataclasses import dataclass

from src.strategies.activation_function import ActivationFunction
from src.strategies.weight_initializers import WeightInitializer
from src.core.neuron import PerceptronNeuron

"""
Classe que representa uma camada com pesos dentro da rede.
Parâmetros nessa classe:
- `number_of_neurons`: número de neurônios declarado na camada;
- `number_of_inputs`: número de entradas que cada neurônio daquela camada recebe;
- `activation`: classe abstrata da função de ativação usada na camada;
- `weight_initializer`: classe abstrata do inicializador de pesos da camada.
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

        #Inicializador de pesos contém os parâmetros `n_in` e `n_out` que indicam respectivamente
        #o número de entradas atuais e o número de neurônios daquela camada (que é o número de saída).
        #embora tenha inicializadores que não usarão essas informações, como Uniform e Normal,
        #os inicializadores He e Xavier-Glorot precisam dessa informação (que existe por camada)
        #logo não é possível passar os parâmetros de modo que não seja durante a inicialização do PerceptronLayer. 
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

Usamos @dataclass porque essa classe serve só para armazenar configurações da camada, e daí
não é preciso escrever manualmente o __init__.
"""
@dataclass
class LayerConfig:
    n_neurons: int
    activation: ActivationFunction
    initializer: WeightInitializer 