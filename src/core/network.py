"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from typing import List
from src.core.layer import PerceptronLayer, LayerConfig
from src.strategies.loss_functions import LossFunction

"""
Classe que representa uma configuração de rede MultilayerPerceptron.
Parâmetros nessa classe:
- `layer_configs` uma lista de LayerConfig, classe de dados que define as configurações
de uma camada (número de neurônios, função de ativação, inicializador de pesos);
- `input_size` indica o número de entradas (padrões) recebidos na camada inicial.

Essa classe implementa somente a estrutura da rede com camadas (PerceptronLayers)
e um método de forward (gerar saída com base nos pesos e neurônios atuais).
"""
class MultilayerPerceptron:
    def __init__(
        self,
        layer_configs: List[LayerConfig],
        input_size: int
    ):
        self.layers = []

        #Listas de cache usadas no backpropagation
        self.last_inputs: List[List[float]] = []
        self.last_outputs: List[List[float]] = []
        
        #Conforme as camadas são iniciadas, é necessário indicar o número de neurônios na camada
        #anterior, como sendo o número de entradas (parâmetro `current_input_size`), que inicialmente
        #é o número de padrões recebido pela rede no inicial. Podemos fazer assim porque o número
        #de neurônios na camada anterior é igual ao número de entradas recebidas por neurônio da
        #camada posterior (no caso, implementamos um MLP fully-connected)
        current_input_size = input_size
        for config in layer_configs:
            layer = PerceptronLayer(
                number_of_neurons=config.n_neurons,
                number_of_inputs=current_input_size,
                activation=config.activation,
                weight_initializer=config.initializer
            )
            self.layers.append(layer)
            current_input_size = config.n_neurons
    
    """
    Executa a propagação direta (forward propagation) da rede.
    Parâmetros:
    - `input_data` o vetor de entrada (padrões) da rede (x).
    Retorna o vetor de saída produzido pela úlktima camada da rede.
    """
    def forward(
        self,
        input_data: List[float],
    ) -> List[float]:
        #Limpa o cache da execução anterior, assumindo que já foi usado em caso de treinamento
        #pelo Trainer durante o processo de retropropagação e atualização de pesos.
        #i.e, não é esperado que se chame essa função duas vezes seguidas durante o treinamento
        self.last_inputs = []
        self.last_outputs = []

        current_data: List[float] = input_data
        
        for layer in self.layers:
            self.last_inputs.append(current_data)

            next_data: List[float] = []
            for neuron in layer.neurons:
                next_data.append(neuron.feedforward(current_data))
            
            self.last_outputs.append(next_data)
            current_data = next_data
        
        return current_data
