from typing import List
from src.core.layer import PerceptronLayer, LayerConfig
from src.utils.loss_functions import LossFunction

class MultilayerPerceptron:
    def __init__(
        self,
        layer_configs: List[LayerConfig],
        input_size: int # número de entradas
    ):
        self.layers = []

        #Listas de cache usadas no backpropagation
        self.last_inputs: List[List[float]] = []
        self.last_outputs: List[List[float]] = []
        
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

    def forward(
        self,
        input_data: List[float], # corresponde ao vetor x de entrada
    ) -> List[float]:
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
