from typing import List
from src.utils.loss_functions import LossFunction
from src.core.network import MultilayerPerceptron

class Trainer:
    def __init__(
        self,
        model: MultilayerPerceptron,
        loss_function: LossFunction,
        learning_rate: float = 0.01
    ):
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate
    
    def train_one_epoch(
        self,
        entry: List[float],
        y_real: List[float]
    ) -> float:
        y_prediction = self.model.forward(entry)
        loss = self.loss_function.compute(y_prediction, y_real)
        self.backpropagate(y_real)
        self.update_weights()
        
        return loss
    
    #Função de backpropagation que só calcula os deltas, mas não realiza atualização
    def backpropagate(
        self,
        y_real: List[float]
    ):
        loss_gradient_list = self.loss_function.derivative(
            self.model.last_outputs[-1],
            y_real
        )

        #Cálculo do delta para última camada
        last_layer = self.model.layers[-1]
        for neuron, loss_gradient in zip(last_layer.neurons, loss_gradient_list):
            neuron.delta_k = neuron.activation.derivative(
                neuron.last_local_induced_field
            ) * loss_gradient

        #Cálculo do delta para camadas seguintes
        for l in reversed(range(len(self.model.layers) - 1)):
            actual_layer = self.model.layers[l]
            next_layer = self.model.layers[l + 1]

            for i, neuron in enumerate(actual_layer.neurons):
                sum_delta_k : float = 0.0
                for neuron_k in next_layer.neurons:
                    sum_delta_k += neuron_k.delta_k * neuron_k.weight_list[i]
                
                neuron.delta_k = sum_delta_k * neuron.activation.derivative(
                    neuron.last_local_induced_field
                )
    
    def update_weights(self):
        for layer in self.model.layers:
            for neuron in layer.neurons:
                for i in range(len(neuron.weight_list)):
                    neuron.weight_list[i] -= self.learning_rate * neuron.delta_k * neuron.last_entry[i]
                neuron.bias -= self.learning_rate * neuron.delta_k
            
    def train(
        self,
        dataset: List,
        epochs: int
    ) -> List[float]:
        loss_history = []
        for epoch in range(epochs):
            total_loss: float = 0.0
            for x, y in dataset:
                total_loss += self.train_one_epoch(x, y)
            
            average_loss = total_loss / len(dataset)
            loss_history.append(average_loss)
            print(f"Época {epoch} | Erro: {average_loss:.6f}")
        
        return loss_history