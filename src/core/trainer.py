"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from typing import List
from src.strategies.loss_functions import LossFunction
from src.strategies.trainer_optimizer import Optimizer
from src.core.network import MultilayerPerceptron

class Trainer:
    def __init__(
        self,
        model: MultilayerPerceptron,
        loss_function: LossFunction,
        optimizer: Optimizer,
        learning_rate: float = 0.01,
        patience: int = None,
        min_delta: float = 0.0
    ):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_delta = min_delta
    
    def train_one_sample(
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

            #popula gradiente para última camada
            neuron.parameter.bias_gradient = neuron.delta_k
            for i in range(len(neuron.weights)):
                neuron.parameter.weights_gradient[i] = neuron.delta_k * neuron.last_entry[i]

        #Cálculo do delta para camadas seguintes
        for l in reversed(range(len(self.model.layers) - 1)):
            actual_layer = self.model.layers[l]
            next_layer = self.model.layers[l + 1]

            for i, neuron in enumerate(actual_layer.neurons):
                sum_delta_k : float = 0.0
                for neuron_k in next_layer.neurons:
                    sum_delta_k += neuron_k.delta_k * neuron_k.parameter.weights[i]
                
                neuron.delta_k = sum_delta_k * neuron.activation.derivative(
                    neuron.last_local_induced_field
                )

                #popula os gradientes para camadas seguintes
                neuron.parameter.bias_gradient = neuron.delta_k
                for j in range(len(neuron.weights)):
                    neuron.parameter.weights_gradient[j] = neuron.delta_k * neuron.last_entry[j]

    def update_weights(self):
        for layer in self.model.layers:
            self.optimizer.step(layer_neurons=layer.neurons, learning_rate=self.learning_rate)
    
    def evaluate(self, dataset: List) -> tuple[float, float]:
        total_loss = 0.0
        correct = 0
        for x, y in dataset:
            prediction = self.model.forward(x)
            if prediction is None:
                continue
            
            #Acumula perda
            total_loss += self.loss_function.compute(prediction, y)
            
            #Acumula acertos
            if prediction.index(max(prediction)) == y.index(max(y)):
                correct += 1
            
        avg_loss = total_loss / len(dataset) if len(dataset) > 0 else 0.0
        accuracy = correct / len(dataset) if len(dataset) > 0 else 0.0
        return avg_loss, accuracy
            
    def train(
        self,
        train_dataset: List,
        val_dataset: List,
        epochs: int
    ):
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        #Variáveis de Early Stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights_snapshot = None

        for epoch in range(epochs):
            total_train_loss: float = 0.0

            for x, y in train_dataset:
                total_train_loss += self.train_one_sample(x, y)
            
            average_train_loss = total_train_loss / len(train_dataset)
            average_val_loss, average_val_acc = self.evaluate(val_dataset)
            _, average_train_acc = self.evaluate(train_dataset)

            history["train_loss"].append(average_train_loss)
            history["val_loss"].append(average_val_loss)
            history["train_acc"].append(average_train_acc)
            history["val_acc"].append(average_val_acc)

            print(
                f"Época {epoch} | "
                f"Train Loss: {average_train_loss:.6f} | "
                f"Val Loss: {average_val_loss:.6f} | "
                f"Val Acc: {average_val_acc:.4f}"
            )

            #Lógica de Early Stopping
            if self.patience is not None:
                if average_val_loss < (best_val_loss - self.min_delta):
                    best_val_loss = average_val_loss
                    patience_counter = 0
                    best_weights_snapshot = self._get_model_weights_snapshot()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"\nEarly stop na época {epoch}.")
                        print(f"O erro de validação não melhorou por {self.patience} épocas seguidas.")
                        #Restaura o melhor modelo
                        self._restore_model_weights(best_weights_snapshot)
                        print("[Early Stopping] Melhores pesos restaurados.")
                        break

        return history
    
    def _get_model_weights_snapshot(self):
        #Cria uma cópia profunda dos valores atuais dos pesos e biases de cada neurônio
        return [
            [(list(neuron.weights), neuron.bias) for neuron in layer.neurons]
            for layer in self.model.layers
        ]
    def _restore_model_weights(self, snapshot):
        #Restaura os pesos e biases a partir do snapshot salvo
        for layer, layer_snapshot in zip(self.model.layers, snapshot):
            for neuron, (w_list, bias) in zip(layer.neurons, layer_snapshot):
                neuron.weights = list(w_list)
                neuron.bias = bias