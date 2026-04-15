from typing import List
import math, random

import numpy as np
from math_functions import MathFunctions
from data_loader import DataLoader

"""
Definir o problema: implementar Multilayer Perceptron (MLP)

Implementar um MLP, definindo
(i) Função de ativação: ReLU (talvez depois eu tente usar um TeLU)
- phi(vk) = {
    vk se vk > 0
    0 caso contrário
}
- phi'(vk) = { 1 se vk > 0; 0 se vk < 0 }; definindo phi'(0) = 0

(ii) Neurônio usado é o Perceptron padrão
(iii) O algoritmo de aprendizado se baseia em retropropagação
"""

class PerceptronLayer:
    def __init__(
        self, 
        number_of_neurons : int, 
        number_of_input: int,
        activation_function, derivative_activation_function): 
        self.number_of_neurons = number_of_neurons
        self.neurons = []

        for i in range(number_of_neurons):
            self.neurons.append(
                PerceptronNeuron(
                    self.random_weight(number_of_input),
                    activation_function, derivative_activation_function
                )
            )
    
    # Função que inicia pesos aleatórios em uma lista
    def random_weight(self, numberOfConnections : int) -> List[float]:
        weight_list : List[float] = []
        for weight in range(numberOfConnections):
            weight_list.append(random.uniform(-1, 1))
        return weight_list

class PerceptronNeuron:
    def __init__(
        self, 
        weight_list : List[float], 
        activation_function,
        derivative_activation_function,
        bias : float = 0
    ):
        self.weight_list = weight_list
        self.bias = bias

        self.last_entry = []
        self.last_local_induced_field = 0
        self.y = 0
        self.delta_k = 0 # gradiente a frente?

        self.activation_function = activation_function
        self.derivative_activation_function = derivative_activation_function 
    
    # gera uma saida y (separado do treino porque é no passo 1)
    def feedforward(self, entry_list : List[float]) -> float:
        self.last_entry = entry_list
        self.last_local_induced_field = MathFunctions.sum_function(self.last_entry, self.weight_list) + self.bias
        self.y = self.activation_function(self.last_local_induced_field)
        return self.y

    #gera uma atualização de peso caso de errado uma saida
    def train_epoch(
        self,
        learning_rate : float = 0.01
    ):
        # Como aqui ja vai ter ocorrido um feedforward temos guardado last_local_induced_field e y
        for i in range(len(self.last_entry)):
            delta_w = learning_rate * self.delta_k * self.last_entry[i]
            self.weight_list[i] += delta_w

        self.bias += learning_rate * self.delta_k
    
    def calculate_local_gradient(self) -> float:
        return self.derivative_activation_function(self.last_local_induced_field)
 
class MultilayerPerceptron:
    def __init__(
        self, layer_topology : List[int], last_layer_size : int,
        activation_function, derivative_activation_function
    ):
        # Layer_topology documenta o número de neurônios e de camadas [3, 4] significa 3 na camada 1 (oculta) e 4 na camada de saída por exemplo
        self.layer_topology = layer_topology
        self.layers = []
        self.last_layer_size = last_layer_size #representa o número de entradas de uma camada, considerando o número de saídas da camada anterior

        #Start the hidden layers & exit
        for number_of_neurons in layer_topology:
            # Esse código considera que se uma camada anterior tem N neuronios, a camada da frente terá N inputs
            self.layers.append(PerceptronLayer(
                number_of_neurons,
                self.last_layer_size,
                activation_function, derivative_activation_function
            ))
            self.last_layer_size = number_of_neurons # faz com que a próxima camada saiba quantos neuronios tem na camada anterior
    
    def backpropagate(self, target_list : List[float]):
        # calcula o delta para ultima camada
        last_layer = self.layers[-1]
        k=0
        for neuron in last_layer.neurons:
            neuron.delta_k = neuron.calculate_local_gradient() * (target_list[k] - neuron.y)
            k+=1
        
        i = len(self.layers) - 2
        # calcula o delta para camadas seguintes
        for actual_layer in reversed(self.layers[:-1]):# para olhar no sentido contrário sem olhar o último layer
            prox_layer = self.layers[i+1]
                
            for j, neuron_j in enumerate(actual_layer.neurons):
                soma_delta_k = 0
                for neuron_k in prox_layer.neurons:
                    # j é a posição do neurônio na camada atual, 
                    # que corresponde ao índice do peso no neurônio da camada seguinte
                    soma_delta_k += neuron_k.delta_k * neuron_k.weight_list[j]
                
                neuron_j.delta_k = neuron_j.calculate_local_gradient() * soma_delta_k
            
            i-=1
    
    def train(self, learning_rate = 0.01):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.train_epoch(learning_rate)
    
    def forward(self, input_data : List[float]) -> List[float]:
        dados_atuais = input_data

        for layer in self.layers:
            proximas_entry = []
            for neuron in layer.neurons:
                #guardas os yk de cada neuron
                proximas_entry.append(neuron.feedforward(dados_atuais))
            dados_atuais = proximas_entry
        
        return dados_atuais #retorna a saída da ultima camada
    
    def calculate_mse(self, dataset):
        error_sum = 0
        for entry, dk in dataset:
            prediction = self.forward(entry)

            erro_camada = 0
            for i in range(len(dk)):
                erro_camada += (dk[i] - prediction[i]) ** 2

            # (dk - y)^2
            error_sum += erro_camada / len(dk)
        
        return error_sum / len(dataset)

    def run_trains(self, dataset, epochs, learning_rate=0.01, stop_error = 0.001):
        for epoch in range(epochs):
            for entry, dk in dataset:
                self.forward(entry)
                self.backpropagate(dk)
                self.train(learning_rate)

            #calcula MSE
            if (epoch % 1 == 0):
                mse = self.calculate_mse(dataset)
                print(f"Época {epoch} e MSE: {mse:.6f}")
                # faz early stop quadno erro chegar a um valor minimo
                if (mse <= stop_error):
                    break

dataset_CARACTERES = DataLoader.carregar_dados_alfabeto('X.npy', 'Y_classe.npy')

random.seed(3)
mlp = MultilayerPerceptron([6, 26], 120, MathFunctions.leakyRELU, MathFunctions.leakyRELUDerivative)
# Eu usei 4 neuronios na hidden layer porque aparentemente 2 faz o RELU puro ter neurônios mortos
# leaky RELU resolve
mlp.run_trains(dataset_CARACTERES, 10000, learning_rate = 0.01, stop_error=0.000001)

print("\n--- RESULTADOS APÓS 10.000 ÉPOCAS ---")
for entry, dk in dataset_CARACTERES:
    resultado = mlp.forward(entry)
    #print(f"Entrada: {entry} | Alvo: {dk} | Saída Rede: {resultado[0]:.4f}")
    predicao = np.argmax(resultado)
    alvo = np.argmax(dk)
    print(f"Alvo (índice): {alvo} | Predição: {predicao} | Confiança: {resultado[predicao]:.4f}")

y_classe = np.load('Y_classe.npy')

# Encontra todos os valores únicos presentes no array
valores_encontrados = np.unique(y_classe)

print(f"Valores únicos no arquivo Y_classe.npy: {valores_encontrados}")
print(f"Formato do arquivo (shape): {y_classe.shape}")
