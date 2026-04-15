from typing import List
import math, random

import numpy as np

"""
Definir o problema: implementar Multilayer Perceptron (MLP)
Autor: Hugo Cardoso Ferreira de Araújo (15459500)
Data: 01/04/2026

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

""" Classe para congregar funções matemáticas úteis """
class MathFunctions:
    def RELU(vk: float) -> float:
        return max(0, vk)

    def Derivada_RELU(vk: float) -> float:
        # para tirar indefinição no ponto 0
        if (vk == 0):
             return 0

        #função degrau corresponde à derivada
        if (vk > 0): 
            return 1
        return 0
    
    def leakyRELU(vk: float) -> float:
        return vk if vk > 0 else 0.01 * vk
    
    def leakyRELUDerivative(vk : float) -> float:
        if (vk > 0): 
            return 1
        return 0.01
    
    def sum_function(list_1 : List[float], list_2 : List[float]) -> float:
        sum : float = 0

        # soma ponderada de listas de tamanho diferentes
        if (len(list_1) != len(list_2)): 
            return -1
        
        for i in range(len(list_1)):
            sum += list_1[i] * list_2[i]

        return sum
    
    def sigmoid(x: float) -> float: #essa versão de implementação é mais estável numericamente do que a versão tradicional
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)
        
    def sigmoid_derivada(x: float) -> float:
        s = MathFunctions.sigmoid(x)
        return s * (1 - s)


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

            # calcula o erro para cada camada
            for neuron_j in actual_layer.neurons:
                soma_delta_k = 0

                # OBS: O neuronio j soma os deltas da camada seguinte vezes o peso que SAEM dele para a camada seguinte

                # Para saber qual peso do neuron_k pegar, precisamos saber 
                # qual é a posição do neuron_j na sua camada
                posicao_j = 0
                for busca_j in actual_layer.neurons:
                    if busca_j == neuron_j:
                        break
                    posicao_j += 1
                

                #o neurônio j (da camada atual) está conectado a todos os neurônios k da camada seguinte.
                #o que precisamos especificar é qual o PESO equivalente da camada seguinte para j
                for neuron_k in prox_layer.neurons:
                    # Pegamos o delta do k e multiplicamos pelo peso que liga o j ao k.
                    soma_delta_k += neuron_k.delta_k * neuron_k.weight_list[posicao_j]

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
            prediction = self.forward(entry)[0] 
            # (dk - y)^2
            error_sum += (dk - prediction) ** 2
        
        return error_sum / len(dataset)

    def run_trains(self, dataset, epochs, learning_rate=0.01, stop_error = 0.001):
        for epoch in range(epochs):
            for entry, dk in dataset:
                self.forward(entry)
                self.backpropagate([dk])
                self.train(learning_rate)

            #calcula MSE
            if (epoch % 250 == 0):
                mse = self.calculate_mse(dataset)
                print(f"Época {epoch} e MSE: {mse:.6f}")
                # faz early stop quadno erro chegar a um valor minimo
                if (mse <= stop_error):
                    break



class DataLoader:
    
    #Método que retorna um dataset com os dados do conjunto CARACTERES COMPLETO. Usamos o arquivo .npy
    def carregar_dados_alfabeto(caminho_x, caminho_y):
        x_raw = np.load(caminho_x)
        y_raw = np.load(caminho_y)
        
        #usamos o método shape para ver o formato dos dados e depois reshape para deixar no formato certo
        x_flat = x_raw.reshape(1326, 120) #Achata as imagens de 10x12 para um vetor de 120 posições, sendo 1326 amostras

        dataset_CARACTERES = []

        for i in range(len(x_flat)):
            dataset_CARACTERES.append([x_flat[i].tolist(), y_raw[i].tolist()]) #adiciona ao dataset a entrada (a letra) e o rótulo. São 120 entradas e 26 saídas.

        return dataset_CARACTERES

dataset = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0]
] # dataset no padrão [entradas], retorno. Dataset de XOR

number_of_exit_neuron = 1
number_of_entry_neuron = len([dataset[0]])

dataset_CARACTERES = DataLoader.carregar_dados_alfabeto('X.npy', 'Y_classe.npy')

random.seed(314159265)
mlp = MultilayerPerceptron([4, 1], 2, MathFunctions.leakyRELU, MathFunctions.leakyRELUDerivative) # 4 neuronios na camada oculta, 1 na saida, recebendo 2 entradas
# Eu usei 4 neuronios na hidden layer porque aparentemente 2 faz o RELU puro ter neurônios mortos
# leaky RELU resolve
mlp.run_trains(dataset, 100000, learning_rate = 0.01, stop_error=0.000001)

print("\n--- RESULTADOS APÓS 10.000 ÉPOCAS ---")
for entry, dk in dataset:
    resultado = mlp.forward(entry)
    print(f"Entrada: {entry} | Alvo: {dk} | Saída Rede: {resultado[0]:.4f}")



y_classe = np.load('Y_classe.npy')

# Encontra todos os valores únicos presentes no array
valores_encontrados = np.unique(y_classe)

print(f"Valores únicos no arquivo Y_classe.npy: {valores_encontrados}")
print(f"Formato do arquivo (shape): {y_classe.shape}")
