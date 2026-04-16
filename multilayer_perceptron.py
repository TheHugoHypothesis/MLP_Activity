from typing import List
import math, random

import numpy as np
from math_functions import MathFunctions
from data_loader import DataLoader

import json

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
            weight_list.append(random.uniform(-0.1, 0.1))
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
        hidden_activation_function, hidden_derivative_activation_function,
        exit_activation_function, exit_derivative_activation_function
    ):
        # Layer_topology documenta o número de neurônios e de camadas [3, 4] significa 3 na camada 1 (oculta) e 4 na camada de saída por exemplo
        self.layer_topology = layer_topology
        self.layers = []
        self.last_layer_size = last_layer_size #representa o número de entradas de uma camada, considerando o número de saídas da camada anterior
        
        count = 0
        for number_of_neurons in layer_topology:
            if (count == len(layer_topology) - 1):
                #aqui ta na ultima camada
                self.layers.append(PerceptronLayer(
                    number_of_neurons,
                    self.last_layer_size,
                    exit_activation_function, exit_derivative_activation_function
                ))
                break

            # Esse código considera que se uma camada anterior tem N neuronios, a camada da frente terá N inputs
            self.layers.append(PerceptronLayer(
                number_of_neurons,
                self.last_layer_size,
                hidden_activation_function, hidden_derivative_activation_function
            ))
            self.last_layer_size = number_of_neurons # faz com que a próxima camada saiba quantos neuronios tem na camada anterior
            count += 1
    
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

    def run_trains(self, dataset_treino, dataset_validacao, epochs, learning_rate=0.01, stop_error = 0.001):
        for epoch in range(epochs):
            for entry, dk in dataset_treino:
                self.forward(entry)
                self.backpropagate(dk)
                self.train(learning_rate)

            #calcula MSE
            if (epoch % 10 == 0):
                mse_treino = self.calculate_mse(dataset_treino)
                mse_validacao = self.calculate_mse(dataset_validacao)
                print(f"Época {epoch}: Treino MSE: {mse_treino:.6f} | Validação MSE: {mse_validacao:.6f}")
        
        mse_f_t = self.calculate_mse(dataset_treino)
        mse_f_v = self.calculate_mse(dataset_validacao)
        print(f"Estado Final (Época {epochs}): Treino MSE: {mse_f_t:.6f} | Validação MSE: {mse_f_v:.6f}")

    def prever(self, input_data: List[float]) -> dict:
        saida_bruta = self.forward(input_data)
        
        indice_vencedor = int(np.argmax(saida_bruta))
        confianca = float(saida_bruta[indice_vencedor])
        
        alfabeto = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if indice_vencedor < 26:
            letra_prevista = alfabeto[indice_vencedor]
        else:
            letra_prevista = str(indice_vencedor)
        
        return {
            "letra": letra_prevista,
            "confianca": round(confianca * 100, 2), # Em porcentagem
            "indice": indice_vencedor
        }
            
def salvar_relatorio_externo(mlp_instancia, dataset, filename: str):
    relatorio = []
    
    for entry, dk in dataset:
        # Usamos o método forward da instância passada
        resultado = mlp_instancia.forward(entry)
        
        predicao = int(np.argmax(resultado))
        alvo = int(np.argmax(dk))
        confianca = float(resultado[predicao])
        
        relatorio.append({
            "indice_alvo": alvo,
            "indice_predito": predicao,
            "confianca": round(confianca, 4),
            "sucesso": alvo == predicao
        })

    with open(filename, 'w') as f:
        json.dump(relatorio, f, indent=4)
    print(f"Relatório salvo externamente em {filename}")

def salvar_pesos_externo(mlp_instancia, filename: str):
    dados = {
        "layer_topology": mlp_instancia.layer_topology,
        "layers": []
    }

    for layer in mlp_instancia.layers:
        layer_data = []
        for neuron in layer.neurons:
            layer_data.append({
                "pesos": neuron.weight_list,
                "bias": neuron.bias
            })
        dados["layers"].append(layer_data)

    with open(filename, 'w') as f:
        json.dump(dados, f, indent=4)
    print(f"Pesos salvos externamente em {filename}")

def carregarJson(filename: str, 
            hidden_activation, hidden_derivative, 
            exit_activation, exit_derivative):
        with open(filename, 'r') as f:
            dados = json.load(f)

        topology = dados["layer_topology"]
        input_size = len(dados["layers"][0][0]["pesos"])

        mlp = MultilayerPerceptron(
            topology, input_size,
            hidden_activation, hidden_derivative,
            exit_activation, exit_derivative
        )

        # Injeta os pesos e bias guardados em cada neurónio
        for i_layer, layer_data in enumerate(dados["layers"]):
            for i_neuron, neuron_data in enumerate(layer_data):
                mlp.layers[i_layer].neurons[i_neuron].weight_list = neuron_data["pesos"]
                mlp.layers[i_layer].neurons[i_neuron].bias = neuron_data["bias"]

        print(f"Modelo carregado com sucesso de {filename}")
        return mlp

dataset_CARACTERES = DataLoader.carregar_dados_alfabeto('X.npy', 'Y_classe.npy')

random.seed(3)

def separar_dataset(dataset, percentual_treino=0.8):
    dados_misturados = dataset[:]
    random.shuffle(dados_misturados)
    
    limite = int(len(dados_misturados) * percentual_treino)
    
    treino = dados_misturados[:limite]
    validacao = dados_misturados[limite:]
    
    return treino, validacao

treino_conjunto, validacao_conjunto = separar_dataset(dataset_CARACTERES, 0.8)

mlp = MultilayerPerceptron(
    [64, 26], 120, 
    MathFunctions.leakyRELU, MathFunctions.leakyRELUDerivative,
    MathFunctions.sigmoid, MathFunctions.sigmoid_derivada    
)

mlp.run_trains(treino_conjunto, validacao_conjunto, 130, learning_rate = 0.01, stop_error=0.000001)
salvar_pesos_externo(mlp, "modelo_mlp.json")
salvar_relatorio_externo(mlp, dataset_CARACTERES, "relatorio.json")

# print("\n--- RESULTADOS APÓS 10.000 ÉPOCAS ---")
# for entry, dk in dataset_CARACTERES:
#     resultado = mlp.forward(entry)
#     #print(f"Entrada: {entry} | Alvo: {dk} | Saída Rede: {resultado[0]:.4f}")
#     predicao = np.argmax(resultado)
#     alvo = np.argmax(dk)
#     print(f"Alvo (índice): {alvo} | Predição: {predicao} | Confiança: {resultado[predicao]:.4f}")

# h_act = MathFunctions.leakyRELU
# h_der = MathFunctions.leakyRELUDerivative
# e_act = MathFunctions.sigmoid
# e_der = MathFunctions.sigmoid_derivada

# mlp =   carregarJson(
#     "modelo_mlp.json", 
#     h_act, h_der, e_act, e_der
# )

# exemplo_entrada = dataset_CARACTERES[4][0]
# resultado = mlp.prever(exemplo_entrada)

# print(f"Letra prevista: {resultado['letra']} ({resultado['confianca']}%)")