from typing import List
import math, random

import numpy as np
from gerar_grafico import apresentar_matriz_confusao
from math_functions import MathFunctions
from data_loader import DataLoader

import json

import matplotlib.pyplot as plt

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
        historico_erros = []
        for epoch in range(epochs):
            for entry, dk in dataset_treino:
                self.forward(entry)
                self.backpropagate(dk)
                self.train(learning_rate)

            # Calcula o erro de treino e validação em cada iteração (época)
            mse_treino = self.calculate_mse(dataset_treino)
            mse_validacao = self.calculate_mse(dataset_validacao)
            
            # Guarda no histórico
            historico_erros.append({
                "epoca": epoch,
                "mse_treino": round(mse_treino, 6),
                "mse_validacao": round(mse_validacao, 6)
            })

            if (epoch % 10 == 0):
                print(f"Época {epoch}: Treino MSE: {mse_treino:.6f} | Validação MSE: {mse_validacao:.6f}")
        
        mse_f_t = self.calculate_mse(dataset_treino)
        mse_f_v = self.calculate_mse(dataset_validacao)
        print(f"Estado Final (Época {epochs}): Treino MSE: {mse_f_t:.6f} | Validação MSE: {mse_f_v:.6f}")

        with open("historico_erros.json", "w") as f:
            json.dump(historico_erros, f, indent=4)
        print("Arquivo 'historico_erros.json' salvo com sucesso!") # produz um arquivo contendo o erro cometido pela rede neural em cada iteração dotreinamento.

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

def avaliacao_teste(mlp_instancia, dataset_teste):
    print("\n--- AVALIAÇÃO NO CONJUNTO DE TESTE ---")
    acertos = 0
    for entry, dk in dataset_teste:
        resultado = mlp_instancia.forward(entry)
        predicao = int(np.argmax(resultado))
        alvo = int(np.argmax(dk))
        if predicao == alvo:
            acertos += 1

    mse_final = mlp_instancia.calculate_mse(dataset_teste)
    acuracia_final = (acertos / len(dataset_teste)) * 100

    print(f"Erro Final (MSE) no Teste: {mse_final:.6f}")
    print(f"Acurácia Final no Teste: {acuracia_final:.2f}% ({acertos}/{len(dataset_teste)})")

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

def separar_dataset_estratificado(dataset, p_treino=0.7, p_validacao=0.15):
    classes = {} #Primeiro agrupamos os dados de acordo com a classe (letra) deles
    for entry, dk in dataset:
        classe_id = int(np.argmax(dk))
        if classe_id not in classes:
            classes[classe_id] = []
        classes[classe_id].append((entry, dk))
    
    treino_final = []
    validacao_final = []
    teste_final = []
    
    for classe_id, amostras in classes.items(): #então separamos proporcionalmente a quantidade de cada letra que vai para cada conjunto
        random.shuffle(amostras)
        
        total_amostras = len(amostras)
        limite_treino = int(total_amostras * p_treino)
        limite_val = int(total_amostras * (p_treino + p_validacao))
        
        treino_final.extend(amostras[:limite_treino])
        validacao_final.extend(amostras[limite_treino:limite_val])
        teste_final.extend(amostras[limite_val:])
        
    random.shuffle(treino_final)
    random.shuffle(validacao_final)
    random.shuffle(teste_final)
    
    return treino_final, validacao_final, teste_final

treino_conjunto, validacao_conjunto = separar_dataset(dataset_CARACTERES, 0.8)
#treino_conjunto, validacao_conjunto, teste_conjunto = separar_dataset_estratificado(dataset_CARACTERES, 0.7, 0.15)

mlp = MultilayerPerceptron(
    [64, 26], 120, 
    MathFunctions.leakyRELU, MathFunctions.leakyRELUDerivative,
    MathFunctions.sigmoid, MathFunctions.sigmoid_derivada    
)

salvar_pesos_externo(mlp, "modelo_mlp_pesos_iniciais.json")#produz um arquivo contendo os pesos iniciais da rede, antes do treinamento

mlp.run_trains(treino_conjunto, validacao_conjunto, 130, learning_rate = 0.01, stop_error=0.000001) #executa o treinamento da rede

salvar_pesos_externo(mlp, "modelo_mlp_pesos_finais.json") #produz um arquivo contendo os pesos finais da rede
salvar_relatorio_externo(mlp, validacao_conjunto, "saidas_teste.json") #produz o arquivo com as saídas produzidas pela rede neural para cada um dos dados de teste 

avaliacao_teste(mlp, validacao_conjunto) #realiza a avaliação do modelo no conjunto de teste, calculando o erro final (MSE) e a acurácia final
apresentar_matriz_confusao(mlp, validacao_conjunto)






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