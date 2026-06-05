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
from src.strategies.early_stopping import EarlyStopping
from src.strategies.classification_strategy import ClassificationStrategy
from src.core.network import MultilayerPerceptron

"""
Classe de Trainer, que pega uma rede Multilayer Perceptron e é capaz de:
- fazer backpropagation;
- atualizar os pesos usando um Otimizador específico;
- rodar uma época de treino e medir acurácia;
- usar uma estratégia de Early Stopping

model: corresponde à classe do MLP instanciada
loss_function: qual medida de erro usar
optimizer: qual otimizador usar
classification_strategy: qual estratégia de escolha de saida usar
learning_rate: taxa de aprendizado
early_stopping: define a estratégia de early stopping adotada

OBS:
- Inicialmente, fizemos atualizações de pesos por meio de listas mais explícitas do python,
fazendo for-range diretamente. Contudo, o Python tem um grande downside nesse sentido,
uma vez que loops explícitos são menos eficientes porque lidam com objetos genéricos Python,
ao invés de funcoes em C otimizadas. Isso limitou bastante a velocidade da nossa rede.
Por isso, em certas partes, substituímos loops-explícitos por list-comprehension e pelo
uso da função zip(...) que são C-nativas, e cerca de 30% mais rápidas em nossos testes.
"""
class Trainer:
    def __init__(
        self,
        model: MultilayerPerceptron,
        loss_function: LossFunction,
        optimizer: Optimizer,
        classification_strategy: ClassificationStrategy,
        learning_rate: float = 0.01,        
        early_stopping: EarlyStopping = None #por padrão, nao usa Early Stopping
    ):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.classification_strategy = classification_strategy
    
    #Função de backpropagation que só calcula os deltas (os gradientes), mas não realiza atualização
    def backpropagate(self, y_real: List[float]):

        #Para calcular a perda da ultima camada, calcula a derivada da função de perda comparando
        #com a saída da rede na com o valor esperado (que está no vetor y_real). Retorna um vetor
        #de mesmo tamanho em que para cada i-ésima posição corresponde
        loss_gradient_list = self.loss_function.derivative(
            self.model.last_outputs[-1],
            y_real
        )

        #Cálculo do delta para última camada
        last_layer = self.model.layers[-1]

        # o zip combina cada neurônio da saída com seu respectivo gradiente de erro calculado acima
        for neuron, loss_gradient in zip(last_layer.neurons, loss_gradient_list):
            # aplica a regra da cadeia: delta_k = derivada da ativação * gradiente do erro
            neuron.delta_k = neuron.activation.derivative(
                neuron.last_local_induced_field,
                neuron.output
            ) * loss_gradient

            #popula gradiente para última camada
            self.update_neuron_gradients(neuron=neuron)

        #Cálculo do delta para camadas seguintes (percorrendo de trás para frente)
        for l in reversed(range(len(self.model.layers) - 1)):
            actual_layer = self.model.layers[l]
            next_layer = self.model.layers[l + 1]
            
            if self.model.use_numpy:
                import numpy as np
                deltas = np.array([neuron_k.delta_k for neuron_k in next_layer.neurons])

            #Calcula o delta de cada neurônio da camada atual
            for i, neuron in enumerate(actual_layer.neurons):
                #Implementa o produto escalar de delta da próxima camada * peso da próxima
                # para estimar o erro da camada anterior.
                if self.model.use_numpy:
                    weights_i = np.array([neuron_k.parameter.weights[i] for neuron_k in next_layer.neurons])
                    sum_delta_k = np.dot(deltas, weights_i)
                else:
                    sum_delta_k : float = 0.0
                    for neuron_k in next_layer.neurons:
                        sum_delta_k += neuron_k.delta_k * neuron_k.parameter.weights[i]
                
                #Regra da cadeia para camada oculta: Somatório dos erros propagados * derivada da ativação local
                neuron.delta_k = sum_delta_k * neuron.activation.derivative(
                    neuron.last_local_induced_field,
                    neuron.output
                )

                #popula os gradientes para camadas seguintes
                self.update_neuron_gradients(neuron=neuron)
    
    #Método que serve para guardar os gradientes internos do neurônio
    #durante a fase de backpropagation, antes do ajuste de pesos.
    def update_neuron_gradients(self, neuron):
        neuron.parameter.bias_gradient = neuron.delta_k

        if self.model.use_numpy:
            import numpy as np
            neuron.parameter.weights_gradient = neuron.delta_k * np.array(neuron.last_entry)
        else:
            neuron.parameter.weights_gradient = [neuron.delta_k * x for x in neuron.last_entry]
    
    #Aciona o otimizador para aplicar os gradientes acumulados e alterar os pesos reais da rede
    def update_weights(self):
        for layer in self.model.layers:
            self.optimizer.step(layer_neurons=layer.neurons, learning_rate=self.learning_rate)
    
    #Função que serve para avaliar o modelo em um conjunto de dados sem treinar
    #isso serve para Early stopping, já que retorna o erro de validação.
    def evaluate(self, dataset: List) -> tuple[float, float]:
        total_loss = 0.0
        correct = 0
        for x, y in dataset:
            prediction = self.model.forward(x)
            
            #Acumula perda
            total_loss += self.loss_function.compute(prediction, y)
            
            #Acumula acertos
            predicted_class = self.classification_strategy.predict_class(prediction)
            expected_class = self.classification_strategy.predict_class(y)

            if predicted_class == expected_class:
                correct += 1
        

        if len(dataset) > 0:
            avg_loss = total_loss / len(dataset)
            accuracy = correct / len(dataset)
        else:
            avg_loss = 0.0
            accuracy = 0.0

        return avg_loss, accuracy

    #Executa o ciclo completo de treinamento da MLP (Épocas, Forward, Backpropagation, Ajuste de Pesos e Validação)
    def train(
        self,
        train_dataset: List,
        val_dataset: List,
        epochs: int
    ):
        #Dicionário de histórico para armazenar as métricas de evolução ao longo das épocas
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        for epoch in range(epochs):
            total_train_loss: float = 0.0
            correct_train = 0

            for x, y in train_dataset:
                y_prediction = self.model.forward(x)
                loss = self.loss_function.compute(y_prediction, y)
                self.backpropagate(y)
                self.update_weights()

                total_train_loss += loss
                
                #usa a estratégia de classificação p/ ver se classificou corretamente
                #obs: garantir que a saída de y seja sempre um vetor mesmo que unitário
                if self.classification_strategy.predict_class(y_prediction) == self.classification_strategy.predict_class(y):
                    correct_train += 1

            
            average_train_loss = total_train_loss / len(train_dataset)
            average_train_acc = correct_train / len(train_dataset)

            # Roda a validação para checar se a rede está generalizando bem ou sofrendo Overfitting
            average_val_loss, average_val_acc = self.evaluate(val_dataset)

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
            if self.early_stopping is not None:
                should_stop = self.early_stopping.should_stop(average_val_loss, self.model)

                if should_stop:
                    print(f"Early stop na época {epoch}.")
                    break

        return history
