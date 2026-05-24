from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def step(self, layer_neurons, learning_rate: float):
        pass

class SGD(Optimizer):
    def step(self, layer_neurons, learning_rate: float):
        for neuron in layer_neurons:
            for i in range(len(neuron.weight_list)):
                neuron.weight_list[i] -= learning_rate * neuron.delta_k * neuron.last_entry[i]
            neuron.bias -= learning_rate * neuron.delta_k

class SGD_momentum(Optimizer):
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum

        self.v_w = {}  # Mapeia: neurônio -> lista de velocidades dos pesos
        self.v_b = {}  # Mapeia: neurônio -> velocidade do bias
    def step(self, layer_neurons, learning_rate: float):
        for neuron in layer_neurons:
            #Se for a primeira iteração do neurônio, inicializa as velocidades com zero
            if neuron not in self.v_w:
                self.v_w[neuron] = [0.0] * len(neuron.weight_list)
                self.v_b[neuron] = 0.0

            #Atualização de graidente
            for i in range(len(neuron.weight_list)):
                gradient = neuron.delta_k * neuron.last_entry[i]
                
                # Equação do momentum: v = beta * v_anterior + lr * gradiente
                self.v_w[neuron][i] = (self.momentum * self.v_w[neuron][i]) + (learning_rate * gradient)
                
                # w = w - v
                neuron.weight_list[i] -= self.v_w[neuron][i]
            #Atualização com Momento para o Viés
            bias_gradient = neuron.delta_k
            self.v_b[neuron] = (self.momentum * self.v_b[neuron]) + (learning_rate * bias_gradient)
            neuron.bias -= self.v_b[neuron]