from abc import ABC, abstractmethod

"""
Classe abstrata que implementa algoritmos de otimização
Ela facilita o uso no main(), uma vez que pode ser usada
como parâmetro configurável do Trainer().

A função de step() implementa um passo de atualização de pesos
É esperado tambem que seja passado já uma lista de neurônios na camada em
`layer_neurons`, ao invés de passar o objeto de PerceptronLayer.
"""
class Optimizer(ABC):
    @abstractmethod
    def step(self, layer_neurons, learning_rate: float):
        pass

""" Implementação de SGD padrão
Atualização dos pesos:
w_i = w_i - η * δ_k * x_i

Atualização do bias:
b = b - η * δ_k

η  -> learning rate
δ_k -> gradiente/erro do neurônio
x_i -> entrada do neurônio
"""
class SGD(Optimizer):
    def step(self, layer_neurons, learning_rate: float):
        for neuron in layer_neurons:
            for i in range(len(neuron.weight_list)):
                neuron.weight_list[i] -= learning_rate * neuron.delta_k * neuron.last_entry[i]
            neuron.bias -= learning_rate * neuron.delta_k

""" Implementação de SGD com Momentum
Gradiente: g_i = δ_k * x_i
Velocidade (momentum): v_i(t) = μ * v_i(t-1) + η * g_i
Atualização do peso: w_i = w_i - v_i
Bias:
    v_b(t) = μ * v_b(t-1) + η * δ_k
    b = b - v_b

μ -> momentum
η -> learning rate
δ_k -> erro do neurônio
x_i -> entrada

No caso, basicamente, fazemos uma média móvel exponencial dos gradientes
"""
class SGD_momentum(Optimizer):
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum

        #Vetores que guardam as velocidades acumladas no momentum
        self.weight_velocity = {}
        self.bias_velocity = {}
    
    def step(self, layer_neurons, learning_rate: float):
        for neuron in layer_neurons:

            #Inicializa as velocidades na primeira atualização
            if neuron not in self.weight_velocity:
                initial_velocity = []

                for i in neuron.weight_list:
                    initial_velocity.append(0.0)

                self.weight_velocity[neuron] = initial_velocity
                self.bias_velocity[neuron] = 0.0

            for i in range(len(neuron.weight_list)):
                input_value = neuron.last_entry[i]
                gradient = neuron.delta_k * input_value
                previous_velocity = self.weight_velocity[neuron][i]

                #Atualização da velocidade (faz média móvel exponencial dos gradientes)
                new_velocity = self.momentum * previous_velocity + learning_rate * gradient

                self.weight_velocity[neuron][i] = new_velocity

                current_weight = neuron.weight_list[i]
                updated_weight = current_weight - new_velocity

                #Atualização do peso usando a velocidade acumulada
                neuron.weight_list[i] = updated_weight

            #Cálculo de velocidade e momentum do bias
            #basicamente é uma repetição do código anterior mas para o bias sozinho
            bias_gradient = neuron.delta_k
            previous_bias_velocity = self.bias_velocity[neuron]
            new_bias_velocity = self.momentum * previous_bias_velocity + learning_rate * bias_gradient
            self.bias_velocity[neuron] = new_bias_velocity
            neuron.bias -= new_bias_velocity