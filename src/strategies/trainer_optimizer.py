"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from abc import ABC, abstractmethod
from typing import List
from src.core.parameter import Parameter

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
            weights = neuron.weights
            param = neuron.parameter

            for i in range(len(weights)):
                weights[i] -= learning_rate * param.weights_gradient[i]
            neuron.bias -= learning_rate * param.bias_gradient

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
Foi adicionado também o termo de regularização L2: `l2_decay`
- Ela é desligada por padrão;
- Aqui como aplicamos no otimizador, não usamos o quadrado e sim a derivada disso,
portanto a fórmula de atualização é a simples multiplicação.
"""
class SGD_momentum(Optimizer):
    def __init__(self, momentum: float = 0.9, l2_decay: float = 0.0):
        self.momentum = momentum
        self.l2_decay = l2_decay

        #Vetores que guardam as velocidades acumladas no momentum
        self.weight_velocity = {}
        self.bias_velocity = {}
    
    def step(self, layer_neurons, learning_rate: float):
        for neuron in layer_neurons:
            weights = neuron.weights
            param = neuron.parameter
            gradients = param.weights_gradient

            #Inicializa as velocidades na primeira atualização
            if param not in self.weight_velocity:
                self.weight_velocity[param] = [0.0] * len(weights)
                self.bias_velocity[param] = 0.0
            
            weight_vel = self.weight_velocity[param]

            for i in range(len(weights)):
                gradient = param.weights_gradient[i]
                if self.l2_decay > 0.0:
                    gradient += self.l2_decay * weights[i]

                #Atualização da velocidade (faz média móvel exponencial dos gradientes)
                weight_vel[i] = self.momentum * weight_vel[i] + learning_rate * gradient
                weights[i] -= weight_vel[i]

            #Cálculo de velocidade e momentum do bias
            #basicamente é uma repetição do código anterior mas para o bias sozinho
            bias_vel = self.bias_velocity[param]
            new_bias_vel = self.momentum * bias_vel + learning_rate * param.bias_gradient
            self.bias_velocity[param] = new_bias_vel
            neuron.bias -= new_bias_vel