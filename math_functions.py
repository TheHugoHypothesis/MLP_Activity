import math, random

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