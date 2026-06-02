"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from typing import List
import numpy as np
from src.core.layer import LayerConfig

class MatrixMultilayerPerceptron:
    def __init__(
        self,
        layer_configs: List[LayerConfig],
        input_size: int
    ):
        self.W = []
        self.b = []
        
        self.act_fns = []
        self.deriv_fns = []
        
        self.last_A = []
        self.last_Z = []
        
        current_input_size = input_size
        for config in layer_configs:
            W_layer = []
            for _ in range(config.n_neurons):
                W_layer.append(config.initializer.initialize(current_input_size, config.n_neurons))
            self.W.append(np.array(W_layer, dtype=np.float64))
            self.b.append(np.zeros((config.n_neurons, 1), dtype=np.float64))
            
            act_obj = config.activation
            act_name = act_obj.__class__.__name__
            if act_name == "Sigmoid":
                self.act_fns.append(lambda Z: 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500))))
                self.deriv_fns.append(lambda Z, A: A * (1.0 - A))
            elif act_name == "RELU":
                self.act_fns.append(lambda Z: np.maximum(0.0, Z))
                self.deriv_fns.append(lambda Z, A: (Z > 0.0).astype(np.float64))
            elif act_name == "LeakyRELU":
                self.act_fns.append(lambda Z: np.where(Z > 0.0, Z, 0.01 * Z))
                self.deriv_fns.append(lambda Z, A: np.where(Z > 0.0, 1.0, 0.01))
            elif act_name == "Linear":
                self.act_fns.append(lambda Z: Z)
                self.deriv_fns.append(lambda Z, A: np.ones_like(Z))
                
            current_input_size = config.n_neurons

        self.dW = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(bias) for bias in self.b]

    def forward(self, X):
        is_single = False
        if isinstance(X, list) or (isinstance(X, np.ndarray) and X.ndim == 1):
            is_single = True
            X = np.array(X, dtype=np.float64).reshape(-1, 1)
            
        self.last_A = [X]
        self.last_Z = []
        
        A = X
        for W, b, act in zip(self.W, self.b, self.act_fns):
            Z = np.dot(W, A) + b
            self.last_Z.append(Z)
            A = act(Z)
            self.last_A.append(A)
            
        if is_single:
            return A.flatten().tolist()
        return A

    def backward(self, dY: np.ndarray):
        batch_size = dY.shape[1]
        
        Z_L = self.last_Z[-1]
        A_L = self.last_A[-1]
        act_deriv_L = self.deriv_fns[-1]
        
        dZ = dY * act_deriv_L(Z_L, A_L)
        
        A_prev = self.last_A[-2]
        self.dW[-1] = np.dot(dZ, A_prev.T) / batch_size
        self.db[-1] = np.sum(dZ, axis=1, keepdims=True) / batch_size
        
        for l in reversed(range(len(self.W) - 1)):
            W_next = self.W[l + 1]
            Z_l = self.last_Z[l]
            A_l = self.last_A[l + 1]
            act_deriv_l = self.deriv_fns[l]
            
            dZ = np.dot(W_next.T, dZ) * act_deriv_l(Z_l, A_l)
            
            A_prev = self.last_A[l]
            self.dW[l] = np.dot(dZ, A_prev.T) / batch_size
            self.db[l] = np.sum(dZ, axis=1, keepdims=True) / batch_size

    @property
    def layers(self):
        class MockNeuron:
            def __init__(self, parent_W, parent_b, index):
                self.parent_W = parent_W
                self.parent_b = parent_b
                self.index = index
            
            @property
            def weights(self):
                return self.parent_W[self.index]
                
            @weights.setter
            def weights(self, new_w):
                self.parent_W[self.index] = new_w
                
            @property
            def bias(self):
                return float(self.parent_b[self.index, 0])
                
            @bias.setter
            def bias(self, new_b):
                self.parent_b[self.index, 0] = new_b
                
        class MockLayer:
            def __init__(self, neurons):
                self.neurons = neurons
                
        mock_layers = []
        for W_l, b_l in zip(self.W, self.b):
            neurons = []
            for i in range(W_l.shape[0]):
                neurons.append(MockNeuron(W_l, b_l, i))
            mock_layers.append(MockLayer(neurons))
        return mock_layers
