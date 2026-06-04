"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from src.core.network import MultilayerPerceptron

""" Classe auxiliar para EarlyStopping

patience: qual número de épocas sem melhora de validação é suportada
min_delta: qual a melhoria mínima para considerar uma "melhora de validação"

OBS: A implementação de EarlyStopping aqui para por não-melhoria do erro de validação
e não da acurácia

- essa definição de estratégia mais genérica poderia ser usada para ter várias estratégias
diferentes de EarlyStopping, mas no caso só implementamos a padrão com snapshot.
"""
class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta

        #variáveis usadas p/ fazer o cálculo do earlystopping usando snapshot
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_snapshot = None
    
    def save_snapshot(self, model: MultilayerPerceptron):
        self.best_snapshot = [
            [(list(neuron.weights), neuron.bias) for neuron in layer.neurons]
            for layer in model.layers
        ]
    
    def restore_snapshot(self, model: MultilayerPerceptron):
        for layer, layer_snapshot in zip(model.layers, self.best_snapshot):
            for neuron, (weights, bias) in zip(layer.neurons, layer_snapshot):
                neuron.weights = list(weights)
                neuron.bias = bias
    
    #método que avalia se o modelo melhorou, retornando True se o treino deve ser interrompido, ou falso caso contrário
    def should_stop(self, current_val_loss: float, model: MultilayerPerceptron) -> bool:
        if current_val_loss < (self.best_val_loss - self.min_delta):
            #se teve melhora significativa, salva o melhor valor, zera o contador e salva snapshot
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
            self.save_snapshot(model)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"\nO erro de validação não melhorou por {self.patience} épocas seguidas.")
                self.restore_snapshot(model)
                print("[Early Stopping] Melhores pesos restaurados.")
                return True
        return False
