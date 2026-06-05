"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from typing import List, Tuple

Dataset = List[Tuple[List[float], List[float]]]


class ConfusionMatrix:
    """
    Serve para calcular e representar a Matriz de Confusão de modelos de classificação
    """

    def __init__(
        self,
        matriz: List[List[int]]
    ):
        self.matriz = matriz

    @staticmethod
    def compute(dataset: Dataset, model, num_classes: int, classification_strategy):
        #Gera uma nova Matriz de Confusão avaliando as predições do modelo sobre um dataset.
        #Retorna uma nova instância da classe preenchida com os dados tabulados de ConfusionMatrix.

        matriz = [[0] * num_classes for _ in range(num_classes)]

        for x, y in dataset:
            pred_out = model.forward(x)
            pred = classification_strategy.predict_class(pred_out)
            true = classification_strategy.predict_class(y)

            matriz[true][pred] += 1

        return ConfusionMatrix(matriz)
    
    def tolist(self) -> List[List[int]]:
        #exporta a matriz de confusão interna para o formato de lista de listas nativa do Python
        return self.matriz
    
    def __str__(self) -> str:
        #Formata a matriz de confusão em uma string textual para exibição no console
        lines = []
        for row in self.matriz:
            lines.append(" ".join(f"{val:4d}" for val in row))
        return "\n".join(lines)

