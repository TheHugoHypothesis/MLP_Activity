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
    def __init__(
        self,
        matriz: List[List[int]]
    ):
        self.matriz = matriz

    @staticmethod
    def compute(dataset: Dataset, model, num_classes: int):
        matriz = [[0] * num_classes for _ in range(num_classes)]

        for x, y in dataset:
            pred_out = model.forward(x)
            pred = pred_out.index(max(pred_out))
            true = y.index(max(y))

            matriz[true][pred] += 1

        return ConfusionMatrix(matriz)
    
    def tolist(self) -> List[List[int]]:
        return self.matriz
    
    def __str__(self) -> str:
        lines = []
        for row in self.matriz:
            lines.append(" ".join(f"{val:4d}" for val in row))
        return "\n".join(lines)

