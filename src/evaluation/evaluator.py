from src.evaluation.metrics import Metrics
from src.evaluation.confusion_matrix import ConfusionMatrix
import numpy as np


class Evaluator:
    def __init__(self, model):
        self.model = model


    def evaluate(self, dataset, num_classes: int = None):
        total_mse = 0.0
        correct = 0

        for x, y in dataset:
            out = self.model.forward(x)

            # MSE por amostra
            total_mse += Metrics.mse(out, y)

            # accuracy
            if np.argmax(out) == np.argmax(y):
                correct += 1

        mse = total_mse / len(dataset)
        acc = correct / len(dataset)

        print("\n--- Avaliação ---")
        print(f"MSE: {mse:.6f}")
        print(f"Acurácia: {acc * 100:.2f}%")

        # matriz de confusão opcional
        if num_classes is not None:
            cm = ConfusionMatrix.compute(dataset, self.model, num_classes)
            print("\nConfusion Matrix:")
            print(cm)

        return {
            "mse": mse,
            "accuracy": acc,
            "confusion_matrix": cm.tolist()
        }