from src.evaluation.metrics import Metrics
from src.evaluation.confusion_matrix import ConfusionMatrix
import numpy as np


class Evaluator:
    def __init__(self, model, loss_function=None):
        self.model = model
        self.loss_function = loss_function

    def evaluate(self, dataset, num_classes: int = None):
        total_loss = 0.0
        correct = 0

        for x, y in dataset:
            out = self.model.forward(x)

            if self.loss_function is not None:
                total_loss += self.loss_function.compute(out, y)

            # accuracy
            if np.argmax(out) == np.argmax(y):
                correct += 1

        acc = correct / len(dataset)

        print("\n--- Avaliação ---")
        if self.loss_function is not None:
            avg_loss = total_loss / len(dataset)
            loss_name = self.loss_function.__class__.__name__
            print(f"{loss_name}: {avg_loss:.6f}")

        print(f"Acurácia: {acc * 100:.2f}%")

        cm = None

        # matriz de confusão opcional
        if num_classes is not None:
            cm = ConfusionMatrix.compute(dataset, self.model, num_classes)
            print("\nConfusion Matrix:")
            print(cm)

            results = {
                "accuracy": acc,
                "confusion_matrix": cm.tolist() if cm is not None else None
            }

            if self.loss_function is not None:
                results[self.loss_function.__class__.__name__.lower()] = avg_loss

        return results