"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from src.evaluation.confusion_matrix import ConfusionMatrix

class Evaluator:
    def __init__(self, model, classification_strategy, loss_function=None):
        self.model = model
        self.classification_strategy = classification_strategy
        self.loss_function = loss_function

    def evaluate(self, dataset, num_classes: int = None):
        total_loss = 0.0
        correct = 0

        for x, y in dataset:
            out = self.model.forward(x)

            if self.loss_function is not None:
                total_loss += self.loss_function.compute(out, y)

            # accuracy
            if self.classification_strategy.predict_class(out) == self.classification_strategy.predict_class(y):
                correct += 1

        acc = correct / len(dataset)

        print("\n--- Avaliação ---")
        if self.loss_function is not None:
            avg_loss = total_loss / len(dataset)
            loss_name = self.loss_function.__class__.__name__
            print(f"{loss_name}: {avg_loss:.6f}")

        print(f"Acurácia: {acc * 100:.2f}%")

        results = {
            "accuracy": acc,
            "confusion_matrix": None
        }

        if self.loss_function is not None:
            results[self.loss_function.__class__.__name__.lower()] = avg_loss

        # matriz de confusão opcional
        if num_classes is not None:
            cm = ConfusionMatrix.compute(dataset, self.model, num_classes, classification_strategy=self.classification_strategy)
            print("\nConfusion Matrix:")
            print(cm)
            if cm is not None:
                results["confusion_matrix"] = cm.tolist()

        return results