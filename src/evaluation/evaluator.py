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
    """
    Classe de utildiades para avaliação final de desempenho do modelo
    Ou seja, é usado pós-convergência para extrair as métricas do conjunto de teste ou validação
    como Erro Médio, Acurácia e gerar a matriz de confusão
    """

    def __init__(self, model, classification_strategy, loss_function=None):
        self.model = model
        self.classification_strategy = classification_strategy
        self.loss_function = loss_function

    def evaluate(self, dataset, num_classes: int = None):
        """
        Executa a inferência em todo o dataset fornecido e calcula as métricas estatísticas

        O método realiza o passo de alimentação direta (forward) para cada amostra, 
        contabiliza a taxa de acerto absoluto (acurácia) e opcionalmente calcula a perda 
        média e constrói a matriz de confusão correspondente.
        """

        total_loss = 0.0
        correct = 0

        #Loop de feedforward sobre os dados de teste
        for x, y in dataset:
            out = self.model.forward(x)

            #Avaliação de erro com base na função de perda
            if self.loss_function is not None:
                total_loss += self.loss_function.compute(out, y)

            #Avaliação de acurácia: Compara se a classe inferida pelo modelo coincide com a classe real (alvo)
            if self.classification_strategy.predict_class(out) == self.classification_strategy.predict_class(y):
                correct += 1

        acc = correct / len(dataset)

        #Exibe os dados de avaliação formatados no console
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
        
        #Adiciona dinamicamente o nome e valor da Loss obtida ao dicionário de resultados
        #Aqui, como essa classe não sabe qual a função de perda usada, ele usa pelo nome da classe
        #em __class__.__name__.
        if self.loss_function is not None:
            results[self.loss_function.__class__.__name__.lower()] = avg_loss

        #Geração da Matriz de Confusão (Executada apenas se 'num_classes' for explicitamente informado)
        if num_classes is not None:
            cm = ConfusionMatrix.compute(dataset, self.model, num_classes, classification_strategy=self.classification_strategy)
            print("\nConfusion Matrix:")
            print(cm)
            if cm is not None:
                results["confusion_matrix"] = cm.tolist()

        return results