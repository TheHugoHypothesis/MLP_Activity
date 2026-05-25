from src.core.trainer_optimizer import *
from src.core.network import MultilayerPerceptron
from src.core.trainer import Trainer

from src.utils.data_loader import DataLoader
from src.utils.dataset_utils import DatasetUtils
from src.utils.io_manager import IOManager

from src.core.layer import LayerConfig
from src.utils.loss_functions import *
from src.utils.activation_function import *
from src.utils.weight_initializers import *
from src.core.cross_validation import run_stratified_k_fold

from src.evaluation.evaluator import Evaluator
import numpy as np
import random

def build_model():
    return MultilayerPerceptron(
        layer_configs=[
            LayerConfig(
                n_neurons=64,
                activation=RELU(),
                initializer=HeInitializer()
            ), # Camada escondida
            LayerConfig(
                n_neurons=26,
                activation=Linear(),
                initializer=XavierGlorotInitializer()
            ) # Camada de saída
        ],
        input_size=120
    )


def build_trainer(model):
    return Trainer(
        model=model,
        loss_function=SoftmaxCrossEntropy(),
        optimizer=SGD_momentum(momentum=0.9, l2_decay=0.0001),
        learning_rate=0.001,
        patience=10,
        min_delta=0.0001
    )


def main():

    random.seed(3)

    #IO MANAGER
    io = IOManager()
    experiment_id = "exp_001"  
    use_cross_validation = True

    #DATASET
    dataset = DataLoader.load_character_from_alphabet(
        "data/raw/X.npy",
        "data/raw/Y_classe.npy"
    )

    x, y = dataset[0]
    print(min(x), max(x))

    if use_cross_validation:
        result = run_stratified_k_fold(
            dataset=dataset,
            k=3,
            build_model=build_model,
            build_trainer=build_trainer,
            epochs=400
        )

        io.save_report({
            "summary": result["summary"],
            "folds": result["folds"]
        }, experiment_id + "_cv_report")
    else:
        train_set, val_set, test_set = DatasetUtils.stratified_split(
            dataset=dataset,
            p_train=0.7,
            p_val=0.15
        )

        mlp = build_model()

        #SALVA MODELO INICIAL
        io.save_model(mlp, experiment_id + "_init")

        trainer = build_trainer(mlp)

        #LOOP TREINO
        epochs = 400
        history = trainer.train(
            train_dataset=train_set,
            val_dataset=val_set,
            epochs=epochs
        )

        evaluator = Evaluator(mlp, loss_function=SoftmaxCrossEntropy())

        print("\n=== CONJUNTO DE TREINO ===")
        train_metrics = evaluator.evaluate(train_set, num_classes=26)

        print("\n=== CONJUNTO DE VALIDAÇÃO ===")
        val_metrics = evaluator.evaluate(val_set, num_classes=26)

        #SAVE FINAL MODELO + REPORT
        io.save_model(mlp, experiment_id + "_final")
        io.save_report({
            "history": history,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }, experiment_id + "_report")


if __name__ == "__main__":
    main()