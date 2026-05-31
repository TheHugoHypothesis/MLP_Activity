"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from src.core.network import MultilayerPerceptron
from src.core.trainer import Trainer

from src.utils.data_loader import DataLoader
from src.utils.dataset_utils import DatasetUtils
from src.utils.io_manager import IOManager
from src.utils.cross_validation import run_stratified_k_fold

from src.core.layer import LayerConfig

from src.strategies.trainer_optimizer import *
from src.strategies.loss_functions import *
from src.strategies.activation_function import *
from src.strategies.weight_initializers import *

from src.evaluation.evaluator import Evaluator

from gerar_grafico import plot_confusion_matrix

import numpy as np
import random
import os

def get_model_layer_configs():
    return [
        LayerConfig(
            n_neurons=64,
            activation=RELU(),
            initializer=HeInitializer()
        ),
        LayerConfig(
            n_neurons=26,
            activation=Linear(),
            initializer=XavierGlorotInitializer()
        )
    ]


def get_experiment_config(use_cross_validation: bool, epochs: int = 400):
    layer_configs = get_model_layer_configs()

    return {
        "mode": "cross_validation" if use_cross_validation else "train_val_test",
        "model": {
            "input_size": 120,
            "layers": [
                {
                    "n_neurons": config.n_neurons,
                    "activation": config.activation.__class__.__name__,
                    "activation_params": dict(config.activation.__dict__),
                    "initializer": config.initializer.__class__.__name__,
                    "initializer_params": dict(config.initializer.__dict__),
                }
                for config in layer_configs
            ]
        },
        "training": {
            "epochs": epochs,
            "learning_rate": 0.001,
            "patience": 10,
            "min_delta": 0.0001,
            "loss_function": "SoftmaxCrossEntropy",
            "optimizer": {
                "name": "SGD_momentum",
                "momentum": 0.9,
                "l2_decay": 0.0001,
            }
        },
        "data_split": {
            "train": 0.7,
            "val": 0.15,
            "test": 0.15,
        },
        "cross_validation": {
            "enabled": use_cross_validation,
            "k": 3 if use_cross_validation else None,
        },
        "evaluation": {
            "num_classes": 26,
        }
    }


def build_model():
    return MultilayerPerceptron(
        layer_configs=get_model_layer_configs(),
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
    use_cross_validation = False
    num_epochs = 400

    #DATASET
    dataset = DataLoader.load_character_from_alphabet(
        "data/raw/X.npy",
        "data/raw/Y_classe.npy"
    )

    x, y = dataset[0]
    print(min(x), max(x))

    
    run_name = io.start_run(experiment_id)
    io.save_experiment_config(
        get_experiment_config(use_cross_validation, epochs=num_epochs),
        experiment_id + ("_cross_validation_experiment_config" if use_cross_validation else "_experiment_config")
    )

    if use_cross_validation:
        result = run_stratified_k_fold(
            dataset=dataset,
            k=3,
            build_model=build_model,
            build_trainer=build_trainer,
            epochs=num_epochs,
            num_classes=26,
            io_manager=io,
            experiment_id=experiment_id
        )
        
    else:
        train_set, val_set, test_set = DatasetUtils.stratified_split(
            dataset=dataset,
            p_train=0.7,
            p_val=0.15
        )

        mlp = build_model()

        #SALVA MODELO INICIAL
        io.save_model(mlp, experiment_id + "_initial_weights")

        trainer = build_trainer(mlp)

        #LOOP TREINO
        history = trainer.train(
            train_dataset=train_set,
            val_dataset=val_set,
            epochs=num_epochs
        )

        io.save_training_history(history, experiment_id + "_training_history")

        evaluator = Evaluator(mlp, loss_function=SoftmaxCrossEntropy())

        print("\n=== CONJUNTO DE TREINO ===")
        train_metrics = evaluator.evaluate(train_set, num_classes=26)

        print("\n=== CONJUNTO DE VALIDAÇÃO ===")
        val_metrics = evaluator.evaluate(val_set, num_classes=26)

        try:
            if val_metrics.get("confusion_matrix") is not None:
                plot_confusion_matrix(
                    np.array(val_metrics["confusion_matrix"]),
                    io.figures_dir,
                    labels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                    out_filename=experiment_id + "_validation_confusion_matrix.png"
                )
        except Exception as e:
            print(f"Aviso: falha ao salvar matriz de validação: {e}")

        print("\n=== CONJUNTO DE TESTE ===")
        test_metrics = evaluator.evaluate(test_set, num_classes=26)

        try:
            if test_metrics.get("confusion_matrix") is not None:
                plot_confusion_matrix(
                    np.array(test_metrics["confusion_matrix"]),
                    io.figures_dir,
                    labels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                    out_filename=experiment_id + "_test_confusion_matrix.png"
                )
        except Exception as e:
            print(f"Aviso: falha ao salvar matriz de teste: {e}")

        #SAVE FINAL MODELO + REPORT
        io.save_model(mlp, experiment_id + "_final_weights")
        io.save_predictions(mlp, test_set, experiment_id + "_test_outputs")   


if __name__ == "__main__":
    main()