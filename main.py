"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

import random

from src.fabric import build_model, build_trainer
from src.evaluation.evaluator import Evaluator

from src.utils.data_loader import DataLoader
from src.utils.dataset_utils import DatasetUtils
from src.utils.io_manager import IOManager
from src.utils.cross_validation import run_stratified_k_fold
from src.utils.console import exibir_dashboard_configuracoes

""" Configuração a ser usada """
CONFIG = {
    "experiment_id": "exp_001",

    #Configurações de separação do Dataset
    "use_cross_validation": False,#false=usa holdout, true=usa cross_validation
    "cross_validation_folds": 3,
    "hold_out_p_train": 0.7, 
    "hold_out_p_validation": 0.15, #o complemento 1 - (hold_out_p_train + hold_out_p_validation) é implicitamente hold_out_p_test

    #Configurações de backpropagation e early stop
    "num_epochs": 400,
    "learning_rate": 0.001,
    "patience": 10,
    "min_delta": 0.0001,

    #Configurações de entradas/saídas
    "input_size": 120,
    "num_classes": 26,

    #configurações de treinamento
    "loss_function": "softmax_cross_entropy",
    "optimizer": {
        "type": "sgd_momentum",
        "momentum": 0.9,
        "l2_decay": 0.0001
    },

    #configurações de camadas
    "layers": [
        {
            "n_neurons": 64,
            "activation": "relu",
            "initializer": "he"
        },
        {
            "n_neurons": 26,
            "activation": "linear",
            "initializer": "xavier"
        }
    ],

    #configurações de dados e randomização
    "random_seed": 3,
    "x_path": "data/raw/X.npy",
    "y_path": "data/raw/Y_classe.npy"
}

def main():
    exibir_dashboard_configuracoes(CONFIG)
    random.seed(CONFIG["random_seed"])

    io = IOManager()
    dataset = DataLoader.load_character_from_alphabet(
        CONFIG["x_path"],
        CONFIG["y_path"]
    )
    
    run_name = io.start_run(CONFIG["experiment_id"])
    io.save_experiment_config(CONFIG, f"{CONFIG['experiment_id']}_experiment_config")

    if CONFIG["use_cross_validation"]:
        result = run_stratified_k_fold(
            dataset=dataset,
            k=CONFIG["cross_validation_folds"],
            build_model=lambda: build_model(CONFIG),
            build_trainer=lambda m: build_trainer(m, CONFIG),
            epochs=CONFIG["num_epochs"],
            num_classes=CONFIG["num_classes"],
            io_manager=io,
            experiment_id=CONFIG["experiment_id"]
        )

        io.save_training_history(result, f"{CONFIG["experiment_id"]}_cross_validation_report")
        
    else:
        #holdout estratificado
        train_set, val_set, test_set = DatasetUtils.stratified_split(
            dataset=dataset,
            p_train=CONFIG["hold_out_p_train"],
            p_val=CONFIG["hold_out_p_validation"]
        )

        mlp = build_model(CONFIG)
        io.save_model(mlp, f"{CONFIG['experiment_id']}_initial_weights")

        trainer = build_trainer(mlp, CONFIG)

        #LOOP TREINO
        history = trainer.train(
            train_dataset=train_set,
            val_dataset=val_set,
            epochs=CONFIG["num_epochs"]
        )

        io.save_training_history(history, CONFIG['experiment_id'] + "_training_history")

        evaluator = Evaluator(mlp, loss_function=trainer.loss_function)

        print("\n=== CONJUNTO DE TREINO ===")
        train_metrics = evaluator.evaluate(train_set, num_classes=CONFIG["num_classes"])

        print("\n=== CONJUNTO DE VALIDAÇÃO ===")
        val_metrics = evaluator.evaluate(val_set, num_classes=CONFIG["num_classes"])

        print("\n=== CONJUNTO DE TESTE ===")
        test_metrics = evaluator.evaluate(test_set, num_classes=CONFIG["num_classes"])

        report = {
            "experiment_id": CONFIG["experiment_id"],
            "history": history,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics
        }

        #SAVE FINAL MODELO + REPORT
        io.save_model(mlp, CONFIG['experiment_id'] + "_final_weights")
        io.save_predictions(mlp, test_set, CONFIG['experiment_id'] + "_test_outputs")   
        io.save_report(report, f"{CONFIG['experiment_id']}_report")

if __name__ == "__main__":
    main()