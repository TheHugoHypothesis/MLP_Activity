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

from src.utils.console import exibir_dashboard_configuracoes, Timer, exibir_dashboard_tempos

""" Configuração a ser usada """
CONFIG = {
    "experiment_id": "exp_008",
    "use_numpy": True, #True para usar numpy (mais rápido e saida equivalente) ou False (mais lento, mas menos lib externa)

    #Configurações de separação do Dataset
    "use_cross_validation": False,#false=usa holdout, true=usa cross_validation
    "cross_validation_folds": 3,
    "hold_out_p_train": 0.7, 
    "hold_out_p_validation": 0.15, #o complemento 1 - (hold_out_p_train + hold_out_p_validation) é implicitamente hold_out_p_test

    #Configurações de backpropagation e early stop
    "num_epochs": 600,
    "learning_rate": 0.1, #0.001 funciona bem para softmax_cross_entropy, 0.01 funciona bem para MSE
    "patience": 20,
    "min_delta": 0.0001,

    #Configurações de entradas/saídas
    "input_size": 120,
    "num_classes": 26,

    #configurações de treinamento
    "loss_function": "mse", #possíveis valores: ["mse", "mae", "softmax_cross_entropy"]
    "classification_strategy": "argmax", #possíveis valores: ["argmax", "argmax_random", "threshold"]
    "classification_threshold": 0.5, #valor para usar com classification_strategy = "threshold", caso contrário ignorado
    "optimizer": {
        "type": "sgd_momentum", #possíveis valores: ["sgd", "sgd_momentum"]
        "momentum": 0.9,
        "l2_decay": 0.0
    },

    #configurações de camadas
    "layers": [
        {
            "n_neurons": 72,#40 neuronios parece funcionar bem para softmax_cross_entropy e MSE
            "activation": "sigmoid", #possíveis valores: ["relu", "leaky_relu", "linear", "sigmoid"].
            "initializer": "xavier" #possíveis valores: ["uniform", "normal", "he", "xavier"]
        },
        {
            "n_neurons": 26,
            "activation": "sigmoid", #possíveis valores: ["relu", "leaky_relu", "linear", "sigmoid"]. Use linear na saída ao usar softmax_cross_entropy, sigmoid para MSE.
            "initializer": "xavier"  #possíveis valores: ["uniform", "normal", "he", "xavier"]
        }
    ],

    #configurações de dados e randomização
    "random_seed": 3,
    "x_path": "data/raw/X.npy",
    "y_path": "data/raw/Y_classe.npy",

    #configurações de ruído
    #permite mesclar classes informando letras (como D e O, I e J), para desativar e usar as 26, basta definir como None
    #exemplo de mesclagem: "merge_classes": [["D", "O"], ["I", "J"]], 
    "merge_classes": None,
    #habilita estratificação na estratégia escolhida de dataset (seja holdout ou k-fold)
    "use_stratification": True
}

def main():
    random.seed(CONFIG["random_seed"])
    try:
        import numpy as np
        np.random.seed(CONFIG["random_seed"])
    except ImportError:
        pass
    tempos = {}

    io = IOManager()

    with Timer() as t_data:
        dataset = DataLoader.load_character_from_alphabet(
            CONFIG["x_path"],
            CONFIG["y_path"]
        )
    tempos["load_data"] = t_data.interval

    #Modificações do dataset de ruído
    num_classes = CONFIG["num_classes"]
    if CONFIG.get("merge_classes"):
        dataset, num_classes = DatasetUtils.merge_classes(dataset, CONFIG["merge_classes"])
        CONFIG["num_classes"] = num_classes
        CONFIG["layers"][-1]["n_neurons"] = num_classes
        print(f"\n[Dataset] Mesclagem de classes. Nova dimensão de saída: {num_classes} classes.")
    
    exibir_dashboard_configuracoes(CONFIG)
    run_name = io.start_run(CONFIG["experiment_id"])
    io.save_experiment_config(CONFIG, f"{CONFIG['experiment_id']}_experiment_config")

    if CONFIG["use_cross_validation"]:
        with Timer() as t_train:
            result = run_stratified_k_fold(
                dataset=dataset,
                k=CONFIG["cross_validation_folds"],
                build_model=lambda: build_model(CONFIG),
                build_trainer=lambda m: build_trainer(m, CONFIG),
                epochs=CONFIG["num_epochs"],
                num_classes=CONFIG["num_classes"],
                io_manager=io,
                experiment_id=CONFIG["experiment_id"],
                use_stratification=CONFIG.get("use_stratification", True)
            )
        tempos["train"] = t_train.interval
        
        result["experiment_id"] = CONFIG["experiment_id"]
        result["num_epochs"] = CONFIG["num_epochs"]
        result["patience"] = CONFIG.get("patience")
        io.save_training_history(result, f"{CONFIG['experiment_id']}_cross_validation_report")
        epocas_reais = sum(len(fold["history"]["train_loss"]) for fold in result["folds"])
        exibir_dashboard_tempos(tempos, n_epochs=epocas_reais)
        
    else:
        #holdout estratificado
        if CONFIG.get("use_stratification", True):
            train_set, val_set, test_set = DatasetUtils.stratified_split(
                dataset=dataset,
                p_train=CONFIG["hold_out_p_train"],
                p_val=CONFIG["hold_out_p_validation"]
            )
        #holdout randomizado
        else:
            train_set, val_set, test_set = DatasetUtils.random_split(
                dataset=dataset,
                p_train=CONFIG["hold_out_p_train"],
                p_val=CONFIG["hold_out_p_validation"]
            )

        mlp = build_model(CONFIG)
        io.save_model(mlp, f"{CONFIG['experiment_id']}_initial_weights")

        trainer = build_trainer(mlp, CONFIG)

        with Timer() as t_train:
            #LOOP TREINO
            history = trainer.train(
                train_dataset=train_set,
                val_dataset=val_set,
                epochs=CONFIG["num_epochs"]
            )

        tempos["train"] = t_train.interval
        io.save_training_history(history, CONFIG['experiment_id'] + "_training_history")

        evaluator = Evaluator(mlp, classification_strategy=trainer.classification_strategy, loss_function=trainer.loss_function)

        print("\n=== CONJUNTO DE TREINO ===")
        train_metrics = evaluator.evaluate(train_set, num_classes=CONFIG["num_classes"])

        print("\n=== CONJUNTO DE VALIDAÇÃO ===")
        val_metrics = evaluator.evaluate(val_set, num_classes=CONFIG["num_classes"])

        print("\n=== CONJUNTO DE TESTE ===")
        test_metrics = evaluator.evaluate(test_set, num_classes=CONFIG["num_classes"])

        report = {
            "experiment_id": CONFIG["experiment_id"],
            "num_epochs": CONFIG["num_epochs"],
            "patience": CONFIG.get("patience"),
            "history": history,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics
        }

        #SAVE FINAL MODELO + REPORT
        io.save_model(mlp, CONFIG['experiment_id'] + "_final_weights")
        io.save_predictions(mlp, test_set, CONFIG['experiment_id'] + "_test_outputs")   
        io.save_report(report, f"{CONFIG['experiment_id']}_report")

        epocas_reais = len(history["train_loss"])
        exibir_dashboard_tempos(tempos, n_epochs=epocas_reais)

if __name__ == "__main__":
    main()