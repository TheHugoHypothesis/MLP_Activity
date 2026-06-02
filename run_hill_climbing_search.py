"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

"""
Script principal para execução da otimização de hiperparâmetros.
Utiliza a estratégia de Hill Climbing (Subida de Encosta) para encontrar
a melhor combinação de parâmetros para a rede neural a partir de um espaço de busca.
"""

from src.utils.hill_climbing_search import run_hill_climbing_search
from src.utils.data_loader import DataLoader
from src.utils.io_manager import IOManager
from main import CONFIG


def main():
    # Cria uma cópia da configuração base definida no main.py
    base_config = CONFIG.copy()

    dataset = DataLoader.load_character_from_alphabet(
        base_config["x_path"],
        base_config["y_path"]
    )

    io = IOManager()

    # Define o espaço de busca para a otimização.
    # As listas devem estar ordenadas, pois o algoritmo "caminha" por elas iterativamente.
    grid_space = {
        "hidden_neurons": [16, 32, 40, 48, 56, 64, 72, 128],
        "activation": ["relu", "leaky_relu", "sigmoid", "linear"], #ativação para intermediária 
        "loss_function": ["mse"],
        "initializer": ["he", "xavier","uniform", "normal"],
        "num_epochs": [200, 400, 600, 800, 1600],
        "learning_rate": [0.0001, 0.001, 0.01, 0.05, 0.1],
        "patience": [3, 5, 10, 20, 40],
        "optimizer_type": ["sgd", "sgd_momentum"],
        "momentum": [0.0, 0.5, 0.8, 0.9, 0.99],
        "l2_decay": [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
        "p_train": [0.7, 0.8],
    }

    # Executa a busca através do Hill Climbing e salva os relatórios resultantes
    report = run_hill_climbing_search(
        dataset=dataset,
        base_config=base_config,
        grid_space=grid_space,
        io_manager=io,
        experiment_id=base_config["experiment_id"] + "_hill"
    )

    # Exibe os resultados no terminal para facilitar a cópia para o main.py
    print("\n" + "="*50)
    print(" BUSCA FINALIZADA! MELHOR CONFIGURAÇÃO ENCONTRADA ")
    print("="*50)
    
    best_combo = report["best_model"]["combo"]
    best_acc = report["best_model"]["val_accuracy"]
    
    for param, value in best_combo.items():
        print(f" - {param}: {value}")
        
    print(f"\n Acurácia de Validação: {best_acc:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
