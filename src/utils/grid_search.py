"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from copy import deepcopy
import time
from typing import Dict, Any, List

from src.fabric import build_model, build_trainer
from src.utils.cross_validation import run_stratified_k_fold
from src.utils.dataset_utils import DatasetUtils
from src.evaluation.evaluator import Evaluator


def evaluate_combo(
    dataset,
    base_config: Dict[str, Any],
    combo: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Avalia uma única combinação de hiperparâmetros.
    A função modifica uma cópia do `base_config` com os parâmetros fornecidos e executa
    o treinamento (via validação cruzada ou holdout), retornando as métricas obtidas.
    """
    cfg = deepcopy(base_config)

    # Aplica a combinação atual nas configurações base
    if "hidden_neurons" in combo:
        cfg["layers"][0]["n_neurons"] = combo["hidden_neurons"]
    if "activation" in combo:
        cfg["layers"][0]["activation"] = combo["activation"]
    if "initializer" in combo:
        cfg["layers"][0]["initializer"] = combo["initializer"]
    if "num_epochs" in combo:
        cfg["num_epochs"] = combo["num_epochs"]
    if "learning_rate" in combo:
        cfg["learning_rate"] = combo["learning_rate"]
    if "patience" in combo:
        cfg["patience"] = combo["patience"]
    if "optimizer_type" in combo:
        cfg["optimizer"]["type"] = combo["optimizer_type"]
    if "momentum" in combo:
        cfg["optimizer"]["momentum"] = combo["momentum"]
    if "l2_decay" in combo:
        cfg["optimizer"]["l2_decay"] = combo["l2_decay"]
    if "p_train" in combo:
        cfg["hold_out_p_train"] = combo["p_train"]
    if "p_val" in combo:
        cfg["hold_out_p_validation"] = combo["p_val"]
    if "cross_validation_folds" in combo:
        cfg["cross_validation_folds"] = combo["cross_validation_folds"]

    start = time.time()

    # Executa o experimento (Validação Cruzada ou Holdout)
    if cfg.get("use_cross_validation", False):
        result = run_stratified_k_fold(
            dataset=dataset,
            k=cfg.get("cross_validation_folds", 3),
            build_model=lambda: build_model(cfg),
            build_trainer=lambda m: build_trainer(m, cfg),
            epochs=cfg.get("num_epochs", 100),
            num_classes=cfg.get("num_classes"),
            io_manager=None,
            experiment_id=None,
            use_stratification=cfg.get("use_stratification", True)
        )

        val_acc = result["summary"]["accuracy"]["val"]["mean"]
        val_loss = result["summary"]["loss"]["val"]["mean"]
        run_info = {"type": "k_fold", "result": result}

    else:
        # Modo Holdout (estratificado ou randomizado)
        if cfg.get("use_stratification", True):
            train_set, val_set, test_set = DatasetUtils.stratified_split(
                dataset=dataset,
                p_train=cfg.get("hold_out_p_train", 0.7),
                p_val=cfg.get("hold_out_p_validation", 0.15)
            )
        else:
            train_set, val_set, test_set = DatasetUtils.random_split(
                dataset=dataset,
                p_train=cfg.get("hold_out_p_train", 0.7),
                p_val=cfg.get("hold_out_p_validation", 0.15)
            )

        model = build_model(cfg)
        trainer = build_trainer(model, cfg)

        history = trainer.train(
            train_dataset=train_set,
            val_dataset=val_set,
            epochs=cfg.get("num_epochs", 100)
        )

        evaluator = Evaluator(model, loss_function=trainer.loss_function)
        val_metrics = evaluator.evaluate(val_set, num_classes=cfg.get("num_classes"))

        val_acc = val_metrics.get("accuracy", 0.0)
        val_loss_key = trainer.loss_function.__class__.__name__.lower()
        val_loss = val_metrics.get(val_loss_key, None)

        run_info = {
            "type": "holdout",
            "history": history,
            "val_metrics": val_metrics
        }

    elapsed = time.time() - start

    return {
        "combo": combo,
        "val_accuracy": val_acc,
        "val_loss": val_loss,
        "elapsed_seconds": elapsed,
        "detail": run_info
    }

def run_hill_climbing_search(
    dataset,
    base_config: Dict[str, Any],
    grid_space: Dict[str, List[Any]],
    io_manager,
    experiment_id: str
):
    """
    Executa otimização Hill Climbing no espaço de hiperparâmetros fornecido em `grid_space`.
    Nesta estratégia iterativa, a busca navega de um vizinho ao outro mudando 1 parâmetro 
    por vez, até chegar a um ponto onde nenhuma alteração melhore a acurácia de validação.
    """
    # 1. Inicia a partir do meio das opções disponíveis no grid_space
    current_combo = {k: v[len(v)//2] for k, v in grid_space.items() if v}
    
    print(f"\n[Hill Climbing] Iniciando estado base: {current_combo}")
    
    # Avalia a combinação inicial em memória
    current_result = evaluate_combo(dataset, base_config, current_combo)
    best_acc = current_result.get("val_accuracy", 0.0)
    
    visited = {str(current_combo): current_result}
    step = 1
    
    while True:
        print(f"\n--- [Hill Climbing] Passo {step} | Atual Melhor Acurácia: {best_acc:.4f} ---")
        
        # 2. Gera os vizinhos (move 1 passo para a esquerda ou para a direita na lista de opções)
        neighbors = []
        for key, allowed_values in grid_space.items():
            if len(allowed_values) <= 1:
                continue
            
            idx = allowed_values.index(current_combo[key])
            if idx > 0:
                n_left = current_combo.copy()
                n_left[key] = allowed_values[idx - 1]
                neighbors.append(n_left)
            if idx < len(allowed_values) - 1:
                n_right = current_combo.copy()
                n_right[key] = allowed_values[idx + 1]
                neighbors.append(n_right)
                
        # Descarta vizinhos já testados anteriormente
        unvisited = [n for n in neighbors if str(n) not in visited]
        if not unvisited:
            print("[Hill Climbing] Nenhum vizinho novo para explorar. Ótimo local atingido!")
            break
            
        print(f"[Hill Climbing] Avaliando {len(unvisited)} vizinhos ao redor da configuração atual...")
        best_neighbor_result = None
        
        # 3. Testa todos os vizinhos
        for n_combo in unvisited:
            n_res = evaluate_combo(dataset, base_config, n_combo)
            visited[str(n_combo)] = n_res
            
            if best_neighbor_result is None or n_res.get("val_accuracy", 0.0) > best_neighbor_result.get("val_accuracy", 0.0):
                best_neighbor_result = n_res
                
        # 4. Checa se o melhor vizinho supera a nossa configuração atual
        if best_neighbor_result and best_neighbor_result.get("val_accuracy", 0.0) > best_acc:
            print(f"[Hill Climbing] Sucesso! Acurácia subiu de {best_acc:.4f} para {best_neighbor_result.get('val_accuracy', 0.0):.4f}")
            current_combo = best_neighbor_result["combo"]
            best_acc = best_neighbor_result.get("val_accuracy", 0.0)
            current_result = best_neighbor_result
        else:
            print("[Hill Climbing] Nenhuma melhoria entre os vizinhos testados. Fim da busca!")
            break
            
        step += 1
        
    # Salva relatório do Hill Climbing
    final_report = {
        "experiment_id": experiment_id,
        "base_config": base_config,
        "grid_space": grid_space,
        "results": list(visited.values()),
        "best_model": current_result
    }
    
    if io_manager is not None:
        io_manager.save_report(final_report, f"{experiment_id}_hill_climbing_final")
        
    return final_report
