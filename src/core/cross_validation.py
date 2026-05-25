"""
Este módulo fornece a função `run_stratified_k_fold(...)` que executa um
experimento de k-fold estratificado. Para cada fold a função:
- cria um modelo novo via `build_model()`;
- instancia um `Trainer` via `build_trainer(model)`;
- treina o modelo no conjunto de treino do fold e valida no conjunto de validação;
- coleta histórico de treino e métricas de avaliação por fold.

O retorno é um dicionário com os resultados por fold e um resumo com média e
desvio padrão das métricas principais.
"""

from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

from src.evaluation.evaluator import Evaluator
from src.utils.dataset_utils import DatasetUtils, Dataset


def _mean_and_std(values: List[float]) -> Dict[str, float]:
    return {
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def run_stratified_k_fold(
    dataset: Dataset,
    k: int,
    build_model,
    build_trainer,
    epochs: int,
    num_classes: int = None
) -> Dict[str, Any]:
    folds = DatasetUtils.stratified_k_fold_split(dataset, k)
    fold_results: List[Dict[str, Any]] = []

    train_accuracies: List[float] = []
    val_accuracies: List[float] = []
    train_losses: List[float] = []
    val_losses: List[float] = []

    # itera sobre cada fold, treinando um modelo novo e coletando métricas
    for fold_index, (train_set, val_set) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_index}/{k} ===")

        model = build_model()
        trainer = build_trainer(model)
        history = trainer.train(
            train_dataset=train_set,
            val_dataset=val_set,
            epochs=epochs
        )

        evaluator = Evaluator(model, loss_function=trainer.loss_function)

        print("\n--- Conjunto de treino ---")
        train_metrics = evaluator.evaluate(train_set, num_classes=num_classes)

        print("\n--- Conjunto de validação ---")
        val_metrics = evaluator.evaluate(val_set, num_classes=num_classes)

        fold_results.append({
            "fold_index": fold_index,
            "history": history,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        })

        if train_metrics and "accuracy" in train_metrics:
            train_accuracies.append(train_metrics["accuracy"])
        if val_metrics and "accuracy" in val_metrics:
            val_accuracies.append(val_metrics["accuracy"])

        train_loss_key = trainer.loss_function.__class__.__name__.lower()
        if train_metrics and train_loss_key in train_metrics:
            train_losses.append(train_metrics[train_loss_key])
        if val_metrics and train_loss_key in val_metrics:
            val_losses.append(val_metrics[train_loss_key])

    summary = {
        "accuracy": {
            "train": _mean_and_std(train_accuracies),
            "val": _mean_and_std(val_accuracies),
        },
        "loss": {
            "train": _mean_and_std(train_losses),
            "val": _mean_and_std(val_losses),
        }
    }

    print("\n=== Resumo do k-fold ===")
    print(
        f"Accuracy train: {summary['accuracy']['train']['mean']:.4f} ± {summary['accuracy']['train']['std']:.4f} | "
        f"val: {summary['accuracy']['val']['mean']:.4f} ± {summary['accuracy']['val']['std']:.4f}"
    )
    print(
        f"Loss train: {summary['loss']['train']['mean']:.6f} ± {summary['loss']['train']['std']:.6f} | "
        f"val: {summary['loss']['val']['mean']:.6f} ± {summary['loss']['val']['std']:.6f}"
    )

    return {
        "folds": fold_results,
        "summary": summary,
    }
