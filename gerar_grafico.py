"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

import sys
import json
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

def carregar_report(report_path: str):
    with open(report_path, "r") as file:
        return json.load(file)

def plotar_curva_aprendizado(history: dict, figures_dir: str, filename_prefix: str):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epocas = list(range(len(train_loss)))

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epocas, train_loss, label="Erro Treino", color="#1E88E5", linewidth=2.5, zorder=3)
    plt.plot(epocas, val_loss, label="Erro Validação", color="#D81B60", linestyle="--", linewidth=2.5, zorder=3)
    
    plt.title("Evolução do Erro (Loss/MSE) no Treinamento", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Épocas", fontsize=11, labelpad=8)
    plt.ylabel("Erro (MSE)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), frameon=True, facecolor="#FFFFFF", edgecolor="#E0E0E0", fontsize=10)
    
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)
        
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, f"{filename_prefix}_train_loss.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Gráfico de erro salvo em: {out_path}")

def plotar_curva_acuracia(history: dict, figures_dir: str, filename_prefix: str):
    if "train_acc" not in history or "val_acc" not in history:
        return
    train_acc = [acc * 100.0 for acc in history["train_acc"]]
    val_acc = [acc * 100.0 for acc in history["val_acc"]]
    epocas = list(range(len(train_acc)))

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epocas, train_acc, label="Acurácia Treino", color="#1E88E5", linewidth=2.5, zorder=3)
    plt.plot(epocas, val_acc, label="Acurácia Validação", color="#D81B60", linestyle="--", linewidth=2.5, zorder=3)
    
    plt.title("Evolução da Acurácia ao longo do Treinamento", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Épocas", fontsize=11, labelpad=8)
    plt.ylabel("Acurácia (%)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), frameon=True, facecolor="#FFFFFF", edgecolor="#E0E0E0", fontsize=10)
    
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)
        
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, f"{filename_prefix}_accuracy_curves.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Gráfico de evolução da acurácia salvo em: {out_path}")

def plotar_comparativo_acuracia(train_acc: float, val_acc: float, test_acc: float, figures_dir: str, filename_prefix: str, mode_label: str):
    plt.figure(figsize=(8, 6), dpi=300)
    
    labels = ["Treino", "Validação", "Teste"]
    accs = [train_acc * 100.0, val_acc * 100.0, test_acc * 100.0]
    colors = ["#1E88E5", "#D81B60", "#004D40"]
    
    bars = plt.bar(labels, accs, color=colors, width=0.5, edgecolor="#555555", linewidth=0.8, zorder=3)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0, 
            yval + 1.5 if yval < 95.0 else yval - 4.5, 
            f"{yval:.2f}%", 
            ha="center", 
            va="bottom" if yval < 95.0 else "top",
            fontsize=10, 
            fontweight="bold",
            color="black" if yval < 95.0 else "white"
        )
        
    plt.title(f"Acurácia Geral do Modelo ({mode_label})\nComparativo de Treino, Validação e Teste", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Acurácia (%)", fontsize=11, labelpad=8)
    plt.ylim(0, 110)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    handles = [plt.Rectangle((0,0),1,1, color=colors[i], ec="#555555", lw=0.8) for i in range(3)]
    plt.legend(
        handles, 
        [f"Treino ({accs[0]:.2f}%)", f"Validação ({accs[1]:.2f}%)", f"Teste ({accs[2]:.2f}%)"],
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#E0E0E0",
        fontsize=10
    )
    
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)
        
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, f"{filename_prefix}_comparativo_acuracia.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Gráfico de comparativo de acurácia salvo em: {out_path}")

def plotar_comparativo_folds(fold_accs: list, mean_val_acc: float, final_test_acc: float, figures_dir: str, filename_prefix: str):
    if not fold_accs:
        return
    plt.figure(figsize=(10, 6), dpi=300)
    
    folds = [f"Fold {i+1}" for i in range(len(fold_accs))]
    accs = [acc * 100.0 for acc in fold_accs]
    
    bars = plt.bar(folds, accs, color="#4682B4", edgecolor="#555555", width=0.5, linewidth=0.8, zorder=3)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0, 
            yval + 1.0, 
            f"{yval:.2f}%", 
            ha="center", 
            va="bottom",
            fontsize=9, 
            fontweight="bold"
        )
        
    plt.axhline(mean_val_acc * 100.0, color="#D81B60", linestyle="--", linewidth=1.8, label=f"Média CV Validação ({mean_val_acc*100.0:.2f}%)", zorder=4)
    plt.axhline(final_test_acc * 100.0, color="#004D40", linestyle=":", linewidth=1.8, label=f"Modelo Final no Teste ({final_test_acc*100.0:.2f}%)", zorder=4)
    
    plt.title("Estabilidade do Modelo: Acurácia por Fold da Validação Cruzada", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Acurácia de Validação (%)", fontsize=11, labelpad=8)
    plt.ylim(0, 110)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#E0E0E0",
        fontsize=10
    )
    
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)
        
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, f"{filename_prefix}_acuracia_folds.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Gráfico de estabilidade dos folds salvo em: {out_path}")

def plotar_matriz_confusao(matrix_list: list, figures_dir: str, filename_prefix: str, label_prefix: str):
    if matrix_list is None:
        return
        
    matrix = np.array(matrix_list)
    n = matrix.shape[0]
    labels = [chr(65 + i) for i in range(n)] if n == 26 else [str(i) for i in range(n)]
    plt.figure(figsize=(12, 10), dpi=300)
    plt.imshow(matrix, cmap="Blues", interpolation="nearest")
    plt.title(f"Matriz de Confusão ({label_prefix.capitalize()})", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar()
    tick_marks = np.arange(n)
    plt.xticks(tick_marks, labels, fontsize=9)
    plt.yticks(tick_marks, labels, fontsize=9)
    plt.xlabel("Predição", fontsize=11, labelpad=8)
    plt.ylabel("Valor Real", fontsize=11, labelpad=8)
    for i in range(n):
        for j in range(n):
            value = matrix[i][j]
            if value > 0:
                plt.text(
                    j, i, str(value),
                    ha="center", va="center",
                    color="white" if value > matrix.max()/2 else "black",
                    fontsize=8, fontweight="bold"
                )
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, f"{filename_prefix}_{label_prefix}_confusion_matrix.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Matriz de confusão de {label_prefix} salva em: {out_path}")

def main():
    if len(sys.argv) > 1:
        report_path = sys.argv[1]
    else:
        report_path = "outputs/exp_008/reports/exp_008_report.json"
    if not os.path.exists(report_path):
        print(f"Erro: O arquivo de relatório {report_path} não existe.")
        return
    print(f"Lendo relatório de experimentos: {report_path}")
    report = carregar_report(report_path)
    
    experiment_dir = os.path.dirname(os.path.dirname(report_path))
    figures_dir = os.path.join(experiment_dir, "figures")
        
    base_name = os.path.basename(report_path)
    exp_id = base_name.split("_")[0]

    # Fluxo de validação cruzada
    if "folds" in report:
        print("\n[Modo: Validação Cruzada Detectado]")
        for fold in report["folds"]:
            fold_idx = fold["fold_index"]
            prefix = f"{exp_id}_fold_{fold_idx}"
            
            if "history" in fold:
                plotar_curva_aprendizado(fold["history"], figures_dir, prefix)
                plotar_curva_acuracia(fold["history"], figures_dir, prefix)
                
            val_m = fold.get("val_metrics", {})
            train_m = fold.get("train_metrics", {})
            if "confusion_matrix" in val_m:
                plotar_matriz_confusao(val_m["confusion_matrix"], figures_dir, prefix, "validation")
            
            if val_m and train_m:
                plotar_comparativo_acuracia(
                    train_m.get("accuracy", 0.0), 
                    val_m.get("accuracy", 0.0), 
                    0.0, 
                    figures_dir, 
                    prefix, 
                    f"Fold {fold_idx}"
                )

        if "final_training_history" in report and report["final_training_history"]:
            plotar_curva_aprendizado(report["final_training_history"], figures_dir, f"{exp_id}_final")
            plotar_curva_acuracia(report["final_training_history"], figures_dir, f"{exp_id}_final")
            
        final_test_m = report.get("final_test_metrics", {})
        if "confusion_matrix" in final_test_m:
            plotar_matriz_confusao(final_test_m["confusion_matrix"], figures_dir, f"{exp_id}_final", "test")
            
        # Comparativo final de acurácias para CV
        final_test_acc = final_test_m.get("accuracy", 0.0)
        final_history = report.get("final_training_history", {})
        final_train_acc = final_history.get("train_acc", [0.0])[-1] if final_history else 0.0
        final_val_acc = final_history.get("val_acc", [0.0])[-1] if final_history else 0.0
        plotar_comparativo_acuracia(
            final_train_acc, 
            final_val_acc, 
            final_test_acc, 
            figures_dir, 
            f"{exp_id}_final", 
            "CV Modelo Final"
        )
        
        # Comparativo de Folds
        fold_accs = [fold["val_metrics"].get("accuracy", 0.0) for fold in report["folds"] if "val_metrics" in fold]
        mean_val_acc = report.get("summary", {}).get("accuracy", {}).get("val", {}).get("mean", 0.0)
        plotar_comparativo_folds(fold_accs, mean_val_acc, final_test_acc, figures_dir, exp_id)
                
    # Fluxo holdout normal
    else:
        print("\n[Modo: Holdout Padrão Detectado]")
        plotar_curva_aprendizado(report["history"], figures_dir, exp_id)
        plotar_curva_acuracia(report["history"], figures_dir, exp_id)
        
        val_m = report.get("val_metrics", {})
        if "confusion_matrix" in val_m:
            plotar_matriz_confusao(val_m["confusion_matrix"], figures_dir, exp_id, "validation")
            
        test_m = report.get("test_metrics", {})
        if "confusion_matrix" in test_m:
            plotar_matriz_confusao(test_m["confusion_matrix"], figures_dir, exp_id, "test")
            
        train_acc = report.get("train_metrics", {}).get("accuracy", 0.0)
        val_acc = report.get("val_metrics", {}).get("accuracy", 0.0)
        test_acc = report.get("test_metrics", {}).get("accuracy", 0.0)
        
        plotar_comparativo_acuracia(train_acc, val_acc, test_acc, figures_dir, exp_id, "Holdout Padrão")

if __name__ == "__main__":
    main()