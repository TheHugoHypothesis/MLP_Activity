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
import numpy as np

def carregar_report(report_path: str):
    with open(report_path, "r") as file:
        return json.load(file)

def plotar_curva_aprendizado(history: dict, figures_dir: str, filename_prefix: str):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epocas = list(range(len(train_loss)))

    plt.figure(figsize=(10, 6))
    plt.plot(epocas, train_loss, label="Loss Treino", marker="o", linewidth=2)
    plt.plot(epocas, val_loss, label="Validation Loss", marker="s", linestyle="--", linewidth=2)
    
    plt.title("Evolução do Erro no Treinamento")
    plt.xlabel("Épocas")
    plt.ylabel("Erro")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, f"{filename_prefix}_train_loss.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfico de erro salvo em: {out_path}")


def plotar_matriz_confusao(matrix_list: list, figures_dir: str, filename_prefix: str, label_prefix: str):
    if matrix_list is None:
        return
        
    matrix = np.array(matrix_list)
    n = matrix.shape[0]
    labels = [chr(65 + i) for i in range(n)] if n == 26 else [str(i) for i in range(n)]
    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, cmap="Blues", interpolation="nearest")
    plt.title(f"Matriz de Confusão ({label_prefix.capitalize()})")
    plt.colorbar()
    tick_marks = np.arange(n)
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predição")
    plt.ylabel("Valor Real")
    for i in range(n):
        for j in range(n):
            value = matrix[i][j]
            if value > 0:
                plt.text(
                    j, i, str(value),
                    ha="center", va="center",
                    color="white" if value > matrix.max()/2 else "black"
                )
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, f"{filename_prefix}_{label_prefix}_confusion_matrix.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Matriz de confusão de {label_prefix} salva em: {out_path}")

def main():
    if len(sys.argv) > 1:
        report_path = sys.argv[1]
    else:
        report_path = "outputs/exp_001/reports/exp_001_report.json"
    if not os.path.exists(report_path):
        print(f"Erro: O arquivo de relatório {report_path} não existe.")
        return
    print(f"Lendo relatório de experimentos: {report_path}")
    report = carregar_report(report_path)
    
    experiment_dir = os.path.dirname(os.path.dirname(report_path))
    figures_dir = os.path.join(experiment_dir, "figures")
        
    #Extrai o ID do experimento do nome do arquivo ou do JSON
    base_name = os.path.basename(report_path)
    exp_id = base_name.split("_")[0]

    #Fluxo de validação cruzada
    if "folds" in report:
        print("\n[Modo: Validação Cruzada Detectado]")
        for fold in report["folds"]:
            fold_idx = fold["fold_index"]
            prefix = f"{exp_id}_fold_{fold_idx}"
            
            #Plota curva de perda do fold
            if "history" in fold:
                plotar_curva_aprendizado(fold["history"], figures_dir, prefix)
                
            #Plota matriz de confusão da validação do fold
            val_m = fold.get("val_metrics", {})
            if "confusion_matrix" in val_m:
                plotar_matriz_confusao(val_m["confusion_matrix"], figures_dir, prefix, "validation")
                
    #Fluxo holdout normal
    else:
        print("\n[Modo: Holdout Padrão Detectado]")
        # Curva de aprendizado global
        plotar_curva_aprendizado(report["history"], figures_dir, exp_id)
        
        #Matriz de validação
        val_m = report.get("val_metrics", {})
        if "confusion_matrix" in val_m:
            plotar_matriz_confusao(val_m["confusion_matrix"], figures_dir, exp_id, "validation")
            
        #Matriz de teste
        test_m = report.get("test_metrics", {})
        if "confusion_matrix" in test_m:
            plotar_matriz_confusao(test_m["confusion_matrix"], figures_dir, exp_id, "test")

if __name__ == "__main__":
    main()