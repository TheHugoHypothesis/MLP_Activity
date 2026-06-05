import json
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np

def main():
    if len(sys.argv) > 1:
        report_path = sys.argv[1]
        if not os.path.exists(report_path):
            print(f"[Erro] O arquivo de relatório {report_path} não existe.")
            return
        reports_dir = os.path.dirname(report_path) or "."
    else:
        #Localiza o relatório de busca local mais recente
        reports_dir = "outputs/reports"
        report_pattern = os.path.join(reports_dir, "*_hill_climbing_final.json")
        report_files = glob.glob(report_pattern)

        if not report_files:
            print(f"[Erro] Nenhum arquivo de relatório encontrado em '{reports_dir}'")
            return

        report_files.sort(key=os.path.getmtime, reverse=True)
        report_path = report_files[0]
        print(f"[Plot] Carregando o relatório mais recente: {os.path.basename(report_path)}")

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    experiment_id = data.get("experiment_id", "exp")
    
    if not results:
        print("[Erro] O arquivo de relatório não contém resultados válidos na chave 'results'.")
        return

    #Ordenar modelos pelo desempenho de validação
    sorted_results = sorted(results, key=lambda x: (-x.get("val_accuracy", 0.0), x.get("val_loss", float('inf'))))

    #Configurações estéticas globais (Estilo Minimalista Sleek)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#EEEEEE"
    plt.rcParams["grid.linestyle"] = "--"

    #Gráfico da curva de erro de todos os modelos
    plt.figure(figsize=(12, 7), dpi=300)
    
    top_n = 5
    colors = ["#1E88E5", "#D81B60", "#004D40", "#FFC107", "#8E24AA"] # Paleta vibrante e contrastante
    
    #Plota primeiro os modelos menos eficientes
    background_plotted = 0
    for idx, r in enumerate(sorted_results[top_n:]):
        history = r.get("detail", {}).get("history", {})
        val_loss = history.get("val_loss", [])
        if val_loss:
            plt.plot(val_loss, color="#CCCCCC", alpha=0.25, linewidth=0.8, zorder=1)
            background_plotted += 1

    #Plota em destaque os Top 5 melhores modelos
    for idx, r in enumerate(sorted_results[:top_n]):
        c = r["combo"]
        history = r.get("detail", {}).get("history", {})
        val_loss = history.get("val_loss", [])
        if val_loss:
            label = (
                f"Top {idx+1}: {c.get('hidden_neurons')} neu, {c.get('activation')}, "
                f"lr={c.get('learning_rate')}, p_train={int(c.get('p_train', 0.7)*100)}% "
                f"(Acc Val: {r['val_accuracy']*100:.2f}%)"
            )
            plt.plot(
                val_loss, 
                color=colors[idx % len(colors)], 
                linewidth=2.5, 
                label=label, 
                zorder=10 - idx
            )

    plt.title(
        f"Curvas de Erro de Validação (Total: {len(results)} modelos)\n"
        f"Busca Local Hill Climbing - Experimento: {experiment_id}",
        fontsize=14, 
        fontweight="bold", 
        pad=15, 
        color="#222222"
    )
    plt.xlabel("Épocas", fontsize=11, labelpad=8)
    plt.ylabel("Validação Loss (MSE)", fontsize=11, labelpad=8)
    plt.grid(True)
    plt.legend(
        loc="upper right", 
        frameon=True, 
        framealpha=0.95, 
        facecolor="#FFFFFF", 
        edgecolor="#E0E0E0", 
        fontsize=9
    )
    
    # Remove bordas excessivas
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)

    plt.tight_layout()
    chart1_path = os.path.join(reports_dir, f"{experiment_id}_curvas_erro_todos_modelos.png")
    plt.savefig(chart1_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Gráfico 1 de erro salvo em: {chart1_path}")

    #Acurácia e validação dos top 15
    top_k = min(15, len(sorted_results))
    labels = []
    train_accs = []
    val_accs = []

    for idx, r in enumerate(sorted_results[:top_k]):
        c = r["combo"]
        history = r.get("detail", {}).get("history", {})
        
        # Acurácia de validação
        val_accs.append(r.get("val_accuracy", 0.0) * 100.0)
        
        # Acurácia final de treino do histórico
        train_acc = 0.0
        if "train_acc" in history and history["train_acc"]:
            train_acc = history["train_acc"][-1]
        train_accs.append(train_acc * 100.0)
        
        # Rótulo conciso para a barra
        label = f"#{idx+1}\n{c.get('hidden_neurons')}N\n{c.get('activation')}\nlr={c.get('learning_rate')}"
        labels.append(label)

    # Configuração das barras agrupadas
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    
    rects1 = ax.bar(x - width/2, train_accs, width, label="Acurácia de Treino", color="#4682B4", zorder=3) # Steel Blue
    rects2 = ax.bar(x + width/2, val_accs, width, label="Acurácia de Validação", color="#FF7F50", zorder=3) # Coral

    ax.set_title(
        f"Comparativo de Generalização: Acurácia de Treino vs. Validação (Top {top_k} Modelos)\n"
        f"Identificação Visual de Overfitting e Desempenho",
        fontsize=14, 
        fontweight="bold", 
        pad=15, 
        color="#222222"
    )
    ax.set_ylabel("Acurácia (%)", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 108)
    ax.grid(True, axis="y")
    ax.legend(
        loc="upper right", 
        frameon=True, 
        facecolor="#FFFFFF", 
        edgecolor="#E0E0E0", 
        fontsize=10
    )

    # Adiciona rótulos numéricos sobre cada barra
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", 
                va="bottom", 
                fontsize=8, 
                color="#555555",
                fontweight="semibold"
            )

    autolabel(rects1)
    autolabel(rects2)

    # Remove bordas excessivas
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    chart2_path = os.path.join(reports_dir, f"{experiment_id}_comparativo_top_modelos.png")
    plt.savefig(chart2_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Gráfico 2 de comparação de acurácia salvo em: {chart2_path}")

if __name__ == "__main__":
    main()
