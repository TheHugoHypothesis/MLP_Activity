"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""
import json
import os
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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

    plt.figure(figsize=(12, 7), dpi=300)
    
    top_n = 5
    colors = ["#1E88E5", "#D81B60", "#004D40", "#FFC107", "#8E24AA"]
    linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
    linewidths = [4.5, 3.8, 3.0, 2.2, 1.5]
    
    background_plotted = 0
    for idx, r in enumerate(sorted_results[top_n:]):
        history = r.get("detail", {}).get("history", {})
        if not history and "result" in r.get("detail", {}):
            folds = r["detail"]["result"].get("folds", [])
            if folds:
                history = folds[0].get("history", {})
        val_loss = history.get("val_loss", [])
        if val_loss:
            plt.plot(val_loss, color="#E0E0E0", alpha=0.4, linewidth=1.2, zorder=1)
            background_plotted += 1

    for idx, r in enumerate(sorted_results[:top_n]):
        c = r["combo"]
        history = r.get("detail", {}).get("history", {})
        if not history and "result" in r.get("detail", {}):
            folds = r["detail"]["result"].get("folds", [])
            if folds:
                history = folds[0].get("history", {})
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
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=linewidths[idx % len(linewidths)], 
                label=label, 
                zorder=10 - idx
            )

    max_val_losses = []
    for r in sorted_results[:top_n]:
        history = r.get("detail", {}).get("history", {})
        if not history and "result" in r.get("detail", {}):
            folds = r["detail"]["result"].get("folds", [])
            if folds:
                history = folds[0].get("history", {})
        val_loss = history.get("val_loss", [])
        if len(val_loss) > 2:
            max_val_losses.append(max(val_loss[2:]))
        elif val_loss:
            max_val_losses.append(max(val_loss))
    if max_val_losses:
        plt.ylim(0.0, 1.2 * max(max_val_losses))

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
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
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
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
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

    neurons = [r["combo"].get("hidden_neurons") for r in results if r["combo"].get("hidden_neurons") is not None]
    accs = [r.get("val_accuracy", 0.0) * 100.0 for r in results if r["combo"].get("hidden_neurons") is not None]
    if neurons:
        plt.figure(figsize=(8, 5), dpi=300)
        unique_neurons = sorted(list(set(neurons)))
        mean_accs = []
        for un in unique_neurons:
            mean_accs.append(np.mean([a for n, a in zip(neurons, accs) if n == un]))
        plt.scatter(neurons, accs, color="#1E88E5", alpha=0.5, edgecolor="none", zorder=3)
        plt.plot(unique_neurons, mean_accs, color="#D81B60", linewidth=2.0, marker="o", zorder=4)
        plt.title("Acurácia de Validação vs. Neurônios Ocultos", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Neurônios Ocultos")
        plt.ylabel("Acurácia (%)")
        plt.grid(True)
        best_neuron_idx = np.argmax(accs)
        best_neuron = neurons[best_neuron_idx]
        best_neuron_acc = accs[best_neuron_idx]
        scatter_handle = mlines.Line2D([], [], color="#1E88E5", marker="o", linestyle="None", alpha=0.5, label=f"Modelos (N={len(neurons)})")
        mean_handle = mlines.Line2D([], [], color="#D81B60", marker="o", linewidth=2.0, label="Acurácia Média por Neurônios")
        best_handle = mlines.Line2D([], [], color="none", label=f"Melhor: {best_neuron} Neurônios ({best_neuron_acc:.1f}%)")
        dist_str = ", ".join([f"{un} (N={neurons.count(un)})" for un in unique_neurons])
        dist_handle = mlines.Line2D([], [], color="none", label=f"Distribuição: {dist_str}")
        plt.legend(handles=[scatter_handle, mean_handle, best_handle, dist_handle], loc="upper left", bbox_to_anchor=(1.05, 1), frameon=True, facecolor="#FFFFFF", edgecolor="#E0E0E0", fontsize=9)
        for spine in ["top", "right"]:
            plt.gca().spines[spine].set_visible(False)
        plt.tight_layout()
        chart_neurons_path = os.path.join(reports_dir, f"{experiment_id}_analise_neuronios.png")
        plt.savefig(chart_neurons_path, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Gráfico de neurônios salvo em: {chart_neurons_path}")

    activations = [r["combo"].get("activation") for r in results if r["combo"].get("activation") is not None]
    accs_act = [r.get("val_accuracy", 0.0) * 100.0 for r in results if r["combo"].get("activation") is not None]
    if activations:
        plt.figure(figsize=(8, 5), dpi=300)
        unique_acts = list(set(activations))
        act_data = [[a for act, a in zip(activations, accs_act) if act == ua] for ua in unique_acts]
        plt.boxplot(act_data, tick_labels=unique_acts, patch_artist=True,
                    boxprops=dict(facecolor="#E8F0FE", color="#1E88E5"),
                    medianprops=dict(color="#D81B60", linewidth=2))
        plt.title("Acurácia de Validação por Função de Ativação", fontsize=12, fontweight="bold", pad=12)
        plt.ylabel("Acurácia (%)")
        plt.grid(True, axis="y")
        if accs_act:
            plt.ylim(max(0.0, min(accs_act) - 10.0), 105.0)
        plt.xlim(0.5, len(unique_acts) + 0.5)
        handles = []
        for name, values in zip(unique_acts, act_data):
            median_val = np.median(values)
            max_val = np.max(values)
            count = len(values)
            label = f"{name} (N={count}): Med={median_val:.1f}%, Máx={max_val:.1f}%"
            patch = mpatches.Patch(facecolor="#E8F0FE", edgecolor="#1E88E5", label=label)
            handles.append(patch)
        plt.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1), frameon=True, facecolor="#FFFFFF", edgecolor="#E0E0E0", fontsize=9)
        for spine in ["top", "right"]:
            plt.gca().spines[spine].set_visible(False)
        plt.tight_layout()
        chart_act_path = os.path.join(reports_dir, f"{experiment_id}_analise_ativacao.png")
        plt.savefig(chart_act_path, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Gráfico de ativação salvo em: {chart_act_path}")

    lrs = [r["combo"].get("learning_rate") for r in results if r["combo"].get("learning_rate") is not None]
    accs_lr = [r.get("val_accuracy", 0.0) * 100.0 for r in results if r["combo"].get("learning_rate") is not None]
    if lrs:
        plt.figure(figsize=(8, 5), dpi=300)
        unique_lrs = sorted(list(set(lrs)))
        mean_lr_accs = []
        for ulr in unique_lrs:
            mean_lr_accs.append(np.mean([a for lr, a in zip(lrs, accs_lr) if lr == ulr]))
        plt.scatter(lrs, accs_lr, color="#004D40", alpha=0.5, edgecolor="none", zorder=3)
        plt.plot(unique_lrs, mean_lr_accs, color="#FFC107", linewidth=2.0, marker="o", zorder=4)
        plt.xscale("log")
        plt.title("Acurácia de Validação vs. Taxa de Aprendizado", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Taxa de Aprendizado (Escala Log)")
        plt.ylabel("Acurácia (%)")
        plt.grid(True)
        best_lr_idx = np.argmax(accs_lr)
        best_lr = lrs[best_lr_idx]
        best_lr_acc = accs_lr[best_lr_idx]
        scatter_handle = mlines.Line2D([], [], color="#004D40", marker="o", linestyle="None", alpha=0.5, label=f"Modelos (N={len(lrs)})")
        mean_handle = mlines.Line2D([], [], color="#FFC107", marker="o", linewidth=2.0, label="Acurácia Média por Taxa")
        best_handle = mlines.Line2D([], [], color="none", label=f"Melhor LR: {best_lr} ({best_lr_acc:.1f}%)")
        dist_str = ", ".join([f"{ulr} (N={lrs.count(ulr)})" for ulr in unique_lrs])
        dist_handle = mlines.Line2D([], [], color="none", label=f"Distribuição: {dist_str}")
        plt.legend(handles=[scatter_handle, mean_handle, best_handle, dist_handle], loc="upper left", bbox_to_anchor=(1.05, 1), frameon=True, facecolor="#FFFFFF", edgecolor="#E0E0E0", fontsize=9)
        for spine in ["top", "right"]:
            plt.gca().spines[spine].set_visible(False)
        plt.tight_layout()
        chart_lr_path = os.path.join(reports_dir, f"{experiment_id}_analise_learning_rate.png")
        plt.savefig(chart_lr_path, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Gráfico de learning rate salvo em: {chart_lr_path}")

    opts = [r["combo"].get("optimizer_type") for r in results if r["combo"].get("optimizer_type") is not None]
    accs_opt = [r.get("val_accuracy", 0.0) * 100.0 for r in results if r["combo"].get("optimizer_type") is not None]
    if opts:
        plt.figure(figsize=(8, 5), dpi=300)
        unique_opts = list(set(opts))
        opt_data = [[a for o, a in zip(opts, accs_opt) if o == uo] for uo in unique_opts]
        plt.boxplot(opt_data, tick_labels=unique_opts, patch_artist=True,
                    boxprops=dict(facecolor="#E2F1E8", color="#004D40"),
                    medianprops=dict(color="#FFC107", linewidth=2))
        plt.title("Acurácia de Validação por Tipo de Otimizador", fontsize=12, fontweight="bold", pad=12)
        plt.ylabel("Acurácia (%)")
        plt.grid(True, axis="y")
        if accs_opt:
            plt.ylim(max(0.0, min(accs_opt) - 10.0), 105.0)
        plt.xlim(0.5, len(unique_opts) + 0.5)
        handles = []
        for name, values in zip(unique_opts, opt_data):
            median_val = np.median(values)
            max_val = np.max(values)
            count = len(values)
            label = f"{name} (N={count}): Med={median_val:.1f}%, Máx={max_val:.1f}%"
            patch = mpatches.Patch(facecolor="#E2F1E8", edgecolor="#004D40", label=label)
            handles.append(patch)
        plt.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1), frameon=True, facecolor="#FFFFFF", edgecolor="#E0E0E0", fontsize=9)
        for spine in ["top", "right"]:
            plt.gca().spines[spine].set_visible(False)
        plt.tight_layout()
        chart_opt_path = os.path.join(reports_dir, f"{experiment_id}_analise_otimizador.png")
        plt.savefig(chart_opt_path, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Gráfico de otimizador salvo em: {chart_opt_path}")

    plt.figure(figsize=(12, 6), dpi=300)
    accs_traj = [r.get("val_accuracy", 0.0) * 100.0 for r in results]
    best_so_far = []
    current_best = 0.0
    for a in accs_traj:
        if a > current_best:
            current_best = a
        best_so_far.append(current_best)
        
    plt.scatter(range(len(accs_traj)), accs_traj, color="#4682B4", alpha=0.6, zorder=3)
    plt.step(range(len(best_so_far)), best_so_far, where="post", color="#D81B60", linewidth=2.5, zorder=4)
    
    plt.title(
        f"Progresso da Busca Local Hill Climbing\n"
        f"Evolução da Acurácia de Validação (Total: {len(results)} modelos)",
        fontsize=14,
        fontweight="bold",
        pad=15,
        color="#222222"
    )
    plt.xlabel("Ordem de Avaliação", fontsize=11, labelpad=8)
    plt.ylabel("Acurácia de Validação (%)", fontsize=11, labelpad=8)
    plt.grid(True)
    initial_acc = accs_traj[0]
    best_acc = best_so_far[-1]
    improvement = best_acc - initial_acc
    model_handle = mlines.Line2D([], [], color="#4682B4", marker="o", linestyle="None", alpha=0.6, label=f"Modelos Avaliados (N={len(accs_traj)})")
    step_handle = mlines.Line2D([], [], color="#D81B60", linewidth=2.5, label="Melhor Resultado Acumulado")
    stats_handle1 = mlines.Line2D([], [], color="none", label=f"Acurácia Inicial: {initial_acc:.1f}%")
    stats_handle2 = mlines.Line2D([], [], color="none", label=f"Melhor Acurácia: {best_acc:.1f}%")
    stats_handle3 = mlines.Line2D([], [], color="none", label=f"Melhoria de Busca: +{improvement:.1f}%")
    plt.legend(
        handles=[model_handle, step_handle, stats_handle1, stats_handle2, stats_handle3],
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#E0E0E0",
        fontsize=10
    )
    
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)
        
    plt.tight_layout()
    chart4_path = os.path.join(reports_dir, f"{experiment_id}_trajetoria_busca.png")
    plt.savefig(chart4_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Gráfico 4 de trajetória de busca salvo em: {chart4_path}")

    plt.figure(figsize=(12, 7), dpi=300)
    times = [r.get("elapsed_seconds", 0.0) for r in results]
    accs_pct = [r.get("val_accuracy", 0.0) * 100.0 for r in results]
    neurons = [r["combo"].get("hidden_neurons", 0) for r in results]
    
    sc = plt.scatter(
        times, 
        accs_pct, 
        c=neurons, 
        cmap="viridis", 
        s=80, 
        alpha=0.75, 
        edgecolor="#555555", 
        linewidths=0.7, 
        zorder=3
    )
    
    cbar = plt.colorbar(sc)
    cbar.set_label("Número de Neurônios Ocultos", fontsize=11, labelpad=8)
    
    best_idx = np.argmax(accs_pct)
    plt.scatter(
        times[best_idx], 
        accs_pct[best_idx], 
        color="#D81B60", 
        marker="*", 
        s=220, 
        edgecolor="black", 
        linewidths=1.2, 
        zorder=5
    )
    
    plt.title(
        f"Eficiência da Busca Local: Acurácia vs. Tempo de Treinamento\n"
        f"Análise de Trade-off (Total de {len(results)} modelos avaliados)",
        fontsize=14, 
        fontweight="bold", 
        pad=15, 
        color="#222222"
    )
    plt.xlabel("Tempo de Execução do Treinamento (segundos)", fontsize=11, labelpad=8)
    plt.ylabel("Acurácia de Validação (%)", fontsize=11, labelpad=8)
    plt.grid(True)
    
    if times:
        plt.xlim(min(times) * 0.9, max(times) * 1.1)
    if accs_pct:
        plt.ylim(min(accs_pct) * 0.95, min(max(accs_pct) * 1.05, 105.0))
        
    total_time = sum(times)
    mean_time = np.mean(times) if times else 0.0
    best_time = times[best_idx] if times else 0.0
    best_acc_val = accs_pct[best_idx] if accs_pct else 0.0
    
    best_handle = mlines.Line2D(
        [], [], 
        color="#D81B60", 
        marker="*", 
        markersize=12, 
        linestyle="None", 
        markeredgecolor="black", 
        label="Melhor Configuração Encontrada"
    )
    legend_header = mlines.Line2D([], [], color="none", label="Estatísticas de Execução:")
    stats_total = mlines.Line2D([], [], color="none", label=f"Tempo Total de Busca: {total_time:.1f}s")
    stats_mean = mlines.Line2D([], [], color="none", label=f"Tempo Médio por Modelo: {mean_time:.1f}s")
    stats_best = mlines.Line2D([], [], color="none", label=f"Melhor Modelo: {best_time:.1f}s (Acc: {best_acc_val:.1f}%)")
    
    plt.legend(
        handles=[best_handle, legend_header, stats_total, stats_mean, stats_best],
        loc="upper left",
        bbox_to_anchor=(1.25, 1.0),
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#E0E0E0",
        fontsize=10
    )
    
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)
        
    plt.tight_layout()
    chart5_path = os.path.join(reports_dir, f"{experiment_id}_analise_tempo.png")
    plt.savefig(chart5_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Gráfico 5 de análise de tempo salvo em: {chart5_path}")

if __name__ == "__main__":
    main()
