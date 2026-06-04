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
report_path = r"outputs\reports\exp_008_hill_hill_climbing_final.json"
if not os.path.exists(report_path):
    report_path = r"outputs\reports\exp_008_hill_hill_climbing_final.json"
output_path = "hill_climbing_search.md"
with open(report_path, "r", encoding="utf-8") as f:
    data = json.load(f)
results = data.get("results", [])

# Ordenar por acurácia de validação decrescente, e depois por val_loss crescente
sorted_results = sorted(results, key=lambda x: (-x.get("val_accuracy", 0.0), x.get("val_loss", float('inf'))))
markdown_content = """# Resumo da Otimização por Hill Climbing
Tabela resumida contendo todas as **{total_combos} combinações** de hiperparâmetros testadas e avaliadas pelo algoritmo de Hill Climbing, ordenadas da **melhor para a pior** com base na acurácia do conjunto de validação.
## Melhor Configuração Encontrada
* **Acurácia de Validação:** {best_acc:.4f} ({best_acc_pct:.2f}%)
* **Acurácia de Treino Final:** {best_train_acc_pct:.2f}%
* **Partição do Dataset (Treino):** {best_p_train_pct:.1f}%
* **Função de Perda de Validação:** {best_loss:.6f}
* **Tempo de Execução:** {best_time:.2f} segundos
* **Hiperparâmetros:**
  * Neurônios Ocultos: `{best_combo[hidden_neurons]}`
  * Função de Ativação Oculta: `{best_combo[activation]}`
  * Função de Perda: `{best_combo[loss_function]}`
  * Inicializador: `{best_combo[initializer]}`
  * Learning Rate: `{best_combo[learning_rate]}`
  
  * Otimizador: `{best_combo[optimizer_type]}`
  * Momentum: `{best_combo[momentum]}`
  * L2 Decay: `{best_combo[l2_decay]}`
  * Paciência (Early Stop): `{best_combo[patience]}`
  * Épocas Máximas: `{best_combo[num_epochs]}`
---
## Tabela Geral de Resultados
| # | Neurônios | Ativação | Perda | LR | Inicializador | Otimizador | Momentum | L2 Decay | Partição Treino | Ép. Real / Max | Paciência | Acurácia Treino | Acurácia Val | Val Loss | Tempo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
""".format(
    total_combos=len(results),
    best_acc=data["best_model"]["val_accuracy"],
    best_acc_pct=data["best_model"]["val_accuracy"] * 100.0,
    best_train_acc_pct=data["best_model"]["detail"]["history"]["train_acc"][-1] * 100.0 if "detail" in data["best_model"] and "history" in data["best_model"]["detail"] and "train_acc" in data["best_model"]["detail"]["history"] else 0.0,
    best_p_train_pct=data["best_model"]["combo"].get("p_train", 0.7) * 100.0,
    best_loss=data["best_model"]["val_loss"],
    best_time=data["best_model"]["elapsed_seconds"],
    best_combo=data["best_model"]["combo"]
)
for idx, r in enumerate(sorted_results, 1):
    c = r["combo"]
    val_acc = r["val_accuracy"]
    val_loss = r["val_loss"]
    elapsed = r["elapsed_seconds"]

    # Extrair épocas reais e acurácia final de treino a partir do histórico
    epochs_trained = 0
    train_acc = 0.0

    if "detail" in r and "history" in r["detail"]:
        hist = r["detail"]["history"]
        if "train_loss" in hist:
            epochs_trained = len(hist["train_loss"])
        if "train_acc" in hist and len(hist["train_acc"]) > 0:
            train_acc = hist["train_acc"][-1]

    if epochs_trained == 0:
        epochs_trained = c.get("num_epochs", 400)

    p_train_pct = c.get("p_train", 0.7) * 100.0

    patience = c.get("patience", "N/A")
    max_epochs = c.get("num_epochs", "N/A")
    epochs_str = f"{epochs_trained} / {max_epochs}"

    markdown_content += "| {idx} | {neurons} | {act} | {loss_fn} | {lr} | {init} | {opt} | {momentum} | {l2} | {p_train:.1f}% | {epochs_str} | {patience} | **{train_acc:.2f}%** | **{val_acc:.2f}%** | {val_loss:.6f} | {time:.1f}s |\n".format(
        idx=idx,
        neurons=c.get("hidden_neurons"),
        act=c.get("activation"),
        loss_fn=c.get("loss_function"),
        lr=c.get("learning_rate"),
        init=c.get("initializer"),
        opt=c.get("optimizer_type"),
        momentum=c.get("momentum"),
        l2=c.get("l2_decay"),
        p_train=p_train_pct,
        epochs_str=epochs_str,
        patience=patience,
        train_acc=train_acc * 100.0,
        val_acc=val_acc * 100.0,
        val_loss=val_loss,
        time=elapsed
    )
with open(output_path, "w", encoding="utf-8") as f:
    f.write(markdown_content)
print("Markdown criado com sucesso em:", output_path)
