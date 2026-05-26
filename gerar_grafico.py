import json
import os
import matplotlib.pyplot as plt
import numpy as np


def carregar_report(report_path: str):
    with open(report_path, "r") as file:
        return json.load(file)


def gerar_grafico(report_path: str):
    report = carregar_report(report_path)

    train_loss = report["history"]["train_loss"]
    val_loss = report["history"]["val_loss"]

    epocas = list(range(len(train_loss)))

    plt.figure(figsize=(10, 6))

    plt.plot(
        epocas,
        train_loss,
        label="Loss Treino",
        marker="o",
        linewidth=2
    )

    plt.plot(
        epocas,
        val_loss,
        label="Validation Loss",
        marker="s",
        linestyle="--",
        linewidth=2
    )

    plt.title("Evolução do Erro no Treinamento")
    plt.xlabel("Épocas")
    plt.ylabel("Erro")
    plt.grid(True, linestyle=":")
    plt.legend()

    plt.annotate(
        f'Final: {train_loss[-1]:.6f}',
        xy=(epocas[-1], train_loss[-1]),
        xytext=(
            epocas[-1] - max(5, len(epocas)//10),
            train_loss[-1] + 0.01
        ),
        arrowprops=dict(
            facecolor='black',
            shrink=0.05,
            width=1,
            headwidth=5
        ),
        fontsize=10
    )

    plt.tight_layout()

    plt.savefig("outputs/figures/train_loss.png", dpi=300)

    print("Gráfico salvo em outputs/figures/train_loss.png")

    #plt.show()


def gerar_matriz_confusao(report_path: str):

    report = carregar_report(report_path)

    matrix = np.array(
        report["val_metrics"]["confusion_matrix"]
    )

    alfabeto = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    plt.figure(figsize=(12, 10))

    plt.imshow(matrix, cmap="Blues", interpolation="nearest")

    plt.title("Matriz de Confusão")
    plt.colorbar()

    tick_marks = np.arange(26)

    plt.xticks(tick_marks, alfabeto)
    plt.yticks(tick_marks, alfabeto)

    plt.xlabel("Predição")
    plt.ylabel("Valor Real")

    for i in range(26):
        for j in range(26):

            value = matrix[i][j]

            if value > 0:
                plt.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > matrix.max()/2 else "black"
                )

    plot_confusion_matrix(matrix, "outputs/figures", labels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

    #plt.show()


if __name__ == "__main__":

    REPORT_PATH = "outputs/reports/exp_001_report.json"

    gerar_grafico(REPORT_PATH)

    gerar_matriz_confusao(REPORT_PATH)




def plot_confusion_matrix(matrix: np.ndarray, figures_dir: str, labels=None, out_filename: str = "confusion_matrix.png"):
    n = matrix.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]

    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, cmap="Blues", interpolation="nearest")
    plt.title("Matriz de Confusão")
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
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > matrix.max()/2 else "black"
                )

    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, out_filename)
    plt.savefig(out_path, dpi=300)
    print(f"Matriz salva em {out_path}")