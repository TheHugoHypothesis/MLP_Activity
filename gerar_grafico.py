import matplotlib.pyplot as plt

# 1. Dados extraídos do seu log do terminal
epocas = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

treino_mse = [
    0.037464, 0.016728, 0.007121, 0.004344, 0.003128, 
    0.002495,  0.002057, 0.001743, 0.001503, 0.001348, 
    0.001202, 0.001086, 0.001027, 0.000987
]

validacao_mse = [
    0.037410, 0.019227, 0.010432, 0.008619, 0.007955, 
    0.007614, 0.007398, 0.007208, 0.007039, 0.006935, 
    0.006862, 0.006809, 0.006792, 0.006787
]

def gerar_grafico():
    # Criar a figura
    plt.figure(figsize=(10, 6))
    
    # Plotar os dados
    plt.plot(epocas, treino_mse, label='MSE Treino', color='#1f77b4', marker='o', linewidth=2)
    plt.plot(epocas, validacao_mse, label='MSE Validação', color='#d62728', marker='s', linestyle='--', linewidth=2)

    # Adicionar anotação no ponto final (melhor resultado)
    plt.annotate(f'Final: {validacao_mse[-1]:.4f}', 
                 xy=(epocas[-1], validacao_mse[-1]), 
                 xytext=(epocas[-1]-20, validacao_mse[-1]+0.005),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    # Estilização
    plt.title('Evolução do Erro Médio Quadrático (MSE) - MLP Letras', fontsize=14, pad=15)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.yscale('linear') # Pode usar 'log' se quiser ver detalhes em valores muito pequenos
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    
    # Adicionar uma linha indicando o patamar de estabilidade
    plt.axhline(y=validacao_mse[-1], color='gray', linestyle=':', alpha=0.5)

    # Salvar e mostrar
    plt.tight_layout()
    plt.savefig('resultado_treinamento.png', dpi=300)
    print("Gráfico 'resultado_treinamento.png' gerado com sucesso!")
    plt.show()

if __name__ == "__main__":
    gerar_grafico()