# Resumo da Otimização por Hill Climbing
Tabela resumida contendo todas as **91 combinações** de hiperparâmetros testadas e avaliadas pelo algoritmo de Hill Climbing, ordenadas da **melhor para a pior** com base na acurácia do conjunto de validação.
## Melhor Configuração Encontrada
* **Acurácia de Validação:** 0.9292 (92.92%)
* **Acurácia de Treino Final:** 99.19%
* **Partição do Dataset (Treino):** 70.0%
* **Função de Perda de Validação:** 0.006599
* **Tempo de Execução:** 65.76 segundos
* **Hiperparâmetros:**
  * Neurônios Ocultos: `56`
  * Função de Ativação Oculta: `sigmoid`
  * Função de Perda: `mse`
  * Inicializador: `normal`
  * Learning Rate: `0.05`
  
  * Otimizador: `sgd_momentum`
  * Momentum: `0.9`
  * L2 Decay: `1e-05`
  * Paciência (Early Stop): `20`
  * Épocas Máximas: `600`
---
## Tabela Geral de Resultados
| # | Neurônios | Ativação | Perda | LR | Inicializador | Otimizador | Momentum | L2 Decay | Partição Treino | Ép. Real / Max | Paciência | Acurácia Treino | Acurácia Val | Val Loss | Tempo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.9 | 1e-05 | 70.0% | 296 / 600 | 20 | **99.19%** | **92.92%** | 0.006599 | 66.0s |
| 2 | 56 | sigmoid | mse | 0.1 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 297 / 600 | 20 | **99.19%** | **92.92%** | 0.006606 | 67.2s |
| 3 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 471 / 600 | 20 | **99.09%** | **92.92%** | 0.006976 | 102.6s |
| 4 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 471 / 800 | 20 | **99.09%** | **92.92%** | 0.006976 | 105.0s |
| 5 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 471 / 600 | 20 | **99.09%** | **92.92%** | 0.006976 | 105.0s |
| 6 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 471 / 600 | 20 | **99.09%** | **92.92%** | 0.006976 | 104.4s |
| 7 | 48 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 476 / 600 | 20 | **99.09%** | **92.92%** | 0.007033 | 100.2s |
| 8 | 64 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 466 / 600 | 20 | **99.09%** | **92.92%** | 0.007105 | 111.0s |
| 9 | 48 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 468 / 600 | 20 | **99.11%** | **92.78%** | 0.007306 | 104.4s |
| 10 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 400 / 400 | 20 | **99.11%** | **92.78%** | 0.007548 | 91.8s |
| 11 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 600 / 600 | 40 | **99.29%** | **92.45%** | 0.006499 | 131.4s |
| 12 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 400 / 400 | 20 | **99.09%** | **92.45%** | 0.007287 | 88.8s |
| 13 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 573 / 600 | 40 | **99.41%** | **92.22%** | 0.006947 | 129.6s |
| 14 | 56 | sigmoid | mse | 0.1 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 284 / 600 | 20 | **99.31%** | **92.22%** | 0.006986 | 65.4s |
| 15 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.9 | 1e-05 | 85.0% | 283 / 600 | 20 | **99.31%** | **92.22%** | 0.006988 | 64.2s |
| 16 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 472 / 600 | 20 | **99.31%** | **92.22%** | 0.007252 | 110.4s |
| 17 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 472 / 800 | 20 | **99.31%** | **92.22%** | 0.007252 | 107.4s |
| 18 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 472 / 600 | 20 | **99.31%** | **92.22%** | 0.007252 | 109.2s |
| 19 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 472 / 600 | 20 | **99.31%** | **92.22%** | 0.007252 | 107.4s |
| 20 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.9 | 1e-05 | 85.0% | 233 / 600 | 10 | **99.21%** | **92.22%** | 0.007297 | 53.4s |
| 21 | 56 | sigmoid | mse | 0.1 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 233 / 600 | 10 | **99.21%** | **92.22%** | 0.007299 | 52.8s |
| 22 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 349 / 600 | 10 | **98.98%** | **91.98%** | 0.007808 | 78.0s |
| 23 | 64 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 452 / 600 | 20 | **99.11%** | **91.67%** | 0.007497 | 109.2s |
| 24 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 319 / 600 | 10 | **98.92%** | **91.67%** | 0.008294 | 72.0s |
| 25 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 319 / 400 | 10 | **98.92%** | **91.67%** | 0.008294 | 73.2s |
| 26 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 319 / 800 | 10 | **98.92%** | **91.67%** | 0.008294 | 75.6s |
| 27 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 319 / 600 | 10 | **98.92%** | **91.67%** | 0.008294 | 75.0s |
| 28 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 319 / 600 | 10 | **98.92%** | **91.67%** | 0.008294 | 74.4s |
| 29 | 48 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 301 / 600 | 10 | **98.62%** | **91.67%** | 0.008454 | 65.4s |
| 30 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 202 / 600 | 20 | **99.19%** | **90.57%** | 0.006861 | 45.0s |
| 31 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 152 / 600 | 10 | **98.98%** | **90.57%** | 0.007178 | 34.2s |
| 32 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-06 | 85.0% | 165 / 600 | 10 | **99.11%** | **90.56%** | 0.006637 | 37.2s |
| 33 | 56 | sigmoid | mse | 0.05 | xavier | sgd_momentum | 0.8 | 1e-05 | 85.0% | 154 / 600 | 10 | **99.11%** | **90.56%** | 0.006841 | 34.8s |
| 34 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 209 / 600 | 20 | **99.21%** | **90.56%** | 0.007004 | 47.4s |
| 35 | 56 | sigmoid | mse | 0.1 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 106 / 600 | 10 | **99.21%** | **90.56%** | 0.007076 | 25.2s |
| 36 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.9 | 1e-05 | 85.0% | 106 / 600 | 10 | **99.21%** | **90.56%** | 0.007078 | 24.6s |
| 37 | 64 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 322 / 600 | 10 | **98.92%** | **90.56%** | 0.008331 | 75.6s |
| 38 | 48 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 164 / 600 | 10 | **99.02%** | **90.00%** | 0.007078 | 35.4s |
| 39 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 158 / 600 | 10 | **99.11%** | **90.00%** | 0.007310 | 36.6s |
| 40 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 158 / 400 | 10 | **99.11%** | **90.00%** | 0.007310 | 36.0s |
| 41 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 158 / 800 | 10 | **99.11%** | **90.00%** | 0.007310 | 36.6s |
| 42 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 158 / 600 | 10 | **99.11%** | **90.00%** | 0.007310 | 35.4s |
| 43 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 158 / 600 | 10 | **99.11%** | **90.00%** | 0.007310 | 36.6s |
| 44 | 64 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 152 / 600 | 10 | **99.11%** | **90.00%** | 0.007311 | 36.6s |
| 45 | 56 | sigmoid | mse | 0.01 | xavier | sgd_momentum | 0.8 | 1e-05 | 85.0% | 402 / 600 | 10 | **97.64%** | **90.00%** | 0.008241 | 96.0s |
| 46 | 48 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 409 / 600 | 10 | **97.93%** | **90.00%** | 0.008752 | 91.2s |
| 47 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 386 / 600 | 10 | **97.26%** | **89.62%** | 0.008663 | 86.4s |
| 48 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.5 | 1e-05 | 70.0% | 600 / 600 | 20 | **97.87%** | **89.62%** | 0.009145 | 133.8s |
| 49 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 0.0001 | 70.0% | 203 / 600 | 20 | **95.12%** | **89.62%** | 0.013194 | 45.0s |
| 50 | 56 | leaky_relu | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 59 / 600 | 10 | **99.02%** | **89.44%** | 0.007242 | 12.6s |
| 51 | 56 | leaky_relu | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 140 / 600 | 10 | **98.92%** | **89.44%** | 0.007512 | 30.0s |
| 52 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 116 / 600 | 5 | **98.82%** | **89.44%** | 0.007842 | 26.4s |
| 53 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 511 / 600 | 20 | **98.72%** | **89.44%** | 0.008038 | 117.0s |
| 54 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.9 | 1e-05 | 85.0% | 255 / 600 | 10 | **98.72%** | **89.44%** | 0.008063 | 58.2s |
| 55 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.5 | 1e-05 | 85.0% | 255 / 600 | 10 | **98.72%** | **89.44%** | 0.008067 | 58.2s |
| 56 | 56 | sigmoid | mse | 0.05 | uniform | sgd | 0.8 | 1e-05 | 85.0% | 383 / 600 | 10 | **98.23%** | **88.89%** | 0.008311 | 66.0s |
| 57 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-06 | 85.0% | 383 / 600 | 10 | **98.13%** | **88.89%** | 0.008361 | 88.2s |
| 58 | 64 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 387 / 600 | 10 | **97.44%** | **88.89%** | 0.008773 | 93.6s |
| 59 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 372 / 600 | 10 | **97.24%** | **88.89%** | 0.008976 | 86.4s |
| 60 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 372 / 400 | 10 | **97.24%** | **88.89%** | 0.008976 | 85.8s |
| 61 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 372 / 800 | 10 | **97.24%** | **88.89%** | 0.008976 | 86.4s |
| 62 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 372 / 600 | 10 | **97.24%** | **88.89%** | 0.008976 | 84.6s |
| 63 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 372 / 600 | 10 | **97.24%** | **88.89%** | 0.008976 | 86.4s |
| 64 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.5 | 1e-05 | 85.0% | 600 / 600 | 20 | **98.23%** | **88.89%** | 0.009143 | 136.2s |
| 65 | 56 | linear | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 113 / 600 | 10 | **98.03%** | **88.89%** | 0.009579 | 22.8s |
| 66 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.5 | 1e-05 | 85.0% | 600 / 600 | 10 | **95.47%** | **88.89%** | 0.011076 | 136.2s |
| 67 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 0.0001 | 85.0% | 130 / 600 | 10 | **95.37%** | **88.89%** | 0.013758 | 29.4s |
| 68 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 0.0001 | 85.0% | 393 / 600 | 10 | **94.69%** | **88.89%** | 0.014588 | 91.2s |
| 69 | 56 | linear | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 32 / 600 | 10 | **97.74%** | **88.33%** | 0.010025 | 6.6s |
| 70 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 0.0001 | 85.0% | 195 / 600 | 20 | **94.98%** | **88.33%** | 0.013325 | 44.4s |
| 71 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 284 / 600 | 5 | **96.16%** | **87.78%** | 0.010152 | 66.0s |
| 72 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 0.0001 | 85.0% | 141 / 600 | 10 | **94.78%** | **87.78%** | 0.013635 | 32.4s |
| 73 | 56 | leaky_relu | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 474 / 600 | 20 | **95.43%** | **87.74%** | 0.008988 | 101.4s |
| 74 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-06 | 70.0% | 400 / 600 | 20 | **98.17%** | **87.26%** | 0.009805 | 88.2s |
| 75 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 209 / 600 | 5 | **97.74%** | **87.22%** | 0.009804 | 48.0s |
| 76 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-06 | 85.0% | 376 / 600 | 20 | **98.13%** | **86.67%** | 0.009971 | 85.8s |
| 77 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.5 | 1e-05 | 85.0% | 436 / 600 | 10 | **97.05%** | **86.11%** | 0.010535 | 100.8s |
| 78 | 56 | sigmoid | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-06 | 85.0% | 263 / 600 | 10 | **97.83%** | **84.44%** | 0.010702 | 61.8s |
| 79 | 56 | sigmoid | mse | 0.01 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 600 / 600 | 20 | **95.12%** | **83.96%** | 0.012108 | 132.0s |
| 80 | 56 | sigmoid | mse | 0.01 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 600 / 600 | 20 | **95.18%** | **83.89%** | 0.012162 | 137.4s |
| 81 | 56 | sigmoid | mse | 0.05 | normal | sgd | 0.8 | 1e-05 | 70.0% | 600 / 600 | 20 | **94.11%** | **83.49%** | 0.012960 | 100.8s |
| 82 | 56 | sigmoid | mse | 0.05 | normal | sgd | 0.8 | 1e-05 | 85.0% | 600 / 600 | 20 | **94.09%** | **83.33%** | 0.013133 | 102.6s |
| 83 | 56 | sigmoid | mse | 0.01 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 514 / 600 | 10 | **93.80%** | **82.22%** | 0.013039 | 119.4s |
| 84 | 56 | sigmoid | mse | 0.05 | normal | sgd | 0.8 | 1e-05 | 85.0% | 462 / 600 | 10 | **92.32%** | **81.11%** | 0.014171 | 80.4s |
| 85 | 56 | sigmoid | mse | 0.01 | uniform | sgd | 0.8 | 1e-05 | 85.0% | 600 / 600 | 10 | **86.12%** | **80.00%** | 0.019376 | 102.6s |
| 86 | 56 | leaky_relu | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 117 / 600 | 20 | **39.27%** | **32.22%** | 0.029874 | 25.8s |
| 87 | 56 | leaky_relu | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 24 / 600 | 10 | **24.21%** | **18.89%** | 0.033587 | 5.4s |
| 88 | 56 | sigmoid | mse | 0.001 | uniform | sgd_momentum | 0.8 | 1e-05 | 85.0% | 55 / 600 | 10 | **12.60%** | **10.00%** | 0.036928 | 13.2s |
| 89 | 56 | linear | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 29 / 600 | 20 | **11.59%** | **6.60%** | 0.037195 | 6.0s |
| 90 | 56 | linear | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 17 / 600 | 10 | **10.83%** | **5.56%** | 0.037842 | 3.6s |
| 91 | 56 | linear | mse | 0.05 | normal | sgd_momentum | 0.8 | 1e-05 | 85.0% | 27 / 600 | 20 | **10.73%** | **5.56%** | 0.037842 | 6.0s |
