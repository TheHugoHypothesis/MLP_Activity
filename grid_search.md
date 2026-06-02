# 📊 Resumo da Otimização por Hill Climbing
Tabela resumida contendo todas as **52 combinações** de hiperparâmetros testadas e avaliadas pelo algoritmo de Hill Climbing, ordenadas da **melhor para a pior** com base na acurácia do conjunto de validação.
## 🏆 Melhor Configuração Encontrada
* **Acurácia de Validação:** 0.9183 (91.83%)
* **Acurácia de Treino Final:** 98.13%
* **Partição do Dataset (Treino):** 70.0%
* **Função de Perda de Validação:** 0.008068
* **Tempo de Execução:** 28.80 segundos
* **Hiperparâmetros:**
  * Neurônios Ocultos: `56`
  * Função de Ativação Oculta: `sigmoid`
  * Função de Perda: `mse`
  * Inicializador: `uniform`
  * Learning Rate: `0.01`
  * Otimizador: `sgd_momentum`
  * Momentum: `0.8`
  * L2 Decay: `1e-05`
---
## 📋 Tabela Geral de Resultados
| # | Neurônios | Ativação | Perda | LR | Inicializador | Otimizador | Momentum | L2 Decay | Partição Treino | Épocas | Acurácia Treino | Acurácia Val | Val Loss | Tempo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 217 | **99.23%** | **91.83%** | 0.006998 | 10.0s |
| 2 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.9 | 1e-05 | 70.0% | 382 | **99.23%** | **91.83%** | 0.007459 | 18.8s |
| 3 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-06 | 70.0% | 570 | **98.46%** | **91.83%** | 0.007520 | 27.3s |
| 4 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 600 | **98.24%** | **91.83%** | 0.007859 | 29.0s |
| 5 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 574 | **98.13%** | **91.83%** | 0.008068 | 28.8s |
| 6 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 574 | **98.13%** | **91.83%** | 0.008068 | 28.3s |
| 7 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.9 | 1e-05 | 70.0% | 285 | **97.91%** | **91.83%** | 0.008088 | 13.8s |
| 8 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 167 | **99.23%** | **91.35%** | 0.007307 | 8.1s |
| 9 | 56 | sigmoid | mse | 0.01 | xavier | sgd_momentum | 0.8 | 1e-05 | 70.0% | 600 | **98.79%** | **91.35%** | 0.007735 | 29.4s |
| 10 | 64 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 576 | **98.35%** | **91.35%** | 0.007744 | 28.8s |
| 11 | 56 | leaky_relu | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 194 | **98.24%** | **90.87%** | 0.007249 | 9.4s |
| 12 | 56 | leaky_relu | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 139 | **97.91%** | **90.87%** | 0.007565 | 6.8s |
| 13 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 447 | **96.81%** | **90.87%** | 0.008905 | 21.8s |
| 14 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 447 | **96.81%** | **90.87%** | 0.008905 | 21.6s |
| 15 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-06 | 70.0% | 446 | **97.36%** | **90.38%** | 0.008345 | 21.6s |
| 16 | 48 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 599 | **97.58%** | **90.38%** | 0.008521 | 28.0s |
| 17 | 64 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 419 | **97.25%** | **90.38%** | 0.008788 | 21.1s |
| 18 | 56 | sigmoid | mse | 0.01 | xavier | sgd_momentum | 0.8 | 1e-05 | 70.0% | 450 | **97.14%** | **89.90%** | 0.008844 | 21.7s |
| 19 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 400 | **96.26%** | **89.90%** | 0.009316 | 19.3s |
| 20 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 400 | **96.26%** | **89.90%** | 0.009316 | 19.0s |
| 21 | 48 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 457 | **96.81%** | **89.90%** | 0.009446 | 21.8s |
| 22 | 56 | sigmoid | mse | 0.05 | uniform | sgd_momentum | 0.8 | 1e-05 | 80.0% | 142 | **98.94%** | **89.42%** | 0.008337 | 7.8s |
| 23 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 80.0% | 498 | **98.37%** | **89.42%** | 0.008876 | 27.2s |
| 24 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.9 | 1e-05 | 80.0% | 250 | **98.37%** | **89.42%** | 0.008876 | 14.2s |
| 25 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-06 | 80.0% | 383 | **97.31%** | **89.42%** | 0.009102 | 20.2s |
| 26 | 56 | linear | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 151 | **97.69%** | **89.42%** | 0.009236 | 7.0s |
| 27 | 56 | linear | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 126 | **97.25%** | **89.42%** | 0.009337 | 5.5s |
| 28 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 80.0% | 384 | **97.21%** | **89.42%** | 0.009608 | 20.5s |
| 29 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 80.0% | 384 | **97.21%** | **89.42%** | 0.009608 | 20.2s |
| 30 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 80.0% | 384 | **97.21%** | **89.42%** | 0.009608 | 21.7s |
| 31 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 343 | **95.60%** | **89.42%** | 0.010289 | 16.8s |
| 32 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 0.0001 | 70.0% | 573 | **94.29%** | **89.42%** | 0.014355 | 26.8s |
| 33 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 0.0001 | 70.0% | 459 | **93.19%** | **89.42%** | 0.015091 | 22.1s |
| 34 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 80.0% | 296 | **95.96%** | **88.46%** | 0.010794 | 16.0s |
| 35 | 56 | leaky_relu | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 80.0% | 117 | **97.79%** | **87.98%** | 0.009265 | 6.4s |
| 36 | 48 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 80.0% | 382 | **96.63%** | **87.98%** | 0.009826 | 20.2s |
| 37 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.5 | 1e-05 | 80.0% | 600 | **95.00%** | **87.98%** | 0.012062 | 33.6s |
| 38 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.5 | 1e-05 | 70.0% | 600 | **92.64%** | **87.98%** | 0.013293 | 28.9s |
| 39 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.5 | 1e-05 | 70.0% | 600 | **92.64%** | **87.98%** | 0.013293 | 28.9s |
| 40 | 64 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 80.0% | 377 | **96.73%** | **87.50%** | 0.009417 | 22.2s |
| 41 | 56 | sigmoid | mse | 0.01 | xavier | sgd_momentum | 0.8 | 1e-05 | 80.0% | 405 | **97.12%** | **87.50%** | 0.009591 | 21.8s |
| 42 | 56 | linear | mse | 0.01 | uniform | sgd_momentum | 0.8 | 1e-05 | 80.0% | 94 | **97.02%** | **86.06%** | 0.010499 | 4.7s |
| 43 | 56 | sigmoid | mse | 0.01 | uniform | sgd_momentum | 0.8 | 0.0001 | 80.0% | 391 | **94.23%** | **84.62%** | 0.015558 | 21.4s |
| 44 | 56 | sigmoid | mse | 0.01 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 600 | **93.30%** | **81.73%** | 0.014008 | 28.4s |
| 45 | 56 | sigmoid | mse | 0.01 | normal | sgd_momentum | 0.8 | 1e-05 | 70.0% | 600 | **93.30%** | **81.73%** | 0.014008 | 28.8s |
| 46 | 56 | sigmoid | mse | 0.01 | normal | sgd_momentum | 0.8 | 1e-05 | 80.0% | 521 | **95.38%** | **80.77%** | 0.014287 | 28.4s |
| 47 | 56 | sigmoid | mse | 0.01 | uniform | sgd | 0.8 | 1e-05 | 80.0% | 600 | **83.75%** | **77.88%** | 0.020026 | 24.9s |
| 48 | 56 | sigmoid | mse | 0.01 | uniform | sgd | 0.8 | 1e-05 | 70.0% | 600 | **75.71%** | **70.19%** | 0.022846 | 21.6s |
| 49 | 56 | sigmoid | mse | 0.001 | uniform | sgd_momentum | 0.8 | 1e-05 | 80.0% | 65 | **15.96%** | **16.35%** | 0.036777 | 3.9s |
| 50 | 56 | sigmoid | mse | 0.001 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 91 | **13.30%** | **13.46%** | 0.036806 | 4.4s |
| 51 | 56 | sigmoid | mse | 0.01 | uniform | sgd | 0.8 | 1e-05 | 70.0% | 50 | **12.97%** | **12.50%** | 0.036756 | 1.9s |
| 52 | 56 | sigmoid | mse | 0.001 | uniform | sgd_momentum | 0.8 | 1e-05 | 70.0% | 66 | **14.29%** | **12.50%** | 0.036906 | 3.3s |
