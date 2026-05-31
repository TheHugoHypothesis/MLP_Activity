"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from typing import List

""" Módulo que contém funções relacionadas à Álgebra Linear """

""" 
Realiza o produto escalar entre duas listas Python.
Ou seja, para duas listas L1 e L2, faz o somatório
de L1[i] . L2[i] para todo 0 <= i <= len(L1) = len(L2)
"""
def scalar_product(
        list_1 : List[float], 
        list_2 : List[float]
    ) -> float:
    total: float = 0.0

    #Soma ponderada de listas de tamanho diferentes
    if (len(list_1) != len(list_2)): 
        raise ValueError("Tentativa de cálculo de produto escalar entre listas de tamanhos distintos.")
    
    for i in range(len(list_1)):
        total += list_1[i] * list_2[i]

    return total