import numpy as np

class DataLoader:
    #Método que retorna um dataset com os dados do conjunto CARACTERES COMPLETO. Usamos o arquivo .npy
    def carregar_dados_alfabeto(caminho_x, caminho_y):
        x_raw = np.load(caminho_x)
        y_raw = np.load(caminho_y)
        
        #usamos o método shape para ver o formato dos dados e depois reshape para deixar no formato certo
        x_flat = x_raw.reshape(1326, 120) #Achata as imagens de 10x12 para um vetor de 120 posições, sendo 1326 amostras

        dataset_CARACTERES = []

        for i in range(len(x_flat)):
            dataset_CARACTERES.append([x_flat[i].tolist(), y_raw[i].tolist()]) #adiciona ao dataset a entrada (a letra) e o rótulo. São 120 entradas e 26 saídas.

        return dataset_CARACTERES