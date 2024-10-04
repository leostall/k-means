# O código implementa o Método do Cotovelo para decidir o número ideal de clusters (k). 
# Ele gera um conjunto de dados e executa o K-means para valores de k de 1 a 10, calculando 
# a inércia (soma das distâncias quadradas dos pontos até seus centroides). O gráfico final exibe a inércia 
# em função de k, assim conseguimos identificar o "cotovelo", que indica o valor ideal de clusters onde o ganho de 
# ajuste começa a diminuir.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=0)

inercias = []

# Testando diferentes valores de k (de 1 a 10)
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    # Guardando o valor da inércia (soma das distâncias quadradas)
    inercias.append(kmeans.inertia_)

plt.plot(k_values, inercias, 'bo-', markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia (Soma das Distâncias Quadradas)')
plt.title('Método do Cotovelo para Identificar o Melhor k')
plt.grid(True)
plt.show()




