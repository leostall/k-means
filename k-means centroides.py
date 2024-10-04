# O código usa o algoritmo K-means para agrupar dados sintéticos em 3 clusters. 
# Primeiro, ele gera e visualiza os dados em um gráfico. Em seguida, aplica o K-means 
# com 3 clusters e exibe os clusters.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Visualizando os dados gerados
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Dados Gerados")
plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

centroides = kmeans.cluster_centers_
rótulos = kmeans.labels_

plt.scatter(X[:, 0], X[:, 1], c=rótulos, s=50, cmap='viridis')

plt.scatter(centroides[:, 0], centroides[:, 1], s=200, c='red', marker='X')
plt.title("Clusters Identificados pelo K-means")
plt.show()
