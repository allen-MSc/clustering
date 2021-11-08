import numpy as np
from scipy.spatial.distance import cdist

class Kmeans:
    def __init__(self, data, num_clusters, n_iterations, seed):
        np.random.seed(seed)
        self.data = data
        self.centroids = []
        self.num_iterations = n_iterations

        # Creamos un arreglo que almacenará el centroide
        # asociado a cada de los puntos de mi data
        self.idx_centroid_assigned = [0] * len(data)

        # Inicializamos los centroides con puntos aleatorios de mi data
        random_indices = np.random.randint(data.shape[0], size=num_clusters)
        self.centroids = data[random_indices, :]

    def fit(self):
        # Iteramos n veces para ajustar los centroides
        for n in range(self.num_iterations):
            old_centroids = np.copy(self.centroids)

            # Asignamos el centroide mas cercano a cada punto de mi data
            distances = cdist(self.data, self.centroids ,'euclidean')
            self.idx_centroid_assigned = np.array([np.argmin(i) for i in distances])

            # Computamos los nuevos centroides
            temp_centroids = []
            for c in range(len(self.centroids)):
                # Computo la media para obtener el nuevo centroide
                centroid_mean = self.data[self.idx_centroid_assigned==c].mean(axis=0)
                temp_centroids.append(centroid_mean)
            self.centroids = np.vstack(temp_centroids)

            if np.all(old_centroids == self.centroids):
                print("Break on interation:", n)
                break
