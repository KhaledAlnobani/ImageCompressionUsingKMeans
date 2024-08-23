import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, K, ite=100, plot_step=False):
        self.K = K
        self.ite = ite
        self.plot_step = plot_step
        self.centroids = None

    def predict(self, X):
        self.centroids = self._initalize_centroids(X, self.K)
        for i in range(self.ite):

            old_centroids = self.centroids
            ind = self._compute_closet_centroids(X, self.centroids)

            if self.plot_step:
                self.plot(X, ind)
            self.centroids = self._compute_centroid(ind, self.K, X)

            if self._converged(old_centroids, self.centroids):
                break
            if self.plot_step:
                self.plot(X, ind)

        return self.centroids, ind

    def _initalize_centroids(self, X, K):
        centroids = np.random.choice(X.shape[0], K, replace=False)
        return X[centroids]

    def _compute_closet_centroids(self, X, centroid):
        n = X.shape[0]
        ind = np.zeros(n, dtype=int)  # Initialize with int instead of float

        for i in range(n):
            dist = []
            for k in range(centroid.shape[0]):
                dist.append(np.linalg.norm(X[i] - centroid[k]))
            ind[i] = np.argmin(dist)

        return ind

    def _compute_centroid(self, ind, K, X):
        centroids = np.zeros((K, X.shape[1]))
        for k in range(K):
            point = X[ind == k]
            centroids[k] = np.mean(point, axis=0)
        return centroids

    def _converged(self, old_centroids, new_centroids, tol=1e-4):
        distance = np.linalg.norm(new_centroids - old_centroids)
        return distance < tol

    def plot(self, X, ind):
        plt.figure(figsize=(4, 3))

        for k in range(self.K):
            plt.scatter(X[ind == k, 0], X[ind == k, 1], label=f'Cluster {k}')

        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=300, c='black', marker='x', label='Centroids')

        plt.title('K-means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()



