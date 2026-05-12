import numpy as np
import math

from scipy import stats

from utility import *


class LinearRegressor:
    def __init__(self, num_features):
        self.w = np.random.rand(num_features)
        self.b = 0.0

    def train(self, X, y, lr: float = 0.01, epochs: int = 10, regularization=None, alpha: float = 0.01):
        N = len(X)

        for _ in range(epochs):
            y_hat = self.forward(X)
            error = y_hat - y
            
            grad_w = (2 / N) * X.T  @ error
            grad_b = (2 / N) * error.sum()

            if not regularization:
                pass

            elif regularization=="L1" or regularization=="Lasso":
                l1_grad = alpha * np.sign(self.w)
                grad_w += l1_grad

            elif regularization=="L2" or regularization=="Ridge":
                grad_w += 2 * alpha * self.w
            
            self.w -= lr * grad_w
            self.b -= lr * grad_b


    def forward(self, X):
        return X @ self.w + self.b


class KNearestNeighbors:
    def __init__(
        self,
        num_neighbors: int,
        p: int,
        index: str = "kdtree",   # "kdtree" | "lsh"
        leaf_size: int = 30,
        lsh_n_planes: int = 10,
        lsh_n_tables: int = 5,
    ):
        self.num_neighbors = num_neighbors
        self.p = p
        self.index = index
        self.leaf_size = leaf_size
        self.lsh_n_planes = lsh_n_planes
        self.lsh_n_tables = lsh_n_tables
        self._index = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.y = y
        if self.index == "lsh":
            self._index = LSHIndex(self.lsh_n_planes, self.lsh_n_tables)
            self._index.fit(X)
        else:
            self._index = KdTree(X, self.leaf_size)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim < 2:
            X = np.expand_dims(X, axis=0)

        if self.index == "lsh":
            indices = np.array([
                self._index.query(x, k=self.num_neighbors, p=self.p) for x in X
            ])
        else:
            indices = np.array([
                self._index.query(x, p=self.p, k=self.num_neighbors) for x in X
            ])

        predictions, _ = stats.mode(self.y[indices], axis=1, keepdims=False)
        return predictions.ravel()