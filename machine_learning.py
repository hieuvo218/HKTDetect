import numpy as np
import math

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
    def __init__(self, num_neighbors: int, p: int, leaf_size: int = 30):
        self.num_neighbors = num_neighbors
        self.p = p
        self.leaf_size = leaf_size
        self.tree = None
        self.y = None

    def fit(self, X, y):
        self.y = y
        self.tree = KdTree(X, self.leaf_size)

    def predict(self, X: np.ndarray):
        # query returns indices of nearest neighbors
        if len(X.shape) < 2:
            X = np.expand_dims(X, axis=0)

        indices = np.array([self.tree.query(x, k=self.num_neighbors) for x in X])
        all_labels = self.y[indices]

        from scipy import stats
        predictions, _ = stats.mode(all_labels, axis=1, keepdims=False)
        
        return predictions.ravel()