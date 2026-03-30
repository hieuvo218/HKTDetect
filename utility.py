import math
import numpy as np


class KdTreeNode:
    def __init__(self, point: np.ndarray, index=None, indices: np.ndarray = None, left: 'KdTreeNode' = None, right: 'KdTreeNode' = None,  axis=None, points=None):
        self.left = left
        self.right = right
        self.point = point
        self.axis = axis
        self.index = index
        self.indices = indices

class KdTree:
    """
    KdTree which is a balanced tree used to look up nearest neighbors
    in that data points have d dimensions.
    """
    def __init__(self, X, leaf_size: int = 10):
        self.X = X
        self.k = self.X.shape[1]
        self.leaf_size = leaf_size
        self.root = self._build(self.X, np.arange(len(X)), depth=0)

    def _build(self, X, indices, depth):
        if len(X) == 0:
            return None
        
        if len(X) <= self.leaf_size:
            return KdTreeNode(point=X, indices=indices)
        
        # Choose node as i % k
        axis = depth % self.k
        sorted_indices= np.argsort(X[:, axis])
        X_sorted = X[sorted_indices]
        indices_sorted = indices[sorted_indices]

        median = len(X_sorted) // 2
        X_1 = X_sorted[:median]
        X_2 = X_sorted[median+1:]

        return KdTreeNode(point=X_sorted[median], 
                          index=indices_sorted[median],
                          left=self._build(X_1, indices_sorted[:median], depth+1), 
                          right=self._build(X_2, indices_sorted[median+1:], depth+1), 
                          axis=axis)

    def query(self, x, p=1, metric: str='Minkowski', k: int=1):
        # Use min heap to store neighbors
        import heapq
        best = []

        def _distance(x1, x2, p=1, metric: str='Minkowski'):
            if metric == 'Minkowski':
                return np.linalg.norm(x=x1-x2, ord=p)
            
        def _search(node):
            if node is None:
                return 
            
            # Base case
            if node.left is None and node.right is None:
                points = node.point if len(node.point.shape) > 1 else node.point.reshape(1, -1)
                for i, point in enumerate(points):
                    d = _distance(x, point, p=p, metric=metric)
                    idx = node.indices[i]

                    if len(best) < k:
                        heapq.heappush(best, (-d, idx))
                    elif -best[0][0] > d:
                        heapq.heapreplace(best, (-d, idx))
                
                return

            # Consider internal node
            d = _distance(x, node.point, p, metric)
            if len(best) < k:
                heapq.heappush(best, (-d, node.index))
            elif -best[0][0] > d:
                heapq.heapreplace(best, (-d, node.index))
            
            # Calculate different with divide boundary
            axis = node.axis
            diff= x[axis] - node.point[axis]

            # Choose which branch to go
            if diff < 0:
                _search(node.left)
                if len(best) < k or abs(diff) < -best[0][0]:
                    _search(node.right)
            else:
                _search(node.right)
                if len(best) < k or abs(diff) < -best[0][0]:
                    _search(node.left)

        _search(self.root)

        result_indices = sorted([idx for _, idx in best], key=lambda i: _distance(x, self.X[i], p=p, metric=metric))
        return np.array(result_indices[:k])