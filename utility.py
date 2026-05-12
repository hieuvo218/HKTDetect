import math
import numpy as np
from scipy import stats


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


class LSHIndex:
    def __init__(self, n_planes: int = 10, n_tables: int = 5, random_state: int = 42):
        """
        n_planes : hyperplanes per table (more = fewer candidates, faster but less accurate)
        n_tables : number of hash tables (more = better recall, slower)
        """
        self.n_planes = n_planes
        self.n_tables = n_tables
        self.rng = np.random.default_rng(random_state)
        self.planes = []       # shape: (n_tables, n_planes, dim)
        self.tables = []       # list of dicts: hash_key -> list of indices
        self.X = None

    def fit(self, X: np.ndarray):
        self.X = X
        dim = X.shape[1]
        self.planes = self.rng.standard_normal((self.n_tables, self.n_planes, dim)).astype(X.dtype)

        for t in range(self.n_tables):
            projections = X @ self.planes[t].T          # (N, n_planes)
            bits = (projections >= 0).astype(np.uint8)  # binarize
            table = {}
            for i, b in enumerate(bits):
                key = b.tobytes()
                table.setdefault(key, []).append(i)
            self.tables.append(table)

    def query(self, x: np.ndarray, k: int, p: int = 2) -> np.ndarray:
        candidates = set()
        for t, table in enumerate(self.tables):
            bits = (x @ self.planes[t].T >= 0).astype(np.uint8)
            key = bits.tobytes()
            candidates.update(table.get(key, []))

        if not candidates:
            candidates = set(range(len(self.X)))  # fallback: brute-force

        cands = np.array(list(candidates))
        dists = np.linalg.norm(self.X[cands] - x, ord=p, axis=1)

        # sort candidates by distance and return top k indices
        top_k_idx = np.argpartition(dists, min(k, len(dists)) - 1)[:k]
        top_k = cands[top_k_idx[np.argsort(dists[top_k_idx])]]

        return top_k


def preprocess_mnist(X_train, X_test=None, y_train=None, y_test=None, flatten: bool = True, normalize: bool = True):
    """Prepare MNIST arrays for classical classifiers.

    Parameters
    ----------
    X_train, X_test:
        Image arrays with shape (N, 28, 28), (N, 28, 28, 1), or already flattened.
    y_train, y_test:
        Optional label arrays returned unchanged except for flattening.
    flatten:
        Convert images to shape (N, 784). This is the safest default for KNN.
    normalize:
        Scale pixel values to [0, 1] and cast to float32.

    Returns
    -------
    Tuple containing the transformed inputs, and labels when provided.
    """

    def _prepare_images(X):
        X = np.asarray(X)

        if normalize:
            X = X.astype(np.float32) / 255.0
        else:
            X = X.astype(np.float32)

        if flatten:
            if X.ndim == 4 and X.shape[-1] == 1:
                X = X.reshape(X.shape[0], -1)
            elif X.ndim == 3:
                X = X.reshape(X.shape[0], -1)
            elif X.ndim != 2:
                raise ValueError(f"Expected MNIST images with 2, 3, or 4 dimensions, got shape {X.shape}")
        else:
            if X.ndim == 3:
                X = X[..., np.newaxis]
            elif X.ndim not in (4,):
                raise ValueError(f"Expected MNIST images with 3 or 4 dimensions, got shape {X.shape}")

        return X

    def _prepare_labels(y):
        if y is None:
            return None
        return np.asarray(y).reshape(-1)

    X_train = _prepare_images(X_train)
    outputs = [X_train]

    if X_test is not None:
        outputs.append(_prepare_images(X_test))

    if y_train is not None:
        outputs.append(_prepare_labels(y_train))

    if y_test is not None:
        outputs.append(_prepare_labels(y_test))

    return tuple(outputs)