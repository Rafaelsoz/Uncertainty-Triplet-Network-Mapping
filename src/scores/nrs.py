import numpy as np
from sklearn.neighbors import KDTree


class NeighborhoodReliabilityScore:
    def __init__(self):
        self.kdtrees = {}
        self.classes = None


    def fit(self, X: np.ndarray, targets: np.ndarray):
        
        targets = targets.astype(int)
        
        self.classes = np.unique(targets).astype(int)

        for cls_ in self.classes:
            mask = targets == cls_
            class_data = X[mask]
            self.kdtrees[cls_] = KDTree(class_data)


    def _compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:

        num_samples = X.shape[0]
        num_classes = len(self.classes)

        distance_matrix  = np.empty((num_samples, num_classes))

        for cls_ in self.classes:
            current_distances = self.kdtrees[cls_].query(X, k=1)[0][:, 0]
            distance_matrix[:, cls_] = current_distances

        return distance_matrix

    def get_score(
        self,
        X: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:

        y_pred = y_pred.astype(int)

        num_samples = X.shape[0]
        indexes = np.arange(num_samples)

        distance_matrix = self._compute_distance_matrix(X)
        same_class_distances = distance_matrix[indexes, y_pred]

        distance_matrix[indexes, y_pred] = np.inf
        other_class_distances = np.min(distance_matrix, axis=1)

        ratio = same_class_distances / (np.maximum(other_class_distances, 1e-15))

        scores = 1 / (ratio + 1)

        return scores