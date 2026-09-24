"""Serializable PLS and RBF SVM fitted to individual Mouse2Vec windows."""

from collections import Counter

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class WindowPLSSVM:
    """Fit on windows, with equal total SVM weight per participant and class."""

    def __init__(self, n_components: int, C: float):
        self.n_components = n_components
        self.C = C

    def fit(self, z: np.ndarray, demographics: np.ndarray, y: np.ndarray, ids: np.ndarray):
        self.mouse_scaler = StandardScaler().fit(z)
        mouse = self.mouse_scaler.transform(z)
        self.pls = PLSRegression(n_components=self.n_components, scale=False).fit(mouse, y)
        projected = self.pls.transform(mouse)
        self.pls_scaler = StandardScaler().fit(projected)
        unique_ids, first_indices = np.unique(ids, return_index=True)
        self.demo_scaler = StandardScaler().fit(demographics[first_indices])
        X = self._combine(projected, demographics)

        # Equal total SVM weight for each participant, and for each class.
        counts = Counter(ids)
        class_people = Counter(y[first_indices])
        weights = np.asarray(
            [len(z) / (2 * class_people[label] * counts[sid]) for sid, label in zip(ids, y)]
        )
        self.svm = SVC(kernel="rbf", C=self.C, gamma="scale", cache_size=500)
        self.svm.fit(X, y, sample_weight=weights)
        self.train_people = len(unique_ids)
        return self

    def _combine(self, projected: np.ndarray, demographics: np.ndarray) -> np.ndarray:
        mouse = self.pls_scaler.transform(projected) / np.sqrt(self.n_components)
        demo = self.demo_scaler.transform(demographics) / np.sqrt(3)
        return np.column_stack((mouse, demo))

    def decision_function(self, z: np.ndarray, demographics: np.ndarray) -> np.ndarray:
        projected = self.pls.transform(self.mouse_scaler.transform(z))
        return self.svm.decision_function(self._combine(projected, demographics))
