"""Reduce the 96-dim invariant descriptor to N_QUBITS scalar angles.

This is purely a classical, PCA-based dimensionality reduction fit on the
training split only. It does not touch rotation/reflection/permutation
invariance (already exact from features/invariant.py) -- it only controls how
many invariant scalar channels are exposed to the quantum encoding, which is
an explicit engineering choice for a tractable simulated qubit count.
"""
import numpy as np
from sklearn.decomposition import PCA


class AngleEncoder:
    def __init__(self, n_qubits: int = 8, seed: int = 42):
        self.n_qubits = n_qubits
        self.pca = PCA(n_components=n_qubits, random_state=seed)
        self.lo = None
        self.hi = None

    def fit(self, X_train: np.ndarray):
        z = self.pca.fit_transform(X_train)
        self.lo = z.min(axis=0)
        self.hi = z.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = self.pca.transform(X)
        z = (z - self.lo) / np.clip(self.hi - self.lo, 1e-8, None)  # [0,1]
        angles = z * np.pi  # [0, pi]
        return angles
