"""Random O(3)/SO(3) transforms for symmetry-error diagnostics."""
import numpy as np


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Random proper rotation matrix (Haar-random via QR of a Gaussian matrix)."""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def random_reflection(rng: np.random.Generator) -> np.ndarray:
    """Random improper transform: a random rotation composed with a parity flip."""
    Q = random_rotation(rng)
    Q[:, 0] *= -1  # det(Q) = -1
    return Q


def apply_transform(coords: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """coords: [...,21,3] -> rotated/reflected coords, same shape."""
    return coords @ Q.T


def symmetry_error(predict_fn, coords: np.ndarray, n_trials: int = 5, seed: int = 0,
                    include_reflections: bool = True):
    """Delta_sym = |h(X) - h(g X)| for random g in O(3), reported mean/max."""
    rng = np.random.default_rng(seed)
    base_pred = predict_fn(coords)
    errs = []
    for t in range(n_trials):
        Q = random_reflection(rng) if (include_reflections and t % 2 == 1) else random_rotation(rng)
        transformed = apply_transform(coords, Q)
        pred_t = predict_fn(transformed)
        errs.append(np.abs(pred_t - base_pred))
    errs = np.concatenate(errs)
    return {"mean": float(errs.mean()), "max": float(errs.max())}
