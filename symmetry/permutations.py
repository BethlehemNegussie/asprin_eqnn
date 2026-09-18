"""Species preserving atom permutations for symmetry-error diagnostics."""
import numpy as np


def random_species_permutation(Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a permutation of indices 0..20 that only shuffles atoms within the
    same chemical species (C among C, H among H, O among O)."""
    perm = np.arange(len(Z))
    for z in np.unique(Z):
        idx = np.where(Z == z)[0]
        shuffled = rng.permutation(idx)
        perm[idx] = shuffled
    return perm


def apply_permutation(coords: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """coords: [...,21,3] -> permuted along the atom axis."""
    return coords[..., perm, :]


def permutation_error(predict_fn, coords: np.ndarray, Z: np.ndarray, n_trials: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    base_pred = predict_fn(coords)
    errs = []
    for _ in range(n_trials):
        perm = random_species_permutation(Z, rng)
        permuted = apply_permutation(coords, perm)
        pred_t = predict_fn(permuted)
        errs.append(np.abs(pred_t - base_pred))
    errs = np.concatenate(errs)
    return {"mean": float(errs.mean()), "max": float(errs.max())}
