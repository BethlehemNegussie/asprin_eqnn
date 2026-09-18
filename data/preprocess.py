"""Mass-weighted COM centering and training only scale"""
import numpy as np

# Standard atomic masses (amu)
ATOMIC_MASS = {1: 1.008, 6: 12.011, 8: 15.999}


def center_of_mass_remove(coords: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """coords: [N,21,3], Z: [21] -> centered coords [N,21,3].

    R_CM = (sum_j m_j r_j) / (sum_j m_j); r'_j = r_j - R_CM.
    By centering the molecule the absolute position disappears.  
    Translation invariant, rotation covariant.
    """
    m = np.array([ATOMIC_MASS[int(z)] for z in Z])  # [21]
    m = m / m.sum() # normalize mass to 1
    R_CM = np.einsum("j,njk->nk", m, coords)  # [N,3], calculates the mass-weighted average position along each coordinate axis (x, y, z) for every individual frame
    return coords - R_CM[:, None, :] # reshaping the center of mass tensor so broadcasting works, every atom in a particular molecule gets the same translation removed


def compute_train_scale(centered_train: np.ndarray) -> float:
    """s = a coordinate scale scalar value that measures the overall geometric size or spatial spread of the centered molecules.
    It is computed as the Root Mean Square (RMS) distance of all atoms from the center of mass across the entire training set.
    Square the centered training coordinates for each atom, return the summed coordinate dimensions, average that 
    over the whole training set, for all 21 atoms and then take the sqrt.
    s = sqrt(mean_{i,j} ||r'_ij||^2) computed on train split only.
    This is used for the purpose of standardizing input magnitude:
    Molecular coordinates are typically expressed in Angstroms 
    where atom positions might range from -5 to +5 Angstroms.
    Dividing coordinates by s scales the typical atom distance down to roughly approx 1.0.
    This is only calculated over the training set; because otherwise information about the test distribution would leak into the preprocessing
    """
    sq_norms = np.sum(centered_train ** 2, axis=-1)  # [N,21], axis=-1 tells NumPy to sum across the very last axis[coordinate axis] of the array.
    return float(np.sqrt(sq_norms.mean()))


def make_splits(n: int, seed: int = 42, ratios=(0.8, 0.1, 0.1)):
    """Fixed seed 80/10/10 split by frame."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n) # prevents the dataset's original ordering from determining which frames become training/test data
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_idx = idx[:n_train] # Actual splitting
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


class EnergyNormalizer:
    """Class that normalizes the target energy.
    mu = training energy mean  and sigma = training energy standard deviation  
    E_norm = (E - mu)/sigma, fit on train only."""

    def fit(self, E_train: np.ndarray): # learns the normalization parameters from training energies only
        self.mu = float(E_train.mean()) # calculates the training mean 
        self.sigma = float(E_train.std()) # calculates the training standard deviation
        return self

    def transform(self, E):
        return (E - self.mu) / self.sigma

    def inverse(self, E_norm):
        """This reverses the transformation since the model may predict normalized energy but the 
        final reported prediction needs to be in the original physical units, otherwise the reported MAE 
        would be in normalized units rather than the original energy units. """
        return E_norm * self.sigma + self.mu
