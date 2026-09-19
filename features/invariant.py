"""Track-A invariant point-cloud features.

For each atom pair define d_jk = ||x_j - x_k||. Distances are invariant under
translation, rotation and reflection: ||Q(x_j - x_k)|| = ||x_j - x_k|| for Q in O(3).
Species-aware pair features are aggregated with permutation invariant sums into a
smooth radial basis histogram, one histogram per unordered species pair
(HH, CC, OO, HC, HO, CO) -> a fixed length, rotation/reflection/permutation
invariant molecule level descriptor.
"""
import numpy as np
from itertools import combinations_with_replacement

SPECIES = [1, 6, 8]  # H, C, O
SPECIES_PAIRS = list(combinations_with_replacement(SPECIES, 2))  # 6 unordered pairs(n(n+1)/2) of species, including self-pairs (H,H), (C,C), (O,O) and cross-pairs (H,C), (H,O), (C,O)
#permutation because distance is symmetric

class InvariantDescriptor:
    """A class representing the entire feature extraction procedure.

    Covers distances from d_min = 0.5 Å to d_max = 6.0 Å. Placing 16 evenly spaced bin centers 
    across this interval creates a smooth grid where each RBF specializes in detecting atomic 
    pairs separated by a specific distance range. Expanding 6 species-pair types across 16 radial 
    channels yields a 96-dimensional descriptor (6 * 16 = 96), providing sufficient expressive 
    capacity for the downstream network to distinguish fine grained chemical bond configurations 
    and non-bonded spatial interactions.

    An RBF measures similarity based on how far an input is from a center point, showing 
    hyperspherical symmetry around that center; direction does not alter the output value, only the radius.

    (Z, R) -> f, where f ∈ R^96
    """
    def __init__(self, n_bins: int = 16, d_min: float = 0.5, d_max: float = 6.0):
        self.n_bins = n_bins
        self.centers = np.linspace(d_min, d_max, n_bins) # produces 16 evenly spaced centers
        # the radial basis width to determine how wide each radial basis function is, chosen so adjacent bins have sensible overlap
        self.beta = 1.0 / (2.0 * (self.centers[1] - self.centers[0]) ** 2) # beta is a width parameter

    def _rbf(self, d: np.ndarray) -> np.ndarray:
        # function that converts distances into radial-basis responses, d: [...,] -> [..., n_bins]
        diff = d[..., None] - self.centers  # broadcast
        return np.exp(-self.beta * diff ** 2)

    def transform(self, coords: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        method that converts 3D atomic coordinates into a fixed length, 
        rotation and translation invariant 96-dimensional feature vector for each frame.
        coords: [N,21,3] (already COM-centered and scaled), Z: [21] -> [N, 6*n_bins]."""
        N = coords.shape[0]
        # Broadcasting computes the displacement vector r_j - r_k 
        # for every atom pair (j, k) across all N frames, creating a 21 * 21 * 3 coordinate 
        # difference tensor per frame
        diff = coords[:, :, None, :] - coords[:, None, :, :]  # computing pair-wise distances for every atom against every other atom.  [N,21,21,3]
        # convert displacement vectors to distances and store it in a Euclidean matrix D [N,21,21], and add a numerical stabilizer
        D = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12) 

        iu, ju = np.triu_indices(21, k=1)  
        """Extracts the upper triangular indices of the distance 
        matrix (excluding self distances on the diagonal since d_ij - d_ji = 0), rendering 
        unique unordered atom pairs (permutation-invariant sum).
        
        rbf_pairs passes the 210 pair distances through 16 Radial Basis Function (Gaussian) bins,
        converting each distance scalar into a 16-dimensional smooth activation vector."""

        # for every frame in the dataset the d_pairs compresses the full 21 x 21 distance grid to a simple 1D array of 210 scalar distances
        d_pairs = D[:, iu, ju]  # [N, n_pairs]
        rbf_pairs = self._rbf(d_pairs)  # [N, n_pairs, n_bins]

        z_pairs = np.stack([np.minimum(Z[iu], Z[ju]), np.maximum(Z[iu], Z[ju])], axis=-1)  # tells us which species pairs we have out of the 6 channels, [n_pairs,2]

        """
        allocate feature tensor, loop through species pairs, select only the relevant pairs and aggregate them
        """
        feats = np.zeros((N, len(SPECIES_PAIRS), self.n_bins))
        for k, (za, zb) in enumerate(SPECIES_PAIRS):
            mask = (z_pairs[:, 0] == za) & (z_pairs[:, 1] == zb)
            if mask.any():
                feats[:, k, :] = rbf_pairs[:, mask, :].sum(axis=1)
        return feats.reshape(N, -1)  # flatten to [N, 6*n_bins]

    @property
    def out_dim(self):
        return len(SPECIES_PAIRS) * self.n_bins
