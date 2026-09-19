"""End-to-end data pipeline.
The full aspirin dataset contains 100,000 frames; this experiment uses
a fixed 3,000-frame subsample and an equal 400-example training budget."""
import numpy as np
from data.load_aspirin import load_raw, validate
from data.preprocess import center_of_mass_remove, compute_train_scale, make_splits, EnergyNormalizer
from features.invariant import InvariantDescriptor

SEED = 42
N_SUBSAMPLE = 3000   # of 100,000 frames
N_BINS = 16
TRAIN_LIMIT = 400    # all 3 models trained on this many examples for a fair, equal budget comparison


def build_dataset(n_subsample: int = N_SUBSAMPLE, seed: int = SEED):
    raw = load_raw()
    validate(raw)
    Z = raw["nuclear_charges"]

    rng = np.random.default_rng(seed)
    sub_idx = rng.choice(len(raw["energies"]), size=n_subsample, replace=False)
    coords = raw["coords"][sub_idx]
    energies = raw["energies"][sub_idx]

    # mass weighted COM removal (translation-invariant, rotation-covariant)
    centered = center_of_mass_remove(coords, Z)

    # fixed seed 80/10/10 split
    train_idx_full, val_idx, test_idx = make_splits(n_subsample, seed=seed)
    # Equal, fixed budget training subset shared by all models (fair comparison,
    # does symmetry reduce sample burden?)
    train_idx = train_idx_full[:TRAIN_LIMIT]

    # scale computed on train split only
    scale = compute_train_scale(centered[train_idx])
    scaled = centered / scale

    # energy normalization fit on train split only
    norm = EnergyNormalizer().fit(energies[train_idx])
    energies_norm = norm.transform(energies)

    #  invariant descriptor (rotation+reflection+permutation invariant)
    descriptor = InvariantDescriptor(n_bins=N_BINS)
    inv_feats = descriptor.transform(scaled, Z)  # [N, 96]

    # Geometry agnostic classical input: flattened raw (centered, unscaled invariance) coords
    flat_coords = scaled.reshape(scaled.shape[0], -1)  # [N, 63]

    data = {
        "Z": Z,
        "coords_raw": coords,
        "coords_centered_scaled": scaled,
        "energies_raw": energies,
        "energies_norm": energies_norm,
        "energy_normalizer": norm,
        "scale": scale,
        "inv_feats": inv_feats,
        "flat_coords": flat_coords,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "descriptor": descriptor,
    }
    return data


if __name__ == "__main__":
    d = build_dataset()
    print("inv_feats", d["inv_feats"].shape, "flat_coords", d["flat_coords"].shape)
    print("train/val/test sizes:", len(d["train_idx"]), len(d["val_idx"]), len(d["test_idx"]))
    print("energy scale mu/sigma:", d["energy_normalizer"].mu, d["energy_normalizer"].sigma)
