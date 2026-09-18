"""Load and validate the rmd17_aspirin.npz dataset """
import numpy as np

NPZ_PATH = "data/rmd17_aspirin.npz"

# Expected composition: C9 H8 O4  (Z counts: 6->9, 1->8, 8->4)
EXPECTED_Z_COUNTS = {6: 9, 1: 8, 8: 4}


def load_raw(path: str = NPZ_PATH) -> dict:
    d = np.load(path)
    return {
        "nuclear_charges": d["nuclear_charges"],
        "coords": d["coords"],
        "energies": d["energies"],
        "forces": d["forces"],
        # returning only model-relevant information
    }


def validate(raw: dict) -> None:
    Z = raw["nuclear_charges"]
    coords = raw["coords"]
    energies = raw["energies"]
    forces = raw["forces"]

    N = coords.shape[0]
    assert coords.shape == (N, 21, 3), f"bad coords shape {coords.shape}"
    assert energies.shape == (N,), f"bad energies shape {energies.shape}"
    assert forces.shape == (N, 21, 3), f"bad forces shape {forces.shape}"
    assert Z.shape == (21,), f"bad nuclear_charges shape {Z.shape}"

    counts = {int(z): int((Z == z).sum()) for z in np.unique(Z)}
    assert counts == EXPECTED_Z_COUNTS, f"unexpected composition {counts}"

    print(f"[load_aspirin] OK: N={N} frames, 21 atoms/frame, composition C9H8O4")
    print(f"[load_aspirin] energy range: [{energies.min():.2f}, {energies.max():.2f}]")
    print(f"[load_aspirin] coord range:  [{coords.min():.3f}, {coords.max():.3f}]")
    print(f"[load_aspirin] force RMS:    {np.sqrt((forces ** 2).mean()):.3f}")


if __name__ == "__main__":
    raw = load_raw()
    validate(raw)
