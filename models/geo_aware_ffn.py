"""Geometry-aware classical baseline.

Input: the InvariantDescriptor features (species-conditioned RBF distance
histograms); invariant to translation, rotation, reflection and
species-preserving atom permutation. Same 4-hidden-layer depth as the
geometry-agnostic model, so the only difference between the two classical
baselines is whether geometry is encoded invariantly or fed in raw.
"""
from sklearn.neural_network import MLPRegressor

HIDDEN = (64, 64, 32, 16)  # 4 hidden layers, same as geo-agnostic model


def build_model(seed: int = 42) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=HIDDEN,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=seed,
    )
