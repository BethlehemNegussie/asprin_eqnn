"""Geometry-agnostic classical baseline.

Input: flattened, centered (but not invariant) 21x3 Cartesian coordinates.
This model has no built in rotation, reflection, or permutation invariance;
it just sees a 63-dim vector of raw numbers. Architecture: input and 4 hidden
layers and linear regression output.
"""
from sklearn.neural_network import MLPRegressor

HIDDEN = (64, 64, 32, 16)  # 4 hidden layers


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
