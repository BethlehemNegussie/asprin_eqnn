import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import build_dataset
from models.geo_agnostic_ffn import build_model as build_agnostic
from models.geo_aware_ffn import build_model as build_aware
from features.reduce_for_quantum import AngleEncoder
from quantum.eqnn_track_a import EQNNTrackA, N_QUBITS
from symmetry.rotations import symmetry_error
from symmetry.permutations import permutation_error

RESULTS_DIR = "results"
SEED = 42


def metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    print("=" * 70)
    print("Building dataset...")
    d = build_dataset()
    Z = d["Z"]
    norm = d["energy_normalizer"]
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    print(f"Fixed training budget shared by all 3 models: {len(tr)} frames "
          f"(val={len(va)}, test={len(te)})")

    y_tr_n, y_va_n, y_te_n = d["energies_norm"][tr], d["energies_norm"][va], d["energies_norm"][te]
    y_tr_raw, y_va_raw, y_te_raw = d["energies_raw"][tr], d["energies_raw"][va], d["energies_raw"][te]

    results = {}
    histories = {}
    predictions = {}

    # ---------------- Mean baseline  ----------------
    mean_pred_test = np.full_like(y_te_raw, y_tr_raw.mean())
    results["Mean baseline"] = {**metrics(y_te_raw, mean_pred_test),
                                 "qubits": "-", "params": 1, "depth": "-"}

    # ---------------- Geometry-agnostic classical FFN ----------------
    print("\n[1/3] Training geometry-agnostic classical FFN (raw flattened coords)...")
    t0 = time.time()
    X_tr, X_va, X_te = d["flat_coords"][tr], d["flat_coords"][va], d["flat_coords"][te]
    m_agnostic = build_agnostic(seed=SEED)
    m_agnostic.fit(X_tr, y_tr_n)
    pred_te_n = m_agnostic.predict(X_te)
    pred_te_raw = norm.inverse(pred_te_n)
    n_params_agnostic = sum(w.size for w in m_agnostic.coefs_) + sum(b.size for b in m_agnostic.intercepts_)
    results["Geometry-agnostic classical (raw coords)"] = {
        **metrics(y_te_raw, pred_te_raw),
        "qubits": "-", "params": int(n_params_agnostic), "depth": len(m_agnostic.coefs_),
    }
    histories["agnostic"] = m_agnostic.loss_curve_
    predictions["agnostic"] = (y_te_raw, pred_te_raw)
    print(f"  done in {time.time()-t0:.1f}s, test MAE={results['Geometry-agnostic classical (raw coords)']['MAE']:.4f}")

    # ---------------- Geometry-aware classical FFN  ----------------
    print("\n[2/3] Training geometry-aware classical FFN (invariant distance/RBF features)...")
    t0 = time.time()
    Xi_tr, Xi_va, Xi_te = d["inv_feats"][tr], d["inv_feats"][va], d["inv_feats"][te]
    m_aware = build_aware(seed=SEED)
    m_aware.fit(Xi_tr, y_tr_n)
    pred_te_n = m_aware.predict(Xi_te)
    pred_te_raw = norm.inverse(pred_te_n)
    n_params_aware = sum(w.size for w in m_aware.coefs_) + sum(b.size for b in m_aware.intercepts_)
    results["Geometry-aware classical (invariant features)"] = {
        **metrics(y_te_raw, pred_te_raw),
        "qubits": "-", "params": int(n_params_aware), "depth": len(m_aware.coefs_),
    }
    histories["aware"] = m_aware.loss_curve_
    predictions["aware"] = (y_te_raw, pred_te_raw)
    print(f"  done in {time.time()-t0:.1f}s, test MAE={results['Geometry-aware classical (invariant features)']['MAE']:.4f}")

    # ---------------- EQNN ----------------
    print(f"\n[3/3] Training EQNN Track A ({N_QUBITS} qubits, invariant encoding + SU(2)-equivariant layers)...")
    t0 = time.time()
    encoder = AngleEncoder(n_qubits=N_QUBITS, seed=SEED).fit(Xi_tr)
    A_tr, A_va, A_te = encoder.transform(Xi_tr), encoder.transform(Xi_va), encoder.transform(Xi_te)

    eqnn = EQNNTrackA(seed=SEED)
    hist = eqnn.train(A_tr, y_tr_n, A_va, y_va_n, epochs=6, batch_size=20, lr=0.08, seed=SEED)
    pred_te_n = eqnn.predict(A_te)
    pred_te_raw = norm.inverse(pred_te_n)
    results["EQNN Track A (quantum)"] = {
        **metrics(y_te_raw, pred_te_raw),
        "qubits": f"{N_QUBITS}->2", "params": eqnn.n_params(), "depth": "2 conv blocks + 2 pools",
    }
    histories["eqnn"] = hist
    predictions["eqnn"] = (y_te_raw, pred_te_raw)
    print(f"  done in {time.time()-t0:.1f}s, test MAE={results['EQNN Track A (quantum)']['MAE']:.4f}")

    # ---------------- Symmetry diagnostics ----------------
    print("\nRunning rotation/reflection and permutation invariance diagnostics...")
    n_sym = 30
    sym_coords = d["coords_centered_scaled"][te[:n_sym]]

    def agnostic_predict(coords):
        flat = coords.reshape(coords.shape[0], -1)
        return norm.inverse(m_agnostic.predict(flat))

    def aware_predict(coords):
        feats = d["descriptor"].transform(coords, Z)
        return norm.inverse(m_aware.predict(feats))

    def eqnn_predict(coords):
        feats = d["descriptor"].transform(coords, Z)
        angles = encoder.transform(feats)
        return norm.inverse(eqnn.predict(angles))

    sym_results = {}
    for name, fn in [("Geometry-agnostic classical (raw coords)", agnostic_predict),
                      ("Geometry-aware classical (invariant features)", aware_predict),
                      ("EQNN Track A (quantum)", eqnn_predict)]:
        rot_err = symmetry_error(fn, sym_coords, n_trials=4, seed=1)
        perm_err = permutation_error(fn, sym_coords, Z, n_trials=4, seed=1)
        sym_results[name] = {"rotation_error_mean": rot_err["mean"], "rotation_error_max": rot_err["max"],
                              "permutation_error_mean": perm_err["mean"], "permutation_error_max": perm_err["max"]}
        results[name]["rotation_error"] = f"{rot_err['mean']:.2e} (max {rot_err['max']:.2e})"
        results[name]["permutation_error"] = f"{perm_err['mean']:.2e} (max {perm_err['max']:.2e})"
        print(f"  {name}: rot_err(mean/max)={rot_err['mean']:.3e}/{rot_err['max']:.3e}  "
              f"perm_err(mean/max)={perm_err['mean']:.3e}/{perm_err['max']:.3e}")

    for k in ["Mean baseline"]:
        results[k]["rotation_error"] = "-"
        results[k]["permutation_error"] = "-"

    # ---------------- Save results ----------------
    with open(f"{RESULTS_DIR}/results.json", "w") as f:
        json.dump({"metrics": results, "symmetry": sym_results,
                    "n_train": len(tr), "n_val": len(va), "n_test": len(te)}, f, indent=2)

    make_plots(results, histories, predictions, y_te_raw)
    print_table(results)
    print(f"\nSaved: {RESULTS_DIR}/results.json and plots/*.png")


def print_table(results):
    print("\n" + "=" * 100)
    header = f"{'Model':45s} {'Qubits':8s} {'Params':7s} {'MAE':>9s} {'RMSE':>9s} {'R2':>8s}"
    print(header)
    print("-" * 100)
    for name, r in results.items():
        print(f"{name:45s} {str(r['qubits']):8s} {str(r['params']):7s} "
              f"{r['MAE']:9.4f} {r['RMSE']:9.4f} {r['R2']:8.4f}")
    print("=" * 100)


def make_plots(results, histories, predictions, y_te_raw):
    # 1. Training/validation loss curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(histories["agnostic"])
    axes[0].set_title("Geometry-agnostic classical\ntraining loss")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("loss")

    axes[1].plot(histories["aware"])
    axes[1].set_title("Geometry-aware classical\ntraining loss")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("loss")

    axes[2].plot(histories["eqnn"]["train_loss"], label="train")
    axes[2].plot(histories["eqnn"]["val_loss"], label="val")
    axes[2].set_title("EQNN Track A\ntraining/validation loss")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("MSE (normalized energy)")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig("plots/01_training_curves.png", dpi=130)
    plt.close()

    # 2. Predicted vs true scatter 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = ["agnostic", "aware", "eqnn"]
    titles = ["Geometry-agnostic classical", "Geometry-aware classical", "EQNN Track A (quantum)"]
    for ax, name, title in zip(axes, names, titles):
        y_true, y_pred = predictions[name]
        ax.scatter(y_true, y_pred, s=8, alpha=0.5)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("True energy"); ax.set_ylabel("Predicted energy")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig("plots/02_pred_vs_true.png", dpi=130)
    plt.close()

    # 3. Absolute error histograms
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, title in zip(names, titles):
        y_true, y_pred = predictions[name]
        ax.hist(np.abs(y_pred - y_true), bins=25, alpha=0.5, label=title)
    ax.set_xlabel("Absolute error (energy units)"); ax.set_ylabel("count")
    ax.set_title("Absolute error distribution (test set)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("plots/03_abs_error_hist.png", dpi=130)
    plt.close()

    # 4. MAE comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(results.keys())
    maes = [results[k]["MAE"] for k in labels]
    ax.barh(labels, maes)
    ax.set_xlabel("Test MAE (energy units)")
    ax.set_title("MAE comparison across models")
    plt.tight_layout()
    plt.savefig("plots/04_mae_comparison.png", dpi=130)
    plt.close()

    # 5. Rotation/permutation invariance error bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    model_names = ["Geometry-agnostic classical (raw coords)",
                   "Geometry-aware classical (invariant features)",
                   "EQNN Track A (quantum)"]
    rot_means = [results[m].get("rotation_error", "-") for m in model_names]
    perm_means = [results[m].get("permutation_error", "-") for m in model_names]

    def parse_mean(s):
        try:
            return float(s.split(" ")[0])
        except Exception:
            return 0.0

    axes[0].bar(model_names, [parse_mean(s) for s in rot_means])
    axes[0].set_yscale("log")
    axes[0].set_title("Rotation/reflection invariance error\n(mean |h(X)-h(gX)|)")
    axes[0].tick_params(axis="x", rotation=20, labelsize=7)

    axes[1].bar(model_names, [parse_mean(s) for s in perm_means])
    axes[1].set_yscale("log")
    axes[1].set_title("Species-permutation invariance error\n(mean |h(X)-h(PX)|)")
    axes[1].tick_params(axis="x", rotation=20, labelsize=7)
    plt.tight_layout()
    plt.savefig("plots/05_symmetry_errors.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
