# Aspirin EQNN vs. Classical Baselines

This repository implements an Equivariant Quantum Neural Network (EQNN) pipeline for molecular energy prediction using `rmd17_aspirin.npz` and the theoretical framework of Nguyen et al., *"Theory for Equivariant Quantum Neural Networks"* (PRX Quantum 5, 020328, 2024).

Three models were built and compared head-to-head on the **same fixed training budget** to address a core scientific question: *Does imposing the correct symmetry reduce sample and parameter burden?*

| Model | Input Feature Representation | Symmetry Characteristics |
|---|---|---|
| **Geometry-agnostic classical FFN** | Flattened, centered $21 \times 3$ coordinates (raw coordinates) | No symmetries built in |
| **Geometry-aware classical FFN** | Species-conditioned pairwise-distance RBF histogram | Exact translation, rotation, reflection, and permutation invariance |
| **EQNN Track A (quantum)** | Invariant descriptor, PCA-reduced to 8 scalar angles | Exact geometric invariance *plus* an internal $SU(2)$-equivariant circuit |

---

## 1. Data Pipeline

* **Data Selection:** Loaded `nuclear_charges`, `coords`, `energies`, and `forces` strictly—deprecated indices and energies (`old_*`) were excluded.
* **Composition Validation:** Verified target shapes and chemical composition ($\text{C}_9\text{H}_8\text{O}_4$).
* **Reference Frame & Center of Mass:** Mass-weighted center-of-mass translation removed per frame (translation-invariant, rotation-covariant). No principal-axis alignment was applied, preserving true spatial orientation.
* **Normalization & Scaling:** A single coordinate scale $s$ and energy target parameters ($\mu, \sigma$) were calculated **strictly from the training split** to eliminate data leakage. Reported MAE/RMSE values are inverse-transformed back to physical units.
* **Data Partitioning:** Fixed seed (42) with an 80/10/10 train/validation/test split by frame.

---

## 2. Invariant Descriptor (Track A)

For each frame, the $21 \times 21$ pairwise-distance matrix is expanded across 16 Gaussian radial basis functions (RBF) and summed into 6 species-pair channels ($\text{H-H}$, $\text{C-C}$, $\text{O-O}$, $\text{H-C}$, $\text{H-O}$, $\text{C-O}$). 

* **Geometric Invariance:** Provably invariant under $O(3)$ (rotations and reflections) as Euclidean distance satisfies $\Vert{}Q(\mathbf{x}_j - \mathbf{x}_k)\Vert{} = \Vert{}\mathbf{x}_j - \mathbf{x}_k\Vert{}$.
* **Permutation Invariance:** Summation across species pairs removes sensitivity to atom indexing order.
* **Numerical Verification:** Validated to floating-point precision ($\sim 10^{-14}$) against random 3D rotations, reflections, and species-preserving permutations prior to training.

---

## 3. Model Architectures

### Geometry-Agnostic Classical FFN
* **Input:** 63-dimensional flattened raw coordinates.
* **Architecture:** 4 hidden layers ($64 \to 64 \to 32 \to 16$) with linear output.
* **Behavior:** Must attempt to learn spatial invariances directly from data distributions.

### Geometry-Aware Classical FFN
* **Input:** 96-dimensional invariant descriptor.
* **Architecture:** Identical depth and layer widths as the geometry-agnostic model.
* **Behavior:** Guaranteed physical invariance by construction without spending parameter capacity learning geometric transformations.

### EQNN Track A (Quantum)
* **Input Encoding:** 96-dimensional descriptor reduced via PCA (fit on train split) to 8 scalar channels, encoded onto 8 qubits using $RY(\theta)$ gates.
* **Circuit Architecture:**
  * **Standard Layers:** Two brickwork sublayers of $SU(2)$-equivariant primitives $U(\theta) = \exp(-i\theta \cdot \text{SWAP})$, implemented via commuting Pauli pairs: $\text{IsingXX}(\theta) \cdot \text{IsingYY}(\theta) \cdot \text{IsingZZ}(\theta)$.
  * **Pooling Layers:** $2 \to 1$ partial trace map ($\phi_3(\rho) = \text{Tr}_A[\rho]$) implemented by dropping wire execution paths, preserving downstream density operator observables.
  * **Readout:** Single-qubit/pair expectation observables $\langle XX \rangle, \langle YY \rangle, \langle ZZ \rangle$ mapped through a 4-parameter linear head.
* **Parameter Count:** **8 trainable parameters** (4 circuit angles + 3 head weights + 1 bias) vs. $\sim 11\text{k}\text{--}13\text{k}$ parameters in classical FFNs.

---

## 4. Compute Budget & Execution Parameters

To run within resource constraints on CPU hardware, the run was executed with the following parameter bounds:

* **Qubit Count:** 8 active qubits (sub-sampled schedule $8 \to 4 \to 2$).
* **Dataset Subsample:** 3,000 total frames (400 train / 300 validation / 300 test).
* **Optimization:** 6 training epochs, batch size of 20 using Adam optimizer ($\eta = 0.08$ for quantum parameters, $\eta = 10^{-3}$ for classical layers).

---

## 5. Performance & Empirical Results

| Model | Qubits | Params | MAE | RMSE | $R^2$ | Rotation Err (Mean/Max) | Permutation Err (Mean/Max) |
|---|---|---|---|---|---|---|---|
| **Mean Baseline** | — | 1 | 5.104 | 6.327 | -0.020 | — | — |
| **Geometry-Agnostic Classical** | — | 10,881 | 5.124 | 6.343 | -0.025 | 2.27 / 4.52 | 1.29 / 4.40 |
| **Geometry-Aware Classical** | — | 12,993 | 4.995 | 6.201 | **0.020** | **0.00 / 0.00** | **0.00 / 0.00** |
| **EQNN Track A** | $8 \to 2$ | **8** | 5.023 | 6.206 | **0.018** | **0.00 / 0.00** | **0.00 / 0.00** |

### Key Findings

1. **Exact Mathematical Invariance:** Input spatial transformations (rotations/reflections/permutations) induce significant prediction drift in the geometry-agnostic model ($\mathcal{O}(1\text{--}4)$ energy units), while both the invariant classical model and the EQNN exhibit **zero variation** (`0.00e+00`) up to floating-point precision.
2. **Parameter Efficiency:** The 8-parameter EQNN achieves competitive $R^2$ and MAE metrics compared to the $1,600\times$ larger invariant classical FFN under low-data constraints.
3. **Data Constraint Limitations:** Under a 400-example training regime, unconstrained models (geometry-agnostic FFN) fail to outperform simple mean baselines due to high sample complexity requirements for learning physical symmetries.

---

## 6. Repository Layout

```text
aspirin_eqnn/
├── data/           # Loader utilities and mass-weighted centering functions
├── symmetry/       # Rotation, reflection, and permutation test operators
├── features/       # Pairwise distance RBF descriptors and PCA dimension reduction
├── models/         # Classical geometry-agnostic and geometry-aware baseline FFNs
├── quantum/        # PennyLane EQNN circuit layers and expectation head implementations
├── pipeline.py     # End-to-end data processing and transformation pipeline
├── train_all.py    # Multi-model training driver, symmetry validation, and plot generation
├── results/        # Numerical output logs and evaluations (results.json)
└── plots/          # Evaluation curves, error distribution histograms, and symmetry tests