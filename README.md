# Aspirin EQNN vs. Classical Baselines

This repository implements an Equivariant Quantum Neural Network (EQNN) pipeline for molecular energy prediction using `rmd17_aspirin.npz` and the theoretical framework of Nguyen et al., *"Theory for Equivariant Quantum Neural Networks"* (PRX Quantum 5, 020328, 2024).

Three models were built and compared head-to-head using the **same fixed set of 400 training examples** to investigate whether explicitly encoding the relevant molecular symmetries can reduce the parameter and data burden of molecular prediction.

| Model | Input Feature Representation | Symmetry Characteristics |
|---|---|---|
| **Geometry-agnostic classical FFN** | Flattened, centered \(21 \times 3\) Cartesian coordinates | No geometric symmetries built into the model |
| **Geometry-aware classical FFN** | Species-conditioned pairwise-distance RBF histogram | Exact translation, rotation, reflection, and species-preserving permutation invariance |
| **EQNN Track A (quantum)** | Invariant descriptor, PCA-reduced to 8 scalar angles | Exact geometric invariance plus an internal \(SU(2)\)-equivariant circuit |

---

## 1. Data Pipeline

* **Data Selection:** Loaded `nuclear_charges`, `coords`, `energies`, and `forces`. Legacy fields such as `old_indices`, `old_energies`, and `old_forces` were intentionally excluded from the model inputs. Forces are retained for dataset validation but are **not used as model inputs or training targets**.

* **Composition Validation:** Verified target shapes and chemical composition:
  \[
  \mathrm{C}_9\mathrm{H}_8\mathrm{O}_4.
  \]
  Each configuration contains 21 atoms: 9 carbon, 8 hydrogen, and 4 oxygen atoms.

* **Reference Frame & Center of Mass:** A mass-weighted center-of-mass translation was removed independently for every frame:
  \[
  \mathbf R_{\mathrm{CM}}
  =
  \frac{\sum_j m_j\mathbf r_j}
       {\sum_j m_j},
  \qquad
  \mathbf r'_j
  =
  \mathbf r_j-\mathbf R_{\mathrm{CM}}.
  \]
  This removes global translation while preserving the physical spatial orientation of the molecule. No principal-axis alignment was applied.

* **Normalization & Scaling:** A coordinate scale \(s\) and energy target parameters \((\mu,\sigma)\) were calculated **strictly from the training data** to prevent data leakage. Reported MAE and RMSE values are inverse-transformed back to the original physical energy units.

* **Data Partitioning:** A fixed random seed of 42 was used for reproducible frame-level partitioning. From the 3,000 sampled configurations:
  \[
  2400\ \text{candidate training},
  \quad
  300\ \text{validation},
  \quad
  300\ \text{test}.
  \]
  Only **400 of the 2,400 candidate training configurations** were used for model fitting. Therefore, all three models were trained on exactly the same 400 examples and evaluated on the same 300 validation and 300 test examples.

---

## 2. Invariant Descriptor (Track A)

For each frame, all unique unordered atom pairs were considered. With 21 atoms, the number of unique pairs is

\[
\binom{21}{2}
=
210.
\]

For every atom pair \((j,k)\), the Euclidean distance was computed:

\[
d_{jk}
=
\|\mathbf r_j-\mathbf r_k\|.
\]

Each pairwise distance was expanded using **16 Gaussian radial basis functions (RBFs)** and aggregated into six unordered species-pair channels:

\[
\mathrm{H-H},
\quad
\mathrm{C-C},
\quad
\mathrm{O-O},
\quad
\mathrm{H-C},
\quad
\mathrm{H-O},
\quad
\mathrm{C-O}.
\]

Therefore, the final invariant descriptor contains

\[
6\times16=96
\]

classical features per molecular configuration.

The RBF expansion is

\[
\phi_b(d)
=
\exp\left[-\beta(d-c_b)^2\right],
\]

where \(c_b\) is the center of the \(b\)-th radial basis function and \(\beta\) determines its width.

For each species pair \((a,b)\), the corresponding feature channel is obtained by summing the RBF responses over all atom pairs of that species combination.

### Geometric Invariance

The descriptor is invariant under orthogonal spatial transformations.

For

\[
\mathbf r_j' = Q\mathbf r_j,
\qquad
Q^TQ=I,
\]

the pairwise distance satisfies

\[
\|\mathbf r_j'-\mathbf r_k'\|
=
\|Q(\mathbf r_j-\mathbf r_k)\|
=
\|\mathbf r_j-\mathbf r_k\|.
\]

Therefore the representation is invariant to rotations and reflections. Together with center-of-mass removal, the representation is invariant to global translations as well.

The resulting spatial symmetry is therefore **\(O(3)\) invariance** rather than only \(SO(3)\) invariance.

### Permutation Invariance

The descriptor aggregates atom pairs according to species rather than preserving individual atom indices.

For a species-pair channel,

\[
F_{ab,b'}
=
\sum_{(j,k)\in P_{ab}}
\phi_{b'}(d_{jk}),
\]

where \(P_{ab}\) contains all pairs belonging to the species combination \((a,b)\).

Consequently, permutations that preserve atomic species leave the descriptor unchanged. The implementation therefore provides **species-preserving permutation invariance**.

### Numerical Verification

The invariant descriptor was numerically tested against transformed molecular configurations, including random 3D rotations, reflections, and species-preserving atom permutations.

The resulting descriptor differences were at approximately floating-point precision:

\[
\sim 10^{-14}.
\]

This verifies the intended invariance of the implemented representation up to normal numerical error.

---

## 3. Model Architectures

### Geometry-Agnostic Classical FFN

* **Input:** 63-dimensional flattened centered Cartesian coordinates:
  \[
  21\times3=63.
  \]

* **Architecture:**
  \[
  63
  \rightarrow
  64
  \rightarrow
  64
  \rightarrow
  32
  \rightarrow
  16
  \rightarrow
  1.
  \]

* **Behavior:** No geometric invariance is explicitly encoded. The model must learn relationships involving rotations, reflections, translations, and atom ordering from the available training data.

* **Trainable Parameters:** 10,881.

---

### Geometry-Aware Classical FFN

* **Input:** 96-dimensional invariant descriptor.

* **Architecture:**
  \[
  96
  \rightarrow
  64
  \rightarrow
  64
  \rightarrow
  32
  \rightarrow
  16
  \rightarrow
  1.
  \]

* **Behavior:** The molecular representation is invariant by construction, so the network receives the same feature vector for symmetry-equivalent molecular configurations.

* **Trainable Parameters:** 12,993.

---

### EQNN Track A (Quantum)

#### Input Reduction and Encoding

The 96-dimensional invariant descriptor is reduced using Principal Component Analysis (PCA):

\[
96\rightarrow8.
\]

PCA is fitted only on the training data.

The eight resulting scalar features are scaled to the interval used for angle encoding and converted to quantum rotation angles:

\[
\theta_k\in[0,\pi].
\]

Each scalar is encoded onto one qubit using

\[
R_y(\theta_k)
=
e^{-i\theta_kY/2}.
\]

Thus the circuit uses **8 input qubits**.

PCA is a classical dimensionality-reduction step. It does not perform an \(SO(3)\rightarrow SU(2)\) conversion.

#### Quantum Convolution Blocks

The circuit contains two brickwork interaction blocks.

For every active pair of wires, the implementation applies

\[
\mathrm{IsingXX}(\theta),
\qquad
\mathrm{IsingYY}(\theta),
\qquad
\mathrm{IsingZZ}(\theta).
\]

Because the Pauli-pair generators commute, their product is

\[
U(\theta)
=
e^{-i\frac{\theta}{2}(XX+YY+ZZ)}.
\]

Using the SWAP identity

\[
\mathrm{SWAP}
=
\frac12(I+XX+YY+ZZ),
\]

we obtain

\[
U(\theta)
=
e^{i\theta/2}
e^{-i\theta\,\mathrm{SWAP}}.
\]

The factor \(e^{i\theta/2}\) is a global phase, so physically

\[
U(\theta)\sim e^{-i\theta\,\mathrm{SWAP}}.
\]

Since

\[
[\mathrm{SWAP},\,g\otimes g]=0,
\qquad
g\in SU(2),
\]

the interaction has the desired internal \(SU(2)\)-equivariant structure.

The circuit therefore uses shared interaction parameters rather than independent parameters for every two-qubit gate.

#### Quantum Pooling

After the first interaction block, the active wires are reduced

\[
8\rightarrow4.
\]

After the second interaction block, they are reduced again:

\[
4\rightarrow2.
\]

The implementation retains every second active wire and removes the remaining wires from subsequent execution and measurement.

For downstream observables, discarding those subsystems is equivalent to tracing them out:

\[
\rho_{\mathrm{kept}}
=
\operatorname{Tr}_{\mathrm{discarded}}(\rho).
\]

This gives the effective quantum architecture

\[
8\rightarrow4\rightarrow2.
\]

#### Quantum Readout

The final two active qubits are measured using the three pairwise correlators

\[
\langle XX\rangle,
\qquad
\langle YY\rangle,
\qquad
\langle ZZ\rangle.
\]

The normalized energy prediction is then

\[
\hat E_{\mathrm{norm}}
=
b
+
w_x\langle XX\rangle
+
w_y\langle YY\rangle
+
w_z\langle ZZ\rangle.
\]

The classical readout therefore contains:

- 3 trainable weights,
- 1 trainable bias.

#### Parameter Count

The quantum circuit contains four trainable interaction parameters.

The readout contains four additional trainable parameters.

Therefore,

\[
4+3+1=8
\]

total trainable parameters.

The final EQNN consequently uses only **8 trainable parameters**, compared with 10,881 parameters in the geometry-agnostic FFN and 12,993 parameters in the geometry-aware FFN.

---

## 4. Compute Budget & Execution Parameters

The experiment was intentionally kept small to make repeated quantum simulation practical on CPU hardware.

* **Full Dataset:** 100,000 molecular configurations.
* **Sampled Dataset:** 3,000 configurations.
* **Candidate Training Set:** 2,400 configurations.
* **Validation Set:** 300 configurations.
* **Test Set:** 300 configurations.
* **Actual Model-Fitting Budget:** 400 training configurations.
* **Qubit Count:** 8 input qubits.
* **Quantum Pooling Schedule:** \(8\rightarrow4\rightarrow2\).
* **Optimization:** Adam.
* **Quantum Training:** 6 epochs, batch size 20, learning rate \(0.08\).
* **Classical Training:** Adam with learning rate \(10^{-3}\).

All models use the same final 400-example training budget and the same validation/test configurations.

---

## 5. Performance & Empirical Results

| Model | Qubits | Params | MAE | RMSE | \(R^2\) | Rotation Err (Mean/Max) | Permutation Err (Mean/Max) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Mean Baseline** | — | 1 | 5.104 | 6.327 | -0.020 | — | — |
| **Geometry-Agnostic Classical** | — | 10,881 | 5.124 | 6.343 | -0.025 | 2.27 / 4.52 | 1.29 / 4.40 |
| **Geometry-Aware Classical** | — | 12,993 | 4.995 | 6.201 | 0.020 | **0.00 / 0.00** | **0.00 / 0.00** |
| **EQNN Track A** | \(8\rightarrow2\) | **8** | 5.023 | 6.206 | 0.018 | **0.00 / 0.00** | **0.00 / 0.00** |

### Key Findings

1. **Symmetry preservation:** Rotations and species-preserving permutations produce substantial prediction changes in the geometry-agnostic model, whereas the invariant classical model and EQNN remain unchanged up to floating-point precision.

2. **Prediction performance:** The geometry-aware classical FFN obtains MAE \(=4.995\), RMSE \(=6.201\), and \(R^2=0.020\). The EQNN obtains MAE \(=5.023\), RMSE \(=6.206\), and \(R^2=0.018\). Their predictive performance is therefore numerically close under the same 400-example training budget.

3. **Parameter efficiency:** The EQNN achieves similar predictive metrics to the invariant classical model with only **8 trainable parameters**, compared with 12,993 trainable parameters for the invariant classical FFN.

4. **Geometry-agnostic behavior:** The geometry-agnostic FFN obtains MAE \(=5.124\), RMSE \(=6.343\), and \(R^2=-0.025\), which is slightly worse than the mean baseline under this limited training budget.

5. **Interpretation of the result:** These results demonstrate that symmetry can be enforced directly in the representation and quantum architecture rather than being learned entirely from finite data. However, the experiment does not establish that the EQNN universally requires fewer samples or parameters; it demonstrates parameter-efficient performance for this particular dataset, descriptor, architecture, and 400-example training regime.

---

## 6. Symmetry Structure of the Implementation

The model contains two distinct symmetry mechanisms.

### Molecular Geometric Symmetry

The classical molecular descriptor provides

\[
O(3)
\]

invariance through pairwise distances.

This covers:

- translation invariance through center-of-mass removal,
- rotation invariance,
- reflection invariance.

Species-preserving atom permutations are handled through aggregation within species-pair channels.

### Internal Quantum Symmetry

The quantum circuit uses SWAP-generated interactions that commute with simultaneous \(SU(2)\) transformations:

\[
[\mathrm{SWAP},g\otimes g]=0.
\]

This gives the internal two-qubit operations an \(SU(2)\)-equivariant structure.

### Relationship Between the Two

The implementation does **not** map the molecular \(SO(3)\) group directly into \(SU(2)\).

Instead, it combines

\[
\boxed{\text{molecular }O(3)\text{-invariant features}}
\]

with

\[
\boxed{\text{internally }SU(2)\text{-equivariant quantum layers}}.
\]

The molecular symmetry is handled before quantum encoding, while the \(SU(2)\) structure is a property of the quantum circuit itself.

---

## 7. Benefits of the EQNN Design

### Strong Symmetry Enforcement

The invariant input descriptor ensures that symmetry-equivalent molecular configurations produce identical classical features before entering the quantum circuit.

### Very Small Trainable Parameter Count

Only 8 trainable parameters are required:

\[
4\ \text{quantum}
+
3\ \text{readout weights}
+
1\ \text{bias}.
\]

This is substantially smaller than the classical FFN parameter counts.

### Structured Quantum Architecture

The brickwork connectivity and shared interaction parameters give the quantum model a structured architecture rather than a collection of unrelated two-qubit gates.

### Computational Tractability

Reducing the descriptor from 96 dimensions to 8 PCA channels makes an 8-qubit statevector simulation practical for this proof-of-concept experiment.

---

## 8. Limitations

### Invariant Descriptor Discards Orientation Information

The pairwise-distance representation is invariant to reflections as well as rotations. Consequently, it cannot distinguish mirror-image configurations or encode chirality.

### Pairwise Representation

The descriptor contains only pairwise geometric information aggregated through RBF channels. Higher-order angular or many-body geometric information is not represented explicitly.

### PCA Compression

Reducing

\[
96\rightarrow8
\]

may discard information contained in the original invariant descriptor.

### Small Training Budget

Only 400 molecular configurations are used for fitting. The experiment therefore demonstrates behavior in a low-data regime rather than full-dataset molecular-energy prediction.

### Small Quantum Circuit

The circuit uses only 8 input qubits and four trainable quantum parameters. This is deliberately compact and should be regarded as a proof-of-concept architecture rather than a complete large-scale molecular EQNN.

### Quantum Simulation

The reported results are obtained using classical simulation of the quantum circuit rather than execution on physical quantum hardware.

### Energy-Only Learning

Although forces are present in the dataset, the implementation predicts only molecular energy. The model is therefore not trained as a joint energy-and-force potential.

### No Direct \(SO(3)\rightarrow SU(2)\) Representation Map

The implementation does not construct molecular irreducible representations or explicitly transform physical \(SO(3)\) features into \(SU(2)\) quantum representations. The molecular geometry is handled by an invariant classical descriptor, while the quantum circuit separately enforces its internal \(SU(2)\)-equivariant structure.

---

## 9. Repository Layout

```text
aspirin_eqnn/
├── data/
│   ├── load_aspirin.py
│   ├── preprocess.py
│   └── rmd17_aspirin.npz
│
├── symmetry/
│   ├── rotations.py
│   └── permutations.py
│
├── features/
│   ├── invariant.py
│   └── reduce_for_quantum.py
│
├── models/
│   ├── geo_agnostic_ffn.py
│   └── geo_aware_ffn.py
│
├── quantum/
│   └── eqnn_track_a.py
│
├── pipeline.py
├── train_all.py
├── requirements.txt
│
├── results/
│   └── results.json
│
└── plots/
    ├── 01_training_curves.png
    ├── 02_pred_vs_true.png
    ├── 03_abs_error_hist.png
    ├── 04_mae_comparison.png
    └── 05_symmetry_errors.png

## 10. Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt

## 11. Running the Experiment

Run the complete training and evaluation pipeline with:

```bash
python train_all.py

The script trains the classical and quantum models, evaluates the test-set metrics, performs symmetry checks, and generates the result files and plots.

## 12. Overall Pipeline

The complete computational pipeline is:

```bash
Asprin coordinates $\rightarrow$   COM centering $\rightarrow$   Pairwise distances $\rightarrow$   Species-aware RBF descriptor $\rightarrow$   96 invariant features $\rightarrow$   8 PCA channels $\rightarrow$   Ry encoding $\rightarrow$   8-qubit EQNN -> 8$\rightarrow$  4$\rightarrow$  2 pooling $\rightarrow$   \langle  XX \rangle  , \langle  YY \rangle  , \langle  ZZ \rangle   $\rightarrow$  Energy prediction


The classical baselines branch from the same preprocessed molecular data, allowing the three approaches to be compared under the same data budget and evaluation protocol.