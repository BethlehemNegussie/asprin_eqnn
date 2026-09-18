"""EQNN 

Design:
  - K=8 qubits, each encoded with one globally rotation/reflection/permutation
    -invariant scalar feature (a PCA-reduced descriptor), via RY.
    Because the *inputs* to the circuit are already O(3)+permutation invariant
    scalars, the whole model's invariance to molecular rotation/reflection/atom
    relabeling is exact and architecture-independent.
  - Local 2-to-2 standard layers use U(theta) = exp(-i*theta*SWAP), an
    SU(2)-equivariant primitive. Up to a global phase (which
    never affects expectation values) this equals
    IsingXX(theta) * IsingYY(theta) * IsingZZ(theta) on the same wire pair,
    because SWAP = (I + XX + YY + ZZ)/2 and XX, YY, ZZ mutually commute.
  - Pooling is implemented as phi3 = Tr_A[rho] equivariant map
    i.e. a qubit is simply never touched again after a
    pooling stage, which is mathematically identical to explicitly tracing it
    out for every expectation value computed downstream.
  - Readout: <XX>, <YY>, <ZZ> on the final 2 active qubits (their combination
    (1+<XX>+<YY>+<ZZ>)/2 recovers the <SWAP> invariant observable;
    exposing the three separately just gives the linear head more freedom).
"""
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

N_QUBITS = 8


def _conv_block(active_wires, theta_a, theta_b):
    """Two brickwork sublayers of shared parameter exp(-i*theta*SWAP) gates."""
    pairs_a = list(zip(active_wires[0::2], active_wires[1::2]))
    shifted = active_wires[1:] + active_wires[:1]
    pairs_b = list(zip(shifted[0::2], shifted[1::2]))

    for (w1, w2) in pairs_a:
        qml.IsingXX(theta_a, wires=[w1, w2])
        qml.IsingYY(theta_a, wires=[w1, w2])
        qml.IsingZZ(theta_a, wires=[w1, w2])
    for (w1, w2) in pairs_b:
        if w1 == w2:
            continue
        qml.IsingXX(theta_b, wires=[w1, w2])
        qml.IsingYY(theta_b, wires=[w1, w2])
        qml.IsingZZ(theta_b, wires=[w1, w2])


def _pool(active_wires):
    """2-to-1 equivariant pooling := partial trace
    keep the first wire of each pair, never act on the dropped wire again."""
    kept = active_wires[0::2]
    return kept


dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="autograd")
def _circuit(angles, theta):
    """angles: [8] invariant encoding angles. theta: [4] circuit params
    (theta_A1, theta_B1, theta_A2, theta_B2)."""
    wires = list(range(N_QUBITS))
    for w in wires:
        qml.RY(angles[w], wires=w)

    active = wires  # 8
    _conv_block(active, theta[0], theta[1])
    active = _pool(active)  # 8 -> 4

    _conv_block(active, theta[2], theta[3])
    active = _pool(active)  # 4 -> 2

    w1, w2 = active
    return (
        qml.expval(qml.PauliX(w1) @ qml.PauliX(w2)),
        qml.expval(qml.PauliY(w1) @ qml.PauliY(w2)),
        qml.expval(qml.PauliZ(w1) @ qml.PauliZ(w2)),
    )


def circuit_batch(angles_batch, theta):
    """angles_batch: [B,8] -> features [B,3] (loop - default.qubit broadcasting
    over non-trainable inputs is used where possible, but we loop explicitly
    for robustness/clarity given the small qubit count)."""
    out = np.zeros((angles_batch.shape[0], 3))
    for i in range(angles_batch.shape[0]):
        xx, yy, zz = _circuit(angles_batch[i], theta)
        out[i] = [xx, yy, zz]
    return out


class EQNNTrackA:
    """Wraps the circuit and a small linear classical head, trained jointly."""

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.theta = pnp.array(rng.uniform(-0.1, 0.1, size=4), requires_grad=True)
        self.w = pnp.array(rng.uniform(-0.1, 0.1, size=3), requires_grad=True)
        self.b = pnp.array(0.0, requires_grad=True)

    def n_params(self):
        return len(self.theta) + len(self.w) + 1

    def _forward_one(self, angles, theta, w, b):
        xx, yy, zz = _circuit(angles, theta)
        return b + w[0] * xx + w[1] * yy + w[2] * zz

    def predict(self, angles_batch):
        preds = np.zeros(angles_batch.shape[0])
        for i in range(angles_batch.shape[0]):
            preds[i] = float(self._forward_one(angles_batch[i], self.theta, self.w, self.b))
        return preds

    def train(self, angles_train, y_train, angles_val, y_val,
              epochs=25, batch_size=32, lr=0.05, seed=42, verbose=True):
        opt = qml.AdamOptimizer(stepsize=lr)
        rng = np.random.default_rng(seed)
        n = angles_train.shape[0]
        history = {"train_loss": [], "val_loss": []}

        def huber(pred, target, delta=1.0):
            r = pred - target
            abs_r = pnp.abs(r)
            quad = 0.5 * r ** 2
            lin = delta * (abs_r - 0.5 * delta)
            return pnp.where(abs_r <= delta, quad, lin)

        def batch_loss(theta, w, b, ang_b, y_b):
            total = 0.0
            for i in range(ang_b.shape[0]):
                pred = self._forward_one(ang_b[i], theta, w, b)
                total = total + huber(pred, y_b[i])
            return total / ang_b.shape[0]

        # Monitoring uses a fixed random subset (not the full set) each epoch,
        # purely to keep this demonstration run's wall-clock time bounded 
        # final test-set metrics always use the full held-out test set.
        
        mon_n = min(80, n, len(y_val))
        mon_train_idx = rng.choice(n, size=mon_n, replace=False)
        mon_val_idx = rng.choice(len(y_val), size=mon_n, replace=False)

        for ep in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                ang_b, y_b = angles_train[idx], y_train[idx]

                (self.theta, self.w, self.b), _ = opt.step_and_cost(
                    lambda th, w, b: batch_loss(th, w, b, ang_b, y_b),
                    self.theta, self.w, self.b,
                )

            train_pred = self.predict(angles_train[mon_train_idx])
            val_pred = self.predict(angles_val[mon_val_idx])
            tr_loss = float(np.mean((train_pred - y_train[mon_train_idx]) ** 2))
            va_loss = float(np.mean((val_pred - y_val[mon_val_idx]) ** 2))
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            if verbose:
                print(f"  [EQNN] epoch {ep+1:02d}/{epochs}  train_mse={tr_loss:.4f}  val_mse={va_loss:.4f}", flush=True)

        return history
