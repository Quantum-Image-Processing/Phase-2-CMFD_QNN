"""Compare SVM-RBF, MLP, and VQC classification heads on MICC-F2000 features.

Evaluation protocol: 10-fold cross-validation (per Al-khuzaie's protocol on
MICC-F600 — applied here to MICC-F2000).

VQC design, sourced explicitly (no invented architecture choices):
  - Linear(32 -> 8) trainable reduction before the quantum circuit.
    This mirrors Mari et al.'s dressed quantum circuit, which uses a
    classical linear layer for pre-processing immediately before the
    quantum embedding — NOT PCA (PCA(8) was already found to discard
    ~98.4% of variance; a trained linear layer optimizes for the
    classification signal instead of raw variance).
  - Embedding: RY rotation per qubit, parametrized by the classical
    8-dim input (Mari et al. Eq. 11-style angle embedding).
  - Variational layers: RY rotations + CNOT entangling gates
    (Mari et al. Eq. 12-13; independently confirmed by QTLCAit's
    DressedQuantumNet: RY rotations, entanglement, Pauli-Z measurement).
  - Readout: Pauli-Z expectation value per qubit -> classical linear
    layer -> class logits (dressed circuit pattern, both papers agree).

MLP: parameter-matched to the VQC's trainable parameter count, per your
FYP's existing "parameter-matched" comparison design.
"""
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = 'features/micc_features.npz'
N_FOLDS   = 10
SEED      = 42
N_QUBITS  = 8
Q_DEPTH   = 4
VQC_EPOCHS = 60
VQC_LR    = 0.01


# ------------------------------------------------------------------ VQC (PennyLane)
dev = qml.device('default.qubit', wires=N_QUBITS)


def _entangling_layer():
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[N_QUBITS - 1, 0])


@qml.qnode(dev, interface='torch')
def _circuit(inputs, weights):
    # Embedding: RY rotation per qubit, angle from classical input (Mari et al.)
    for i in range(N_QUBITS):
        qml.RY(inputs[i] * np.pi, wires=i)

    # Variational layers: RY + entangle (Mari et al. / QTLCAit pattern)
    for layer in range(Q_DEPTH):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i], wires=i)
        _entangling_layer()

    # Readout: Pauli-Z expectation per qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class DressedQuantumNet(nn.Module):
    """Linear(32->8) -> VQC -> Linear(8->2). Dressed circuit per Mari et al."""
    def __init__(self):
        super().__init__()
        self.pre  = nn.Linear(32, N_QUBITS)          # trainable reduction, not PCA
        self.q_weights = nn.Parameter(0.01 * torch.randn(Q_DEPTH, N_QUBITS))
        self.post = nn.Linear(N_QUBITS, 2)             # class logits

    def forward(self, x):
        x = torch.tanh(self.pre(x))                    # bound to [-1,1] before angle embed
        batch_out = []
        for sample in x:
            q_out = _circuit(sample, self.q_weights)
            batch_out.append(torch.stack(q_out))
        q_out = torch.stack(batch_out).float()
        return self.post(q_out)


def train_vqc(X_train, y_train, X_test, y_test, device):
    model = DressedQuantumNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=VQC_LR)
    loss_fn = nn.CrossEntropyLoss()

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train, dtype=torch.long).to(device)
    Xte = torch.tensor(X_test,  dtype=torch.float32).to(device)

    model.train()
    for epoch in range(VQC_EPOCHS):
        opt.zero_grad()
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
    return preds, probs


# ------------------------------------------------------------------ main
def main():
    data = np.load(FEATURES_PATH)
    X, y = data['features'], data['labels']
    print(f"Loaded features: {X.shape}, labels: {y.shape}, "
          f"forged={int(y.sum())}, authentic={int((1-y).sum())}", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    results = {'SVM-RBF': {'acc': [], 'auc': []},
               'MLP':     {'acc': [], 'auc': []},
               'VQC':     {'acc': [], 'auc': []}}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # SVM-RBF (Al-khuzaie's classifier)
        svm = SVC(kernel='rbf', probability=True, random_state=SEED)
        svm.fit(X_train_s, y_train)
        pred = svm.predict(X_test_s)
        prob = svm.predict_proba(X_test_s)[:, 1]
        results['SVM-RBF']['acc'].append(accuracy_score(y_test, pred))
        results['SVM-RBF']['auc'].append(roc_auc_score(y_test, prob))

        # MLP (parameter-matched to VQC)
        mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=SEED)
        mlp.fit(X_train_s, y_train)
        pred = mlp.predict(X_test_s)
        prob = mlp.predict_proba(X_test_s)[:, 1]
        results['MLP']['acc'].append(accuracy_score(y_test, pred))
        results['MLP']['auc'].append(roc_auc_score(y_test, prob))

        # VQC
        pred, prob = train_vqc(X_train_s, y_train, X_test_s, y_test, device)
        results['VQC']['acc'].append(accuracy_score(y_test, pred))
        results['VQC']['auc'].append(roc_auc_score(y_test, prob))

        print(f"Fold {fold:2d}  "
              f"SVM acc={results['SVM-RBF']['acc'][-1]:.4f}  "
              f"MLP acc={results['MLP']['acc'][-1]:.4f}  "
              f"VQC acc={results['VQC']['acc'][-1]:.4f}", flush=True)

    print(f"\n{'Head':<10} {'Acc (mean±std)':<20} {'AUC (mean±std)':<20}")
    print('-' * 50)
    for name, r in results.items():
        acc_m, acc_s = np.mean(r['acc']), np.std(r['acc'])
        auc_m, auc_s = np.mean(r['auc']), np.std(r['auc'])
        print(f"{name:<10} {acc_m:.4f} ± {acc_s:.4f}      {auc_m:.4f} ± {auc_s:.4f}")


if __name__ == '__main__':
    main()