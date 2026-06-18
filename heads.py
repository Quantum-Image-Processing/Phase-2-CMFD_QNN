"""The two heads under comparison plus matched-capacity helpers.
VQC is Mari Example 2 to the gate level: H + Ry(x*pi/2) embedding, L layers of
per-qubit Ry followed by a linear CNOT chain, <Z> readout, then linear K->1."""
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pennylane as qml


def make_vqc(n_qubits, n_layers, device="default.qubit"):
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for k in range(n_qubits):                       # embedding (Mari Eq. 11)
            qml.Hadamard(wires=k)
            qml.RY(inputs[:, k] * torch.pi / 2, wires=k)
        for l in range(n_layers):                       # L variational layers
            for k in range(n_qubits):
                qml.RY(weights[l, k], wires=k)
            for k in range(n_qubits - 1):               # linear CNOT chain
                qml.CNOT(wires=[k, k + 1])
        return tuple(qml.expval(qml.PauliZ(k)) for k in range(n_qubits))

    return qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits)})


class DressedVQCHead(nn.Module):
    def __init__(self, n_qubits, n_layers, device="default.qubit"):
        super().__init__()
        self.vqc = make_vqc(n_qubits, n_layers, device)
        self.post = nn.Linear(n_qubits, 1)              # L_{K->1}
    def forward(self, x):
        q = self.vqc(x)
        q = q.permute(1, 0, 2).squeeze(-1)             # (batch, n_qubits)
        return self.post(q).squeeze(-1)                # logit


class MLPHead(nn.Module):
    def __init__(self, k, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def vqc_param_count(n_qubits, n_layers):
    return n_layers * n_qubits + (n_qubits + 1)         # rotations + post-linear


def hidden_for_param_match(k, target_params):
    # MLP params = (k+1)*h + (h+1) = h*(k+2) + 1  ->  solve for h
    return max(1, round((target_params - 1) / (k + 2)))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def make_svm():
    grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 1e-3, 1e-2, 1e-1, 1]}
    return GridSearchCV(SVC(kernel="rbf"), grid, cv=3, n_jobs=-1)


def fit_torch_head(head, Xtr, ytr, Xte, epochs, lr, seed):
    """Full-batch training with a fixed validation split and early stopping.
    Returns (test_probabilities, train_loss_curve, val_loss_curve).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)

    n_train = Xtr.shape[0]
    if n_train == 0:
        raise ValueError("Training set is empty.")

    if n_train > 1:
        val_size = max(1, int(np.floor(0.15 * n_train)))
        if val_size >= n_train:
            val_size = n_train - 1
        perm = np.random.RandomState(seed).permutation(n_train)
        val_idx = perm[:val_size]
        tr_idx = perm[val_size:]
        Xtr_train = Xtr[tr_idx]
        ytr_train = ytr[tr_idx]
        Xtr_val = Xtr[val_idx]
        ytr_val = ytr[val_idx]
    else:
        val_size = 0
        Xtr_train, ytr_train = Xtr, ytr
        Xtr_val, ytr_val = Xtr, ytr

    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []
    best_state = None
    best_val = np.inf
    wait = 0

    for epoch in range(epochs):
        head.train()
        opt.zero_grad()
        loss = lossf(head(Xtr_train), ytr_train)
        loss.backward()
        opt.step()

        train_loss = loss.item()
        train_losses.append(train_loss)

        if val_size > 0:
            head.eval()
            with torch.no_grad():
                val_loss = lossf(head(Xtr_val), ytr_val).item()
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1:02d}/{epochs}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f}")
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = deepcopy(head.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= 10:
                    break
        else:
            print(f"Epoch {epoch + 1:02d}/{epochs}: train_loss={train_loss:.4f}")

    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    with torch.no_grad():
        probs = torch.sigmoid(head(Xte)).numpy()
    return probs, train_losses, val_losses
