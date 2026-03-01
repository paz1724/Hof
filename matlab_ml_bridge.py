# matlab_ml_bridge.py
# Python 3.10+ recommended
# pip install numpy scikit-learn joblib torch
# optional for GNN: pip install torch-geometric (plus its extra wheels per your CUDA/CPU)

from __future__ import annotations
import os, json
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
import joblib

# n_jobs for GridSearchCV. Set env SKLEARN_N_JOBS=1 when calling from MATLAB
# (MATLAB's embedded Python cannot spawn loky worker subprocesses).
_N_JOBS = int(os.environ.get("SKLEARN_N_JOBS", "-1"))

# --------- Optional torch / torch-geometric imports ----------
_HAS_TORCH = False
_HAS_TG = False
torch = None
nn = None
F = None
Data = None
DataLoader = None
GCNConv = None
global_mean_pool = None
try:
    # On Windows, MATLAB's embedded Python may fail to load PyTorch DLLs
    # unless we explicitly register torch's DLL directory first.
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        import importlib.util as _ilu
        _torch_spec = _ilu.find_spec("torch")
        if _torch_spec is not None and _torch_spec.origin is not None:
            _torch_lib = os.path.join(os.path.dirname(_torch_spec.origin), "lib")
            if os.path.isdir(_torch_lib):
                os.add_dll_directory(_torch_lib)
            # Also register the torch bin directory (contains some DLLs on newer versions)
            _torch_bin = os.path.join(os.path.dirname(_torch_spec.origin), "bin")
            if os.path.isdir(_torch_bin):
                os.add_dll_directory(_torch_bin)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
    _HAS_TG = True
    try:
        from torch_geometric.data import Data  # pyright: ignore[reportMissingImports]
        try:
            from torch_geometric.loader import DataLoader  # pyright: ignore[reportMissingImports]
        except Exception:
            # Compatibility fallback for older torch-geometric versions.
            from torch_geometric.data import DataLoader  # pyright: ignore[reportMissingImports]
        from torch_geometric.nn import GCNConv, global_mean_pool  # pyright: ignore[reportMissingImports]
    except Exception:
        _HAS_TG = False
except Exception:
    _HAS_TORCH = False
    _HAS_TG = False

# When torch is unavailable in-process (e.g. MATLAB DLL conflict),
# store the real Python path so we can fall back to subprocess execution.
import sys as _sys
_REAL_PYTHON = None
if not _HAS_TORCH:
    _candidate = os.path.join(_sys.prefix, "python.exe" if os.name == "nt" else "python")
    if os.path.isfile(_candidate):
        _REAL_PYTHON = _candidate


def _to_numpy(x):
    """Accept list / nested lists / numpy arrays; return np.array."""
    return np.asarray(x)


def _parse_input(d: dict):
    """
    Expected d fields (edit to match your MATLAB struct):
      - d["X"] : shape [N, F] numeric features (simple tabular baseline)
      - d["y"] : shape [N] labels 0/1
    For GNN mode (graph per sample), expected:
      - d["node_feats"] : shape [N, V, F] (N graphs, V nodes each, F features per node)
      - d["edge_index"] : shape [2, E] or [E, 2] edges shared across graphs (node indices 0..V-1)
      - d["y"] : shape [N]
    You can provide BOTH; we will train BOTH models if possible.
    """
    if "y" not in d:
        raise ValueError('Missing required field "y".')

    y = _to_numpy(d["y"]).astype(np.int64).reshape(-1)
    if y.size == 0:
        raise ValueError('"y" must contain at least one sample.')
    unique_y = np.unique(y)
    if not np.isin(unique_y, np.array([0, 1], dtype=np.int64)).all():
        raise ValueError(f'Expected binary labels in "y" with values 0/1. Got {unique_y.tolist()}.')

    out = {"y": y}
    counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"[parse_input] y: {y.shape[0]} samples, class distribution: {counts}", flush=True)

    if "X" in d:
        X = _to_numpy(d["X"]).astype(np.float32)
        if X.ndim != 2 or X.shape[1] < 1:
            raise ValueError(f'Expected X as [N,F] with F>=1. Got {X.shape}.')
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'X and y sample mismatch. X has {X.shape[0]} rows, y has {y.shape[0]}.')
        out["X"] = X
        print(f"[parse_input] X: {X.shape[0]} samples, {X.shape[1]} features", flush=True)

    # Optional: which models to train
    _VALID_MODELS = {"random_forest", "extra_trees", "hist_gradient_boosting", "svm", "cnn", "transformer", "rl", "gnn"}
    if "models" in d:
        raw = d["models"]
        # A single string (e.g. from MATLAB scalar string) must not be
        # iterated character-by-character; wrap it in a list first.
        if isinstance(raw, str):
            models = [raw]
        else:
            models = [str(m) for m in raw]
        unknown = set(models) - _VALID_MODELS
        if unknown:
            raise ValueError(f"Unknown model names: {unknown}. Valid: {sorted(_VALID_MODELS)}")
        out["models"] = models
        print(f"[parse_input] models: {models}", flush=True)

    if "node_feats" in d and "edge_index" in d:
        node_feats = _to_numpy(d["node_feats"]).astype(np.float32)
        if node_feats.ndim != 3 or node_feats.shape[1] < 2 or node_feats.shape[2] < 1:
            raise ValueError(
                f'Expected node_feats as [N,V,F] with V>=2 and F>=1. Got {node_feats.shape}.'
            )
        if node_feats.shape[0] != y.shape[0]:
            raise ValueError(
                f'node_feats and y sample mismatch. node_feats has {node_feats.shape[0]} rows, y has {y.shape[0]}.'
            )
        edge_index = _to_numpy(d["edge_index"]).astype(np.int64)
        if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.T
        if edge_index.shape[0] != 2:
            raise ValueError(f'edge_index must be [2,E] or [E,2]. Got {edge_index.shape}.')
        n_nodes = node_feats.shape[1]
        if edge_index.size == 0:
            raise ValueError("edge_index cannot be empty.")
        if edge_index.min() < 0 or edge_index.max() >= n_nodes:
            raise ValueError(f"edge_index node ids must be in [0, {n_nodes - 1}].")
        out["node_feats"] = node_feats
        out["edge_index"] = edge_index
        print(f"[parse_input] Graph data: {node_feats.shape[0]} graphs, {node_feats.shape[1]} nodes, {node_feats.shape[2]} features, {edge_index.shape[1]} edges", flush=True)

    return out


def _metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(np.int64)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "auc": auc}


def _safe_stratified_splits(y, preferred=5):
    y = np.asarray(y).astype(np.int64).reshape(-1)
    values, counts = np.unique(y, return_counts=True)
    if values.size < 2:
        raise ValueError("Need at least two classes in y for binary classification.")
    max_splits = int(counts.min())
    if max_splits < 2:
        raise ValueError("Need at least 2 samples in each class for stratified CV.")
    return min(preferred, max_splits)


def _predict_proba_binary(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
    else:
        p = model.decision_function(X)
        if p.ndim == 1 and (p.min() < 0 or p.max() > 1):
            p = 1.0 / (1.0 + np.exp(-p))
    return np.asarray(p, dtype=np.float32).reshape(-1)


def train_sklearn_models(X, y, seed=0, model_keys=None):
    """
    Trains sklearn classifiers with CV and returns best estimator + cv results summary.
    *model_keys*: list of model name strings to evaluate.
                  Default (None) → ``["random_forest", "extra_trees", "hist_gradient_boosting"]``.
    """
    if model_keys is None:
        model_keys = ["random_forest", "extra_trees", "hist_gradient_boosting"]

    all_candidates = {
        "random_forest": lambda: RandomForestClassifier(random_state=seed, class_weight="balanced"),
        "extra_trees": lambda: ExtraTreesClassifier(random_state=seed, class_weight="balanced"),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=seed),
        "svm": lambda: SVC(probability=True, class_weight="balanced", random_state=seed),
    }

    # Small but effective search spaces (edit as needed)
    all_param_grids = {
        "random_forest": {
            "n_estimators": [200, 500],
            "max_depth": [None, 6, 12],
            "min_samples_leaf": [1, 3, 8],
            "max_features": ["sqrt", 0.8],
        },
        "extra_trees": {
            "n_estimators": [300, 700],
            "max_depth": [None, 6, 12],
            "min_samples_leaf": [1, 3, 8],
            "max_features": ["sqrt", 0.8],
        },
        "hist_gradient_boosting": {
            "max_depth": [3, 6, None],
            "learning_rate": [0.05, 0.1],
            "max_iter": [200, 400],
            "l2_regularization": [0.0, 0.1],
        },
        "svm": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
        },
    }

    candidates = [(k, all_candidates[k]()) for k in model_keys if k in all_candidates]
    if not candidates:
        return None, None, None

    cv = StratifiedKFold(n_splits=_safe_stratified_splits(y, preferred=5), shuffle=True, random_state=seed)

    best = None
    best_score = -1.0
    best_name = None
    best_cv = None

    for name, model in candidates:
        print(f"[train_sklearn] Evaluating {name} ...", flush=True)
        gs = GridSearchCV(
            model,
            param_grid=all_param_grids[name],
            scoring="f1",
            cv=cv,
            n_jobs=_N_JOBS,
            refit=True,
        )
        gs.fit(X, y)
        print(f"[train_sklearn] {name} best CV f1={gs.best_score_:.4f}  params={gs.best_params_}", flush=True)
        if gs.best_score_ > best_score:
            best_score = float(gs.best_score_)
            best = gs.best_estimator_
            best_name = name
            best_cv = {
                "best_score_f1": float(gs.best_score_),
                "best_params": gs.best_params_,
            }

    print(f"[train_sklearn] Winner: {best_name} (f1={best_score:.4f})", flush=True)
    return best_name, best, best_cv


# Keep old name as alias for backward compatibility
train_best_tree_model = train_sklearn_models


# -------------------- Torch-based tabular models --------------------

def _train_val_split_torch(X, y, seed, val_frac=0.2):
    """80/20 stratified split, returning torch tensors."""
    idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(idx, test_size=val_frac, stratify=y, random_state=seed)
    Xt = torch.tensor(X[tr_idx], dtype=torch.float32)
    yt = torch.tensor(y[tr_idx], dtype=torch.float32)
    Xv = torch.tensor(X[va_idx], dtype=torch.float32)
    yv = torch.tensor(y[va_idx], dtype=torch.float32)
    return Xt, yt, Xv, yv


# ---- 1D CNN ----

if _HAS_TORCH:
    class TabularCNN(torch.nn.Module):
        def __init__(self, n_features: int, hidden: int = 64):
            super().__init__()
            self.conv1 = nn.Conv1d(1, hidden, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            # x: [N, F] → [N, 1, F]
            x = x.unsqueeze(1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.mean(dim=2)  # global avg pool → [N, hidden]
            return self.fc(x).squeeze(-1)


def train_cnn(X, y, seed=0, epochs=80, lr=1e-3):
    if not _HAS_TORCH:
        return None, {"enabled": False, "reason": "torch not available"}

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt, yt, Xv, yv = _train_val_split_torch(X, y, seed)
    model = TabularCNN(n_features=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_f1, best_state = -1.0, None
    patience, wait = 15, 0
    print(f"[train_cnn] Starting CNN training: {X.shape[0]} samples, {X.shape[1]} features, {epochs} epochs", flush=True)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xv)).numpy()
        m = _metrics(yv.numpy(), probs)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = m
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            print(f"[train_cnn] Early stop at epoch {ep+1}", flush=True)
            break
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"[train_cnn] Epoch {ep+1}/{epochs}  val_f1={m['f1']:.4f}  best_f1={best_f1:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[train_cnn] Done. Best val metrics: {best_metrics}", flush=True)
    return model, {"enabled": True, "val_metrics": best_metrics}


def predict_cnn(model, X):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy().astype(np.float32)
    return probs


# ---- Transformer ----

if _HAS_TORCH:
    class TabularTransformer(torch.nn.Module):
        def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
            super().__init__()
            self.feature_embed = nn.Linear(1, d_model)
            self.pos_embed = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            # x: [N, F] → [N, F, 1] → embed → [N, F, d_model]
            x = self.feature_embed(x.unsqueeze(-1)) + self.pos_embed
            x = self.encoder(x)
            x = x.mean(dim=1)  # mean pool over features
            return self.fc(x).squeeze(-1)


def train_transformer(X, y, seed=0, epochs=80, lr=1e-3):
    if not _HAS_TORCH:
        return None, {"enabled": False, "reason": "torch not available"}

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt, yt, Xv, yv = _train_val_split_torch(X, y, seed)
    model = TabularTransformer(n_features=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_f1, best_state = -1.0, None
    patience, wait = 15, 0
    print(f"[train_transformer] Starting Transformer training: {X.shape[0]} samples, {X.shape[1]} features, {epochs} epochs", flush=True)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xv)).numpy()
        m = _metrics(yv.numpy(), probs)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = m
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            print(f"[train_transformer] Early stop at epoch {ep+1}", flush=True)
            break
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"[train_transformer] Epoch {ep+1}/{epochs}  val_f1={m['f1']:.4f}  best_f1={best_f1:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[train_transformer] Done. Best val metrics: {best_metrics}", flush=True)
    return model, {"enabled": True, "val_metrics": best_metrics}


def predict_transformer(model, X):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy().astype(np.float32)
    return probs


# ---- RL DQN Classifier ----

if _HAS_TORCH:
    class DQNClassifier(torch.nn.Module):
        def __init__(self, n_features: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2),  # Q-values for actions {0, 1}
            )

        def forward(self, x):
            return self.net(x)


def train_rl(X, y, seed=0, episodes=5, lr=1e-3):
    """Train a DQN-style classifier via experience replay."""
    if not _HAS_TORCH:
        return None, {"enabled": False, "reason": "torch not available"}

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    n_features = X.shape[1]
    model = DQNClassifier(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Experience replay buffer
    buffer_X = []
    buffer_action = []
    buffer_reward = []
    batch_size = min(64, X.shape[0])

    # Split for validation
    idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    eps_start, eps_end = 1.0, 0.05
    total_steps = episodes * len(X_tr)

    best_f1, best_state, best_metrics = -1.0, None, None
    step = 0
    print(f"[train_rl] Starting RL DQN training: {X.shape[0]} samples, {episodes} episodes", flush=True)

    for ep in range(episodes):
        order = rng.permutation(len(X_tr))
        for i in order:
            state = torch.tensor(X_tr[i], dtype=torch.float32).unsqueeze(0)
            eps = eps_start - (eps_start - eps_end) * min(step / max(total_steps - 1, 1), 1.0)
            step += 1

            # Epsilon-greedy action
            if rng.rand() < eps:
                action = rng.randint(2)
            else:
                model.eval()
                with torch.no_grad():
                    action = int(model(state).argmax(dim=1).item())

            reward = 1.0 if action == int(y_tr[i]) else -1.0
            buffer_X.append(X_tr[i])
            buffer_action.append(action)
            buffer_reward.append(reward)

            # Train from replay buffer
            if len(buffer_X) >= batch_size:
                idxs = rng.choice(len(buffer_X), size=batch_size, replace=False)
                bx = torch.tensor(np.array([buffer_X[j] for j in idxs]), dtype=torch.float32)
                ba = torch.tensor([buffer_action[j] for j in idxs], dtype=torch.long)
                br = torch.tensor([buffer_reward[j] for j in idxs], dtype=torch.float32)

                model.train()
                q_all = model(bx)
                q_selected = q_all.gather(1, ba.unsqueeze(1)).squeeze(1)
                opt.zero_grad()
                loss = loss_fn(q_selected, br)
                loss.backward()
                opt.step()

        # Validation after each episode
        model.eval()
        with torch.no_grad():
            q = model(torch.tensor(X_va, dtype=torch.float32))
            probs = torch.softmax(q, dim=1)[:, 1].numpy()
        m = _metrics(y_va, probs)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = m
        print(f"[train_rl] Episode {ep+1}/{episodes}  val_f1={m['f1']:.4f}  best_f1={best_f1:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[train_rl] Done. Best val metrics: {best_metrics}", flush=True)
    return model, {"enabled": True, "val_metrics": best_metrics}


def predict_rl(model, X):
    model.eval()
    with torch.no_grad():
        q = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.softmax(q, dim=1)[:, 1].numpy().astype(np.float32)
    return probs


# ----------- Subprocess fallback for torch models in MATLAB -----------

def _run_torch_subprocess(model_key, X, y, seed, save_dir):
    """
    Train a torch model in a subprocess when in-process torch is unavailable
    (e.g. DLL conflict in MATLAB's embedded Python).
    Returns (info_dict, y_prob_list) or (disabled_dict, None).
    """
    if _REAL_PYTHON is None:
        return {"enabled": False, "reason": "torch not available and no external Python found"}, None

    import subprocess, tempfile
    # Save data to temp files
    data_path = os.path.join(save_dir, f"_tmp_{model_key}_data.npz")
    np.savez(data_path, X=X, y=y)

    script = f"""
import sys, os, json
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
os.environ.setdefault('SKLEARN_N_JOBS', '1')
import numpy as np
import matlab_ml_bridge as mb

data = np.load({data_path!r})
X, y = data['X'], data['y']
seed = {seed}
save_dir = {save_dir!r}
model_key = {model_key!r}

trainers = {{"cnn": mb.train_cnn, "transformer": mb.train_transformer, "rl": mb.train_rl}}
predictors = {{"cnn": mb.predict_cnn, "transformer": mb.predict_transformer, "rl": mb.predict_rl}}

model, info = trainers[model_key](X, y, seed=seed)
result = {{"info": info}}
if model is not None:
    import torch
    y_prob = predictors[model_key](model, X).tolist()
    pt_path = os.path.join(save_dir, f"{{model_key}}_state.pt")
    torch.save(model.state_dict(), pt_path)
    result["y_prob"] = y_prob
    result["pt_path"] = pt_path

# Write result as JSON
out_path = os.path.join(save_dir, f"_tmp_{{model_key}}_result.json")
# Convert numpy types for JSON
def jsonable(o):
    if isinstance(o, dict): return {{str(k): jsonable(v) for k, v in o.items()}}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.integer, np.floating)): return o.item()
    return o
with open(out_path, 'w') as f:
    json.dump(jsonable(result), f)
"""
    print(f"[subprocess] Running {model_key} training via {_REAL_PYTHON} ...", flush=True)
    proc = subprocess.run(
        [_REAL_PYTHON, "-c", script],
        capture_output=True, text=True, timeout=600,
    )
    # Forward stdout/stderr from subprocess
    if proc.stdout:
        for line in proc.stdout.rstrip().split("\n"):
            print(line, flush=True)
    if proc.returncode != 0:
        print(f"[subprocess] {model_key} failed (exit {proc.returncode}):", flush=True)
        if proc.stderr:
            for line in proc.stderr.rstrip().split("\n"):
                print(f"  {line}", flush=True)
        return {"enabled": False, "reason": f"subprocess failed (exit {proc.returncode})"}, None

    result_path = os.path.join(save_dir, f"_tmp_{model_key}_result.json")
    with open(result_path) as f:
        result = json.load(f)

    # Clean up temp files
    for tmp in [data_path, result_path]:
        try:
            os.remove(tmp)
        except OSError:
            pass

    info = result["info"]
    y_prob = result.get("y_prob")
    return info, y_prob


# ------------------------ GNN (optional) ------------------------

if _HAS_TG:
    class SimpleGCN(torch.nn.Module):
        def __init__(self, in_dim: int, hidden: int = 64):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, hidden)
            self.lin = nn.Linear(hidden, 1)

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            g = global_mean_pool(x, batch)   # graph embedding
            logits = self.lin(g).squeeze(-1)
            return logits


def train_gnn(node_feats, edge_index, y, seed=0, epochs=80, lr=1e-3, batch_size=64):
    if not _HAS_TG:
        return None, {"enabled": False, "reason": "torch_geometric not available"}

    torch.manual_seed(seed)
    np.random.seed(seed)

    N, V, Fdim = node_feats.shape  # V should be 5
    # Build list of Data objects (one graph per sample)
    data_list = []
    for i in range(N):
        x = torch.tensor(node_feats[i], dtype=torch.float32)  # [5,F]
        ei = torch.tensor(edge_index, dtype=torch.long)       # [2,E]
        yi = torch.tensor(int(y[i]), dtype=torch.long)
        data_list.append(Data(x=x, edge_index=ei, y=yi))

    # Split
    idx = np.arange(N)
    train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    train_set = [data_list[i] for i in train_idx]
    val_set = [data_list[i] for i in val_idx]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = SimpleGCN(in_dim=Fdim, hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    print(f"[train_gnn] Starting GNN training: {N} graphs, {V} nodes, {Fdim} features, {epochs} epochs", flush=True)

    best_val_f1 = -1.0
    best_val_metrics = None
    best_state = None

    for _epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            yb = batch.y.float()
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # quick val
        model.eval()
        probs = []
        ys = []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch.x, batch.edge_index, batch.batch)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
                ys.append(batch.y.cpu().numpy())
        probs = np.concatenate(probs)
        ys = np.concatenate(ys)
        m = _metrics(ys, probs)
        if m["f1"] > best_val_f1:
            best_val_f1 = m["f1"]
            best_val_metrics = m
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (_epoch + 1) % 10 == 0 or _epoch == 0:
            print(f"[train_gnn] Epoch {_epoch+1}/{epochs}  val_f1={m['f1']:.4f}  best_f1={best_val_f1:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[train_gnn] Done. Best val metrics: {best_val_metrics}", flush=True)
    return model, {"enabled": True, "val_metrics": best_val_metrics}


def predict_gnn(model, node_feats, edge_index):
    model.eval()
    N, V, Fdim = node_feats.shape
    # Batch all graphs as one large disconnected graph.
    # Create batch tensors manually
    x_all = torch.tensor(node_feats.reshape(N * V, Fdim), dtype=torch.float32)
    ei = torch.tensor(edge_index, dtype=torch.long)

    # Shift edges per graph
    E = ei.shape[1]
    ei_all = []
    for i in range(N):
        shift = i * V
        ei_all.append(ei + shift)
    ei_all = torch.cat(ei_all, dim=1)

    batch = torch.arange(N, dtype=torch.long).repeat_interleave(V)
    with torch.no_grad():
        logits = model(x_all, ei_all, batch)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    print(f"[predict_gnn] Predicted {probs.shape[0]} graphs", flush=True)
    return probs


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


# ------------------------ Main entry for MATLAB ------------------------

def train_predict_save(matlab_dict: dict, save_dir: str, seed: int = 0):
    """
    Called from MATLAB.

    Returns a plain Python dict (MATLAB will receive a py.dict):
      {
        "tree": {...},               # sklearn winner (if any sklearn models requested)
        "cnn"|"transformer"|"rl": {},# torch model results
        "gnn": {...},                # if graph data + "gnn" requested
        "chosen": "<best model key>",
        "y_prob": [...],
        "metrics": {...}
      }
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"[bridge] Parsing input ...", flush=True)

    parsed = _parse_input(matlab_dict)
    y = parsed["y"]
    models = parsed.get("models", ["random_forest", "extra_trees", "hist_gradient_boosting"])

    results = {}

    _SKLEARN_KEYS = {"random_forest", "extra_trees", "hist_gradient_boosting", "svm"}
    _TORCH_TABULAR_KEYS = {"cnn", "transformer", "rl"}

    sklearn_keys = [m for m in models if m in _SKLEARN_KEYS]
    torch_keys = [m for m in models if m in _TORCH_TABULAR_KEYS]
    want_gnn = "gnn" in models

    # ---- sklearn models (tabular) ----
    if "X" in parsed and sklearn_keys:
        print(f"[bridge] Training sklearn models {sklearn_keys} ...", flush=True)
        X = parsed["X"]
        _, class_counts = np.unique(y, return_counts=True)
        can_holdout = class_counts.min() >= 2 and X.shape[0] >= 10

        if can_holdout:
            tr_idx, va_idx = train_test_split(
                np.arange(X.shape[0]),
                test_size=0.2,
                stratify=y,
                random_state=seed,
            )
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

            name, tree, tree_cv = train_sklearn_models(X_tr, y_tr, seed=seed, model_keys=sklearn_keys)
            if tree is not None:
                y_prob_val = _predict_proba_binary(tree, X_va)
                tree_select_metrics = _metrics(y_va, y_prob_val)
            else:
                name, tree, tree_cv = None, None, None
                tree_select_metrics = None
        else:
            name, tree, tree_cv = train_sklearn_models(X, y, seed=seed, model_keys=sklearn_keys)
            if tree is not None:
                y_prob_val = _predict_proba_binary(tree, X)
                tree_select_metrics = _metrics(y, y_prob_val)
            else:
                tree_select_metrics = None

        if tree is not None:
            # Refit on all data for final model and exported probabilities.
            tree.fit(X, y)
            y_prob_tree = _predict_proba_binary(tree, X)

            tree_path = os.path.join(save_dir, f"best_tree_{name}.joblib")
            joblib.dump(tree, tree_path)
            print(f"[bridge] sklearn model saved: {tree_path}", flush=True)

            results["tree"] = {
                "model": name,
                "cv": tree_cv,
                "path": tree_path,
                "metrics_select": tree_select_metrics,
                "metrics_train": _metrics(y, y_prob_tree),
                "y_prob": y_prob_tree.tolist(),
            }

    # ---- Torch tabular models ----
    if "X" in parsed and torch_keys:
        X = parsed["X"]
        if _HAS_TORCH:
            # In-process path (normal Python, pytest, etc.)
            torch_trainers = {
                "cnn": (train_cnn, predict_cnn),
                "transformer": (train_transformer, predict_transformer),
                "rl": (train_rl, predict_rl),
            }
            for tk in torch_keys:
                train_fn, pred_fn = torch_trainers[tk]
                print(f"[bridge] Training {tk} model ...", flush=True)
                model, info = train_fn(X, y, seed=seed)
                if model is not None:
                    y_prob_torch = pred_fn(model, X)
                    pt_path = os.path.join(save_dir, f"{tk}_state.pt")
                    torch.save(model.state_dict(), pt_path)
                    print(f"[bridge] {tk} model saved: {pt_path}", flush=True)
                    results[tk] = {
                        "enabled": True,
                        "path": pt_path,
                        "train_info": info,
                        "metrics_select": info.get("val_metrics"),
                        "metrics_train": _metrics(y, y_prob_torch),
                        "y_prob": y_prob_torch.tolist(),
                    }
                else:
                    results[tk] = info
        else:
            # Subprocess fallback (MATLAB DLL conflict or torch not installed)
            for tk in torch_keys:
                print(f"[bridge] Training {tk} model (subprocess) ...", flush=True)
                info, y_prob_list = _run_torch_subprocess(tk, X, y, seed, save_dir)
                if y_prob_list is not None:
                    y_prob_arr = np.array(y_prob_list, dtype=np.float32)
                    results[tk] = {
                        "enabled": True,
                        "path": info.get("pt_path", os.path.join(save_dir, f"{tk}_state.pt")),
                        "train_info": info,
                        "metrics_select": info.get("val_metrics"),
                        "metrics_train": _metrics(y, y_prob_arr),
                        "y_prob": y_prob_list,
                    }
                else:
                    results[tk] = info

    # ---- GNN model (graph per sample) ----
    if want_gnn and "node_feats" in parsed and "edge_index" in parsed:
        print(f"[bridge] Training GNN model ...", flush=True)
        if _HAS_TG:
            model, info = train_gnn(parsed["node_feats"], parsed["edge_index"], y, seed=seed)
            if model is not None:
                y_prob_gnn = predict_gnn(model, parsed["node_feats"], parsed["edge_index"])
                gnn_path = os.path.join(save_dir, "gnn_state.pt")
                torch.save(model.state_dict(), gnn_path)

                results["gnn"] = {
                    "enabled": True,
                    "path": gnn_path,
                    "train_info": info,
                    "metrics_select": info.get("val_metrics"),
                    "metrics_train": _metrics(y, y_prob_gnn),
                    "y_prob": y_prob_gnn.tolist(),
                }
            else:
                results["gnn"] = info
        else:
            results["gnn"] = {"enabled": False, "reason": "torch/torch_geometric not available"}

    # ---- Choose best model by selection F1 ----
    cand = []
    if "tree" in results and "metrics_select" in results["tree"]:
        cand.append(("tree", results["tree"]["metrics_select"]["f1"], results["tree"]["y_prob"], results["tree"]["metrics_select"]))

    for tk in list(_TORCH_TABULAR_KEYS) + ["gnn"]:
        if (
            tk in results
            and results[tk].get("enabled", False)
            and results[tk].get("metrics_select") is not None
        ):
            cand.append((tk, results[tk]["metrics_select"]["f1"], results[tk]["y_prob"], results[tk]["metrics_select"]))

    if len(cand) == 0:
        # Build informative message listing disabled models
        disabled = {k: v.get("reason", "unknown") for k, v in results.items()
                    if isinstance(v, dict) and not v.get("enabled", True)}
        msg = "No model could be trained."
        if disabled:
            msg += f" Disabled models: {disabled}."
        msg += " Provide X and/or (node_feats, edge_index) with valid binary y."
        raise ValueError(msg)

    cand.sort(key=lambda t: t[1], reverse=True)
    chosen, _, y_prob, chosen_metrics = cand[0]

    results["chosen"] = chosen
    results["y_prob"] = y_prob
    results["metrics"] = chosen_metrics
    print(f"[bridge] Chosen model: {chosen} | metrics: acc={chosen_metrics['acc']:.4f} f1={chosen_metrics['f1']:.4f} auc={chosen_metrics['auc']:.4f}", flush=True)

    # also save a small json summary
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, indent=2)
    print(f"[bridge] Summary saved: {os.path.join(save_dir, 'summary.json')}", flush=True)

    return results


def predict_tree_model(model_path: str, X):
    """
    Load a saved tree model and return probabilities for class 1.
    Useful for MATLAB-side test/evaluation on unseen data.
    """
    if not isinstance(model_path, str) or len(model_path.strip()) == 0:
        raise ValueError("model_path must be a non-empty string.")

    X = _to_numpy(X).astype(np.float32)
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError(f"Expected X as [N,F] with F>=1. Got {X.shape}.")

    model = joblib.load(model_path)
    print(f"[predict_tree] Loaded model from {model_path}", flush=True)
    y_prob = _predict_proba_binary(model, X)
    print(f"[predict_tree] Predicted {y_prob.shape[0]} samples", flush=True)
    return {"y_prob": y_prob.tolist()}