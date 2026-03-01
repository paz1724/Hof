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
import joblib

# n_jobs for GridSearchCV. Set env SKLEARN_N_JOBS=1 when calling from MATLAB
# (MATLAB's embedded Python cannot spawn loky worker subprocesses).
_N_JOBS = int(os.environ.get("SKLEARN_N_JOBS", "-1"))

# --------- Optional GNN imports (fallback if not installed) ----------
_HAS_TG = False
torch = None
nn = None
F = None
Data = None
DataLoader = None
GCNConv = None
global_mean_pool = None
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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
    _HAS_TG = False


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


def train_best_tree_model(X, y, seed=0):
    """
    Trains a strong tree ensemble with CV and returns best estimator + cv results summary.
    Uses only scikit-learn (no exotic deps).
    """
    # Candidates
    candidates = [
        ("rf", RandomForestClassifier(random_state=seed, class_weight="balanced")),
        ("et", ExtraTreesClassifier(random_state=seed, class_weight="balanced")),
        ("hgb", HistGradientBoostingClassifier(random_state=seed)),
    ]

    # Small but effective search spaces (edit as needed)
    param_grids = {
        "rf": {
            "n_estimators": [200, 500],
            "max_depth": [None, 6, 12],
            "min_samples_leaf": [1, 3, 8],
            "max_features": ["sqrt", 0.8],
        },
        "et": {
            "n_estimators": [300, 700],
            "max_depth": [None, 6, 12],
            "min_samples_leaf": [1, 3, 8],
            "max_features": ["sqrt", 0.8],
        },
        "hgb": {
            "max_depth": [3, 6, None],
            "learning_rate": [0.05, 0.1],
            "max_iter": [200, 400],
            "l2_regularization": [0.0, 0.1],
        },
    }

    cv = StratifiedKFold(n_splits=_safe_stratified_splits(y, preferred=5), shuffle=True, random_state=seed)

    best = None
    best_score = -1.0
    best_name = None
    best_cv = None

    for name, model in candidates:
        print(f"[train_tree] Evaluating {name} ...", flush=True)
        gs = GridSearchCV(
            model,
            param_grid=param_grids[name],
            scoring="f1",
            cv=cv,
            n_jobs=_N_JOBS,
            refit=True,
        )
        gs.fit(X, y)
        print(f"[train_tree] {name} best CV f1={gs.best_score_:.4f}  params={gs.best_params_}", flush=True)
        if gs.best_score_ > best_score:
            best_score = float(gs.best_score_)
            best = gs.best_estimator_
            best_name = name
            best_cv = {
                "best_score_f1": float(gs.best_score_),
                "best_params": gs.best_params_,
            }

    print(f"[train_tree] Winner: {best_name} (f1={best_score:.4f})", flush=True)
    return best_name, best, best_cv


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
        "tree": {...},
        "gnn": {...} (if possible),
        "chosen": "tree" or "gnn",
        "y_prob": [...],
        "metrics": {...}
      }
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"[bridge] Parsing input ...", flush=True)

    parsed = _parse_input(matlab_dict)
    y = parsed["y"]

    results = {}

    # ---- Tree model (tabular) ----
    if "X" in parsed:
        print(f"[bridge] Training tree model ...", flush=True)
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

            name, tree, tree_cv = train_best_tree_model(X_tr, y_tr, seed=seed)
            y_prob_val = _predict_proba_binary(tree, X_va)
            tree_select_metrics = _metrics(y_va, y_prob_val)
        else:
            # Small datasets may not support a stable stratified holdout split.
            name, tree, tree_cv = train_best_tree_model(X, y, seed=seed)
            y_prob_val = _predict_proba_binary(tree, X)
            tree_select_metrics = _metrics(y, y_prob_val)

        # Refit on all data for final model and exported probabilities.
        tree.fit(X, y)
        y_prob_tree = _predict_proba_binary(tree, X)

        tree_path = os.path.join(save_dir, f"best_tree_{name}.joblib")
        joblib.dump(tree, tree_path)
        print(f"[bridge] Tree model saved: {tree_path}", flush=True)

        results["tree"] = {
            "model": name,
            "cv": tree_cv,
            "path": tree_path,
            "metrics_select": tree_select_metrics,
            "metrics_train": _metrics(y, y_prob_tree),
        }
        results["tree"]["y_prob"] = y_prob_tree.tolist()

    # ---- GNN model (graph per sample) ----
    if "node_feats" in parsed and "edge_index" in parsed:
        print(f"[bridge] Training GNN model ...", flush=True)
        if _HAS_TG:
            model, info = train_gnn(parsed["node_feats"], parsed["edge_index"], y, seed=seed)
            if model is not None:
                y_prob_gnn = predict_gnn(model, parsed["node_feats"], parsed["edge_index"])
                gnn_path = os.path.join(save_dir, "gnn_state.pt")
                import torch
                torch.save(model.state_dict(), gnn_path)

                results["gnn"] = {
                    "enabled": True,
                    "path": gnn_path,
                    "train_info": info,
                    "metrics_select": info.get("val_metrics"),
                    "metrics_train": _metrics(y, y_prob_gnn),
                }
                results["gnn"]["y_prob"] = y_prob_gnn.tolist()
            else:
                results["gnn"] = info
        else:
            results["gnn"] = {"enabled": False, "reason": "torch/torch_geometric not available"}

    # ---- Choose model by F1 on train (replace with proper held-out if you want) ----
    chosen = None
    y_prob = None
    chosen_metrics = None

    cand = []
    if "tree" in results and "metrics_select" in results["tree"]:
        cand.append(("tree", results["tree"]["metrics_select"]["f1"], results["tree"]["y_prob"], results["tree"]["metrics_select"]))
    if (
        "gnn" in results
        and results["gnn"].get("enabled", False)
        and results["gnn"].get("metrics_select") is not None
    ):
        cand.append(("gnn", results["gnn"]["metrics_select"]["f1"], results["gnn"]["y_prob"], results["gnn"]["metrics_select"]))

    if len(cand) == 0:
        raise ValueError("No model could be trained. Provide X and/or (node_feats, edge_index) with valid binary y.")

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