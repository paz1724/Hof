"""Tests for matlab_ml_bridge.py"""
import os
import json
import tempfile

import numpy as np
import pytest

import matlab_ml_bridge as mb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tabular(n=60, f=4, seed=42):
    """Return a simple tabular dict with balanced binary labels."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, f).astype(np.float32)
    y = np.array([0, 1] * (n // 2), dtype=np.int64)
    return {"X": X.tolist(), "y": y.tolist()}


def _make_graph(n=40, v=5, f=3, e=8, seed=42):
    """Return a graph-mode dict with balanced binary labels."""
    rng = np.random.RandomState(seed)
    node_feats = rng.randn(n, v, f).astype(np.float32)
    edge_index = rng.randint(0, v, size=(2, e)).astype(np.int64)
    y = np.array([0, 1] * (n // 2), dtype=np.int64)
    return {
        "node_feats": node_feats.tolist(),
        "edge_index": edge_index.tolist(),
        "y": y.tolist(),
    }


# ===================================================================
# 1. _parse_input
# ===================================================================

class TestParseInput:
    def test_valid_tabular(self):
        d = _make_tabular(n=20, f=3)
        out = mb._parse_input(d)
        assert out["X"].shape == (20, 3)
        assert out["X"].dtype == np.float32
        assert out["y"].shape == (20,)
        assert out["y"].dtype == np.int64

    def test_valid_graph(self):
        d = _make_graph(n=10, v=4, f=2, e=6)
        out = mb._parse_input(d)
        assert out["node_feats"].shape == (10, 4, 2)
        assert out["edge_index"].shape[0] == 2
        assert out["y"].shape == (10,)

    def test_edge_index_transposed(self):
        """edge_index provided as [E,2] should be transposed to [2,E]."""
        d = _make_graph(n=10, v=4, f=2, e=6)
        # Transpose edge_index to [E,2]
        ei = np.array(d["edge_index"])  # currently [2,E]
        d["edge_index"] = ei.T.tolist()  # now [E,2]
        out = mb._parse_input(d)
        assert out["edge_index"].shape[0] == 2

    def test_missing_y_raises(self):
        with pytest.raises(ValueError, match="Missing required field"):
            mb._parse_input({"X": [[1, 2], [3, 4]]})

    def test_empty_y_raises(self):
        with pytest.raises(ValueError, match="at least one sample"):
            mb._parse_input({"X": np.empty((0, 2)).tolist(), "y": []})

    def test_non_binary_labels_raises(self):
        with pytest.raises(ValueError, match="binary labels"):
            mb._parse_input({"X": [[1], [2], [3]], "y": [0, 1, 2]})

    def test_xy_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            mb._parse_input({"X": [[1, 2]], "y": [0, 1]})

    def test_bad_x_shape_raises(self):
        with pytest.raises(ValueError, match="Expected X"):
            mb._parse_input({"X": [1, 2, 3], "y": [0, 1, 0]})


# ===================================================================
# 2. _metrics
# ===================================================================

class TestMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        m = mb._metrics(y_true, y_prob)
        assert m["acc"] == 1.0
        assert m["f1"] == 1.0
        assert m["auc"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        m = mb._metrics(y_true, y_prob)
        assert m["acc"] == 0.0
        assert m["f1"] == 0.0

    def test_mixed_values_in_range(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.2, 0.9, 0.6, 0.8, 0.3, 0.1])
        m = mb._metrics(y_true, y_prob)
        assert 0.0 <= m["acc"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0
        assert 0.0 <= m["auc"] <= 1.0


# ===================================================================
# 3. _safe_stratified_splits
# ===================================================================

class TestSafeStratifiedSplits:
    def test_balanced_returns_preferred(self):
        y = np.array([0] * 50 + [1] * 50)
        assert mb._safe_stratified_splits(y, preferred=5) == 5

    def test_small_dataset(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        assert mb._safe_stratified_splits(y, preferred=5) == 3

    def test_single_class_raises(self):
        y = np.array([0, 0, 0])
        with pytest.raises(ValueError, match="at least two classes"):
            mb._safe_stratified_splits(y)

    def test_one_sample_per_class_raises(self):
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="at least 2 samples"):
            mb._safe_stratified_splits(y)


# ===================================================================
# 4. _predict_proba_binary
# ===================================================================

class TestPredictProbaBinary:
    def test_with_fitted_sklearn_model(self):
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.RandomState(0)
        X = rng.randn(40, 3).astype(np.float32)
        y = np.array([0, 1] * 20)
        clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        p = mb._predict_proba_binary(clf, X)
        assert p.shape == (40,)
        assert p.dtype == np.float32
        assert np.all((p >= 0) & (p <= 1))


# ===================================================================
# 5. train_best_tree_model
# ===================================================================

class TestTrainBestTreeModel:
    @pytest.fixture(scope="class")
    def trained(self):
        d = _make_tabular(n=80, f=4, seed=7)
        X = np.array(d["X"], dtype=np.float32)
        y = np.array(d["y"], dtype=np.int64)
        return mb.train_best_tree_model(X, y, seed=7)

    def test_returns_valid_name(self, trained):
        name, _, _ = trained
        assert name in ("random_forest", "extra_trees", "hist_gradient_boosting")

    def test_model_can_predict(self, trained):
        _, model, _ = trained
        X_new = np.random.randn(5, 4).astype(np.float32)
        pred = model.predict(X_new)
        assert pred.shape == (5,)

    def test_cv_dict_keys(self, trained):
        _, _, cv_dict = trained
        assert "best_score_f1" in cv_dict
        assert "best_params" in cv_dict
        assert isinstance(cv_dict["best_score_f1"], float)


# ===================================================================
# 6. train_predict_save (end-to-end)
# ===================================================================

class TestTrainPredictSave:
    def test_tabular_end_to_end(self, tmp_path):
        d = _make_tabular(n=60, f=4, seed=99)
        save_dir = str(tmp_path / "model_out")
        result = mb.train_predict_save(d, save_dir, seed=99)

        assert "chosen" in result
        assert "tree" in result
        assert "y_prob" in result
        assert "metrics" in result

        # Check saved artefacts
        assert os.path.isfile(os.path.join(save_dir, "summary.json"))
        tree_files = [f for f in os.listdir(save_dir) if f.startswith("best_tree_")]
        assert len(tree_files) == 1

        # summary.json should be valid JSON
        with open(os.path.join(save_dir, "summary.json")) as f:
            summary = json.load(f)
        assert "chosen" in summary

    def test_raises_when_no_data(self):
        with pytest.raises(ValueError):
            mb.train_predict_save({"y": [0, 1, 0, 1]}, "/tmp/nowhere")


# ===================================================================
# 7. predict_tree_model
# ===================================================================

class TestPredictTreeModel:
    def test_load_and_predict(self, tmp_path):
        d = _make_tabular(n=40, f=3, seed=8)
        save_dir = str(tmp_path / "pred_out")
        mb.train_predict_save(d, save_dir, seed=8)

        tree_files = [f for f in os.listdir(save_dir) if f.startswith("best_tree_")]
        model_path = os.path.join(save_dir, tree_files[0])
        X_new = np.random.randn(5, 3).astype(np.float32)
        out = mb.predict_tree_model(model_path, X_new)
        assert len(out["y_prob"]) == 5

    def test_empty_path_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            mb.predict_tree_model("", [[1, 2]])

    def test_bad_x_shape_raises(self, tmp_path):
        d = _make_tabular(n=40, f=3, seed=8)
        save_dir = str(tmp_path / "pred_bad")
        mb.train_predict_save(d, save_dir, seed=8)

        tree_files = [f for f in os.listdir(save_dir) if f.startswith("best_tree_")]
        model_path = os.path.join(save_dir, tree_files[0])
        with pytest.raises(ValueError, match="Expected X"):
            mb.predict_tree_model(model_path, [1, 2, 3])


# ===================================================================
# 8. _to_jsonable
# ===================================================================

class TestToJsonable:
    def test_numpy_scalars(self):
        result = mb._to_jsonable(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

        result = mb._to_jsonable(np.float64(3.14))
        assert isinstance(result, float)

    def test_nested_dict_and_list(self):
        obj = {"a": np.int32(1), "b": [np.float64(2.5), {"c": np.int64(3)}]}
        result = mb._to_jsonable(obj)
        assert result == {"a": 1, "b": [2.5, {"c": 3}]}
        # Ensure JSON-serialisable
        json.dumps(result)

    def test_plain_values_passthrough(self):
        assert mb._to_jsonable("hello") == "hello"
        assert mb._to_jsonable(42) == 42
        assert mb._to_jsonable(None) is None


# ===================================================================
# 9. GNN functions (conditional)
# ===================================================================

@pytest.mark.skipif(not mb._HAS_TG, reason="torch-geometric not installed")
class TestGNN:
    def test_train_gnn(self):
        d = _make_graph(n=40, v=5, f=3, e=8, seed=0)
        parsed = mb._parse_input(d)
        model, info = mb.train_gnn(
            parsed["node_feats"], parsed["edge_index"], parsed["y"],
            seed=0, epochs=5,
        )
        assert model is not None
        assert info["enabled"] is True
        assert "val_metrics" in info

    def test_predict_gnn(self):
        d = _make_graph(n=40, v=5, f=3, e=8, seed=0)
        parsed = mb._parse_input(d)
        model, _ = mb.train_gnn(
            parsed["node_feats"], parsed["edge_index"], parsed["y"],
            seed=0, epochs=5,
        )
        probs = mb.predict_gnn(model, parsed["node_feats"], parsed["edge_index"])
        assert probs.shape == (40,)
        assert np.all((probs >= 0) & (probs <= 1))


class TestTrainGNNFallback:
    """When torch-geometric is NOT installed, train_gnn returns a disabled info dict."""

    @pytest.mark.skipif(mb._HAS_TG, reason="Only test fallback when TG is absent")
    def test_returns_disabled(self):
        d = _make_graph(n=10, v=5, f=3, e=6)
        parsed = mb._parse_input(d)
        model, info = mb.train_gnn(
            parsed["node_feats"], parsed["edge_index"], parsed["y"],
        )
        assert model is None
        assert info["enabled"] is False


# ===================================================================
# 10. Model selection via "models" field
# ===================================================================

class TestModelSelection:
    def test_parse_input_models_field(self):
        d = _make_tabular(n=20, f=3)
        d["models"] = ["random_forest", "svm"]
        out = mb._parse_input(d)
        assert out["models"] == ["random_forest", "svm"]

    def test_parse_input_invalid_model_raises(self):
        d = _make_tabular(n=20, f=3)
        d["models"] = ["random_forest", "bogus"]
        with pytest.raises(ValueError, match="Unknown model names"):
            mb._parse_input(d)

    def test_default_models_backward_compat(self, tmp_path):
        """No models field → trains random_forest/extra_trees/hist_gradient_boosting (old behaviour)."""
        d = _make_tabular(n=60, f=4, seed=99)
        save_dir = str(tmp_path / "compat")
        result = mb.train_predict_save(d, save_dir, seed=99)
        assert "tree" in result
        assert result["tree"]["model"] in ("random_forest", "extra_trees", "hist_gradient_boosting")

    def test_select_only_svm(self, tmp_path):
        d = _make_tabular(n=60, f=4, seed=99)
        d["models"] = ["svm"]
        save_dir = str(tmp_path / "svm_only")
        result = mb.train_predict_save(d, save_dir, seed=99)
        assert "tree" in result
        assert result["tree"]["model"] == "svm"


# ===================================================================
# 11. SVM via train_sklearn_models
# ===================================================================

class TestSVM:
    def test_svm_via_sklearn_models(self):
        d = _make_tabular(n=60, f=4, seed=7)
        X = np.array(d["X"], dtype=np.float32)
        y = np.array(d["y"], dtype=np.int64)
        name, model, cv = mb.train_sklearn_models(X, y, seed=7, model_keys=["svm"])
        assert name == "svm"
        assert hasattr(model, "predict_proba")
        p = mb._predict_proba_binary(model, X)
        assert p.shape == (60,)


# ===================================================================
# 12. CNN (conditional on torch)
# ===================================================================

@pytest.mark.skipif(not mb._HAS_TORCH, reason="torch not installed")
class TestCNN:
    def test_train_and_predict(self):
        d = _make_tabular(n=60, f=4, seed=0)
        X = np.array(d["X"], dtype=np.float32)
        y = np.array(d["y"], dtype=np.int64)
        model, info = mb.train_cnn(X, y, seed=0, epochs=5)
        assert model is not None
        assert info["enabled"] is True
        probs = mb.predict_cnn(model, X)
        assert probs.shape == (60,)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_end_to_end(self, tmp_path):
        d = _make_tabular(n=60, f=4, seed=0)
        d["models"] = ["cnn"]
        save_dir = str(tmp_path / "cnn_e2e")
        result = mb.train_predict_save(d, save_dir, seed=0)
        assert "cnn" in result
        assert result["cnn"]["enabled"] is True
        assert os.path.isfile(os.path.join(save_dir, "cnn_state.pt"))


# ===================================================================
# 13. Transformer (conditional on torch)
# ===================================================================

@pytest.mark.skipif(not mb._HAS_TORCH, reason="torch not installed")
class TestTransformer:
    def test_train_and_predict(self):
        d = _make_tabular(n=60, f=4, seed=0)
        X = np.array(d["X"], dtype=np.float32)
        y = np.array(d["y"], dtype=np.int64)
        model, info = mb.train_transformer(X, y, seed=0, epochs=5)
        assert model is not None
        assert info["enabled"] is True
        probs = mb.predict_transformer(model, X)
        assert probs.shape == (60,)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_end_to_end(self, tmp_path):
        d = _make_tabular(n=60, f=4, seed=0)
        d["models"] = ["transformer"]
        save_dir = str(tmp_path / "transformer_e2e")
        result = mb.train_predict_save(d, save_dir, seed=0)
        assert "transformer" in result
        assert result["transformer"]["enabled"] is True
        assert os.path.isfile(os.path.join(save_dir, "transformer_state.pt"))


# ===================================================================
# 14. RL DQN (conditional on torch)
# ===================================================================

@pytest.mark.skipif(not mb._HAS_TORCH, reason="torch not installed")
class TestRL:
    def test_train_and_predict(self):
        d = _make_tabular(n=60, f=4, seed=0)
        X = np.array(d["X"], dtype=np.float32)
        y = np.array(d["y"], dtype=np.int64)
        model, info = mb.train_rl(X, y, seed=0, episodes=2)
        assert model is not None
        assert info["enabled"] is True
        probs = mb.predict_rl(model, X)
        assert probs.shape == (60,)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_end_to_end(self, tmp_path):
        d = _make_tabular(n=60, f=4, seed=0)
        d["models"] = ["rl"]
        save_dir = str(tmp_path / "rl_e2e")
        result = mb.train_predict_save(d, save_dir, seed=0)
        assert "rl" in result
        assert result["rl"]["enabled"] is True
        assert os.path.isfile(os.path.join(save_dir, "rl_state.pt"))


# ===================================================================
# 15. Mixed model selection
# ===================================================================

class TestMixedModels:
    def test_sklearn_plus_torch(self, tmp_path):
        """Train a mix of sklearn and torch models, check winner is selected."""
        d = _make_tabular(n=80, f=4, seed=42)
        models = ["random_forest", "svm"]
        if mb._HAS_TORCH:
            models.append("cnn")
        d["models"] = models
        save_dir = str(tmp_path / "mixed")
        result = mb.train_predict_save(d, save_dir, seed=42)
        assert "chosen" in result
        assert "y_prob" in result
        assert "metrics" in result
