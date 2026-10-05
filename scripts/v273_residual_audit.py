"""Two-residual audit math and portable replay; no audio or TensorFlow imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ("best_pair_residual_ratio", "best_triplet_residual_ratio")
FOLDS = (0, 1, 2, 4)
REGISTERS = ("low", "mid", "high", "unassigned")
SCHEMA_VERSION = 2
EXPERIMENT = "v273_internal_B_low_minimal_residual_failure_audit"
COUNT_KEYS = ("val_rows", "val_k23_rows", "applied", "corrections", "regressions",
              "other_k_actions", "global_net", "k23_exact_net")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def classifier():
    # Identical to the original exact guard, including its random_state.
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                  class_weight="balanced", random_state=28431)),
    ])


def fit_classifier(X, y):
    # Both entry points use the same dtype/layout as well as hyperparameters.
    return classifier().fit(np.array(X, dtype=np.float64, order="C"), y)


def auc(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    return float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None


def stats_by_class(y, values):
    out = {}
    for label, name in ((1, "K2"), (0, "K3")):
        z = np.asarray(values)[np.asarray(y) == label]
        out[name] = {"n": len(z), **{k: None for k in ("mean", "median", "std", "q25", "q75")}}
        if len(z):
            out[name].update(mean=float(z.mean()), median=float(np.median(z)),
                             std=float(z.std()), q25=float(np.quantile(z, .25)),
                             q75=float(np.quantile(z, .75)))
    return out


def corr(x, y):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def oriented_univariate(yfit, xfit, yval, xval):
    positive, negative = xfit[yfit == 1], xfit[yfit == 0]
    delta = float(positive.mean() - negative.mean())
    orientation = 1 if delta >= 0 else -1
    val_delta = (float(xval[yval == 1].mean() - xval[yval == 0].mean())
                 if len(np.unique(yval)) == 2 else None)
    return {"fit_orientation": orientation, "fit_class_mean_delta": delta,
            "val_class_mean_delta": val_delta, "val_auc": auc(yval, orientation * xval)}


def register_labels(f0, cuts):
    f0 = np.asarray(f0)
    labels = np.full(len(f0), "unassigned", dtype="U10")
    valid = np.isfinite(f0) & (f0 > 0)
    labels[valid & (f0 < cuts[0])] = "low"
    labels[valid & (f0 >= cuts[0]) & (f0 < cuts[1])] = "mid"
    labels[valid & (f0 >= cuts[1])] = "high"
    return labels


def action_metrics(true_k, probability):
    """All inputs already satisfy B_low & base_K3; other K remain in accounting."""
    y, p = np.asarray(true_k), np.asarray(probability)
    require(y.shape == p.shape and np.isfinite(p).all(), "bad action arrays")
    require(np.all((p >= 0) & (p <= 1)), "bad probability")
    applied = p >= .5
    k23 = np.isin(y, (2, 3))
    corrections = int(np.sum(applied & (y == 2)))
    regressions = int(np.sum(applied & (y == 3)))
    return {
        "val_rows": len(y), "val_k23_rows": int(k23.sum()),
        "joint_auc": auc((y[k23] == 2).astype(int), p[k23]),
        "applied": int(applied.sum()), "corrections": corrections,
        "regressions": regressions, "other_k_actions": int(np.sum(applied & ~k23)),
        "global_net": corrections - regressions, "k23_exact_net": corrections - regressions,
        "by_k_net": {str(k): corrections if k == 2 else -regressions if k == 3 else 0
                     for k in range(7)},
    }


def stratify(true_k, probability, labels):
    require(len(labels) == len(true_k), "register length mismatch")
    require(np.isin(labels, REGISTERS).all(), "unknown register")
    parts = {name: action_metrics(true_k[labels == name], probability[labels == name])
             for name in REGISTERS}
    total = action_metrics(true_k, probability)
    for key in COUNT_KEYS:
        require(sum(part[key] for part in parts.values()) == total[key],
                "non-additive register counts: " + key)
    return parts


def model_state(model):
    scale, lr = model.named_steps["scale"], model.named_steps["lr"]
    return {
        "features": list(FEATURES), "classes": lr.classes_.tolist(), "threshold": .5,
        "scale_mean": scale.mean_.tolist(), "scale_scale": scale.scale_.tolist(),
        "scale_var": scale.var_.tolist(), "scale_n_samples_seen": int(scale.n_samples_seen_),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
        "n_iter": lr.n_iter_.tolist(), "parameters": lr.get_params(),
    }


def score_from_state(X, state):
    require(state["features"] == list(FEATURES) and state["classes"] == [0, 1],
            "model schema mismatch")
    require(state["threshold"] == .5, "threshold changed")
    score = ((X - np.asarray(state["scale_mean"])) / np.asarray(state["scale_scale"])) @ np.asarray(state["coef"]) + state["intercept"]
    return score, expit(score)


def analyze(Xf, yf, f0f, Xv, yv, f0v, *, include_register_refit=False):
    """Fit once on K2/K3; stratify these exact global predictions on ALL action rows."""
    for X, y, f0 in ((Xf, yf, f0f), (Xv, yv, f0v)):
        require(X.shape == (len(y), 2) and len(f0) == len(y), "input shape mismatch")
        require(np.isfinite(X).all() and np.isin(y, range(7)).all(), "invalid residual input")
    train, test = np.isin(yf, (2, 3)), np.isin(yv, (2, 3))
    ytr, yte = (yf[train] == 2).astype(int), (yv[test] == 2).astype(int)
    require(len(np.unique(ytr)) == 2, "FIT binary collapse")
    # Keep the original guard's matrix layout and input order.
    m = fit_classifier(Xf[train], ytr)
    state = model_state(m)
    predictions = {}
    for split, X in (("fit", Xf), ("val", Xv)):
        predictions[split + "_score"] = m.decision_function(X) if len(X) else np.empty(0)
        predictions[split + "_probability"] = m.predict_proba(X)[:, 1] if len(X) else np.empty(0)
    p = predictions["val_probability"]
    fit_f0 = f0f[train & np.isfinite(f0f) & (f0f > 0)]
    require(len(fit_f0) > 0, "no finite FIT register values")
    cuts = np.quantile(fit_f0, [1 / 3, 2 / 3])
    regf, regv = register_labels(f0f, cuts), register_labels(f0v, cuts)
    predictions.update(fit_register=regf, val_register=regv)

    uni = {name: oriented_univariate(ytr, Xf[train, j], yte, Xv[test, j])
           for j, name in enumerate(FEATURES)}
    parts = stratify(yv, p, regv)
    for name, part in parts.items():
        mf, mv = (regf == name) & train, (regv == name) & test
        part["fit_rows"] = int(mf.sum())
        for j, feature in enumerate(FEATURES):
            part[("pair_auc", "triplet_auc")[j]] = auc(
                (yv[mv] == 2).astype(int), uni[feature]["fit_orientation"] * Xv[mv, j])

    refits = {"enabled": include_register_refit, "registers": {}}
    if include_register_refit:
        for name in REGISTERS:
            mf, mv = train & (regf == name), regv == name
            entry = {"status": "insufficient_fit", "fit_rows": int(mf.sum()), "val_rows": int(mv.sum())}
            if mf.sum() >= 12 and len(np.unique(yf[mf])) == 2:
                rm = fit_classifier(Xf[mf], (yf[mf] == 2).astype(int))
                rp = rm.predict_proba(Xv[mv])[:, 1] if mv.any() else np.empty(0)
                entry.update(action_metrics(yv[mv], rp), status="completed", model=model_state(rm))
            refits["registers"][name] = entry

    Zf = m.named_steps["scale"].transform(Xf[train])
    cov = np.cov(Zf, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    report = {
        "rows": {"fit": int(train.sum()), "val": int(test.sum()),
                 "fit_K2": int(ytr.sum()), "fit_K3": int((ytr == 0).sum()),
                 "val_K2": int(yte.sum()), "val_K3": int((yte == 0).sum())},
        "feature_stats": {name: {"fit": stats_by_class(ytr, Xf[train, j]),
                                  "val": stats_by_class(yte, Xv[test, j])}
                          for j, name in enumerate(FEATURES)},
        "univariate": uni,
        "joint": {"val_auc": auc(yte, p[test]),
                  "coefficients_standardized": dict(zip(FEATURES, state["coef"])),
                  "intercept": state["intercept"],
                  "score_stats": stats_by_class(yte, predictions["val_score"][test])},
        "correlation": {"fit_all": corr(Xf[train, 0], Xf[train, 1]),
                        "val_all": corr(Xv[test, 0], Xv[test, 1]),
                        **{f"{split}_K{k}": corr(X[y == k, 0], X[y == k, 1])
                           for split, X, y in (("fit", Xf, yf), ("val", Xv, yv)) for k in (2, 3)}},
        "standardized_fit_covariance": cov.tolist(),
        "standardized_fit_condition_number": float(eig.max() / max(float(eig.min()), 1e-12)),
        "register_cuts_hz": {"low_upper": float(cuts[0]), "high_lower": float(cuts[1])},
        "register_definition": "tertiles of finite positive median_triplet_f0 on FIT K2/K3; estimated, not annotated pitch",
        "global_action": action_metrics(yv, p),
        "global_stratified": parts, "register_refit": refits,
        "register_counts_additive": True,
    }
    return report, state, predictions


def extracted_matrix(rows):
    """Preserve all candidate rows; explicitly mark unavailable/nonfinite residuals."""
    X = np.full((len(rows), 2), np.nan)
    f0 = np.full(len(rows), np.nan)
    reasons = np.full(len(rows), "", dtype="U32")
    for i, row in enumerate(rows):
        if row is None:
            reasons[i] = "insufficient_f0_candidates"
            continue
        require(all(k in row for k in (*FEATURES, "median_triplet_f0")), "feature schema drift")
        X[i] = [row[k] for k in FEATURES]
        f0[i] = row["median_triplet_f0"]
        if not np.isfinite(X[i]).all():
            reasons[i] = "nonfinite_residual"
    return X, f0, reasons == "", reasons


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_identity(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(json.dumps({"dtype": a.dtype.str, "shape": a.shape}, sort_keys=True).encode())
    h.update(a.tobytes())
    return {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": h.hexdigest()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def check_partitions(arrays, fold):
    require(fold in FOLDS, "forbidden validation fold")
    fit_folds = [x for x in FOLDS if x != fold]
    require(np.isin(arrays["fit_fold"], fit_folds).all(), "forbidden FIT fold")
    require(np.all(arrays["val_fold"] == fold), "wrong VAL fold")
    for split in ("fit", "val"):
        ids = arrays[split + "_ids"]
        require(len(np.unique(ids)) == len(ids), "duplicate row IDs")
        require(len(ids) == len(arrays[split + "_fold"]), "fold/ID length mismatch")
    require(not np.intersect1d(arrays["fit_ids"], arrays["val_ids"]).size, "FIT/VAL row overlap")
    require(not np.intersect1d(arrays["fit_recording"], arrays["val_recording"]).size,
            "FIT/VAL recording overlap")


def verify_export(root):
    """Replay the frozen scaler/LR and action counts without fitting or loading audio."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["schema_version"] == SCHEMA_VERSION, "unsupported export schema")
    for name in ("rows.jsonl", "replay.npz", "model.json", "report.json", "report.md"):
        require(sha256_file(root / name) == manifest["files"][name], "file checksum changed: " + name)
    with np.load(root / "replay.npz", allow_pickle=False) as z:
        arrays = dict(z)
    require(set(arrays) == set(manifest["arrays"]), "array schema changed")
    for name, value in arrays.items():
        require(array_identity(value) == manifest["arrays"][name], "array checksum changed: " + name)
    report = json.loads((root / "report.json").read_text())
    state = json.loads((root / "model.json").read_text())
    require(report["schema_version"] == SCHEMA_VERSION, "unsupported report schema")
    fold = report["protocol"]["validation_fold"]
    check_partitions(arrays, fold)
    cuts = [report["register_cuts_hz"][k] for k in ("low_upper", "high_lower")]
    for split in ("fit", "val"):
        action = arrays[split + "_b_low"] & (arrays[split + "_base_k"] == 3)
        valid = arrays[split + "_valid"]
        require(int(action.sum()) == len(valid), "action population mismatch")
        require(np.array_equal(arrays[split + "_action_ids"], arrays[split + "_ids"][action]),
                "action row identity mismatch")
        require(np.array_equal(valid, arrays[split + "_excluded_reason"] == ""), "exclusion mismatch")
        require(np.isfinite(arrays[split + "_X"][valid]).all(), "nonfinite valid features")
        score, probability = score_from_state(arrays[split + "_X"][valid], state)
        require(np.allclose(score, arrays[split + "_score"], rtol=0, atol=1e-12), "score replay mismatch")
        require(np.allclose(probability, arrays[split + "_probability"], rtol=0, atol=1e-12), "probability replay mismatch")
        # Decisions must agree exactly, even if a score is very close to zero.
        require(np.array_equal(probability >= .5, arrays[split + "_probability"] >= .5), "action replay mismatch")
        labels = register_labels(arrays[split + "_f0"][valid], cuts)
        require(np.array_equal(labels, arrays[split + "_register"]), "register replay mismatch")
    action = arrays["val_b_low"] & (arrays["val_base_k"] == 3)
    yv = arrays["val_true_k"][action][arrays["val_valid"]]
    p = arrays["val_probability"]
    require(action_metrics(yv, p) == report["global_action"], "global action metrics mismatch")
    parts = stratify(yv, p, arrays["val_register"])
    for name, part in parts.items():
        require(all(value == report["global_stratified"][name][key] for key, value in part.items()),
                "stratified metrics mismatch: " + name)
    return {"status": "verified", "validation_fold": fold,
            "valid_val_rows": len(yv), "global_net": report["global_action"]["global_net"],
            "outer_fold_3_used": False, "refit": False}
