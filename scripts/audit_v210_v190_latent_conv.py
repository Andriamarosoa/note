"""V21 audit: what does the useful V19 23x64 conv latent actually encode?

No V19 retraining. For each canonical outer fold we load the saved V19 model,
freeze it, fit deterministic ridge probes only on that fold's final-fit rows, and
evaluate only on its outer rows. The probes test:
  * cardinality K=0..6 from compact conv3 vs raw-spectral descriptors;
  * player/style-prefix and spectral-energy separability (nuisance/context checks);
  * local birth-cell information in conv3 embeddings vs raw 3x3 spectral patches,
    using both random negatives and the V19 center head's own hard negatives.
Locked12 is never indexed or evaluated.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v190_dense_birth_centers as v190

BATCH = 96
RIDGE = 1.0
MAX_TRAIN_PER_K = 5000


def one(root: Path, pattern: str) -> Path:
    xs = sorted(root.glob(pattern))
    if len(xs) != 1:
        raise RuntimeError(f"expected one {pattern}, got {len(xs)}")
    return xs[0]


def auc_binary(y, score):
    y = np.asarray(y, dtype=np.int8).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    pos = int(np.sum(y == 1)); neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and s[order[j]] == s[order[i]]:
            j += 1
        rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = rank
        i = j
    sum_pos = float(np.sum(ranks[y == 1]))
    return float((sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def standardize_fit(x):
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std > 1e-7, std, 1.0)
    return mean, std


def ridge_multiclass_fit(x, y, classes):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y)
    mean, std = standardize_fit(x)
    z = (x - mean) / std
    z = np.concatenate([z, np.ones((len(z), 1), dtype=np.float64)], axis=1)
    class_to_col = {c: i for i, c in enumerate(classes)}
    cols = np.asarray([class_to_col[v] for v in y], dtype=np.int32)
    counts = np.bincount(cols, minlength=len(classes)).astype(np.float64)
    w = np.asarray([len(y) / (len(classes) * max(1.0, counts[c])) for c in cols], dtype=np.float64)
    yy = np.eye(len(classes), dtype=np.float64)[cols]
    a = z.T @ (w[:, None] * z)
    reg = np.eye(a.shape[0], dtype=np.float64) * RIDGE
    reg[-1, -1] = 0.0
    b = z.T @ (w[:, None] * yy)
    coef = np.linalg.solve(a + reg, b)
    return {"mean": mean, "std": std, "coef": coef, "classes": np.asarray(classes)}


def ridge_multiclass_predict(model, x):
    z = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["std"]
    z = np.concatenate([z, np.ones((len(z), 1), dtype=np.float64)], axis=1)
    score = z @ model["coef"]
    return model["classes"][np.argmax(score, axis=1)], score


def ridge_binary_fit(x, y):
    return ridge_multiclass_fit(x, np.asarray(y, dtype=np.int32), [0, 1])


def ridge_binary_score(model, x):
    _, score = ridge_multiclass_predict(model, x)
    return score[:, 1] - score[:, 0]


def ridge_regression_fit(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    mean, std = standardize_fit(x)
    z = (x - mean) / std
    z = np.concatenate([z, np.ones((len(z), 1), dtype=np.float64)], axis=1)
    a = z.T @ z
    reg = np.eye(a.shape[0], dtype=np.float64) * RIDGE
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(a + reg, z.T @ y)
    return {"mean": mean, "std": std, "coef": coef}


def ridge_regression_predict(model, x):
    z = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["std"]
    z = np.concatenate([z, np.ones((len(z), 1), dtype=np.float64)], axis=1)
    return (z @ model["coef"]).reshape(-1)


def card_metrics(y, pred):
    y = np.asarray(y, dtype=np.int32); pred = np.asarray(pred, dtype=np.int32)
    per = {}
    vals = []
    for k in range(7):
        m = y == k
        acc = float(np.mean(pred[m] == y[m])) if np.any(m) else None
        per[str(k)] = {"rows": int(np.sum(m)), "exact": acc}
        if acc is not None:
            vals.append(acc)
    poly = y >= 2
    return {
        "exact": float(np.mean(pred == y)),
        "balanced_exact": float(np.mean(vals)),
        "poly_exact": float(np.mean(pred[poly] == y[poly])) if np.any(poly) else None,
        "mae": float(np.mean(np.abs(pred - y))),
        "per_true_k": per,
    }


def classification_metrics(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    classes = sorted(set(y.tolist()))
    per = []
    for c in classes:
        m = y == c
        per.append(float(np.mean(pred[m] == y[m])))
    return {"accuracy": float(np.mean(pred == y)), "balanced_accuracy": float(np.mean(per)), "classes": classes}


def r2(y, pred):
    y = np.asarray(y, dtype=np.float64); pred = np.asarray(pred, dtype=np.float64)
    den = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - np.sum((y - pred) ** 2) / den) if den > 0 else 0.0


def conv_descriptor(feat):
    x = np.asarray(feat, dtype=np.float32)
    return np.concatenate([np.mean(x, axis=(1, 2)), np.max(x, axis=(1, 2)), np.std(x, axis=(1, 2))], axis=1).astype(np.float32)


def raw_descriptor(spec):
    x = np.asarray(spec, dtype=np.float32)
    global_stats = np.concatenate([np.mean(x, axis=(1, 2)), np.max(x, axis=(1, 2)), np.std(x, axis=(1, 2))], axis=1)
    time_profile = np.mean(x, axis=2).reshape((len(x), -1))
    freq_profile = np.mean(x, axis=1).reshape((len(x), -1))
    return np.concatenate([global_stats, time_profile, freq_profile], axis=1).astype(np.float32)


def raw_patch(spec_row, flat):
    t = int(flat // v100.SPECTRAL_BANDS); f = int(flat % v100.SPECTRAL_BANDS)
    out = np.zeros((3, 3, spec_row.shape[-1]), dtype=np.float32)
    for dt in range(-1, 2):
        for df in range(-1, 2):
            tt, ff = t + dt, f + df
            if 0 <= tt < v100.TIME_FRAMES and 0 <= ff < v100.SPECTRAL_BANDS:
                out[dt + 1, df + 1] = spec_row[tt, ff]
    return out.reshape(-1)


def far_mask(truth):
    mask = np.ones(v190.CENTER_CELLS, dtype=bool)
    for flat in truth:
        t = int(flat // v100.SPECTRAL_BANDS); f = int(flat % v100.SPECTRAL_BANDS)
        for dt in range(-1, 2):
            for df in range(-2, 3):
                tt, ff = t + dt, f + df
                if 0 <= tt < v100.TIME_FRAMES and 0 <= ff < v100.SPECTRAL_BANDS:
                    mask[tt * v100.SPECTRAL_BANDS + ff] = False
    return mask


def extract(latent_model, spectral, indices, target, eligible, k):
    idx = np.asarray(indices, dtype=np.int64)
    desc_c, desc_r, energy = [], [], []
    cell_c, cell_r, cell_y, cell_tag, cell_head = [], [], [], [], []
    for s in range(0, len(idx), BATCH):
        rows = idx[s:s + BATCH]
        sb = np.asarray(spectral[rows], dtype=np.float32)
        feat, logits = latent_model.predict(sb, batch_size=BATCH, verbose=0)
        feat = np.asarray(feat, dtype=np.float32)
        logits = np.asarray(logits, dtype=np.float32)
        desc_c.append(conv_descriptor(feat)); desc_r.append(raw_descriptor(sb)); energy.append(np.mean(sb[:, :, :, 0], axis=(1, 2)))
        flat_feat = feat.reshape((len(rows), v190.CENTER_CELLS, feat.shape[-1]))
        for bi, row in enumerate(rows):
            if not eligible[row] or int(k[row]) <= 0:
                continue
            truth = np.flatnonzero(target[row] > 0)
            if len(truth) == 0:
                continue
            fm = far_mask(truth)
            pool = np.flatnonzero(fm)
            if len(pool) < len(truth):
                continue
            hard = pool[np.argsort(logits[bi, pool])[-len(truth):]]
            rng = np.random.default_rng(9100003 + int(row))
            rand = rng.choice(pool, size=len(truth), replace=False)
            for flat in truth:
                cell_c.append(flat_feat[bi, flat]); cell_r.append(raw_patch(sb[bi], int(flat))); cell_y.append(1); cell_tag.append(0); cell_head.append(float(logits[bi, flat]))
            for flat in hard:
                cell_c.append(flat_feat[bi, flat]); cell_r.append(raw_patch(sb[bi], int(flat))); cell_y.append(0); cell_tag.append(1); cell_head.append(float(logits[bi, flat]))
            for flat in rand:
                cell_c.append(flat_feat[bi, flat]); cell_r.append(raw_patch(sb[bi], int(flat))); cell_y.append(0); cell_tag.append(2); cell_head.append(float(logits[bi, flat]))
    return {
        "conv": np.concatenate(desc_c, axis=0),
        "raw": np.concatenate(desc_r, axis=0),
        "energy": np.concatenate(energy, axis=0),
        "cell_conv": np.asarray(cell_c, dtype=np.float32),
        "cell_raw": np.asarray(cell_r, dtype=np.float32),
        "cell_y": np.asarray(cell_y, dtype=np.int8),
        "cell_tag": np.asarray(cell_tag, dtype=np.int8),
        "cell_head": np.asarray(cell_head, dtype=np.float32),
    }


def balanced_train_sample(fit_idx, k, fold):
    rng = np.random.default_rng(16061 + 97 * fold)
    out = []
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    for value in range(7):
        pool = fit_idx[np.asarray(k)[fit_idx] == value]
        if len(pool) > MAX_TRAIN_PER_K:
            pool = rng.choice(pool, size=MAX_TRAIN_PER_K, replace=False)
        out.append(np.asarray(pool, dtype=np.int64))
    return np.sort(np.concatenate(out))


def label_info(members):
    players, styles = [], []
    for m in members:
        stem = Path(str(m)).stem
        bits = stem.split("_")
        players.append(bits[0])
        token = bits[1] if len(bits) > 1 else "unknown"
        mm = re.match(r"[A-Za-z]+", token)
        styles.append(mm.group(0) if mm else token)
    return np.asarray(players), np.asarray(styles)


def cell_auc_pack(y, tag, conv_score, raw_score, head_score):
    y = np.asarray(y); tag = np.asarray(tag)
    out = {}
    for name, mask in {
        "all": np.ones(len(y), dtype=bool),
        "hard": tag != 2,
        "random": tag != 1,
    }.items():
        out[name] = {
            "rows": int(np.sum(mask)),
            "conv_probe_auc": auc_binary(y[mask], conv_score[mask]),
            "raw_patch_probe_auc": auc_binary(y[mask], raw_score[mask]),
            "v19_center_head_auc": auc_binary(y[mask], head_score[mask]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--v190-fold-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = v100._load_spectral_caches(args.cache_dir)
    spectral = np.asarray(cache["spectral"])
    members = np.asarray([str(x) for x in cache["members"]], dtype="U96")
    candidates, recon = v102._reconstruct_candidates(cache)
    pitch, tm, td, ts, supervision = v102._derive_supervision(members, candidates, args.dataset_dir, expected_slot_targets=cache["slot_targets"])
    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), 6)
    target, eligf, distinct = v190._birth_center_targets(cache, pitch, td, k)
    eligible = np.asarray(eligf) > 0.5
    players, styles = label_info(members)
    player_classes = sorted(set(players.tolist())); style_classes = sorted(set(styles.tolist()))

    import tensorflow as tf
    from tensorflow import keras

    folds = []
    pooled = {name: [] for name in ["k", "actual", "conv", "raw", "player", "player_conv", "player_raw", "style", "style_conv", "style_raw", "energy", "energy_conv", "energy_raw"]}
    cell_fold_metrics = []

    for fold in range(5):
        ctx = v172._fold_context(SimpleNamespace(dataset_dir=args.dataset_dir, cache_dir=args.cache_dir, outer_fold=fold))
        model, _, _ = v190._build_model(ctx["final_spec"])
        model.load_weights(one(args.v190_fold_dir, f"**/v190-dense-birth-centers-fold-{fold}.weights.h5"))
        spectral_input = next(t for t in model.inputs if t.name.split(":", 1)[0] == "spectral_map")
        latent = keras.Model(spectral_input, [model.get_layer("v190_dense_conv3").output, model.get_layer("v190_birth_center_logits_flat").output])
        train_idx = balanced_train_sample(ctx["final_fit_idx"], k, fold)
        outer_idx = np.asarray(ctx["outer_idx"], dtype=np.int64)
        tr = extract(latent, spectral, train_idx, target, eligible, k)
        te = extract(latent, spectral, outer_idx, target, eligible, k)

        card_conv = ridge_multiclass_fit(tr["conv"], k[train_idx], list(range(7)))
        card_raw = ridge_multiclass_fit(tr["raw"], k[train_idx], list(range(7)))
        pred_conv, _ = ridge_multiclass_predict(card_conv, te["conv"])
        pred_raw, _ = ridge_multiclass_predict(card_raw, te["raw"])

        player_conv = ridge_multiclass_fit(tr["conv"], players[train_idx], player_classes)
        player_raw = ridge_multiclass_fit(tr["raw"], players[train_idx], player_classes)
        pp_conv, _ = ridge_multiclass_predict(player_conv, te["conv"]); pp_raw, _ = ridge_multiclass_predict(player_raw, te["raw"])
        style_conv = ridge_multiclass_fit(tr["conv"], styles[train_idx], style_classes)
        style_raw = ridge_multiclass_fit(tr["raw"], styles[train_idx], style_classes)
        ps_conv, _ = ridge_multiclass_predict(style_conv, te["conv"]); ps_raw, _ = ridge_multiclass_predict(style_raw, te["raw"])
        energy_conv = ridge_regression_fit(tr["conv"], tr["energy"]); energy_raw = ridge_regression_fit(tr["raw"], tr["energy"])
        pe_conv = ridge_regression_predict(energy_conv, te["conv"]); pe_raw = ridge_regression_predict(energy_raw, te["raw"])

        cb = ridge_binary_fit(tr["cell_conv"], tr["cell_y"])
        rb = ridge_binary_fit(tr["cell_raw"], tr["cell_y"])
        cscore = ridge_binary_score(cb, te["cell_conv"]); rscore = ridge_binary_score(rb, te["cell_raw"])
        cell_metrics = cell_auc_pack(te["cell_y"], te["cell_tag"], cscore, rscore, te["cell_head"])
        cell_fold_metrics.append(cell_metrics)

        with np.load(one(args.v190_fold_dir, f"**/predictions-fold-{fold}.npz"), allow_pickle=False) as z:
            presence = np.asarray(z["presence"], dtype=np.float32)
        actual_k = np.sum(presence >= 0.5, axis=1).astype(np.int32)
        if len(actual_k) != len(outer_idx):
            raise RuntimeError(f"fold {fold}: prediction/outer length mismatch")
        rr = json.loads(one(args.v190_fold_dir, f"**/report-fold-{fold}.json").read_text())
        true = k[outer_idx]
        wrong = actual_k != true
        fold_result = {
            "fold": fold,
            "train_probe_rows": int(len(train_idx)),
            "outer_rows": int(len(outer_idx)),
            "v190_f1": float(rr["strata"]["aggregate"][v190.MODEL_KEY]["metrics"]["global"]["f1"]),
            "actual_v19_cardinality": card_metrics(true, actual_k),
            "conv_cardinality_probe": card_metrics(true, pred_conv),
            "raw_cardinality_probe": card_metrics(true, pred_raw),
            "conv_probe_recovers_actual_v19_count_errors": float(np.mean(pred_conv[wrong] == true[wrong])) if np.any(wrong) else None,
            "conv_probe_breaks_actual_v19_correct_counts": float(np.mean(pred_conv[~wrong] != true[~wrong])) if np.any(~wrong) else None,
            "player_conv": classification_metrics(players[outer_idx], pp_conv),
            "player_raw": classification_metrics(players[outer_idx], pp_raw),
            "style_prefix_conv": classification_metrics(styles[outer_idx], ps_conv),
            "style_prefix_raw": classification_metrics(styles[outer_idx], ps_raw),
            "energy_r2_conv": r2(te["energy"], pe_conv),
            "energy_r2_raw": r2(te["energy"], pe_raw),
            "cell_birth": cell_metrics,
        }
        folds.append(fold_result)

        for name, arr in [
            ("k", true), ("actual", actual_k), ("conv", pred_conv), ("raw", pred_raw),
            ("player", players[outer_idx]), ("player_conv", pp_conv), ("player_raw", pp_raw),
            ("style", styles[outer_idx]), ("style_conv", ps_conv), ("style_raw", ps_raw),
            ("energy", te["energy"]), ("energy_conv", pe_conv), ("energy_raw", pe_raw),
        ]:
            pooled[name].append(np.asarray(arr))
        print(json.dumps(fold_result, sort_keys=True))
        tf.keras.backend.clear_session()

    pooled = {k0: np.concatenate(v, axis=0) for k0, v in pooled.items()}
    if len(pooled["k"]) != len(k):
        raise RuntimeError(f"outer coverage {len(pooled['k'])} != {len(k)}")
    mean_cell = {}
    for subset in ("all", "hard", "random"):
        mean_cell[subset] = {}
        for metric in ("conv_probe_auc", "raw_patch_probe_auc", "v19_center_head_auc"):
            mean_cell[subset][metric] = float(np.mean([f[subset][metric] for f in cell_fold_metrics]))

    actual_wrong = pooled["actual"] != pooled["k"]
    result = {
        "schema_version": 1,
        "protocol": {
            "v210_v190_latent_conv_audit": True,
            "outer_clean_rows": int(len(k)),
            "same_five_saved_v190_models": True,
            "v190_retrained": False,
            "probe_training_rows_are_final_fit_only": True,
            "probe_evaluation_rows_are_outer_only": True,
            "probe_hyperparameter_sweep": False,
            "ridge_lambda": RIDGE,
            "max_train_rows_per_k": MAX_TRAIN_PER_K,
            "runtime_annotations_required": False,
            "locked12_indexed_or_evaluated": False,
        },
        "representation": {
            "conv_layer": "v190_dense_conv3",
            "conv_cell_dim": 96,
            "conv_cardinality_descriptor_dim": int(folds and 288),
            "raw_descriptor_dim": 270,
            "raw_cell_patch_dim": 27,
            "grid": [int(v100.TIME_FRAMES), int(v100.SPECTRAL_BANDS)],
        },
        "labels": {
            "player_classes": player_classes,
            "filename_style_prefix_classes": style_classes,
            "eligible_birth_rows": int(np.sum(eligible & (k > 0))),
            "representable_distinct_center_poly_rate": float(np.mean(distinct[eligible & (k >= 2)] == k[eligible & (k >= 2)])),
        },
        "pooled_outer": {
            "actual_v19_cardinality": card_metrics(pooled["k"], pooled["actual"]),
            "conv_cardinality_probe": card_metrics(pooled["k"], pooled["conv"]),
            "raw_cardinality_probe": card_metrics(pooled["k"], pooled["raw"]),
            "conv_probe_recovers_actual_v19_count_errors": float(np.mean(pooled["conv"][actual_wrong] == pooled["k"][actual_wrong])) if np.any(actual_wrong) else None,
            "conv_probe_breaks_actual_v19_correct_counts": float(np.mean(pooled["conv"][~actual_wrong] != pooled["k"][~actual_wrong])) if np.any(~actual_wrong) else None,
            "player_conv": classification_metrics(pooled["player"], pooled["player_conv"]),
            "player_raw": classification_metrics(pooled["player"], pooled["player_raw"]),
            "style_prefix_conv": classification_metrics(pooled["style"], pooled["style_conv"]),
            "style_prefix_raw": classification_metrics(pooled["style"], pooled["style_raw"]),
            "energy_r2_conv": r2(pooled["energy"], pooled["energy_conv"]),
            "energy_r2_raw": r2(pooled["energy"], pooled["energy_raw"]),
            "cell_birth_mean_fold_auc": mean_cell,
        },
        "folds": folds,
        "supervision": {"source": supervision, "candidate_reconstruction": recon},
    }
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["pooled_outer"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
