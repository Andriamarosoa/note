"""Verified recovered S5 parents and unchanged S8/S9/S13..S18 checkpoints.

Two runtimes: stage s5 uses sklearn 1.8; stage s18 uses sklearn 1.4.2.
New pieces use four separate fold branches. Within a branch, piece-held-out
risk checkpoints are averaged. This new deployment rule is explicit; no
GOAT labels are read and no full S80 or independent accuracy is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

import joblib
import numpy as np

FOLDS = (0, 1, 2, 4)
SOURCE_COMMIT = "7165e1ef54510aa9effc9f597b2783c832ca907a"
GLOBAL = "open_k0k6_series5__prior_corrected__tilt1.5__freeze__mix1"
POLY = "open_k0k6_series5__raw__tilt1.5__freeze__mix1"
POLICY_SHA = "aad15473f3abbb510e1082643c77a0fbdddb288af204c1c69b807e760f614099"
INPUT_SHA = {
    "s97": "949c739f10f9cd4c3743a176b2e2c413e1be2bc99053508f8aaed364a29cdd2d",
    "s116": "0a78175a37e36c6dba2cc9310cdd400ac2c90f48f68cf886ac964e56e36234ec",
}
BUNDLES = (
    ("v273-original-S8-weights-S157-S158-GOAT-proof",
     "v273-S8-original-weights-S157-S158-evidence.tar.gz",
     "26f12d96ef1e24efbf12ab785082bc642edf1597edcc47c12c7acdc39717c1d9"),
    ("v273-original-S13-S14-trained-S153-S154-cascade-archive",
     "v273-S13-S14-original-trained-models-and-S153-S154-evidence.tar.gz",
     "d8c4ec5f54967058b83185a8516a826f4ca7cc19005b60951a1d8147bfef4bb9"),
    ("v273-original-S11-S16-S17-S18-weights-and-GOAT-s149-s150",
     "v273-S11-S16-S17-S18-original-weights-and-S149-S150-evidence.tar.gz",
     "d2e1b80c3656ed0cc2c708dba150293b9b66aff23358542ea110be6b6ed072bc"),
)
SELECTED = {
    "S8": "series8__poly_parent__votes__C0.3__cost0.5",
    "S13": "series13__flow_logistic__p0_gt0.975",
    "S14": "series14__flow_mean__p0_gt0.9",
    "S16": "series16__flow_logistic__k23__pbase_gt0.99",
    "S17": "series17__flow_logistic__k32__pbase_gt0.99",
    "S18": "series18__flow_logistic__k21__pbase_gt0.95",
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as source:
            data = source.read()
        need(hashlib.sha256(data).hexdigest() == expected, "download SHA: " + path.name)
        path.write_bytes(data)
    need(sha(path) == expected, "source SHA: " + path.name)


def read(path, keys, strict=False):
    with np.load(path, allow_pickle=False) as z:
        if strict:
            need(set(z.files) == set(keys), "unexpected runtime input fields: " + str(path))
        return {k: z[k].copy() for k in keys}


def policy_data(path):
    return read(path, ("global_index", "variant_ids", "predictions"))


def selected(archive, name, ids):
    need(np.array_equal(archive["global_index"], ids), "historical policy IDs changed")
    names = list(map(str, archive["variant_ids"]))
    need(name in names, "missing original policy " + name)
    return archive["predictions"][:, names.index(name)].astype(np.int8)


def tilt(prob):
    q = np.asarray(prob, np.float64).copy()
    q[..., 2:] *= 1.5
    return q.argmax(axis=-1).astype(np.int8)


def inputs(s97, s116):
    need(sha(s97) == INPUT_SHA["s97"] and sha(s116) == INPUT_SHA["s116"],
         "pinned GOAT acoustic input mismatch")
    a = read(s97, ("row_id", "summary58", "flow245", "geometry3", "sequence42x49",
                   "morphology490", "physics294", "member", "start_samples", "end_samples"), True)
    b = read(s116, ("row_id", "control111", "freeze_K", "geometry46"), True)
    need(np.array_equal(a["row_id"], b["row_id"]), "external event ID mismatch")
    need(np.array_equal(a["summary58"], b["control111"][:, :58]), "58D input mismatch")
    need(np.array_equal(b["geometry46"], b["control111"][:, 58:104]), "ownership mismatch")
    need(len(a["row_id"]) == len(set(a["row_id"])) == 57, "GOAT event coverage")
    return a, b


def stage_s5(args):
    import sklearn
    from scripts.verify_v273_s5_owned527_recovery import fetch_fold, verify, MODEL_DIGEST
    from scripts.v273_s4_owned527 import build_owned527

    need(sklearn.__version__ == "1.8.0", "original S5 runtime must use sklearn 1.8.0")
    if args.fetch:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda f: fetch_fold(args.cache, f), FOLDS))
    args.out.mkdir(parents=True)
    historical = verify(args.cache, args.out / "historical_constructor")
    policy_path = args.cache / "original-s5-policies.npz"
    fetch(f"https://raw.githubusercontent.com/Andriamarosoa/note/{SOURCE_COMMIT}/"
          "analysis/evidence/v273-open-k0-k6/series5/predictions.npz", policy_path, POLICY_SHA)
    archive = policy_data(policy_path)
    ids = archive["global_index"]
    lookup = {int(value): i for i, value in enumerate(ids)}
    g, p = np.full(len(ids), -1, np.int8), np.full(len(ids), -1, np.int8)
    folds = np.full(len(ids), -1, np.int8)
    a, b = inputs(args.s97, args.s116)
    design = build_owned527(a["summary58"], b["geometry46"], a["sequence42x49"], b["freeze_K"])
    need(np.array_equal(design[:, :111], b["control111"]), "new 111D control parity")
    raw, prior, evidence = [], [], []
    for fold in FOLDS:
        directory = args.cache / "original-s5" / f"fold{fold}"
        state = joblib.load(directory / "model.joblib")
        z = read(directory / "predictions.npz", ("global_index", "raw", "prior_corrected"))
        at = np.asarray([lookup[int(i)] for i in z["global_index"]])
        g[at], p[at], folds[at] = tilt(z["prior_corrected"]), tilt(z["raw"]), fold
        q = state["model"].predict_proba(design)
        c = q / np.asarray(state["class_weights"])[None, :]
        c /= c.sum(axis=1, keepdims=True)
        raw.append(q); prior.append(c)
        evidence.append({"fold": fold, "model_sha256": MODEL_DIGEST[fold],
                         "global_K_histogram": np.bincount(tilt(c), minlength=7).tolist(),
                         "poly_K_histogram": np.bincount(tilt(q), minlength=7).tolist()})
    need(np.array_equal(g, selected(archive, GLOBAL, ids)), "historical global policy mismatch")
    need(np.array_equal(p, selected(archive, POLY, ids)), "historical poly policy mismatch")
    raw, prior = np.stack(raw), np.stack(prior)
    np.savez_compressed(args.out / "historical_s5_parents.npz", global_index=ids, fold=folds,
                        global_K=g, poly_K=p)
    np.savez_compressed(args.out / "external_s5.npz", row_id=a["row_id"], freeze_K=b["freeze_K"],
                        fold_ids=np.asarray(FOLDS), raw=raw, prior_corrected=prior,
                        global_K=tilt(prior), poly_K=tilt(raw))
    report = {"status": "ORIGINAL_S5_FOUR_MODELS_NEW_AUDIO_FORWARD_COMPLETE",
              "external_events": 57, "historical_rows": len(ids),
              "historical_parent_decisions_checked": 2 * len(ids),
              "historical_parent_mismatches": 0, "checkpoints": evidence,
              "external_design_shape": list(design.shape),
              "external_design_sha256": hashlib.sha256(design.tobytes()).hexdigest(),
              "external_predictions_sha256": sha(args.out / "external_s5.npz"),
              "source_code_original_text_recovered": False,
              "historical_constructor_exact": historical["full_historical_probability_equivalence"],
              "truth_labels_read": False, "models_retrained": False,
              "independent_accuracy_measured": False, "S80_promoted": False}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def prepare_models(cache):
    directories = {}
    for index, (tag, name, expected) in enumerate(BUNDLES):
        archive = cache / name
        fetch(f"https://github.com/Andriamarosoa/note/releases/download/{tag}/{name}", archive, expected)
        destination = cache / f"source{index}"
        destination.mkdir(exist_ok=True)
        with tarfile.open(archive) as source:
            members = [m for m in source.getmembers() if m.isfile()
                       and m.name.startswith("original/")
                       and ("/models/" in m.name or m.name.endswith("/predictions.npz"))]
            source.extractall(destination, members=members, filter="data")
        for stage in (destination / "original").iterdir():
            directories[stage.name] = stage
    return directories


def historic_acoustic(cache, ids):
    from scripts.loop_v273_birth_flow import flow_profile
    lookup = {int(value): i for i, value in enumerate(ids)}
    summary = np.empty((len(ids), 58), np.float32)
    flow = np.empty((len(ids), 245), np.float32)
    geometry = np.empty((len(ids), 3), np.float32)
    baseline = np.full(len(ids), -1, np.int8)
    for fold in FOLDS:
        z = read(cache / "features" / f"fold{fold}" / f"features-fold-{fold}.npz",
                 ("global_index", "member", "starts", "ends", "summary", "sequence", "baseline"))
        at = np.asarray([lookup[int(i)] for i in z["global_index"]])
        summary[at], flow[at], baseline[at] = z["summary"], flow_profile(z["sequence"]), z["baseline"]
        last = np.empty(len(at), np.float32)
        for member in np.unique(z["member"]):
            rows = np.flatnonzero(z["member"] == member)
            rows = rows[np.argsort(z["starts"][rows])]
            last[rows[0]] = 2
            last[rows[1:]] = np.clip(np.diff(z["starts"][rows]) / 44100., 0, 2)
        geometry[at] = np.column_stack(((z["ends"] - z["starts"]) / 1764.,
                                        np.log1p(last), np.minimum(last, .2) / .2))
    need(np.isin(baseline, range(7)).all(), "historical acoustic coverage incomplete")
    return summary, flow, geometry, baseline


def acoustic335(summary, flow, geometry, b, g, p):
    from scripts.loop_v273_native_risk import designs
    n = len(b)
    votes = designs({"native_position": np.arange(n), "global_index": np.arange(n),
                     "features": summary}, np.arange(n), b, g, p, geometry)["votes_time"]
    return np.column_stack((votes, summary, flow)).astype(np.float32)


def score(model, scaler, x, advantage=False):
    q = model.predict_proba(np.clip(scaler.transform(x), -6, 6))
    classes = list(map(int, model.classes_))
    need(1 in classes, "original positive target class missing")
    result = q[:, classes.index(1)]
    if advantage and 0 in classes:
        result = result - q[:, classes.index(0)]
    need(np.isfinite(result).all(), "invalid saved classifier output")
    return result


def decode(stage, b, g, p, parent, q):
    if stage == "S8":
        mask = np.isin(b, (2, 3, 4)) & (g != p) & (q < -.5)
        return np.where(mask, g, p).astype(np.int8)
    if stage == "S13":
        mask = (b == 0) & (parent != b) & (q > .975)
    elif stage == "S14":
        mask = (b == 0) & (parent == 1) & (q > .9)
    else:
        source, dest, threshold = {"S16": (2, 3, .99), "S17": (3, 2, .99),
                                    "S18": (2, 1, .95)}[stage]
        mask = (b == source) & (parent == dest) & (q > threshold)
    return np.where(mask, b, parent).astype(np.int8)


def stage_s18(args):
    import sklearn
    from scripts.loop_v273_learned_risk import design as vote_design
    need(sklearn.__version__ == "1.4.2", "original S8..S18 runtime must use sklearn 1.4.2")
    models = prepare_models(args.cache)
    hist = read(args.s5 / "historical_s5_parents.npz", ("global_index", "fold", "global_K", "poly_K"), True)
    ext = read(args.s5 / "external_s5.npz", ("row_id", "freeze_K", "fold_ids", "raw",
                                          "prior_corrected", "global_K", "poly_K"), True)
    a, control = inputs(args.s97, args.s116)
    need(np.array_equal(ext["row_id"], a["row_id"]), "sealed S5 external alignment")
    ids = hist["global_index"]; lookup = {int(v): i for i, v in enumerate(ids)}
    summary, flow, geom, b = historic_acoustic(args.cache, ids)
    g, p = hist["global_K"], hist["poly_K"]
    history_x = acoustic335(summary, flow, geom, b, g, p)
    hx8 = vote_design({"features": summary}, b, g, p)["votes"]
    xb = ext["freeze_K"]; xg, xp = ext["global_K"], ext["poly_K"]
    external_x = [acoustic335(a["summary58"], a["flow245"], a["geometry3"], xb, xg[i], xp[i])
                  for i in range(4)]
    ex8 = [vote_design({"features": a["summary58"]}, xb, xg[i], xp[i])["votes"] for i in range(4)]
    history_parent = p.copy(); external_parent = xp.copy()
    history_outputs = {}; external_outputs = {}; evidence = []; model_proof = []
    patterns = {"S8": ("*__votes__C0.3.joblib",),
                "S13": ("*__flow_logistic.joblib",),
                "S14": ("*__flow_logistic.joblib", "*__flow_hgb.joblib"),
                "S16": ("k23__*__flow_logistic.joblib",),
                "S17": ("k32__*__flow_logistic.joblib",),
                "S18": ("k21__*__flow_logistic.joblib",)}
    for stage in ("S8", "S13", "S14", "S16", "S17", "S18"):
        historical_heads = []; external_heads = []; counts = []
        for pattern in patterns[stage]:
            files = sorted((models[stage] / "models").glob(pattern))
            need(len(files) == 19, stage + " selected checkpoint count changed")
            hq = np.zeros(len(ids), np.float32); seen = np.zeros(len(ids), bool)
            eq = [[] for _ in FOLDS]
            for file in files:
                state = joblib.load(file)
                fold = int(state["fold"]); fi = FOLDS.index(fold)
                held_key = {"S8": "test_ids", "S13": "held_ids",
                            "S14": "held_global_ids"}.get(stage, "held_ids")
                at = np.asarray([lookup[int(v)] for v in state[held_key]], dtype=np.intp)
                need(not seen[at].any() and np.all(hist["fold"][at] == fold), "checkpoint test fold mismatch")
                seen[at] = True
                hx = hx8 if stage == "S8" else history_x
                xx = ex8[fi] if stage == "S8" else external_x[fi]
                hq[at] = score(state["model"], state["scaler"], hx[at], stage == "S8")
                eq[fi].append(score(state["model"], state["scaler"], xx, stage == "S8"))
                model_proof.append({"stage": stage, "file": file.name, "fold": fold,
                                    "sha256": sha(file), "historical_test_rows": len(at)})
            need(all(eq), "missing original fold-specific risk checkpoints")
            historical_heads.append(hq)
            external_heads.append(np.stack([np.mean(v, axis=0) for v in eq]))
            counts.append([len(v) for v in eq])
        hq = np.mean(historical_heads, axis=0); eq = np.mean(external_heads, axis=0)
        current_history_parent = history_outputs["S9"] if stage in ("S13", "S14") else history_parent
        current_external_parent = external_outputs["S9"] if stage in ("S13", "S14") else external_parent
        hout = decode(stage, b, g, p, current_history_parent, hq)
        eout = np.stack([decode(stage, xb, xg[i], xp[i], current_external_parent[i], eq[i]) for i in range(4)])
        original = policy_data(models[stage] / "predictions.npz")
        expected = selected(original, SELECTED[stage], ids)
        mismatch = int(np.count_nonzero(hout != expected))
        need(mismatch == 0, f"historical {stage} replay mismatch: {mismatch}")
        if stage != "S8":
            parent_name = {"S13": "series9_parent", "S14": "series9_parent",
                           "S16": "series15_parent", "S17": "series16_parent",
                           "S18": "series17_parent"}[stage]
            need(np.array_equal(selected(original, parent_name, ids), current_history_parent),
                 "historical " + stage + " input ancestry mismatch")
        history_outputs[stage], external_outputs[stage] = hout, eout
        evidence.append({"stage": stage, "selected_policy": SELECTED[stage],
                         "historical_rows": len(ids), "historical_K_mismatches": mismatch,
                         "checkpoint_counts_by_arm_and_fold": counts,
                         "external_changed_from_parent_by_fold": (eout != current_external_parent).sum(axis=1).tolist(),
                         "external_K_histograms": [np.bincount(v, minlength=7).tolist() for v in eout]})
        history_parent, external_parent = hout, eout
        if stage == "S8":
            history_parent = np.where((p >= 5) & (g <= 3), p, hout).astype(np.int8)
            external_parent = np.where((xp >= 5) & (xg <= 3), xp, eout).astype(np.int8)
            history_outputs["S9"], external_outputs["S9"] = history_parent.copy(), external_parent.copy()
        elif stage == "S14":
            history_parent = np.where((history_outputs["S13"] != history_outputs["S9"]) |
                                      (hout != history_outputs["S9"]), b, history_outputs["S9"]).astype(np.int8)
            external_parent = np.where((external_outputs["S13"] != external_outputs["S9"]) |
                                       (eout != external_outputs["S9"]), xb[None, :], external_outputs["S9"]).astype(np.int8)
            history_outputs["S15"], external_outputs["S15"] = history_parent.copy(), external_parent.copy()
        print(json.dumps(evidence[-1]), flush=True)
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out / "external_s5_to_s18.npz", row_id=a["row_id"], freeze_K=xb,
                        fold_ids=np.asarray(FOLDS), global_K=xg, poly_K=xp,
                        **{stage + "_K": values for stage, values in external_outputs.items()})
    report = {"status": "RECOVERED_S5_TO_S18_ORIGINAL_CHECKPOINT_CASCADE_VERIFIED",
              "historical_rows": len(ids), "historical_selected_stages_verified": evidence,
              "original_selected_checkpoints_executed": len(model_proof), "checkpoints": model_proof,
              "external_events": 57, "external_branches": list(FOLDS),
              "external_S18_histograms": [np.bincount(v, minlength=7).tolist() for v in external_parent],
              "external_prediction_sha256": sha(args.out / "external_s5_to_s18.npz"),
              "deployment_rule": "Preserve four S5 fold branches; average only same-fold original piece-held risk checkpoints; apply unchanged original selected thresholds and eligibility masks. No final cross-fold decision is selected here.",
              "new_piece_aggregation_is_new_policy": True,
              "S135_or_PR16_substitutes_used": False, "models_retrained": False,
              "truth_labels_read": False, "GOAT_is_already_exposed_smoke_data": True,
              "full_S45_16_families_complete": False, "full_external_S80_executed": False,
              "independent_accuracy_measured": False, "S80_promoted": False}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("checkpoints", "historical_selected_stages_verified")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("s5", "s18"), required=True)
    for name in ("cache", "s97", "s116", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--s5", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    need(not args.out.exists(), "refuse to overwrite existing evidence")
    if args.stage == "s5":
        stage_s5(args)
    else:
        need(args.s5 is not None, "sealed S5 output directory required")
        stage_s18(args)
