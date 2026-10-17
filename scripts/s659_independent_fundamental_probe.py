"""S659: causal independent-fundamental spectral guard on S658 octave rescues.

Research-only: unchanged frozen S614 heads, original K, native NoteOn/Off state.
Only a lower-octave NOTE proposal from the S657 gate can be vetoed.
Pre-registered physical spectrum thresholds. No TEST labels, no DEV tuning.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np

from causal_note.strict_live import SR, Selection, CausalSelection, load_heads
from scripts.s613_prepare_strict_live import download, digest, labels
from scripts.s613_train_strict_live import real_notes, event_score, summary
from scripts.s637_guitarset_three_view_groundtruth import (
    ROOT, MODEL, raw_guitar_audio, GUITARSET_PINNED_CONTROL,
)
from scripts.s638_guitarset_true_overlap import overlap_groups
from scripts.s652_tone_filter_multiview import causal_lowpass, predict_causal_view, FILTERS
from scripts.s657_coincident_tone_lower_repair import condition

POLICIES = (
    "S614_original", "gate_now_050",
    "probe_low_2", "probe_low_3", "probe_low_4",
    "probe_low_3_harm3",
)
FFT_LEN = 32768
WINDOW = 8192
TAPER = np.hanning(WINDOW).astype(np.float32)
FREQ = np.fft.rfftfreq(FFT_LEN, 1.0 / SR)

def _local_contrast(spectrum, center):
    """A genuine independent partial should stand above nearby off-pitch energy."""
    half = max(3.0, center * .024)
    signal = spectrum[np.abs(FREQ - center) <= half]
    left = (FREQ > center * .73) & (FREQ < center * .90)
    right = (FREQ > center * 1.10) & (FREQ < center * 1.27)
    control = spectrum[left | right]
    if not len(signal) or not len(control):
        return 0.0
    return float((np.max(signal) + 1.e-8) / (np.median(control) + 1.e-8))

def fundamental_contrasts(wave, end, lower_midi):
    """Only PCM samples with index < end are used. No zero-phase filter."""
    left = max(0, int(end) - WINDOW)
    x = wave[left:int(end)]
    if len(x) < WINDOW:
        x = np.pad(x, (WINDOW - len(x), 0))
    spectrum = np.abs(np.fft.rfft(x * TAPER, n=FFT_LEN))
    f0 = 440.0 * 2 ** ((int(lower_midi) - 69) / 12.0)
    return _local_contrast(spectrum, f0), _local_contrast(spectrum, 3.0 * f0)

def permit(policy, f0_contrast, h3_contrast):
    if policy == "probe_low_2":
        return f0_contrast >= 2
    if policy == "probe_low_3":
        return f0_contrast >= 3
    if policy == "probe_low_4":
        return f0_contrast >= 4
    if policy == "probe_low_3_harm3":
        return f0_contrast >= 3 and h3_contrast >= 1.5
    return True

def measure_one(spec, root, archive, heads, config):
    wave, sha = raw_guitar_audio(spec, archive)
    truth = real_notes(labels(spec, root)["notes"], len(wave) / SR)
    ends, native, native_k = predict_causal_view(wave, heads)
    filtered = []
    for fc in FILTERS.values():
        e, p, _ = predict_causal_view(causal_lowpass(wave, fc), heads)
        if not np.array_equal(e, ends):
            raise ValueError("Tone view changed native clock")
        filtered.append(p)
    mild, dark = filtered
    states = {name: CausalSelection(config) for name in POLICIES}
    emitted = {name: [] for name in POLICIES}
    attempted = Counter()
    accepted = Counter()
    by_pitch = Counter()
    ages = np.full(49, 10**7, np.int32)

    for frame, end in enumerate(ends):
        ages = np.minimum(ages + 1, 10**7)
        onset = (native[frame, :, 0] >= config.onset) & (
            native[frame, :, 1] >= config.active_on)
        ages[onset] = 0
        for name, selector in states.items():
            if name == "S614_original":
                p = native[frame]
            else:
                p = condition(native[frame], mild[frame], dark[frame],
                              config, ages, "gate_now_050",
                              selector.armed, selector.active)
                if name.startswith("probe_"):
                    added = np.flatnonzero(
                        (p[:, 0] >= config.onset) &
                        (native[frame, :, 0] < config.onset)
                    )
                    # Compute one FFT at most per physical hop, only when
                    # a lower-octave candidate is actually being proposed.
                    for low in added:
                        pitch = int(low) + 40
                        f0, h3 = fundamental_contrasts(wave, end, pitch)
                        attempted[name] += 1
                        if not permit(name, f0, h3):
                            p[low] = native[frame, low]
                        else:
                            accepted[name] += 1
                            by_pitch[(name, pitch)] += 1
            emitted[name].extend(selector.step(
                p, native_k[frame], int(end), int(end)))

    for name, selector in states.items():
        emitted[name].extend(selector.panic(len(wave)))

    overlap = overlap_groups(truth)
    result = {}
    original_matches = None
    for name in POLICIES:
        score, pairs = event_score(emitted[name], truth)
        matches = {q for _, q in pairs}
        if name == "S614_original":
            original_matches = matches
        result[name] = dict(
            score=score, matched=matches,
            overlap_TP=len(matches & overlap["onset_overlap"]),
            K3plus_TP=len(matches & overlap["onset_K3plus"]),
            E2_TP=sum(truth[q]["pitch"] == 40 for q in matches),
            attempts=int(attempted[name]),
            accepted=int(accepted[name]),
        )
    for row in result.values():
        matched = row.pop("matched")
        row["new_true"] = len(matched - original_matches)
        row["lost_true"] = len(original_matches - matched)
    return dict(name=spec["name"], composition=spec["composition"],
                audio_sha256=sha, stats=result,
                allowed_pitch_histogram={
                    name: {str(p): int(n) for (policy,p),n in by_pitch.items()
                           if policy == name} for name in POLICIES
                })

def aggregate(rows):
    out = {}
    for name in POLICIES:
        parts = [r["stats"][name] for r in rows]
        out[name] = dict(
            score=summary([p["score"] for p in parts]),
            overlap_TP=sum(p["overlap_TP"] for p in parts),
            K3plus_TP=sum(p["K3plus_TP"] for p in parts),
            E2_TP=sum(p["E2_TP"] for p in parts),
            new_true=sum(p["new_true"] for p in parts),
            lost_true=sum(p["lost_true"] for p in parts),
            attempted=sum(p["attempts"] for p in parts),
            accepted=sum(p["accepted"] for p in parts),
        )
    return out

def train_choice(cal, held):
    c, h = aggregate(cal), aggregate(held)
    baseline = "S614_original"
    winner = max(POLICIES, key=lambda name: (
        c[name]["score"]["F1"], -c[name]["score"]["FP"],
        name == baseline))
    if (c[winner]["score"]["F1"] <
            c[baseline]["score"]["F1"] + .003):
        winner = baseline
    held_by_composition = {}
    for comp in sorted({r["composition"] for r in held}):
        rows = [r for r in held if r["composition"] == comp]
        a = aggregate(rows)
        held_by_composition[comp] = (
            a[winner]["score"]["F1"] - a[baseline]["score"]["F1"])
    enough = sum(v > 0 for v in held_by_composition.values()) >= 3
    selected = winner if (
        winner != baseline and enough and
        h[winner]["score"]["F1"] >= h[baseline]["score"]["F1"] + .002
    ) else baseline
    return selected, dict(winner_calibration=winner, selected=selected,
        calibration_F1={p: c[p]["score"]["F1"] for p in POLICIES},
        held_F1={p: h[p]["score"]["F1"] for p in POLICIES},
        held_gain_by_composition=held_by_composition)

def run(out):
    if out.exists():
        raise FileExistsError("Never overwrite S659 research evidence")
    frozen = json.loads((ROOT / "selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes()) == frozen["model_sha256"]
    assert digest((ROOT / "protocol.json").read_bytes()) == frozen["protocol_sha256"]
    proto = json.loads((ROOT / "protocol.json").read_text())
    train = [s for s in proto["sources"] if s["corpus"] == "GuitarSet"
             and s["split"] == "train"]
    dev = [s for s in proto["sources"] if s["corpus"] == "GuitarSet"
           and s["split"] == "dev"]
    test = [s for s in proto["sources"] if s["split"] == "test"]
    assert len(train) == 36 and len(dev) == 12
    assert not {(s["corpus"],s["composition"]) for s in train} & {
        (s["corpus"],s["composition"]) for s in dev + test}
    comps = sorted({s["composition"] for s in train})
    assert len(comps) == 9
    cal_comps = set(comps[:5])
    heads = load_heads(MODEL, expected_sha256=frozen["model_sha256"])
    config = Selection(**heads.metadata["selection"])
    start = time.perf_counter()

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "data").mkdir()
        for key, fname in (("GuitarSet_audio","GuitarSet_audio.zip"),
                           ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            item = proto["guitar_archives"][key]
            download(item["url"], root / "data" / fname,
                     item["size"], item["sha256"])
        with zipfile.ZipFile(root / "data/GuitarSet_audio.zip") as arc:
            calibration, held = [], []
            for j,s in enumerate(train, 1):
                item = measure_one(s,root,arc,heads,config)
                (calibration if s["composition"] in cal_comps
                 else held).append(item)
                print("S659 TRAIN",j,"/36",s["name"],flush=True)
            selected,choice = train_choice(calibration,held)
            print("S659 TRAIN CHOICE",json.dumps(choice),flush=True)
            dev_rows = []
            for j,s in enumerate(dev, 1):
                item = measure_one(s,root,arc,heads,config)
                dev_rows.append(item)
                print("S659 DEV",j,"/12",s["name"],flush=True)

    scores = aggregate(dev_rows)
    base = scores["S614_original"]["score"]
    assert all(base[k] == v for k,v in GUITARSET_PINNED_CONTROL.items())
    for name,row in scores.items():
        score = row["score"]
        row["delta_TP"] = score["TP"] - base["TP"]
        row["delta_FP"] = score["FP"] - base["FP"]
        row["delta_F1_points"] = round(
            100 * (score["F1"] - base["F1"]), 6)
        print("S659 SCORE",name,"TP",score["TP"],"FP",score["FP"],
              "FN",score["FN"],"F1",round(score["F1"],8),
              "K3+",row["K3plus_TP"],"E2",row["E2_TP"],
              "attempted",row["attempted"],"accepted",row["accepted"],
              flush=True)
    report = dict(
        status="S659_CAUSAL_FUNDAMENTAL_GUARD_RESEARCH_ONLY",
        source_head_sha="544bd5103ddb3ff29fc9941810adf70dc9f188a4",
        original_K_unchanged=True, original_live_unmodified=True,
        no_future_audio=True, no_backdated_notes=True,
        no_promotion=True, test_never_opened=True,
        dev_previously_exposed=True,
        all_selection_done_exclusively_by_train=selected,
        selection_decision=choice,
        dev=scores, per_dev_recording=dev_rows,
        elapsed_seconds=round(time.perf_counter()-start,2),
    )
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S659 FINAL",json.dumps(dict(selected=selected,
          DEV_F1={n:v["score"]["F1"] for n,v in scores.items()})),
          flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    args = ap.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
