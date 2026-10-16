"""S615 train 128/256/512-specific or shared multi-hop native NOTE heads.

Uses ONLY pinned S613 train/dev sources. No hidden test clips or labels.
Original S613 trained checkpoint and S80 are untouched.
Training is CPU-bounded via stratified sampling per SOURCE (seed 615).
All evaluation/tuning results are development-only, NOT held-out F1.

Examples:
 python -m scripts.s615_prepare_multihop --root data/S613
 python -m scripts.s615_train_multihop --root data/S613 --mode hop128 --out results/s615/hop128
 python -m scripts.s615_train_multihop --root data/S613 --mode hop256 --out results/s615/hop256
 python -m scripts.s615_train_multihop --root data/S613 --mode hop512 --out results/s615/hop512
 python -m scripts.s615_train_multihop --root data/S613 --mode shared --out results/s615/shared

For smoke only: --count-epochs 1 --pitch-epochs 1 --max-pitch-rows 200.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from causal_note.strict_live import (
    HOP, SR, SCHEMA, Selection, CausalSelection, NativeLiveHeads,
    count_features,
)
from scripts.s613_train_strict_live import (
    targets, event_score, real_notes, summary,
)


HOPS = (128, 256, 512)
CONFIGS = [
    Selection(onset=onset, active_on=sustain, onset_reset=.2,
              release_frames=release)
    for onset, sustain, release in (
        (.45, .4, 5), (.6, .5, 5), (.7, .5, 5),
        (.45, .4, 8), (.6, .5, 8), (.7, .5, 8),
    )
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def allowed_hops(mode):
    return HOPS if mode == "shared" else (int(mode[3:]),)


def source_data(root, spec, hop):
    fn = root / "multihop" / f"hop{hop}" / f"{spec['id']}.npz"
    receipt = fn.with_suffix(".json")
    origin = root / "cache" / f"{spec['id']}.json"
    if not (fn.is_file() and receipt.is_file() and origin.is_file()):
        raise FileNotFoundError(f"Prepare S615 dev/train cache for {spec['name']} hop{hop}")
    audit = json.loads(receipt.read_text(encoding="utf-8"))
    origin_proof = json.loads(origin.read_text(encoding="utf-8"))
    if (spec["split"] not in ("train", "dev") or audit["source_id"] != spec["id"]
            or audit["hop"] != hop or audit["split"] != spec["split"]
            or "labels" not in origin_proof or not origin_proof["annotations_read"]
            or audit["original_annotation_sha256"] != origin_proof["labels"]["sha256"]):
        raise ValueError("Provenance/annotation gate failed: " + spec["name"])
    with np.load(fn, allow_pickle=False) as z:
        ends = z["ends"]
        features = z["features"]
    if (features.shape != (len(ends), 49, 126)
            or len(ends) != audit["feature_shape"][0]
            or (ends != np.arange(1, len(ends)+1) * hop).any()
            or sha(features.tobytes()) != audit["feature_sha256"]):
        raise ValueError("S615 feature cache digest/shape mismatch")
    note_data = origin_proof["labels"]["notes"]
    labels, k = targets(ends, note_data)
    return ends, features, labels, k, note_data, origin_proof["prepared_samples"]


def sample_train(root, specs, hops, *, max_pitch_rows, seed=615):
    rng = np.random.default_rng(seed)
    pitch_x, pitch_y, pitch_w, count_x, count_y, summary_rows = [], [], [], [], [], []
    for s in specs:
        for hop in hops:
            _, x, y, k, _, _ = source_data(root, s, hop)
            # Time-uniform ~23ms count samples, independent of hop.
            stride = max(1, int(round(4 * HOP / hop)))
            count_x.append(count_features(x[::stride]).astype(np.float32))
            count_y.append(k[::stride].astype(np.int64))
            # Preserve all attacks/offsets as far as the cap permits.
            # Rate-controlled sampling of active/background is explicit.
            full_rate = np.where(
                np.maximum(y[..., 0], y[..., 2]) > 0, 1.,
                np.where(y[..., 1] > 0, .30, .025),
            )
            mask = rng.random(full_rate.shape) < full_rate
            frame, pitch = np.nonzero(mask)
            if len(frame) > max_pitch_rows:
                strong = np.flatnonzero(full_rate[frame, pitch] >= 1.)
                other = np.flatnonzero(full_rate[frame, pitch] < 1.)
                if len(strong) > max_pitch_rows:
                    # Never silently discard positives to fit an arbitrary cap.
                    raise ValueError(
                        f"Increase --max-pitch-rows for {s['name']} hop{hop}: "
                        f"{len(strong)} onset/release rows")
                keep_n = max_pitch_rows - len(strong)
                # Uniform over the sampled non-event pool.
                selected = rng.choice(other, size=keep_n, replace=False)
                chosen = np.sort(np.concatenate((strong, selected)))
                frame, pitch = frame[chosen], pitch[chosen]
            pitch_x.append(x[frame, pitch].copy())
            pitch_y.append(y[frame, pitch].copy())
            pitch_w.append((1. / full_rate[frame, pitch]).astype(np.float32))
            summary_rows.append(dict(source=s["name"], split=s["split"], hop=hop,
                                     count_rows=len(k[::stride]),
                                     pitch_rows=len(frame),
                                     target_note_on_rows=int(y[frame, pitch, 0].sum())))
    return (np.concatenate(pitch_x), np.concatenate(pitch_y),
            np.concatenate(pitch_w), np.concatenate(count_x),
            np.concatenate(count_y), summary_rows)


def mlp(torch, dim, nout):
    from torch import nn
    return nn.Sequential(nn.Linear(dim, 96), nn.ReLU(),
                         nn.Linear(96, 48), nn.ReLU(),
                         nn.Linear(48, nout))


def fit(root, out, mode, *, count_epochs=24, pitch_epochs=28,
        max_pitch_rows=7000, pilot_train_sources=None,
        pilot_dev_sources=None):
    import torch
    from torch import nn
    if out.exists():
        raise FileExistsError("Refusing to overwrite S615 experiment: " + str(out))
    hops = allowed_hops(mode)
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != SCHEMA:
        raise ValueError("Unexpected source protocol schema")
    train = [s for s in protocol["sources"] if s["split"] == "train"]
    dev = [s for s in protocol["sources"] if s["split"] == "dev"]
    test = [s for s in protocol["sources"] if s["split"] == "test"]
    if not train or not dev or not test:
        raise ValueError("All train/dev/test source groups must exist")
    if pilot_train_sources is not None or pilot_dev_sources is not None:
        if not pilot_train_sources or not pilot_dev_sources:
            raise ValueError("Pilot needs positive train AND dev source counts")
        train = train[:pilot_train_sources]
        dev = dev[:pilot_dev_sources]
        print(json.dumps(dict(
            status="PILOT_SUBSET_ONLY_NOT_A_FULL_CORPUS_RESULT",
            train_sources=len(train), dev_sources=len(dev))), flush=True)
    groups = {
        part: {(s["corpus"], s["composition"]) for s in seq}
        for part, seq in (("train", train), ("dev", dev), ("test", test))
    }
    if any(groups[a] & groups[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("Train/dev/test composition leakage")
    torch.set_num_threads(2)
    torch.manual_seed(615)
    np.random.seed(615)
    x, y, sample_weight, cx, cy, receipts = sample_train(
        root, train, hops, max_pitch_rows=max_pitch_rows)
    del test
    if len(np.unique(cy)) < 2 or y[:, 0].sum() < 2:
        raise ValueError("Not enough pitch / count supervision")
    out.mkdir(parents=True, exist_ok=False)
    (out / "training_sources.json").write_text(
        json.dumps(receipts, indent=2) + "\n", encoding="utf-8")

    def normal(data):
        mu = data.mean(axis=0, dtype=np.float64).astype(np.float32)
        sig = np.maximum(data.std(axis=0, dtype=np.float64), .05).astype(np.float32)
        return mu, sig

    cm, cs = normal(cx)
    net_count = mlp(torch, cx.shape[1], 7)
    count_data = torch.from_numpy((cx-cm)/cs)
    count_targets = torch.from_numpy(cy)
    optimizer = torch.optim.Adam(net_count.parameters(), lr=.002)
    for e in range(count_epochs):
        ix = torch.randperm(len(count_data))
        for batch in ix.split(2048):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                net_count(count_data[batch]), count_targets[batch])
            loss.backward()
            optimizer.step()
        print(json.dumps(dict(stage="count", epoch=e+1,
                              cross_entropy=float(loss))), flush=True)
    del count_data, count_targets, cx, cy
    # Calibrate pitch head on PREDICTED count probabilities, never true K.
    # Pitch x has shape (rows,126) while count sees (frames,441).
    cp = []
    # Recompute chosen rows while predicting count in bounded per-source batches.
    # The sampling indices are regenerated using seed 615 and identical traversal.
    rng = np.random.default_rng(615)
    for s in train:
        for hop in hops:
            _, features, label, _, _, _ = source_data(root, s, hop)
            rate = np.where(np.maximum(label[...,0],label[...,2])>0,1.,
                            np.where(label[...,1]>0,.30,.025))
            mask = rng.random(rate.shape) < rate
            frame, pitch = np.nonzero(mask)
            if len(frame)>max_pitch_rows:
                strong=np.flatnonzero(rate[frame,pitch]>=1.)
                other=np.flatnonzero(rate[frame,pitch]<1.)
                if len(strong)>max_pitch_rows:
                    raise ValueError("Source positive cap changed")
                selected=rng.choice(other,size=max_pitch_rows-len(strong),replace=False)
                keep=np.sort(np.concatenate((strong,selected)))
                frame,pitch=frame[keep],pitch[keep]
            result=[]
            with torch.no_grad():
                for start in range(0,len(features),1024):
                    z=torch.from_numpy((count_features(features[start:start+1024])-cm)/cs)
                    probs=torch.softmax(net_count(z),-1).numpy().astype(np.float32)
                    result.append(probs)
            p=np.concatenate(result)
            cp.append(p[frame])
    cp = np.concatenate(cp)
    if len(cp)!=len(x):
        raise AssertionError("Prediction-count feature rows misaligned")
    # Slightly soften 50% of K estimates, mirroring the original S613
    # robustness strategy while never injecting the actual ground-truth K.
    softened=np.random.default_rng(614).random(len(cp))<.5
    cp[softened]=.7*cp[softened]+.3/7
    pitch_x=np.concatenate([x,cp],axis=1)
    del x,cp
    pm,ps=normal(pitch_x)
    xt=torch.from_numpy((pitch_x-pm)/ps)
    yt=torch.from_numpy(y)
    wt=torch.from_numpy(sample_weight/sample_weight.mean())
    del pitch_x,y,sample_weight
    net_pitch=mlp(torch,xt.shape[1],3)
    optimizer=torch.optim.Adam(net_pitch.parameters(),lr=.0015)
    pos_weight=torch.tensor([6.,1.5,6.])
    for e in range(pitch_epochs):
        ix=torch.randperm(len(xt))
        for batch in ix.split(4096):
            optimizer.zero_grad(set_to_none=True)
            loss=(nn.functional.binary_cross_entropy_with_logits(
                net_pitch(xt[batch]),yt[batch],
                reduction="none",pos_weight=pos_weight)*wt[batch,None]).mean()
            loss.backward();optimizer.step()
        print(json.dumps(dict(stage="pitch",epoch=e+1,
                              bce=float(loss))),flush=True)

    arrays = dict(schema=np.asarray(SCHEMA))
    for name,network,mean,scale in (("count",net_count,cm,cs),
                                    ("pitch",net_pitch,pm,ps)):
        arrays[name+".mean"]=mean
        arrays[name+".scale"]=scale
        for i,l in enumerate((network[0],network[2],network[4])):
            arrays[f"{name}.w{i}"]=l.weight.detach().numpy().T.copy()
            arrays[f"{name}.b{i}"]=l.bias.detach().numpy().copy()
    metadata=dict(architecture="native_causal_neural_multihop_v1",
                  training_mode=mode, trained_hops=list(hops),
                  selection=asdict(Selection(onset=.6,active_on=.5)),
                  native_schema=SCHEMA, base_model="S613_train_script_reimplementation",
                  data_protocol_sha256=sha(protocol_path.read_bytes()),
                  training_sources=len(train),development_sources=len(dev),
                  trained_without_test_split=True,
                  lookahead_samples=0,
                  train_hop_distribution=list(hops))
    arrays["metadata"]=np.asarray(json.dumps(metadata))
    checkpoint=out/"trained_raw.npz"
    np.savez_compressed(checkpoint,**arrays)
    scores=validate_dev(root,dev,hops,checkpoint)
    for hop in hops:
        metadata.setdefault("selection_by_hop",{})[str(hop)]=scores[str(hop)]["best"]["selection"]
    arrays["metadata"]=np.asarray(json.dumps(metadata))
    selected=out/"NOTE_multihop_candidate.npz"
    np.savez_compressed(selected,**arrays)
    report=dict(status="EXPERIMENTAL_DEV_ONLY_NOT_PROMOTED",
                model_mode=mode,trained_hops=list(hops),count_epochs=count_epochs,
                pitch_epochs=pitch_epochs,source_count_train=len(train),
                source_count_dev=len(dev),candidate_sha256=sha(selected.read_bytes()),
                data_protocol_sha256=metadata["data_protocol_sha256"],
                dev_by_hop=scores,
                warning="NO HELD-OUT TEST OR LIVE DEVICE VALIDATION. NO EXACT-K CLAIM.")
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n",
                                   encoding="utf-8")
    print(json.dumps(dict(final_checkpoint=str(selected),
                          report=str(out/"report.json"),
                          dev_by_hop={hop:row["best"] for hop,row in scores.items()}),
                     flush=True))
    return report


def validate_dev(root,specs,hops,model):
    heads=NativeLiveHeads(model)
    result={}
    for hop in hops:
        cohorts=[[] for _ in CONFIGS]
        for s in specs:
            ends,x,_,_,notes,length=source_data(root,s,hop)
            truth=real_notes(notes,length/SR)
            with np.errstate(over="ignore"):
                p,k=heads.predict(x)
            for j,config in enumerate(CONFIGS):
                selector=CausalSelection(config,hop=hop)
                events=[]
                for end,pred,count in zip(ends,p,k):
                    events.extend(selector.step(pred,count,int(end),int(end)))
                events.extend(selector.panic(int(length)))
                sc,_=event_score(events,truth,.08)
                cohorts[j].append(dict(source=s["name"],corpus=s["corpus"],**sc))
        choices=[dict(selection=asdict(cfg),scores=summary(items))
                 for cfg,items in zip(CONFIGS,cohorts)]
        best=max(choices,key=lambda row:row["scores"]["F1"])
        result[str(hop)]=dict(best=best,all=choices)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--mode",choices=("hop128","hop256","hop512","shared"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--count-epochs",type=int,default=24)
    parser.add_argument("--pitch-epochs",type=int,default=28)
    parser.add_argument("--max-pitch-rows",type=int,default=7000)
    parser.add_argument("--pilot-train-sources",type=int,default=None)
    parser.add_argument("--pilot-dev-sources",type=int,default=None)
    args=parser.parse_args()
    if min(args.count_epochs,args.pitch_epochs,args.max_pitch_rows)<1:
        parser.error("Epochs and row cap must be positive")
    fit(args.root,args.out,args.mode,count_epochs=args.count_epochs,
        pitch_epochs=args.pitch_epochs,max_pitch_rows=args.max_pitch_rows,
        pilot_train_sources=args.pilot_train_sources,
        pilot_dev_sources=args.pilot_dev_sources)


if __name__=="__main__":
    main()
