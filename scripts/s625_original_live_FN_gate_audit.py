"""S625 causal NOTE dev audit: why do 838 true NoteOn remain missing?

No model changes. Inspect ORIGINAL S614 attack and sustain heads at each
unmatched true onset on the 12 GuitarSet + 5 URMP development recordings.
Separate zero acoustic onset evidence from disjoint learned onset/sustain
evidence and already joint-eligible but state-suppressed attacks. These
classes are observations, not causal proof of a specific selector bug.
Do not touch the already-scored S614 test.
"""
from __future__ import annotations

from collections import Counter,defaultdict
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np
from scipy.io import wavfile

from causal_note.strict_live import SR,Selection,feature_track,load_heads
from scripts.s616_live_dev_ablation import ROOT,MODEL,prepare_dev_source
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import decode,event_score,real_notes,summary


def identify(ends,p,pitch,onset,selection):
    clock=np.asarray(ends)/SR
    inside=(clock>=onset-.080)&(clock<=onset+.080)
    channel=p[inside,pitch-40,:]
    if not len(channel):
        return "no_observed_frames",0.,0.
    peak_o=float(channel[:,0].max())
    peak_a=float(channel[:,1].max())
    o=channel[:,0]>=selection.onset
    a=channel[:,1]>=selection.active_on
    if (o&a).any():category="attack_and_sustain_same_frame_no_matching_emission"
    elif o.any() and a.any():category="attack_sustain_not_coincident"
    elif o.any():category="attack_only_sustain_insufficient"
    elif a.any():category="sustain_only_attack_insufficient"
    else:category="neither_head_above_threshold"
    return category,peak_o,peak_a


def audit_source(spec,ends,x,truth,heads,selection,n_samples):
    p,k=heads.predict(x)
    original=decode(ends,p,k,selection,n_samples)
    score,pairs=event_score(original,truth)
    true_matched={j for i,j in pairs}
    groups=Counter()
    failure=[]
    for j,note in enumerate(truth):
        label,attack,sustain=identify(ends,p,note["pitch"],note["time"],selection)
        if j not in true_matched:
            groups[label]+=1
            failure.append(dict(source=spec["name"],pitch=note["pitch"],
                true_start=float(note["time"]),observed_category=label,
                max_attack=round(attack,5),max_sustain=round(sustain,5)))
    assert sum(groups.values())==score["FN"]
    return score,groups,failure


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    out=parser.parse_args().out
    if out.exists():raise FileExistsError("S625 is immutable")
    out.mkdir(parents=True)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    checkpoint=digest(MODEL.read_bytes())
    assert checkpoint==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    native=load_heads(MODEL,expected_sha256=checkpoint)
    cfg=Selection(**native.metadata["selection"])
    assert cfg.onset==.6 and cfg.active_on==.5
    guitar=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    urmp=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="dev"]
    assert len(guitar)==12 and len(urmp)==5
    begin=time.perf_counter()
    records=[];failures=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        for spec in urmp:
            pkg=prepare_dev_source(spec,root)
            score,categories,failed=audit_source(spec,pkg["ends"],pkg["features"],
                                                 pkg["truth"],native,cfg,pkg["samples"])
            records.append(dict(corpus="URMP",source=spec["name"],score=score,
                                categories=dict(categories)))
            failures.extend(failed)
            print("S625 URMP",spec["name"],categories,flush=True)
        a=protocol["guitar_archives"]["GuitarSet_audio"]
        b=protocol["guitar_archives"]["GuitarSet_annotations"]
        download(a["url"],root/"data/GuitarSet_audio.zip",a["size"],a["sha256"])
        download(b["url"],root/"data/GuitarSet_annotations.zip",b["size"],b["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as zipped:
            for spec in guitar:
                audio=zipped.read(spec["audio"])
                rate,pcm=wavfile.read(io.BytesIO(audio))
                assert rate==SR and pcm.dtype in (np.int16,np.int32)
                y=pcm.astype(np.float32)/(32768 if pcm.dtype==np.int16 else 2147483648)
                if y.ndim==2:y=y.mean(axis=1)
                y=y[:int(spec["seconds"]*SR)].copy()
                ends,x=feature_track(y)
                truth=real_notes(labels(spec,root)["notes"],len(y)/SR)
                score,categories,failed=audit_source(spec,ends,x,truth,native,cfg,len(y))
                records.append(dict(corpus="GuitarSet",source=spec["name"],
                                    score=score,categories=dict(categories)))
                failures.extend(failed)
                print("S625 GUITAR",spec["name"],categories,flush=True)
    corpus_scores={}
    overall=Counter()
    for name,expected in (("GuitarSet",(998,585,582)),("URMP",(243,149,256))):
        same=[r for r in records if r["corpus"]==name]
        aggregate=summary([r["score"] for r in same])
        assert (aggregate["TP"],aggregate["FP"],aggregate["FN"])==expected
        categories=Counter()
        for source in same:categories.update(source["categories"])
        assert sum(categories.values())==aggregate["FN"]
        corpus_scores[name]=dict(TP=aggregate["TP"],FP=aggregate["FP"],
                                 FN=aggregate["FN"],failure_taxonomy=dict(categories))
        overall.update(categories)
    assert sum(overall.values())==582+256==838
    result=dict(status="S625_ROOT_CAUSE_AUDIT_OF_ORIGINAL_S614_DEV_MISSED_NOTES",
        original_S80_S614_unchanged=True,checkpoint_sha256=checkpoint,
        source_count=len(records),published_test_NOT_read=True,
        no_new_independent_claim=True,
        windows_used="reference onset -80ms to +80ms, labels only in DEV AUDIT",
        joint_onset_sustain_required=True,
        failure_categories_all=dict(overall),by_corpus=corpus_scores,
        sources=[dict(corpus=s["corpus"],source=s["source"],categories=s["categories"]) for s in records],
        complete_unmatched_true_events=failures,
        no_promotion=True,elapsed_seconds=round(time.perf_counter()-begin,2))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S625 ROOT CAUSE",json.dumps({"total":dict(overall),"by_corpus":corpus_scores,
        "note":"Overlap is necessary in S614 selection but not sufficient for matched NoteOn"},indent=2),flush=True)


if __name__=="__main__":
    main()
