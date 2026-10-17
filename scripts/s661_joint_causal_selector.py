"""S661 — combine independently generated NOTE bass x2 and Tone octave rescues.

Purpose: an actual end-to-end causal joint selection experiment, NOT a new
production checkpoint. Baselines S614 and S641 are reproduced from frozen
native original NOTE head. The native K posterior is passed unchanged to
every CausalSelection. NOTE Live files and historical checkpoints untouched.
Selection is frozen on unseen TRAIN compositions, and only then inspected
on previously exposed DEV. TEST never opened.
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
from scripts.s613_prepare_strict_live import download, digest
from scripts.s613_train_strict_live import event_score, summary
from scripts.s637_guitarset_three_view_groundtruth import (
    ROOT, MODEL, GUITARSET_PINNED_CONTROL,
)
from scripts.s640_learned_overlap_rescue import (
    iter_acoustic, get_candidates, relay, pcm_truth,
)
from scripts.s641_fundamental_consistency import (
    train_record, build_gate, train_only_group_threshold,
)
from scripts.s638_guitarset_true_overlap import overlap_groups
from scripts.s652_tone_filter_multiview import (
    FILTERS, causal_lowpass, predict_causal_view,
)
from scripts.s657_coincident_tone_lower_repair import condition
from scripts.s660_multiscale_fundamental_probe import (
    multiscale_contrasts, multiscale_permit,
)

POLICIES=(
    "S614_original",
    "S641_41D",
    "S660_recent",
    "S661_union_tone",
    "S661_union_recent",
    "S661_union_strict",
    "S661_s641_physical_veto",
)
MIN_HELD_GAIN=.003
MIN_POSITIVE_HELD_COMPOSITIONS=3


def tone_rescue(native,mild,dark,cfg,ages,selector):
    return condition(native,mild,dark,cfg,ages,"gate_now_050",
                     selector.armed,selector.active)


def _inject(out, proposal, native, selector, keep):
    """Mutate local copy only: original probabilities never modified."""
    for pitch in np.flatnonzero(
            (proposal[:,0] >= selector.config.onset - 1e-6) &
            (native[:,0] < selector.config.onset - 1e-6)):
        i=int(pitch)
        if not keep(i) or selector.active[i] or not selector.armed[i]:
            continue
        out[i,0]=max(float(out[i,0]),float(proposal[i,0]))
        out[i,1]=max(float(out[i,1]),float(proposal[i,1]))
    return out


def joined(native,bass,tone,selector,candidates,scores,
           threshold,variant,contrasts):
    """Separate rescue evidence merged prior to ONE causal selection step."""
    if variant=="S614_original":
        return native
    if variant=="S660_recent":
        return _inject(native.copy(),tone,native,selector,
          lambda i: (i+40 in contrasts and
                     multiscale_permit("probe_recent_2",contrasts[i+40])))
    # S641 rescue is always evaluated independently of the Tone proposals
    p,_=relay(native,bass,selector,scores,threshold)
    if variant=="S641_41D":
        return p
    if variant=="S661_s641_physical_veto":
        # Only additional S641 pitches are filtered; original S614 intact.
        for i in range(len(native)):
            if (float(p[i,0])+1e-6 >= selector.config.onset and
                float(native[i,0])+1e-6 < selector.config.onset and
                not (i+40 in contrasts and
                     multiscale_permit("probe_short_2",contrasts[i+40]))):
                p[i]=native[i]
        return p
    def permitted(i):
        x=contrasts.get(i+40)
        if x is None:
            return False
        if variant=="S661_union_tone":
            return True
        if variant=="S661_union_recent":
            return multiscale_permit("probe_recent_2",x)
        if variant=="S661_union_strict":
            return multiscale_permit("probe_both_2",x)
        raise ValueError("Unknown S661 variant "+variant)
    return _inject(p,tone,native,selector,permitted)


def evaluate(spec,root,archive,heads,cfg,gate,threshold):
    wave,truth,sha=pcm_truth(spec,root,archive)
    # The three Tone posterior streams are computed in offline batches,
    # but each frame is strictly right-aligned, past-PCM-only. No future
    # signal is used in a physical decision. This is NOT a measured Live
    # CPU or microphone latency claim.
    ends_original, native_verify, original_k=predict_causal_view(wave,heads)
    tone_outputs=[]
    for fc in FILTERS.values():
        ends,post,_=predict_causal_view(causal_lowpass(wave,fc),heads)
        if not np.array_equal(ends,ends_original):
            raise ValueError("Tone and original frame clocks differ")
        tone_outputs.append(post)
    mild,dark=tone_outputs
    states={name:CausalSelection(cfg) for name in POLICIES}
    events={name:[] for name in POLICIES}
    ages=np.full(49,10**7,np.int32)
    rescue_attempts=Counter()
    matched=0
    for end,x,native,k,bass,valid in iter_acoustic(wave,heads):
        j=int(np.searchsorted(ends_original,int(end)))
        if j>=len(ends_original) or int(ends_original[j])!=int(end):
            raise AssertionError("S641 vs Tone causal sample-clock mismatch")
        if not np.allclose(k,original_k[j],atol=1e-5,rtol=1e-5):
            raise AssertionError("S641 and S614 original K changed")
        if not np.allclose(native,native_verify[j],atol=1e-5,rtol=1e-5):
            raise AssertionError("S641 vs native original pitch changed")
        matched+=1
        ages=np.minimum(ages+1,10**7)
        new=(native[:,0]>=cfg.onset)&(native[:,1]>=cfg.active_on)
        ages[new]=0
        candidates=get_candidates(x,native,k,bass,valid,cfg.onset)
        confidences={}
        if candidates:
            ids=list(candidates)
            score=gate.predict_proba(
                np.stack([candidates[i] for i in ids]))[:,1]
            confidences={i:float(v) for i,v in zip(ids,score)}
        contrasts={}
        for name,selector in states.items():
            if name=="S614_original":
                result=native
            else:
                tone=tone_rescue(native,mild[j],dark[j],cfg,ages,selector)
                tone_pitches=np.flatnonzero(
                    (tone[:,0]>=cfg.onset)&(native[:,0]<cfg.onset))
                # Calculate spectral evidence for Tone and S641 candidates,
                # cache per physical hop. Only past samples are accessed.
                to_check=set(int(i) for i in tone_pitches)
                if name=="S661_s641_physical_veto":
                    to_check.update(i for i,v in confidences.items()
                                    if v>=threshold)
                for i in to_check:
                    if i+40 not in contrasts:
                        contrasts[i+40]=multiscale_contrasts(wave,end,i+40)
                result=joined(native,bass,tone,selector,candidates,
                              confidences,threshold,name,contrasts)
                rescue_attempts[name]+=int(np.sum(
                    (result[:,0]>=cfg.onset)&(native[:,0]<cfg.onset)))
            events[name].extend(selector.step(result,k,int(end),int(end)))
    if not matched:
        raise AssertionError("S661 never processed a native S641 frame")
    grouped=overlap_groups(truth)
    output={}
    base_matches=None
    for name,selector in states.items():
        events[name].extend(selector.panic(len(wave)))
        sc,pairs=event_score(events[name],truth)
        actual={j for _,j in pairs}
        if name=="S614_original":
            base_matches=actual
        output[name]=dict(score=sc,matched=actual,
            onset_poly=len(actual&grouped["onset_overlap"]),
            K3plus_TP=len(actual&grouped["onset_K3plus"]),
            E2_TP=sum(truth[i]["pitch"]==40 for i in actual),
            attempts=int(rescue_attempts[name]))
    for row in output.values():
        hits=row.pop("matched")
        row["new_true"]=len(hits-base_matches)
        row["lost_true"]=len(base_matches-hits)
    return dict(name=spec["name"],composition=spec["composition"],
        audio_sha256=sha,frames=matched,stats=output)


def aggregate(records):
    return {name:dict(
        score=summary([r["stats"][name]["score"] for r in records]),
        onset_poly=sum(r["stats"][name]["onset_poly"] for r in records),
        K3plus_TP=sum(r["stats"][name]["K3plus_TP"] for r in records),
        E2_TP=sum(r["stats"][name]["E2_TP"] for r in records),
        new_true=sum(r["stats"][name]["new_true"] for r in records),
        lost_true=sum(r["stats"][name]["lost_true"] for r in records),
        rescue_attempts=sum(r["stats"][name]["attempts"] for r in records))
        for name in POLICIES}


def held_winner(records):
    """No selection based on DEV, only held-out TRAIN compositions."""
    agg=aggregate(records)
    base=agg["S641_41D"]["score"]
    groups={}
    for c in sorted({r["composition"] for r in records}):
        g=aggregate([r for r in records if r["composition"]==c])
        groups[c]={n:round(
            g[n]["score"]["F1"]-g["S641_41D"]["score"]["F1"],8)
            for n in POLICIES}
    acceptable=[]
    for name in POLICIES:
        if name in ("S614_original","S641_41D"):
            continue
        score=agg[name]["score"]
        positive=sum(groups[g][name]>1e-8 for g in groups)
        if (score["F1"]>=base["F1"]+MIN_HELD_GAIN and
                positive>=MIN_POSITIVE_HELD_COMPOSITIONS):
            acceptable.append(name)
    winner=max(acceptable,key=lambda n:(agg[n]["score"]["F1"],
               -agg[n]["score"]["FP"])) if acceptable else "S641_41D"
    return winner,dict(
        selected_without_DEV=winner,
        held_F1={n:agg[n]["score"]["F1"] for n in POLICIES},
        held_by_composition=groups,
        min_F1_gain_vs_S641=MIN_HELD_GAIN,
        min_improved_groups=MIN_POSITIVE_HELD_COMPOSITIONS)


def training_arrays(records):
    data=np.concatenate([r["X"] for r in records])
    y=np.concatenate([r["y"] for r in records])
    groups=np.concatenate([np.full(len(r["y"]),r["composition"],dtype=object)
                           for r in records])
    return data,y,groups


def run(out):
    if out.exists():
        raise FileExistsError("S661 report must be an immutable new directory")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    proto=json.loads((ROOT/"protocol.json").read_text())
    train=[s for s in proto["sources"] if
           s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in proto["sources"] if
         s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    test=[s for s in proto["sources"] if s["split"]=="test"]
    assert len(train)==36 and len(dev)==12
    assert not {(s["corpus"],s["composition"]) for s in train} & {
        (s["corpus"],s["composition"]) for s in dev+test}
    compositions=sorted({s["composition"] for s in train})
    assert len(compositions)==9
    calibration=set(compositions[:5])
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    cfg=Selection(**heads.metadata["selection"])
    started=time.perf_counter()
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        for key,fname in (("GuitarSet_audio","GuitarSet_audio.zip"),
                          ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            source=proto["guitar_archives"][key]
            download(source["url"],root/"data"/fname,source["size"],
                     source["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as arc:
            training=[]
            for n,s in enumerate(train,1):
                r=train_record(s,root,arc,heads,cfg)
                training.append(r)
                print("S661 BUILD TRAIN",n,"/36",s["name"],
                      "n",r["candidates"],"positives",r["positives"],
                      flush=True)
            xtrain,ytrain,_=training_arrays(
                [r for r in training if r["composition"] in calibration])
            assert ytrain.sum()>8 and len(np.unique(ytrain))==2
            control_gate=build_gate().fit(xtrain,ytrain)
            holdouts=[s for s in train if s["composition"] not in calibration]
            held=[]
            for n,s in enumerate(holdouts,1):
                r=evaluate(s,root,arc,heads,cfg,control_gate,.90)
                held.append(r)
                print("S661 HELD",n,"/",len(holdouts),s["name"],flush=True)
            selected,held_decision=held_winner(held)
            print("S661 FROZEN TRAIN CHOICE",json.dumps(held_decision),
                  flush=True)
            fullx,fully,fullgroups=training_arrays(training)
            threshold,cv=train_only_group_threshold(fullx,fully,fullgroups)
            assert threshold==.90, (
                "S641 CV threshold not reconstructed",threshold)
            fullgate=build_gate().fit(fullx,fully)
            validation=[]
            for n,s in enumerate(dev,1):
                row=evaluate(s,root,arc,heads,cfg,fullgate,threshold)
                validation.append(row)
                print("S661 DEV",n,"/12",s["name"],flush=True)
    score=aggregate(validation)
    original=score["S614_original"]["score"]
    assert all(original[k]==v for k,v in GUITARSET_PINNED_CONTROL.items())
    # Independent full re-run must reproduce earlier S641 benchmark exactly.
    assert score["S641_41D"]["score"]["TP"]==1009
    assert score["S641_41D"]["score"]["FP"]==594
    for name,r in score.items():
        z=r["score"]; b=score["S641_41D"]["score"]
        r["delta_TP_vs_S641"]=z["TP"]-b["TP"]
        r["delta_FP_vs_S641"]=z["FP"]-b["FP"]
        r["delta_F1_points_vs_S641"]=round(100*(z["F1"]-b["F1"]),6)
        print("S661 SCORE",name,"TP",z["TP"],"FP",z["FP"],
              "FN",z["FN"],"F1",round(z["F1"],8),
              "dF1_vs_S641",r["delta_F1_points_vs_S641"],
              "K3+",r["K3plus_TP"],"E2",r["E2_TP"],flush=True)
    report=dict(
        status="S661_S641_TONE_S660_JOINT_CAUSAL_SELECTION_RESEARCH",
        original_sha256=frozen["model_sha256"],
        baseline_reconstructed_S641=True,
        original_Exact_K_unchanged=True,
        original_LIVE_unmodified=True,
        original_NoteOff_state_preserved=True,
        no_future_pcm_or_backdated_NoteOn=True,
        generated_candidate_sources=("S641_bass_x2","S658_filtered_Tone"),
        train_calibration_compositions=sorted(calibration),
        train_held_compositions=sorted(set(compositions)-calibration),
        threshold_S641_selected_on_TRAIN_CV=.90,
        TRAIN_OOF_CV=cv,
        frozen_selection=selected,
        held_train_selection=held_decision,
        DEV_previously_exposed=True,
        TEST_not_opened=True,
        no_automatic_promotion=True,
        DEV_results=score,
        DEV_by_recording=validation,
        elapsed_seconds=round(time.perf_counter()-started,2))
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S661 FINAL",json.dumps(dict(
        selected=selected,
        F1={n:v["score"]["F1"] for n,v in score.items()})),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
