"""S724 gain stress audit on original GuitarSet CAL PCM, paired by composition.

Two distinct goals:
(a) invariance to recording volume, not to playing technique;
(b) actual F1 of the same emitted NoteOn/Off events at +/-6 dB gain.
Predeclared calibration subset (6 comp + 6 solo), gain factors 0.5, 1, 2.
Clipping is reported explicitly; a clipped variant cannot substantiate
clean-gain robustness. No gain range is selected against HELD/DEV.
No model or Live stack is promoted.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from causal_note.strict_live import SR,feature_track
from causal_note.s706_sharp_spectra import sharp_track
from causal_note.s724_loudness_timbre import augmented_track
from causal_note.s705_temporal_heads import predict_track
from causal_note.s708_cardinality_selection import GuidedPolicy,decode_guided
from causal_note.s707_causal_release import prepare_hazards,render_release
from scripts.s704_joint_heads import BATCH,acoustic_features
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s672_causal_offset_timing_grid import raw_guitar_audio,native_posteriors
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s706_train_sharp_temporal_loops import setup,load_model,record

LEVELS={"soft_0.5":.5,"nominal_1.0":1.,"loud_2.0":2.}


def build_scaled_audio(audio,heads,factor):
    altered=audio.astype(np.float32)*factor
    clipped=int(np.count_nonzero(np.abs(altered)>1.))
    gain_audio=np.clip(altered,-1.,1.)
    ends,feat=feature_track(gain_audio,block_size=4096)
    native,k=native_posteriors(feat,heads)
    selected=np.arange(BATCH-1,len(ends),BATCH)
    decisions=ends[selected]
    x=acoustic_features(feat,native,k)[selected]
    x=np.concatenate((x,sharp_track(gain_audio,decisions)),-1).astype(np.float32)
    if x.shape!=(len(decisions),49,288):
        raise AssertionError("Gain audit must share frozen S708 ABI")
    return dict(x=x,ends=ends,samples=len(audio),decision_ends=decisions,
        native_features=feat[selected].copy(),
        native_post=native[selected].copy(),native_k=k[selected].copy(),
        clip_samples=clipped,clip_fraction=float(clipped/len(audio)),
        raw_gain=factor)


def emit(item,model,policy,offset,release,use_timbre):
    x=augmented_track(item["x"]) if use_timbre else item["x"]
    p,k=predict_track(model,x)
    post=policy.blend*p+(1-policy.blend)*item["native_post"]
    src=decode_guided(item["decision_ends"],post,k,policy,item["samples"])
    out,_=render_release(prepare_hazards(item,src,offset),
                         item["ends"],item["samples"],**release)
    return out


def run(args):
    if args.out.exists():raise FileExistsError("S724 gain stress immutable output required")
    args.out.mkdir(parents=True)
    frozen=json.loads((args.s708_dir/"report.json").read_text())
    trained=json.loads((args.s724_dir/"report.json").read_text())
    assert frozen["source_family"]==706
    assert digest((args.s708_dir/"s708_selected.pt").read_bytes())==frozen["checkpoint_sha256"]
    assert trained["s708_sha256"]==frozen["checkpoint_sha256"]
    source=trained["train"]["s724_epoch2"]
    assert digest((args.s724_dir/source["path"]).read_bytes())==source["sha256"]
    policy=GuidedPolicy(**frozen["frozen_candidate"]["choice"]["policy"])
    release=frozen["frozen_candidate"]["choice"]["release"]
    assert release==dict(gate=.8,persistence=1,grace=24)
    report=dict(schema="S724_PCM_GAIN_STRESS_CAL12",promoted=False,
        official_live_changed=False,TEST_opened=False,CAL_only=True,
        original_S708_sha256=frozen["checkpoint_sha256"],
        new_S724_sha256=source["sha256"],levels=LEVELS,
        no_gain_or_threshold_tuning=True,HELD_DEV_unopened=True,
        limitations="PCM gain is not a different physical pluck; clipping changes waveform timbre")
    def save():
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        protocol,heads,selector,offset=setup(root,args.offset_checkpoint)
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            asset=protocol["guitar_archives"][key]
            download(asset["url"],root/"data"/name,asset["size"],asset["sha256"])
        originals=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        models=dict(S708_frozen=load_model(args.s708_dir/"s708_selected.pt")[0],
                    S724_timbre_epoch2_Kguided=load_model(args.s724_dir/source["path"])[0])
        measures={level:{name:[] for name in models} for level in LEVELS}
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive,zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            compositions=expanded_train_specs(originals,archive,ann)
            cal=[s for s in compositions if s["composition"] in frozen["cal_compositions"]]
            # Only track names (not annotations) decide this symmetric subset.
            pairs=[]
            for texture in ("comp","solo"):
                group=sorted((s for s in cal if s["name"].endswith("_"+texture)),
                             key=lambda s:s["name"])
                if len(group)<6:raise AssertionError("Not enough matching CAL textures")
                pairs+=group[:6]
            assert len(pairs)==12
            rows=[]
            for i,spec in enumerate(pairs):
                raw,raw_sha=raw_guitar_audio(spec,archive)
                variants={}
                for level,factor in LEVELS.items():
                    item=build_scaled_audio(raw,heads,factor)
                    predictions={}
                    for name,model in models.items():
                        predictions[name]=emit(item,model,policy,offset,release,
                                              use_timbre=name.startswith("S724"))
                    variants[level]=(item,predictions)
                # Reproduce ORIGINAL event stream at unity gain before labels.
                historical=record(spec,archive,heads,selector,offset)
                if raw_sha!=historical["sha256"]:
                    raise AssertionError("Original CAL GuitarSet audio SHA changed")
                unity=variants["nominal_1.0"][0]
                np.testing.assert_array_equal(unity["decision_ends"],historical["decision_ends"])
                np.testing.assert_allclose(unity["x"],historical["x"],atol=1e-6,rtol=1e-6)
                # Ground truth only examined once all PCM gains emitted.
                truth=real_notes(labels(spec,root)["notes"],len(raw)/SR)
                result=dict(name=spec["name"],sha256=raw_sha,texture=spec["name"].rsplit("_",1)[-1],
                            results={})
                for level,(item,events) in variants.items():
                    this={name:metric(ev,truth,item["ends"]) for name,ev in events.items()}
                    for name,score in this.items():
                        measures[level][name].append(score)
                    result["results"][level]=dict(
                        clip_fraction=item["clip_fraction"],
                        stats={name:{k:row[k] for k in SCORES} |
                                       {k:row[k] for k in ("tp","fp","fn")}
                               for name,row in this.items()})
                rows.append(result)
                report["completed_tracks"]=len(rows)
                save()
                print("S724 GAIN CAL12",i+1,len(pairs),spec["name"],flush=True)
            aggregate_by_gain={level:{name:aggregate(rows) for name,rows in mods.items()}
                               for level,mods in measures.items()}
            report["per_gain"]=dict(levels=aggregate_by_gain,tracks=rows)
            report["gain_robustness"]={
                level:{name:{k:aggregate_by_gain[level][name][k] -
                            aggregate_by_gain["nominal_1.0"][name][k]
                             for k in SCORES} for name in models}
                for level in LEVELS if level!="nominal_1.0"}
            report["any_clipping"]=any(
                result["results"][lvl]["clip_fraction"]>0.
                for result in rows for lvl in LEVELS)
            save()
    print("S724 GAIN FINAL",json.dumps(dict(
         variants=list(models),gain_robustness=report["gain_robustness"],
         any_clipping=report["any_clipping"],promoted=False)),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",type=Path,required=True)
    ap.add_argument("--s724-dir",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    with threadpool_limits(limits=2):
        run(ap.parse_args())
