"""S777: locate first cross-worker numeric discrepancy in causal audio frontend.

Independent FIT/CAL guitar clips, no labels, no training, no promotion.
Save hashes of raw PCM, native FFT spectra/temporal features,
ONNX/tree native posterior and 96D sharp spectral features separately.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import sys
import tempfile
import zipfile
import numpy as np
import torch
from causal_note.strict_live import feature_track
from causal_note.s706_sharp_spectra import sharp_track
from scripts.s672_causal_offset_timing_grid import raw_guitar_audio,native_posteriors
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s613_prepare_strict_live import download
from scripts.s773_causal_repro_by_stage import fingerprint

def audit(args):
    if args.out.exists():
        raise FileExistsError("S777 immutable report")
    args.out.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    result=dict(schema="S777_RAW_NATIVE_FRONTEND_REPRO",complete=False,
        torch_threads=args.threads,labels_used=False,promoted=False,
        source_MIDI_untouched=True,TEST_opened=False,HELD_opened=False,records=[],python=sys.version,platform=platform.platform(),processor=platform.processor())
    def save():
        (args.out/"report.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    save()
    samples={}
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for name,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                              ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            remote=protocol["guitar_archives"][name]
            download(remote["url"],root/"data"/filename,
                     remote["size"],remote["sha256"])
        original=[s for s in protocol["sources"]
                  if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            specs=expanded_train_specs(original,wav,ann)
            wanted=[next(s for s in specs if s["composition"]==group and s["name"].startswith("00_") and "_comp" in s["name"])
                    for group in (FIT_COMPS[0],CAL_COMPS[0])]
            for spec in wanted:
                audio,sha=raw_guitar_audio(spec,wav)
                ends,feature=feature_track(audio,block_size=4096)
                post,k=native_posteriors(feature,heads)
                selected=np.arange(3,len(ends),4)
                sharp=sharp_track(audio,ends[selected])
                picked=np.unique(np.clip(np.array([0,1,2,7,31,127,len(ends)//2,len(ends)-1]),
                    0,len(ends)-1)).astype(np.int64)
                sharp_indices=np.unique(np.clip(np.array([0,1,2,7,31,len(selected)//2,len(selected)-1]),
                    0,len(selected)-1)).astype(np.int64)
                key=str(len(result["records"]))
                samples[key+"_native_index"]=picked
                samples[key+"_native_FFT"]=np.asarray(feature[picked],np.float32)
                samples[key+"_native_post"]=np.asarray(post[picked],np.float32)
                samples[key+"_native_k"]=np.asarray(k[picked],np.float32)
                samples[key+"_sharp_index"]=sharp_indices
                samples[key+"_sharp"]=np.asarray(sharp[sharp_indices],np.float32)
                samples[key+"_pcm"]=np.asarray(audio[:min(65536,len(audio))],np.float32)

                result["records"].append(dict(name=spec["name"],composition=spec["composition"],
                    decoded_WAV_SHA=sha,samples=len(audio),frames=len(ends),
                    stages=dict(raw_audio=fingerprint(audio),
                        native_FFT_features=fingerprint(feature),
                        native_pitch_post=fingerprint(post),
                        native_K_post=fingerprint(k),
                        sharp_harmonic=fingerprint(sharp),
                        decision_ends=fingerprint(ends[selected]))))
                save()
                print("S777_STAGE",spec["name"],len(ends),flush=True)
    np.savez_compressed(args.out/"sampled_vectors.npz",**samples)
    result["complete"]=True
    save()
    print("S777_COMPLETE",json.dumps(dict(threads=args.threads,
        tracks=[r["name"] for r in result["records"]])),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--threads",type=int,choices=(1,2),required=True)
    audit(p.parse_args())
