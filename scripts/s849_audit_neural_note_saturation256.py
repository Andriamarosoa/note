"""S849 forensic audit for S847 frozen neural shift selector, PCM256.

No training, no GT leakage, no fake model improvement. Three *diagnostic*
runs reset the GRU never / every 8 / every 32 PCM256 blocks. Same real
guitar from S848. Reports softmax class dominance, variation and chosen
notes; diversifying labels by subtraction is NOT a correctness metric.

Reference standard +12 remains a separately published S848 baseline.
"""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from onset256.real_recorded_transposer256 import NeuralShiftSelector256,HOP,SR,ABI
from scripts.s847_train_real_recorded_transposer256 import read_recorded_bank

def audit(real_input,model_path,bank_dir,out):
    out=Path(out)
    if out.exists():raise FileExistsError("immutable audit")
    sr,pcm=wavfile.read(real_input)
    if sr!=SR or pcm.ndim!=1 or pcm.dtype!=np.int16:
        raise ValueError("S849 audit requires identical S848 original mono PCM16")
    audio=pcm.astype(np.float32)/32768
    torch.set_num_threads(1)
    ckpt=torch.load(model_path,map_location='cpu',weights_only=True)
    if ckpt["abi"]!=ABI or ckpt['trained_with_original_PCM_HOP']!=HOP:
        raise ValueError("wrong trained model")
    model=NeuralShiftSelector256().eval()
    model.load_state_dict(ckpt["model_state"],strict=True)
    bank,meta=read_recorded_bank(bank_dir)
    pitches=np.asarray(sorted(bank),np.int64)
    report=dict(schema="S849_TRUE_NEURAL_PITCH_SELECTION_SATURATION_AUDIT_V1",
                source=str(real_input),source_samples=len(audio),
                exact_new_audio_hop=HOP,zero_future_pcm=True,no_hold=True,
                actual_training_horizon_hops=8,
                model_promoted=False,onset_event_F1_NOT_MEASURED=True,
                class_bias_calibration_not_attempted=True,
                protocols=[])
    for restart in (0,8,32):
        state=None;target=[];source=[]
        with torch.inference_mode():
            for i,start in enumerate(range(0,len(audio),HOP)):
                if restart and i%restart==0:state=None
                block=np.zeros(HOP,np.float32)
                end=min(start+HOP,len(audio))
                block[:end-start]=audio[start:end]
                o,state=model(torch.from_numpy(block)[None,None,:],
                               torch.tensor([12],dtype=torch.long),state=state)
                target.append(torch.sigmoid(o['target_logits'][0,0]).numpy())
                source.append(torch.sigmoid(o['source_logits'][0,0]).numpy())
        target=np.asarray(target,dtype=np.float64)[:,pitches-40]
        source=np.asarray(source,dtype=np.float64)[:,pitches-40]
        choices=pitches[target.argmax(axis=1)]
        tops=Counter(map(int,choices))
        source_choices=pitches[source.argmax(axis=1)]
        target_variance=target.std(axis=0)
        report["protocols"].append(dict(
            reset_gru_every_hops=restart,
            frames=len(choices),
            top_midi_frequency={str(p):n for p,n in tops.most_common()},
            top_one_fraction=max(tops.values())/len(choices),
            source_top_midi_frequency={str(p):n for p,n in
                                       Counter(map(int,source_choices)).most_common()},
            median_note_probability_std=float(np.median(target_variance)),
            maximum_note_probability_std=float(target_variance.max()),
            expected_output_is_a_true_transposed_phrase=False))
    out.mkdir(parents=True)
    (out/'resultat.json').write_text(json.dumps(report,indent=2)+"\n")
    print("S849_REAL_NEURAL_NOTE_SATURATION",json.dumps(report["protocols"]),flush=True)
    return report

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--wav",type=Path,required=True)
    a.add_argument("--model",type=Path,required=True)
    a.add_argument("--bank",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    x=a.parse_args()
    audit(x.wav,x.model,x.bank,x.out)
