"""S419 — composition-held note-head gate: S402 vs S416, preserving each S80 K.

A *simple* learned composition-only selector chooses which native MIDI pitch
head to use for each PREDICTED K, using accuracy differences on compositions
OTHER than the held-out test composition. True K never enters inference.
Every row outputs exactly its S80 K; no YourMT3 at train or inference.

Caution: both source heads have their own 4-fold OOF outputs, whose training
can include the gate's held-out fold when used on OTHER folds to estimate the
gate. This is a *development meta-gate*, not an independently nested model.
No baseline or official reference is changed by writing this evidence.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    load,labels,composition_key,composition_fold,need)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha

N=59309
POLY=7385
GLOBAL_K_CORRECT=53529
POLY_K_CORRECT=6054
MIN_TRAIN_NET_ADVANTAGE=12
MIN_BIN_TRAIN_N=130
def exact_match(a,b):
    aa=np.sort(np.asarray(a)[np.asarray(a)>=0])
    bb=np.sort(np.asarray(b)[np.asarray(b)>=0])
    return bool(np.array_equal(aa,bb))
def run(audio,s80,baseline,s416,truth,annotations,out):
    need(not out.exists(),'S419 immutable alternative cannot overwrite')
    a=load(audio);r=load(s80);b=load(baseline);c=load(s416);t=load(truth)
    row=a['row_id'].astype(str)
    k=r['s80_K'].astype(np.int8)
    require_same=(len(row)==N and len(set(row))==N
      and all(np.array_equal(row,z['row_id'].astype(str)) for z in (r,b,c,t))
      and np.array_equal(k,b['s80_K']) and np.array_equal(k,c['s80_K'])
      and np.array_equal((b['midi_pitch_slots']>=0).sum(1),k)
      and np.array_equal((c['midi_pitch_slots']>=0).sum(1),k)
      and int((k==t['true_K']).sum())==GLOBAL_K_CORRECT
      and int(((k==t['true_K'])&(t['true_K']>=2)).sum())==POLY_K_CORRECT)
    need(require_same,'S419 exact original S80 K and source rows drift')
    y,slots,meta=labels(a,annotations,t)
    member=a['member'].astype(str)
    comp=np.array([composition_key(m) for m in member])
    folds=np.array([composition_fold(g) for g in comp],np.int8)
    old=b['midi_pitch_slots'].astype(np.int16)
    proposed=c['midi_pitch_slots'].astype(np.int16)
    correct0=np.array([exact_match(old[i],slots[i]) for i in range(N)],bool)
    correct1=np.array([exact_match(proposed[i],slots[i]) for i in range(N)],bool)
    selected=np.full((N,6),-1,np.int16)
    chosen_byfold={}
    proofs=[]
    for f in range(4):
        train=folds!=f;test=folds==f
        need(train.sum()>30000 and test.sum()>3000
             and not (set(comp[train])&set(comp[test])),
             'S419 cannot select architecture on held test composition')
        bins={}
        for kk in range(7):
            train_bin=train&(k==kk)
            original=int(correct0[train_bin].sum())
            candidate=int(correct1[train_bin].sum())
            # True outcomes TRAIN ONLY; note classifications from models
            # were crossfitted in their OWN folds, not nested here.
            net=candidate-original
            allow=bool(kk>0 and int(train_bin.sum())>=MIN_BIN_TRAIN_N
                      and net>=MIN_TRAIN_NET_ADVANTAGE)
            bins[str(kk)]=dict(train_n=int(train_bin.sum()),S402_correct=original,
                 S416_correct=candidate,net_training=net,use_S416=allow)
            mask=test&(k==kk)
            selected[mask]=proposed[mask] if allow else old[mask]
        chosen_byfold[str(f)]=bins
        proofs.append(dict(held_fold=f,train_compositions=len(set(comp[train])),
             held_compositions=len(set(comp[test])),composition_overlap=0,
             held_events=int(test.sum()),
             K_source_predictions_untouched=True))
    need(np.array_equal((selected>=0).sum(1),k)
         and int((selected>=0).sum())==int(k.sum())==33475,
         'CRITICAL S419 post-choice S80 K regression')
    mixed=np.array([exact_match(selected[i],slots[i]) for i in range(N)],bool)
    bytrue={}
    for kk in range(7):
        mask=t['true_K']==kk
        bytrue[str(kk)]=dict(n=int(mask.sum()),
             S402_correct=int(correct0[mask].sum()),
             S416_correct=int(correct1[mask].sum()),
             S419_correct=int(mixed[mask].sum()),
             corrections_vs_S402=int((mixed[mask]&~correct0[mask]).sum()),
             regressions_vs_S402=int((~mixed[mask]&correct0[mask]).sum()),
             exactK_regressions=0)
    scope=t['true_K']>=2
    corrected=int((mixed[scope]&~correct0[scope]).sum())
    regressed=int((~mixed[scope]&correct0[scope]).sum())
    summary=dict(S402_poly_chord_correct=int(correct0[scope].sum()),
       S416_poly_chord_correct=int(correct1[scope].sum()),
       S419_poly_chord_correct=int(mixed[scope].sum()),
       total_true_poly=POLY,
       corrections_vs_S402_poly=corrected,
       regressions_vs_S402_poly=regressed,
       net_vs_S402_poly=corrected-regressed,
       S402_all_chord_correct=int(correct0.sum()),
       S416_all_chord_correct=int(correct1.sum()),
       S419_all_chord_correct=int(mixed.sum()),
       S80_K_predictions_preserved_all_rows=True,
       exactK_regressions=0)
    out.mkdir(parents=True)
    result=out/'S419_NATIVE_K_PROTECTED_HEAD_GATE_59309.npz'
    np.savez_compressed(result,row_id=row,s80_K=k,midi_pitch_slots=selected,
       composition_fold=folds)
    # Independent exported source event note count audit.
    export=out/'S419_note_onset_counts.csv'
    with export.open('w',encoding='utf8') as fp:
        fp.write('row_id,s80_K,native_MIDI_note_count\n')
        for rid,kk,nn in zip(row,k,(selected>=0).sum(1)):
            fp.write(f'{rid},{int(kk)},{int(nn)}\n')
    with export.open(encoding='utf8') as fp:
        lines=fp.read().splitlines()[1:]
    need(len(lines)==N and all(
        int(fields[1])==int(fields[2]) and fields[0]==row[i] and
        int(fields[1])==int(k[i])
        for i,line in enumerate(lines) if (fields:=line.split(','))),
        'S419 independent CSV exported K count mismatch')
    report=dict(status='S419_NATIVE_K_FIRST_SONGHELD_NOTE_HEAD_GATE',
       model_selection_uses_only_other_composition_labels=True,
       meta_gate_not_nest_fully_with_upstream_S402_S416=True,
       gate_input_only_S80_predicted_K=True,
       both_head_sources_native_without_YourMT3=True,
       true_K_NEVER_used_to_choose_head_for_prediction=True,
       no_external_YourMT3_at_training_or_inference=True,
       actual_exactK_correct_global=GLOBAL_K_CORRECT,
       actual_exactK_correct_poly=POLY_K_CORRECT,
       any_exactK_regression=False,
       number_of_original_events=N,number_of_new_notes=int(k.sum()),
       metrics=summary,by_true_K=bytrue,fold_gates=chosen_byfold,
       song_split_proofs=proofs,
       source_hashes=dict(audio=sha(audio),S80=sha(s80),S402=sha(baseline),
          S416=sha(s416),JAMS=sha(annotations)),
       output_prediction_hash=sha(result),
       exported_CSV_hash=sha(export),
       S402_active_reference_UNCHANGED=True,
       S408_active_offset_reference_UNCHANGED=True,
       validation_entire_S80_pipeline_independent=False,
       no_promotion=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('S419 SCORES',json.dumps(summary),flush=True)
    print('S419 FOLD GATES',json.dumps(chosen_byfold),flush=True)
    print('S419 BY K',json.dumps(bytrue),flush=True)
    print('S419 PASS all 59309 native NoteOn counts EXACT S80 K, zero regression',flush=True)

def selftest():
    base=np.array([48,-1,-1,-1,-1,-1])
    same=np.array([48,-1,-1,-1,-1,-1])
    diff=np.array([60,-1,-1,-1,-1,-1])
    need(exact_match(base,same) and not exact_match(base,diff),
         'S419 strict note identity target')
    need(exact_match([-1]*6,[-1]*6),
         'K0 empty chord correct for both heads')
    print('S419 SELFTEST PASS note chord equality including K0')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    for n in ('audio','s80','s402','s416','truth','annotations','out'):
        p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for n in ('audio','s80','s402','s416','truth','annotations','out'):
            need(getattr(a,n)is not None,'missing '+n)
        run(a.audio,a.s80,a.s402,a.s416,a.truth,a.annotations,a.out)
