"""S508 source-true head coverage and learnability audit (oracle CAPACITY ONLY).

Does the 13-head neural selector have the CORRECT note-set among its
available proposals? If not, can an independent per-pitch recombination
from the observed union of acoustic pitches recover it? Count unobserved
true pitches separately. Oracle uses annotations FOR DIAGNOSIS ONLY;
never feed coverage results into live inference, train on these same rows,
or claim this is a valid learned score.
"""
from __future__ import annotations
import argparse,collections,json,hashlib
from pathlib import Path
import numpy as np
def need(x,s):
    if not x:raise ValueError(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def counts(truth,notes,baseline,mask):
    old=notes[np.arange(len(truth)),baseline]
    orig=np.all(old==truth,axis=1)
    available=mask[:,:,None]
    exact=np.all(notes==truth[:,None,:],axis=2)&mask
    has_any=exact.any(axis=1)
    any_pitch=np.any(notes&available,axis=1)
    covers=np.all((~truth)|any_pitch,axis=1)
    miss=np.sum(truth&(~any_pitch),axis=1)
    note_poly=truth.sum(axis=1)>=2
    return dict(n=len(truth),baseline_correct=int(orig.sum()),
        individually_correct_proposal_available=int(has_any.sum()),
        oracle_additional_correct_if_choose_existing_head=int(np.sum(~orig&has_any)),
        impossible_for_any_whole_head=int(np.sum(~has_any)),
        union_can_cover_all_true_pitches=int(covers.sum()),
        theoretical_extra_correct_if_select_individual_pitches=int(np.sum(~orig&covers)),
        missing_at_least_one_true_pitch_from_all_heads=int(np.sum(miss>0)),
        missing_true_pitch_total=int(miss.sum()),
        baseline_poly_correct=int(np.sum(orig&note_poly)),
        oracle_head_poly_correct=int(np.sum(has_any&note_poly)),
        union_covers_all_true_poly_notes=int(np.sum(covers&note_poly)),
        true_poly_event_count=int(note_poly.sum()),
        oracle_all_correct_maybe_only_with_ground_truth=True)
def run(args):
    r=json.loads(args.report.read_text())
    need(not args.out.exists() and
         r["status"]=="S505_RETROSPECTIVE_REAL_EIGHT_RECORDINGS_HEAD_PITCH_SET_TRAIN_ROWS"
         and r["audio_proposal_truth_table_SHA256"]==sha(args.data),
         "S508 original guitar and piano note-head source must match S505")
    with np.load(args.data,allow_pickle=False) as z:
        d={k:z[k].copy() for k in z.files}
    source=d["corpus"].astype(str)
    truth=d["truth_note_bits"].astype(bool)
    notes=d["head_note_bits"].astype(bool)
    base=d["selected_BP_baseline_head"].astype(int)
    mask=d["head_mask"].astype(bool)
    record=d["recording_sha"].astype(str)
    truthK=truth.sum(axis=1)
    sK=d["source_K"].astype(int)
    output={}
    for corpus in ("guitar_S80","piano_V88"):
        idx=np.flatnonzero(source==corpus)
        y=truth[idx];p=notes[idx];b=base[idx];m=mask[idx]
        grouped={k:counts(y[truthK[idx]==k],p[truthK[idx]==k],
                  b[truthK[idx]==k],m[truthK[idx]==k]) for k in range(7)}
        by_source_k={str(k):counts(y[sK[idx]==k],p[sK[idx]==k],
                  b[sK[idx]==k],m[sK[idx]==k]) for k in range(7)}
        by_rec={}
        for rec in sorted(set(record[idx])):
            chosen=idx[record[idx]==rec]
            by_rec[rec]=counts(truth[chosen],notes[chosen],base[chosen],mask[chosen])
        main=counts(y,p,b,m)
        need(main["baseline_correct"]==r["by_source"][corpus]["raw_BP_base_full_note_sets_correct"],
             "S508 baseline original note set unexpectedly shifted")
        main["positive_true_onsets_without_any_BasicPitch_canonical_group"]=(
            r["by_source"][corpus]["annotated_true_attacks_NOT_matched"])
        main["true_original_attacks_total"]=r["by_source"][corpus]["truth_original_positive_attacks"]
        output[corpus]=dict(summary=main,by_true_K={str(k):v for k,v in grouped.items()},
                            by_original_source_K=by_source_k,by_recording=by_rec)
        print("S508 HEAD COVERAGE UPPER BOUND",json.dumps(dict(source=corpus,**main)),flush=True)
    args.out.mkdir(parents=True)
    rep=dict(status="S508_TRUE_NOTE_PITCH_HEAD_SELECTION_ORACLE_CAPACITY_NOT_MODEL",
       oracle_used_true_notes_ONLY_for_posthoc_diagnostic=True,
       independent_per_note_selection_not_yet_trained=True,
       no_unobserved_pitches_invented=True,
       grouping_trueK_not_an_inference_feature=True,
       eight_sources_already_previously_exposed=True,
       source_K_new_piano_vs_K_active_guitar_different=True,
       no_full_stream_negative_annotations=True,
       no_original_S80_Live_changes=True,no_promotion=True,
       by_source=output,source_S505_SHA256=sha(args.data))
    (args.out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
def selftest():
    y=np.zeros((2,128),bool);y[:,52]=True;y[1,55]=True
    p=np.zeros((2,2,128),bool);p[:,:,52]=True;p[1,1,55]=True
    z=counts(y,p,np.array([0,0]),np.ones((2,2),bool))
    need(z["oracle_additional_correct_if_choose_existing_head"]==1
         and z["union_can_cover_all_true_pitches"]==2,
         "S508 pitch capacity oracle must count actual unseen-note coverage")
    print("S508 SELFTEST PASS candidate head oracle vs independently selectable pitch coverage")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for k in ("data","report","out"):p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("data","report","out"):need(getattr(a,k)is not None,"S508 missing "+k)
        run(a)
