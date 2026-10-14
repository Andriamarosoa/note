"""S453 — original S5/S8/S18/S80 external wrong-K provenance, no retraining.

Replays already frozen original predictions from 506 genuine IDMT dataset3 V88
events, and independently rematches XML pitches with the *unchanged*
S299/S451 protocol. Each source stage is audited against the same actual
annotation-matched native rows. Unmatched events never become true K0.
This exposes the first stage that predicts K0 or K2 instead of K3 without
claiming an intervention or a new independent blind holdout.
"""
from __future__ import annotations
import argparse,collections,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s190_score_frozen_dataset2_XML_exactk import parse_original_pitch_events
from scripts.s451_score_frozen_S450_IDMT_XML_true_notes import annotated_sets,sha_bytes
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s402_k_input_first_supervised_midi_pitch_network import need

STAGES=("V88_freeze","S5_consensus","S8_consensus","S9_consensus",
        "S13_consensus","S14_consensus","S15_consensus",
        "S16_consensus","S17_consensus","S18_consensus",
        "S18_original_S217","S72","S79","S80")

def fold_consensus(p,b):
    """Exact original S217 3-of-4 rule with label-blind V88 fallback."""
    need(p.shape==(4,len(b)),"S453 genuine four source folds needed")
    result=b.copy().astype(np.int8)
    consensus=np.zeros(len(b),bool)
    for i in range(len(b)):
        count=collections.Counter(int(k) for k in p[:,i])
        value,n=count.most_common(1)[0]
        if n>=3:
            result[i]=value;consensus[i]=True
    return result,consensus

def recount(truth,choices,mask):
    ix=np.flatnonzero(mask)
    if not len(ix):return dict(n=0,exact=0,accuracy=None,
                       pred_K0=0,K_under=0,K_over=0)
    t=truth[ix];v=choices[ix]
    return dict(n=int(len(ix)),exact=int(np.sum(t==v)),
                accuracy=float(np.mean(t==v)),
                pred_K0=int(np.sum(v==0)),
                K_under=int(np.sum(v<t)),K_over=int(np.sum(v>t)),
                pred_K_hist=np.bincount(v,minlength=7).tolist())

def run(s276,s277,s280,s281,s282,s297,s450,s451_report,out):
    need(not out.exists(),"S453 immutable external diagnostic already exists")
    earlier=json.loads(s451_report.read_text())
    manifest=json.loads(s276.read_text())
    need(earlier["status"]=="S451_ORIGINAL_S80_EXTERNAL_IDMT_TRUE_NOTE_IDENTITY_PAIRED"
         and earlier["matched_annotation_onsets"]==47
         and earlier["pitches_in_supported_MIDI40to88"]==45
         and earlier["unknown_or_unmatched_events_excluded"]==459
         and not earlier["experimental_no_promotion"] is False,
         "S453 must independently reproduce the exact S451 comparison population")
    need(manifest["status"]=="S276_IDMT_CANDIDATE_COHORT_CENTRAL_DIRECTORY_ONLY_NOT_SCORED"
         and manifest["counts_WAV"]["dataset3"]==5
         and manifest["XML_MIDI_annotation_contents_read"] is False,
         "S453 original IDMT sample selection altered")
    def load(path):
        with np.load(path,allow_pickle=False) as z:
            need("row_id" in z.files,"S453 source missing real event identity")
            return {x:z[x].copy() for x in z.files}
    native=load(s277);five=load(s280);stages=load(s281)
    neutral=load(s282);last=load(s297);notes=load(s450)
    ids=native["row_id"].astype(str);n=len(ids)
    need(n==506 and len(set(ids))==n and
         all(np.array_equal(ids,obj["row_id"].astype(str))
             for obj in (five,stages,neutral,last,notes)),
         "S453 original V88 native row IDs or fourfold alignment drift")
    member=native["member"].astype(str)
    starts=native["start_samples"].astype(np.int64)
    baseline=native["incumbent_K"].astype(np.int8)
    true_original_S80=last["s80_K"].astype(np.int8)
    need(np.array_equal(five["incumbent_freeze_K"],baseline)
         and np.array_equal(stages["freeze_K"],baseline)
         and np.array_equal(neutral["freeze_K"],baseline)
         and np.array_equal(last["incumbent_freeze_K"],baseline)
         and np.array_equal(notes["s80_K"],true_original_S80)
         and np.isin(true_original_S80,range(7)).all()
         and np.array_equal((notes["S450_pitch_slots"]>=0).sum(1),true_original_S80),
         "S453 historical S80 source or downstream named note budget changed")
    original={"V88_freeze":baseline}
    allfold={}
    s5=five["original_S5_K_by_fold"].astype(np.int8)
    allfold["S5_consensus"]=s5
    need(s5.shape==(4,n) and np.array_equal(stages["fold_ids"],[0,1,2,4])
         and np.array_equal(five["fold_ids"],[0,1,2,4]),
         "S453 original S5 fourfold model provenance lost")
    for stage in ("S8","S9","S13","S14","S15","S16","S17","S18"):
        allfold[stage+"_consensus"]=stages[stage+"_K"].astype(np.int8)
    for label,arr in allfold.items():
        result,ok=fold_consensus(arr,baseline)
        original[label]=result
    original["S18_original_S217"]=neutral["S18_piece_neutral_consensus_K"].astype(np.int8)
    need(np.array_equal(original["S18_consensus"],original["S18_original_S217"]),
         "S453 independent fourfold S217 replay differs from original S282 archive")
    for name,field in (("S72","s72_K"),("S79","s79_K"),("S80","s80_K")):
        original[name]=last[field].astype(np.int8)
    need(set(original)==set(STAGES) and all(len(z)==n for z in original.values()),
         "S453 incomplete original source stages")
    # Every XML must be opened only after immutable prediction inventory checks.
    group=manifest["candidate_groups"]["dataset3_first"]
    directory={obj["wav_filename_sha256"]:obj for obj in group}
    need(set(member)==set(directory) and len(directory)==5,
         "S453 original source song provenance altered")
    with urllib.request.urlopen(urllib.request.Request(
        API,headers={"Accept":"application/json","User-Agent":"S453-original-S80-stage-audit"}),
        timeout=60) as response:
        meta=audit(json.load(response))
    zipremote=HttpRangeZIP(meta["source_url"],meta["size_bytes"])
    targets=np.full(n,-1,np.int8)
    supported=np.zeros(n,bool)
    proofs=[]
    with zipfile.ZipFile(zipremote) as z:
        need(sha_bytes("\n".join(sorted(z.namelist())).encode())==
             manifest["zip_central_names_sha256"],
             "S453 original source ZIP changed from S276 preregistration")
        for identity,obj in sorted(directory.items()):
            file=obj["annotation_path_name_only"]
            need(file in z.namelist() and file.lower().endswith(".xml"),
                 "S453 missing unique original paired XML")
            bytes_xml=z.read(file)
            events=parse_original_pitch_events(bytes_xml)
            ii=np.flatnonzero(member==identity)
            ii=ii[np.argsort(starts[ii],kind="stable")]
            labeled=annotated_sets(events,starts[ii]/44100.)
            for i,truth in zip(ii,labeled):
                if truth is None:continue
                targets[i]=len(truth)
                supported[i]=bool(all(40<=p<=88 for p in truth))
            proofs.append(dict(audio_filename_sha256=identity,
                XML_sha256=sha_bytes(bytes_xml),n_events=len(ii),
                note_matched=sum(x is not None for x in labeled)))
    any_matched=targets>=1;valid=any_matched&supported;poly=valid&(targets>=2)
    need(int(any_matched.sum())==47 and int(valid.sum())==45
         and int(poly.sum())==21 and
         int((~any_matched).sum())==459,
         "S453 independently reconstructed labels do not match S451 source")
    report={}
    for label in STAGES:
        pred=original[label]
        report[label]=dict(all_506_prediction_histogram=np.bincount(
              pred,minlength=7).tolist(),
            source_matched=recount(targets,pred,valid),
            true_poly=recount(targets,pred,poly),
            true_K2=recount(targets,pred,valid&(targets==2)),
            true_K3=recount(targets,pred,valid&(targets==3)),
            true_K4=recount(targets,pred,valid&(targets==4)),
            wrong_K0_on_matched=int(np.sum((pred==0)&valid)),
            wrong_K0_true_poly=int(np.sum((pred==0)&poly)))
    for stage in STAGES:
        if stage=="V88_freeze":continue
        before=original["V88_freeze"];after=original[stage]
        report[stage]["paired_vs_original_V88_matched"]=dict(
           fixes=int(np.sum(valid&(before!=targets)&(after==targets))),
           regressions=int(np.sum(valid&(before==targets)&(after!=targets))),
           net=int(np.sum(valid&(after==targets))-np.sum(valid&(before==targets))))
    error_ids=np.flatnonzero(poly&(original["S80"]!=targets))
    cases=[]
    for i in error_ids:
        first=next((s for s in STAGES if original[s][i]==0),None)
        stage_chain={s:int(original[s][i]) for s in STAGES}
        chain_votes={s:dict(Counter=collections.Counter(map(int,allfold[s][:,i]))) for s in allfold}
        cases.append(dict(source_row_id=str(ids[i]),real_song_SHA=str(member[i]),
           true_K=int(targets[i]),original_S80_K=int(original["S80"][i]),
           first_stage_predicted_K0=first,
           exact_stage_chain=stage_chain,
           original_fold4_votes={s:{str(k):int(v) for k,v in
                 chain_votes[s]["Counter"].items()} for s in chain_votes}))
    matched_rows=[dict(source_row_id=str(ids[i]),member_SHA=str(member[i]),
        original_S80_K=int(original["S80"][i]),
        actual_K=int(targets[i]),within_supported_notes=bool(supported[i]),
        each_original_stage_K={stage:int(original[stage][i]) for stage in STAGES})
        for i in np.flatnonzero(any_matched)]
    out.mkdir(parents=True)
    path=out/"S453_independent_original_IDMT_stage_audit.jsonl"
    path.write_text("".join(json.dumps(rec)+"\n" for rec in matched_rows))
    result=dict(status="S453_ORIGINAL_S5_S8_S18_S80_EXTERNAL_IDMT_WRONGK_STAGE_AUDIT",
      genuine_original_S80_all_stages_joined_by_row=True,
      original_source_predictions_untouched=True,
      source_native_events=n,source_WAV_count=5,
      matched_note_annotation_events=int(any_matched.sum()),
      valid_pitch_supported_events=int(valid.sum()),
      matched_true_poly_events=int(poly.sum()),
      unknown_unmatched_not_K0=int((~any_matched).sum()),
      historical_IDMT_annotations_previously_seen=True,
      no_new_blind_cohort=True,no_retraining_or_threshold_search=True,
      no_promotion=True,
      original_models_S5_fourfold=4,
      original_stage_ledger=report,wrong_S80_poly_events=cases,
      source_SHA256=dict(S276=sha(s276),S277=sha(s277),
          S280=sha(s280),S281=sha(s281),S282=sha(s282),
          original_S80=sha(s297),S450_names=sha(s450)),
      original_XML_SHA_proofs=proofs,annotated_rows_SHA256=sha(path))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S453 ROOT-CAUSE",json.dumps(dict(
        true_poly=21,
        S80=report["S80"]["true_poly"],
        freeze=report["V88_freeze"]["true_poly"],
        S5=report["S5_consensus"]["true_poly"],
        S8=report["S8_consensus"]["true_poly"],
        S18=report["S18_consensus"]["true_poly"],
        S72=report["S72"]["true_poly"],
        poly_wrong_K0=report["S80"]["wrong_K0_true_poly"],
        true_K3_by_stage={name:report[name]["true_K3"] for name in (
            "V88_freeze","S5_consensus","S8_consensus","S18_consensus","S80")})),
          flush=True)

def selftest():
    b=np.array([0,2,0,3],np.int8)
    folded=np.array([[2,3,2,3],[2,3,2,2],[2,3,2,2],[0,0,3,0]],np.int8)
    selected,supported=fold_consensus(folded,b)
    need(selected.tolist()==[2,3,2,2] and supported.tolist()==[True,True,True,True],
         "S453 cannot reproduce original heldfold votes")
    weak=np.array([[2,2,2,2],[2,3,3,3],[3,2,3,2],[3,3,2,3]],np.int8)
    out,supported=fold_consensus(weak,b)
    need(np.array_equal(out,b) and not np.any(supported),
         "S453 S217 abstention must fall back exactly to true historical V88 K")
    truth=np.array([2,3,2,-1],np.int8)
    subset=(truth>=2)
    assert recount(truth,np.array([2,0,3,0]),subset)["exact"]==1
    print("S453 SELFTEST PASS fold-neutral historical K0 K2 K3 stage replay")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for flag in ("s276","s277","s280","s281","s282","s297","s450",
                 "s451-report","out"):
        p.add_argument("--"+flag,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        keys=("s276","s277","s280","s281","s282","s297","s450",
              "s451_report","out")
        for key in keys:need(getattr(a,key)is not None,"S453 missing --"+key)
        run(*(getattr(a,k) for k in keys))
