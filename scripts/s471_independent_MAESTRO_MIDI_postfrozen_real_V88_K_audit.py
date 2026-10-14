"""S471: TRUE THIRD CORPUS - external MAESTRO piano MIDI after audio PREMIDI freeze.

Read MIDI ONLY AFTER SHA-verifying sealed genuine original V88+S466 outputs
from S470. Groups MIDI note-ons 20ms, matches native V88 events at <=80ms
using independent maximum-cardinality one-to-one bipartite assignment.
Reports real onset recall, native candidate multiplication, and original
V88-only conditional exact-K, plus S466 birth-gate recall.
Does NOT report a S80 accuracy: full S80 NOT run on this source.
Does NOT tune thresholds or learn from MAESTRO note labels.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,urllib.parse,collections,time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import mido
SR=44100
TOL=.080
GROUP=.020
SOUND=.030
def need(c,s):
    if not c:raise ValueError(s)
def digest(blob):return hashlib.sha256(blob).hexdigest()
def sha(path):return digest(Path(path).read_bytes())
def fetch_midi(mirror,path):
    need(path.endswith(".midi") and ".." not in path and not path.startswith("/"),
         "S471 unsafe MIDI file path outside S469 manifest")
    url=mirror+urllib.parse.quote(path,safe="/")+"?download=true"
    req=urllib.request.Request(url,headers={
        "User-Agent":"S471-PostFreeze-MAESTRO3-midi-only",
        "Accept-Encoding":"identity"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=120) as r:
                data=r.read(12_000_001)
                need(r.status==200 and 100<len(data)<12_000_000 and data[:4]==b"MThd",
                     "S471 third-source paired MIDI inaccessible or not Standard MIDI")
                return data
        except Exception as e:
            if attempt==2:raise
            print("S471 retry MIDI",path,type(e).__name__,flush=True)
            time.sleep(1.5*(attempt+1))
    raise RuntimeError("MIDI unavailable")
def midi_events(raw):
    from io import BytesIO
    mf=mido.MidiFile(file=BytesIO(raw))
    tempo=500000
    t=0.
    active=collections.defaultdict(list)
    events=[]
    onset_raw=0
    for msg in mido.merge_tracks(mf.tracks):
        t+=mido.tick2second(msg.time,mf.ticks_per_beat,tempo)
        if msg.type=="set_tempo":
            tempo=msg.tempo
        elif msg.type=="note_on" and msg.velocity>0:
            key=(getattr(msg,"channel",0),int(msg.note))
            active[key].append(float(t));onset_raw+=1
        elif msg.type=="note_off" or (
            msg.type=="note_on" and getattr(msg,"velocity",1)==0):
            key=(getattr(msg,"channel",0),int(msg.note))
            if active[key]:
                start=active[key].pop(0)
                if t>start:events.append((start,float(t),int(msg.note)))
    events.sort(key=lambda a:(a[0],a[2],a[1]))
    need(onset_raw>0 and len(events)>0,"S471 original MAESTRO MIDI missing note events")
    return events

def onset_groups(events,offset,length):
    relevant=[(st-offset,en-offset,int(p)) for st,en,p in events
              if offset-.05<=st<offset+length+.05]
    groups=[]
    for st,en,p in relevant:
        if not groups or st-groups[-1][0]>GROUP:
            groups.append([float(st),[]])
        groups[-1][1].append((float(st),float(en),int(p)))
    anchors=[]
    for anchor,_ in groups:
        if not (0<=anchor<length):continue
        sounding=sorted({int(p) for st,en,p in relevant
            if st<=anchor+SOUND and en>anchor+SOUND})
        # MIDI notes begun BEFORE the excerpt also sound at current attack.
        sounding=sorted({int(p) for st,en,p in events
            if st<=offset+anchor+SOUND and en>offset+anchor+SOUND})
        anchors.append(dict(onset=float(anchor),pitches=sounding,
                            trueK=len(sounding),
                            all_MIDI40_88=all(40<=q<=88 for q in sounding)))
    return anchors

def match_times(native,labels,maxdist=TOL):
    n=len(native);m=len(labels)
    if not n or not m:return []
    dist=np.abs(native[:,None]-labels[None,:])
    costs=np.concatenate((np.where(dist<=maxdist,dist/(maxdist+1e-9)*.25,1e5),
                           np.ones((n,n),np.float64)),axis=1)
    rows,cols=linear_sum_assignment(costs)
    result=[(int(i),int(j),float(native[i]-labels[j]))
           for i,j in zip(rows,cols) if j<m and dist[i,j]<=maxdist]
    need(len({j for i,j,d in result})==len(result),
         "S471 independent maximum-cardinality matching reused a MIDI note group")
    return result
def run(freeze,freeze_report,manifest,out):
    need(not out.exists(),"S471 post-freeze MIDI independent report immutable")
    meta=json.loads(manifest.read_text())
    proof=json.loads(freeze_report.read_text())
    need(meta["status"]=="S469_PREREGISTERED_MAESTRO_V3_THIRD_SOURCE_FILENAME_ONLY"
         and not meta["MIDI_note_contents_read"] and not meta["WAV_file_contents_read"]
         and proof["status"]=="S470_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI"
         and proof["output_sha256"]==sha(freeze)
         and proof["preregistered_S469_SHA"]==sha(manifest)
         and proof["prelabel_predictions_frozen"]
         and not proof["paired_MIDI_annotation_files_OPENED"]
         and proof["original_complete_S80_NOT_REPLAYED"]
         and proof["three_original_piano_recordings"]==3,
         "S471 cannot open MIDI until original real-source V88 and birth predictions are sealed")
    with np.load(freeze,allow_pickle=False) as z:
        ids=z["row_id"].astype(str)
        member=z["member"].astype(str)
        starts=z["start_samples"].astype(np.int64)
        K=z["original_V88_K"].astype(np.int8)
        suppress=z["suggested_suppression_mask"].astype(bool)
        probs=z["original_GuitarSet_birth4_prob"].astype(float)
    need(len(ids)==proof["original_V88_event_count"] and
         len(set(ids))==len(ids) and
         probs.shape==(4,len(ids)) and
         np.isin(K,range(7)).all() and
         np.all(~suppress[K>0]),
         "S471 original piano event ID or source K changed post-freeze")
    paths={item["filename_SHA256"]:item for item in meta["selected_sources"]}
    need(set(member)==set(paths) and len(paths)==3,
         "S471 source piano filenames not preregistered")
    recordings=[];ledger=[]
    totals=collections.Counter()
    for ident,item in sorted(paths.items()):
        raw=fetch_midi(meta["audio_mirror_directory"],item["MIDI_filename_path_ONLY_not_opened"])
        note=midi_events(raw)
        anchors=onset_groups(note,float(item["audio_excerpt_start_seconds"]),
                                  float(item["audio_excerpt_duration_seconds"]))
        at=np.flatnonzero(member==ident)
        order=at[np.argsort(starts[at],kind="stable")]
        ontime=starts[order]/SR
        expected=np.asarray([g["onset"] for g in anchors],float)
        pairs=match_times(ontime,expected,TOL)
        valid=[]
        lost=[]
        old_correct=0;poly_old_correct=0;poly_n=0
        for i,j,delta in pairs:
            actual=anchors[j]
            idx=order[i]
            if actual["trueK"]<1 or not actual["all_MIDI40_88"] or actual["trueK"]>6:
                continue
            valid.append((i,j,int(idx),float(delta),actual))
            if int(K[idx])==actual["trueK"]:old_correct+=1
            if actual["trueK"]>=2:
                poly_n+=1;poly_old_correct+=int(K[idx]==actual["trueK"])
            if suppress[idx]:lost.append((i,j,idx,actual))
        matchids={i for i,j,d in pairs}
        far=np.min(np.abs(ontime[:,None]-expected[None,:]),axis=1)>TOL if len(expected) else np.ones(len(ontime),bool)
        rec=dict(
            piano_filename_SHA=ident,piano_composer=item["composer"],
            MIDI_bytes_SHA256=digest(raw),
            annotated_MIDI_note_events_total=len(note),
            MIDI_attack_groups_in_excerpt=len(anchors),
            original_V88_native_candidate_count=len(at),
            original_V88_native_K0=int(np.sum(K[at]==0)),
            matched_onsets_independent_80ms=len(pairs),
            original_80ms_midi_onset_recall=(len(pairs)/len(anchors) if anchors else None),
            original_native_candidates_more_than80ms_from_MIDI=int(far.sum()),
            original_distant_predicted_V88_K0=int(np.sum(far&(K[order]==0))),
            note_set_scoreable_pairs_in_MIDI40to88=len(valid),
            poly_scoreable_matched_pairs=poly_n,
            original_V88_correct_K_on_scoreable=old_correct,
            original_V88_correct_K_on_poly_pairs=poly_old_correct,
            S466_birth_gate_proposed_dropped_total=int(np.sum(suppress[at])),
            S466_birth_gate_missed_matched_true_onsets=int(np.sum(
                [suppress[order[i]] for i,j,d in pairs])),
            S466_birth_gate_missed_scoreable_true_poly=int(np.sum(
                [suppress[idx] for i,j,idx,delta,a in valid if a["trueK"]>=2])),
            octave_piano_not_comparable_to_guitar_timbre=True)
        recordings.append(rec)
        totals.update(native=len(at),nativeK0=rec["original_V88_native_K0"],
             midi_groups=len(anchors),matched=len(pairs),
             far=rec["original_native_candidates_more_than80ms_from_MIDI"],
             farK0=rec["original_distant_predicted_V88_K0"],
             scoreable=len(valid),poly=poly_n,
             V88good=old_correct,V88polygood=poly_old_correct,
             proposed_drop=rec["S466_birth_gate_proposed_dropped_total"],
             attacked_lost=rec["S466_birth_gate_missed_matched_true_onsets"],
             poly_lost=rec["S466_birth_gate_missed_scoreable_true_poly"])
        for i,j,delta in pairs:
            actual=anchors[j];idx=order[i]
            ledger.append(dict(row_id=str(ids[idx]),
                 annotated_attack_at_excerpt_seconds=actual["onset"],
                 original_native_attack_at_excerpt_seconds=float(ontime[i]),
                 signed_onset_lag_ms=1000*delta,
                 true_sounding_MIDI_pitches=actual["pitches"],
                 true_K=actual["trueK"],
                 MIDI40_88_and_K1to6=bool(actual["all_MIDI40_88"] and 1<=actual["trueK"]<=6),
                 original_V88_K=int(K[idx]),
                 birth_gate_suggested_drop=bool(suppress[idx])))
        print("S471 THIRD PIANO",json.dumps(rec),flush=True)
    need(totals["native"]==len(ids)
         and totals["midi_groups"]>0
         and totals["matched"]>0
         and len(recordings)==3,
         "S471 true MAESTRO source or MIDI matching missing")
    out.mkdir(parents=True)
    path=out/"S471_POSTFREEZE_REAL_PIANO_MIDI_MATCHED_ONSETS.jsonl"
    path.write_text("".join(json.dumps(x)+"\n" for x in ledger))
    report=dict(status="S471_EXTERNAL_THIRD_REAL_MAESTRO_PIANO_V88_MIDI_POSTFREEZE",
        corpus="MAESTRO_v3_test",
        instrument="acoustic_piano_not_guitar",
        original_GuitarSet_IDMT_weight_tuning_on_this_MAESTRO_data=False,
        original_MIDI_only_read_AFTER_S470_SHA_verified=True,
        original_S80_complete_chain_NOT_RUN=True,
        do_not_compare_V88_as_S80=True,
        retrospective_new_instrument_selection_from_test_split=True,
        note_on_event_annotations_not_continuous_note_activity_after_pedal=True,
        no_promotion=True,
        cohort_sums=dict(totals),
        exact_K_conditional_on_matched_scoreable_V88=(
          totals["V88good"]/totals["scoreable"] if totals["scoreable"] else None),
        exact_K_conditional_on_matched_poly_V88=(
          totals["V88polygood"]/totals["poly"] if totals["poly"] else None),
        per_recording=recordings,
        SHA256=dict(S469_preregistration=sha(manifest),
                    S470_frozen_native_inference=sha(freeze),
                    S471_MIDI_ledger=sha(path)))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S471 POSTFREEZE THIRD-SOURCE VERDICT",json.dumps(
        dict(corpus="MAESTRO_v3_test",sums=dict(totals),
            V88_K_score=report["exact_K_conditional_on_matched_scoreable_V88"],
            V88_poly_K_score=report["exact_K_conditional_on_matched_poly_V88"],
            WARNING="Not full original S80")),flush=True)

def selftest():
    a=np.asarray([0,.03,.11])
    b=np.asarray([.02,.13])
    pairs=match_times(a,b,.04)
    need(len(pairs)==2 and len({i for i,j,d in pairs})==2,
         "S471 original musical onset matching one-to-one needed")
    on=onset_groups([(.1,.5,50),(.11,.5,55),(.4,.6,60)],.0,.9)
    need(len(on)==2 and on[0]["trueK"]==2 and on[1]["trueK"]==1,
         "S471 original MIDI 20ms birth grouping or MIDI sounding cardinality failed")
    print("S471 SELFTEST PASS true note grouping and original native onset 80ms matching")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for name in ("freeze","freeze-report","manifest","out"):
        p.add_argument("--"+name,type=Path)
    args=p.parse_args()
    if args.self_test:selftest()
    else:
        for name in ("freeze","freeze_report","manifest","out"):
            need(getattr(args,name)is not None,"S471 missing "+name)
        run(args.freeze,args.freeze_report,args.manifest,args.out)
