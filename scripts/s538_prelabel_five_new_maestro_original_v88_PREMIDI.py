"""S538: prospective source-only FIVE new composer MAESTRO piano WAV -> original pretrained V84/V88 native candidates.

New five-composer cohort S537; no WAV or MIDI opened before preregister.
THIRD source external to GuitarSet/IDMT. Inputs fixed S469 official filename-only
manifest; download WAVs alone, never MIDI until S471 after frozen SHA.
Reuse actual historical V88 stack and original freeze_local_combo weights.
Compute same S97 1090D acoustic evidence; apply four frozen GuitarSet-only
S466 note-birth models as auxiliary observation (not activated gate).
Crucial: V88 here != full S80. Do NOT claim a new S80 Exact-K score.
"""
from __future__ import annotations
import argparse,hashlib,io,json,urllib.request,urllib.parse,wave,zipfile,os,tempfile,collections,time
from pathlib import Path
from types import SimpleNamespace
import joblib,numpy as np
from scripts.s277_original_freeze_dataset3_native_prelabel import (
    decode_external_wav,pcm)
from scripts.s84_external_freeze_native_count import (
    proposal_args,runtime_records,runtime_stack,_load_frozen_stack,
    _predict_score_tracks,_extended_candidate_features,_clusters_window,
    CLUSTER_WINDOW_MS,_cluster_arrays,capture_timing,MAX_CANDIDATES,
    _spectral_maps_for_runtime,COVERED,build_model,predict,SEED)
from scripts.extract_v273_harmonic_trajectory import _frames,features_from_spectrogram
from scripts.loop_v273_birth_flow import flow_profile
from scripts.v273_extended_head_bank import audio_temporal_views
from scripts.s97_external_s45_original_features import geometry
from scripts.s458_train_independent_audio_only_K_blind_songheld import inputs
from scripts.s466_train_separate_acoustic_birth_v_continuation_guitarset import predict_birth
SR=44100
def need(x,msg):
    if not x:raise ValueError(msg)
def shab(blob):
    return hashlib.sha256(blob).hexdigest()
def sha(p):
    return shab(Path(p).read_bytes())
def fetch_track(base,name):
    address=base+urllib.parse.quote(name,safe="/")+"?download=true"
    request=urllib.request.Request(address,headers={
        "User-Agent":"S538-prelabel-maestro-piano-original-V88/1.0",
        "Accept-Encoding":"identity"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request,timeout=240) as response:
                need(response.status==200 and
                     "text/html" not in response.headers.get("Content-Type",""),
                     "S538 individual WAV mirror returned error/HTML")
                size=response.headers.get("Content-Length")
                if size:need(int(size)<=140_000_000,"S538 mirror WAV unexpectedly large")
                chunks=[];n=0
                while True:
                    b=response.read(1<<20)
                    if not b:break
                    chunks.append(b);n+=len(b)
                    need(n<=140_000_000,"S538 audio download exceeded 140MB cap")
                need(n>100_000 and b"RIFF"==chunks[0][:4],
                     "S538 original mirror file is not RIFF WAV")
                return b"".join(chunks),address
        except Exception as e:
            if attempt==2:raise
            print("S538 retry WAV mirror",name,type(e).__name__,flush=True)
            time.sleep(3*(attempt+1))
    raise RuntimeError("No source")
def selftest():
    temp=np.linspace(-.1,.1,44100*3,dtype=np.float32)
    original=pcm(temp)
    signal,details=decode_external_wav(original)
    need(len(signal)==len(temp) and details["original_sample_rate_hz"]==44100,
         "S538 original PCM conversion not compatible with S277")
    print("S538 SELFTEST PASS original WAV to V88 PCM and no MIDI")
def run(manifest,source_dir,weights,birthmodels,birthreport,out):
    need(not out.exists(),"S538 original preMIDI piano predictions are immutable")
    m=json.loads(manifest.read_text())
    b=json.loads(birthreport.read_text())
    need(m["status"]=="S537_PREREGISTERED_DISTINCT_MAESTRO_V3_TEST_PIANO_COHORT_NO_WAV_NO_MIDI"
         and m["selected_recordings"]==5
         and not m["MIDI_contents_downloaded_or_inspected"]
         and not m["WAV_contents_downloaded_or_inspected"]
         and m["no_promotion"] and m["selected_different_composers_from_S469"]
         and m["preregistration_fixed_before_audio_or_labels"]
         and m["prospective_exclusion_verified_ONLY_AGAINST_S469_not_all_repo_history"]
         and b["status"]=="S466_GUITARSET_TRAIN_ONLY_COMPOSITION_DISJOINT_BINARY_BIRTH_HEAD"
         and b["IDMT_not_read_during_training_or_threshold_selection"]
         and b["model_no_original_S80_K_as_input"]
         and weights.is_file(),
         "S538 preregistered third-corpus track and historical models required")
    selected=m["selected_sources"]
    need(len(selected)==5 and len({s["composer"] for s in selected})==5
         and len({s["WAV_filename_sha256"] for s in selected})==5,
         "S538 MAESTRO selection not distinct composers")
    # Source registered before annotation. Only exact three WAV mirror files read.
    clips={};files=[];proof=[]
    out.mkdir(parents=True)
    tempzip=out/"NEVER_UPLOAD_TEMP_RAW_AUDIO.zip"
    try:
        with zipfile.ZipFile(tempzip,"w",compression=zipfile.ZIP_STORED) as z:
            tracks=[]
            for i,item in enumerate(selected):
                raw,url=fetch_track(m["audio_mirror_directory"],item["audio_filename"])
                need(shab(item["audio_filename"].encode())==item["WAV_filename_sha256"],
                     "S538 filename chosen after original manifest freeze")
                full,fmt=decode_external_wav(raw)
                offset=int(item["frozen_excerpt_offset_s"]*SR)
                length=int(item["frozen_excerpt_length_s"]*SR)
                need(len(full)>=offset+length and length==12*SR,
                     "S538 original third-source recorded WAV too short for predeclared excerpt")
                segment=full[offset:offset+length]
                normwav=pcm(segment)
                ident=item["WAV_filename_sha256"]
                member=f"maestro3_{i:02d}_{ident[:12]}_comp.jams"
                basename=f"maestro3_{i:02d}_{ident[:12]}_comp_mix.wav"
                z.writestr(basename,normwav)
                tracks.append(SimpleNamespace(annotation_member=member,
                            audio_zip=tempzip,audio_member=basename))
                clips[ident]=segment
                files.append(dict(identity=ident,member=member))
                proof.append(dict(audio_filename=item["audio_filename"],
                       WAV_filename_SHA256=ident,
                       original_WAV_SHA256=shab(raw),
                       source_duration_seconds=float(len(full)/SR),
                       excerpt_offset_seconds=item["frozen_excerpt_offset_s"],
                       excerpt_duration_seconds=item["frozen_excerpt_length_s"],
                       data_mirror_host="huggingface.co",
                       WAV_processing=fmt))
                print("S538 SOURCE WAV",json.dumps(dict(
                    recording=i+1,source_sha=ident,bytes=len(raw),
                    native_format=fmt)),flush=True)
        # At this point all raw third-source audio was captured before
        # pretrained models and any note annotation. No MIDI accessed.
        params=proposal_args(source_dir)
        floor,retention,encoder86,encoder87,model88=_load_frozen_stack(params)
        score=_predict_score_tracks(params.base_model,tracks)
        records=runtime_records(tracks,score,floor)
        need(len(records)>0,"S538 original genuine V84 found no piano attack candidates")
        X88,outputs=runtime_stack(records,encoder86,encoder87,model88)
        candidates,fused=_extended_candidate_features(X88,outputs)
        clusters=_clusters_window(records,CLUSTER_WINDOW_MS)
        seq,mask,stats,_,_,_= _cluster_arrays(
            clusters,{},records,candidates,outputs,fused)
        timing=capture_timing(clusters,records,fused,MAX_CANDIDATES)
        nativegeo=_spectral_maps_for_runtime(tracks,clusters,records,window=COVERED)
        cache=dict(sequence=np.asarray(seq,np.float32),
            mask=np.asarray(mask,np.float32),stats=np.asarray(stats,np.float32),
            spectral=np.asarray(nativegeo,np.float32),
            members=np.asarray([q["member"] for q in clusters]),
            cluster_start_samples=np.asarray(timing["cluster_start_samples"],np.int64))
        original=build_model("learned_gate",SEED)
        original.load_weights(weights)
        original_prob,original_K=predict(original,cache,np.arange(len(clusters)))
        starts=np.asarray(timing["cluster_start_samples"],np.int64)
        ends=timing["full_candidate_samples"][timing["full_candidate_offsets"][1:]-1]
        to_ident={x["member"]:x["identity"] for x in files}
        member=np.asarray([to_ident[str(x)] for x in cache["members"].astype(str)])
        ids=np.asarray([v+":"+str(int(s)) for v,s in zip(member,starts)])
        need(len(ids)>0 and len(set(ids))==len(ids) and
             original_prob.shape==(len(ids),7) and
             np.isin(original_K,range(7)).all(),
             "S538 pretrained real V88 original piano row or decision contract lost")
        # Extract historical source 1090D features on SAME detected events.
        n=len(ids)
        summary=np.empty((n,58),np.float32)
        aseq=np.empty((n,42,49),np.float32)
        for ident,waveform in clips.items():
            ii=np.flatnonzero(member==ident)
            for row in ii:
                spec,tt=_frames(waveform,int(starts[row]))
                desc,act=features_from_spectrogram(spec,tt,return_states=True)
                need(len(desc)==58 and act.shape==(42,49),"S538 original S97 feature ABI changed")
                summary[row]=[desc[key] for key in sorted(desc)]
                aseq[row]=np.log1p(act)
        flow=flow_profile(aseq)
        morph,physics=audio_temporal_views(aseq)
        geom=geometry(member,starts,ends)
        bank=dict(summary58=summary,flow245=flow,
            morphology490=morph,physics294=physics,geometry3=geom)
        features=inputs(bank,original_K.astype(np.int8))
        prob=np.empty((4,n),np.float32)
        traincuts=[]
        for fold in range(4):
            modelpath=birthmodels/f"S466_birth_vs_nonbirth_fold{fold}.joblib"
            spec=b["all_folds"][fold]
            need(spec["model_SHA256"]==sha(modelpath) and spec["fold"]==fold,
                 "S538 source original GuitarSet-only binary checkpoint modified")
            pack=joblib.load(modelpath)
            need(pack["no_training_on_IDMT"] and
                 pack["no_original_K_feature"] and pack["fold"]==fold,
                 "S538 source birth head training provenance incorrect")
            prob[fold]=predict_birth(pack["model"],features)
            traincuts.append(spec["inner_selected_threshold"])
        # No external threshold selection. Experimental decision based on
        # ORIGINAL V88 K>=1 protection (full S80 NOT available).
        trained=np.asarray([float(v) if v is not None else np.nan for v in traincuts])
        suppress=np.zeros(n,bool)
        if np.isfinite(trained).all():
            suppress=(original_K==0)&np.all(prob<trained[:,None],axis=0)
        outpath=out/"S538_MAESTRO_ORIGINAL_V88_K_BIRTH_FROZEN_PREMIDI.npz"
        np.savez_compressed(outpath,row_id=ids,member=member,
            start_samples=starts,end_samples=ends,
            original_V88_K=np.asarray(original_K,np.int8),
            original_V88_probabilities=np.asarray(original_prob,np.float32),
            original_GuitarSet_birth4_prob=prob,
            auxiliary_trainonly_birth4_thresholds=trained,
            suggested_suppression_mask=suppress)
        report=dict(status="S538_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI",
            actual_external_instrument="acoustic_piano",
            source_not_GuitarSet_or_IDMT=True,
            five_original_piano_recordings=5,
            source_excerpt_seconds_per_recording=12,
            source_excerpt_offset_seconds=15,
            original_V88_event_count=n,
            original_V88_K_hist=np.bincount(original_K,minlength=7).tolist(),
            original_V88_PREDICTED_K0=int(np.sum(original_K==0)),
            S466_birth_gated_event_candidate_suppression_proposals=int(np.sum(suppress)),
            output_sha256=sha(outpath),
            preregistered_S469_SHA=sha(manifest),
            historical_V88_weights_sha=sha(weights),
            original_4_S466_train_only_thresholds=traincuts,
            exact_original_audio_source_fingerprints=proof,
            prelabel_predictions_frozen=True,
            source_s537_selection_verified_only_against_s469=True,
            actual_new_composer_count=5,
            WAV_contents_read_only_audio=True,
            paired_MIDI_annotation_files_OPENED=False,
            original_complete_S80_NOT_REPLAYED=True,
            no_full_S80_ExactK_claim=True,
            original_live_onsets_offsets_not_modified=True,
            no_promotion=True)
        (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print("S538 PREMIDI THIRD SOURCE",json.dumps(dict(
            recordings=5,events=n,
            V88_hist=report["original_V88_K_hist"],
            suppress=int(suppress.sum()),midi_read=False)),flush=True)
    finally:
        if tempzip.exists():tempzip.unlink()

if __name__=="__main__":
    q=argparse.ArgumentParser()
    q.add_argument("--self-test",action="store_true")
    for name in ("manifest","source-dir","weights","birthmodels","birthreport","out"):
        q.add_argument("--"+name,type=Path)
    z=q.parse_args()
    if z.self_test:selftest()
    else:
        for name in ("manifest","source_dir","weights","birthmodels","birthreport","out"):
            need(getattr(z,name)is not None,"S538 missing "+name)
        run(z.manifest,z.source_dir,z.weights,z.birthmodels,z.birthreport,z.out)
