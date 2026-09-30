"""Audit and label original inner audio with the explicit silence/count states."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

from causal_note.count_states import SilencePolicy, acoustic_evidence, target_state, UNLABELLED
from causal_note.guitarset import index_guitarset
from scripts.rebuild_v273_sources import digest, write_json, verify_dataset
from scripts.train_v100_spectral_string_slots import decode_pcm16_mono_wav
from scripts.v273_high_pitch_registers import assigned_pitches
from scripts.v273_window_experiment import array_hash, require
from scripts.restore_v273_original_backup import validate_original_inventory


def prepare(args):
    require(not args.output.exists(), 'refusing to overwrite a target audit')
    settings = json.loads(args.policy.read_text())
    require(settings['state_order'] == list(range(-1, 7)) and settings['sample_rate'] == 44100,
            'wrong label schema')
    require(settings['annotation_overlap_veto'] and not settings['allow_zero_padding_as_silence'],
            'unsafe silence target policy')
    policy = SilencePolicy(**settings['policy'])
    require((policy.before_samples, policy.after_samples) == (3100,2788),
            'silence support differs from original feature support')
    verify_dataset(args.dataset)
    cfg = json.loads(args.config.read_text())
    validate_original_inventory(args.geometry)
    geometry = json.loads((args.geometry/'report.json').read_text())
    require(geometry['config_sha256'] == digest(args.config), 'wrong source partition')
    with np.load(args.geometry/'rows.npz', allow_pickle=False) as z:
        members, starts = z['member'].copy(), z['start'].copy()
    # Bundle hashes cover complete .npy files, including the dtype/shape header.
    for name,values in (('members',members),('cluster_start_samples',starts)):
        serialized=io.BytesIO();np.save(serialized,values,allow_pickle=False)
        require(hashlib.sha256(serialized.getvalue()).hexdigest()==geometry['bundle_fields'][name]['sha256'],
                'row metadata differs from original bundle: '+name)
    folds = np.asarray([cfg['member_folds'][str(m)] for m in members])
    selected = np.flatnonzero(folds != 3)
    states = np.full(len(members), UNLABELLED, np.int8)
    old_k = np.full(len(members), UNLABELLED, np.int8)
    trainable = np.zeros(len(members), bool)
    overlap = np.zeros(len(members), np.int32)
    quiet = np.zeros(len(members), bool)
    full = np.zeros(len(members), bool)
    rms = np.full(len(members), np.nan, np.float32)
    peak = np.full(len(members), np.nan, np.float32)
    tracks = {t.annotation_member:t for t in index_guitarset(args.dataset)}
    seen = set(); unassigned = 0
    with zipfile.ZipFile(args.dataset/'annotation.zip') as archive:
        for record in geometry['tracks']:
            member = record['member']
            require(cfg['member_folds'][member] != 3 and member not in seen, 'outer or duplicate track')
            seen.add(member)
            timing = args.geometry/record['timing_file']
            require(digest(timing) == record['timing_sha256'], 'candidate timing changed')
            with np.load(timing, allow_pickle=False) as z:
                ids, flat, offsets = z['global_index'], z['full_candidate_samples'], z['full_candidate_offsets']
                groups = [flat[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
            np.testing.assert_array_equal(members[ids], np.repeat(member,len(ids)))
            np.testing.assert_array_equal(starts[ids], [g[0] for g in groups])
            doc = json.loads(archive.read(member))
            events = [(round(d['time']*44100),round((d['time']+d['duration'])*44100),float(d['value']))
                      for a in doc['annotations'] if a['namespace']=='note_midi' for d in a['data']]
            require(events and all(b>a for a,b,_ in events), 'missing or invalid note annotations')
            _, counts, missing = assigned_pitches(groups, [(a,p) for a,_,p in events])
            old_k[ids] = counts; unassigned += missing
            on, off = (np.asarray([e[j] for e in events]) for j in (0,1))
            track = tracks[member]
            audio = np.asarray(decode_pcm16_mono_wav(track.audio_zip,track.audio_member).samples,np.float32)/32768.
            for gid in ids:
                evidence = acoustic_evidence(audio,int(starts[gid]),policy)
                overlap[gid] = np.sum((on < evidence['end']) & (off > evidence['begin']))
                quiet[gid], full[gid] = evidence['quiet'], evidence['fully_observed']
                rms[gid], peak[gid] = evidence['max_frame_rms'], evidence['peak']
                states[gid], trainable[gid] = target_state(int(old_k[gid]),evidence,int(overlap[gid]))
            if len(seen)%25 == 0:
                print(json.dumps(dict(processed_tracks=len(seen),silence_rows=int(np.sum(states==-1)))),flush=True)
    require(seen == set(members[selected]) and len(seen)==190, 'incomplete inner population')
    require(np.all(old_k[selected]>=0) and np.all(states[folds==3]==UNLABELLED), 'wrong scope')
    require(np.all(states[(old_k>0)]==old_k[(old_k>0)]), 'positive count changed')
    fit, val = np.flatnonzero((folds!=3)&(folds!=0)), np.flatnonzero(folds==0)
    summaries={}
    for name, ids in (('fit',fit),('validation',val)):
        summaries[name]=dict(rows=len(ids),trainable=int(trainable[ids].sum()),
            excluded_unknown=int((~trainable[ids]).sum()),
            old_k_histogram=np.bincount(old_k[ids],minlength=7).tolist(),
            state_histogram={str(s):int(np.sum(states[ids]==s)) for s in range(-1,7)},
            quiet_but_annotated=int(np.sum(quiet[ids] & (overlap[ids]>0))),
            k0_without_annotation_but_audible=int(np.sum((old_k[ids]==0)&(overlap[ids]==0)&~quiet[ids])),
            minimum_rms_no_note=(float(rms[ids][(old_k[ids]==0)&(overlap[ids]==0)&full[ids]].min())
                if np.any((old_k[ids]==0)&(overlap[ids]==0)&full[ids]) else None))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'targets.npz',global_index=np.arange(len(members)),member=members,
        start=starts,old_k=old_k,state=states,trainable=trainable,quiet=quiet,fully_observed=full,
        annotated_overlap=overlap,max_frame_rms=rms,peak=peak,fit_index=fit,validation_index=val)
    eligible=all(summaries[p]['state_histogram']['-1']>0 and summaries[p]['state_histogram']['0']>0
                 for p in ('fit','validation'))
    report=dict(status='prepared',schema=settings['schema'],state_order=settings['state_order'],
        policy=settings,policy_sha256=digest(args.policy),config_sha256=digest(args.config),
        geometry_report_sha256=digest(args.geometry/'report.json'),
        annotation_sha256=digest(args.dataset/'annotation.zip'),
        audio_sha256=digest(args.dataset/'audio_mono-pickup_mix.zip'),
        target_sha256=digest(args.output/'targets.npz'),script_sha256=digest(__file__),
        state_builder_sha256=digest('src/causal_note/count_states.py'),
        fit_indices_sha256=array_hash(fit),val_indices_sha256=array_hash(val),
        summary=summaries,training_population_ready=eligible,unassigned_annotations=unassigned,
        original_clock=True,outer_rows_evaluated=0,model_training_performed=False,
        output_corrector=False,thresholds_tuned_on_validation=False,
        limitations=['candidate-conditioned population may contain little or no silence',
            'operational dBFS limits are not a universal audibility threshold',
            'annotation overlap only constructs targets, never model inputs',
            'padding-only or incomplete quiet windows are excluded, not relabelled silence'])
    write_json(args.output/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','geometry','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('analysis/v273-native-paired-config.json'))
    p.add_argument('--policy',type=Path,default=Path('analysis/v273-silence-policy.json'))
    prepare(p.parse_args())
