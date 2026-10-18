"""S702: isolated rescue state plus causal learned accept/reject event head.

Fit four TRAIN composition families, calibrate the fifth, freeze ONE event
policy, then check four HELD families and diagnostic DEV. TEST is unopened.
Reuse frozen S701/S675 checkpoints. No official LIVE or model is replaced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import zipfile

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from causal_note.strict_live import Selection, feature_track, load_heads, SR
from causal_note.s702_event_selection import (
    candidate_track, render_candidates, onset_keys, paired_onsets,
)
from scripts.s613_prepare_strict_live import download, digest, labels
from scripts.s613_train_strict_live import decode, event_score, real_notes
from scripts.s672_causal_offset_timing_grid import (
    ROOT, MODEL, raw_guitar_audio, native_posteriors, note_instances,
    native_rhythm_clock, hazard_predictions, choose_offset,
)
from scripts.s682_onset_four_lock_guard import (
    aggregate, native_metrics, full_aggregate, safe_full,
    BATCH_HOPS, OFFSET_GATE, OFFSET_PERSISTENCE,
)
from scripts.s701_train_causal_onset_shadow import onset_features, predict_onsets, rescue

POLICIES = {'native': (None, None)}
for _native in (None, .20, .35, .50, .65):
    for _extra in (None, .65, .80, .90):
        if _native is None and _extra is None:
            continue
        POLICIES[f'event_native_{_native}_rescue_{_extra}'] = (_native, _extra)
LEGACY = 'head_0.75_floor_0.20'
PROPOSAL = 'head_0.60_floor_0.10'


def prepare(spec, root, archive, heads, selection, onset_model):
    audio, sha = raw_guitar_audio(spec, archive)
    truth = real_notes(labels(spec, root)['notes'], len(audio) / SR)
    ends, features = feature_track(audio, block_size=4096)
    post, k = native_posteriors(features, heads)
    x = onset_features(features, post, k)
    scores = predict_onsets(onset_model, x)
    legacy_post = rescue(post, scores, LEGACY, selection)
    wide_post = rescue(post, scores, PROPOSAL, selection)
    rows, event_x, _ = candidate_track(ends, post, k, wide_post, x, scores, selection)
    fixed_rows, _, _ = candidate_track(ends, post, k, legacy_post, x, scores, selection)
    native = decode(ends, post, k, selection, len(audio))
    legacy = decode(ends, legacy_post, k, selection, len(audio))
    fixed = render_candidates(ends, post, fixed_rows, np.ones(len(fixed_rows)),
                              selection, len(audio), rescue_threshold=.5,
                              protect_native=True)
    if not onset_keys(native).issubset(onset_keys(fixed)):
        raise AssertionError('S702 removed an original native NoteOn')
    no_op = render_candidates(ends, post, rows, np.ones(len(rows)), selection, len(audio))
    if no_op != native:
        raise AssertionError('S702 no-op failed exact native event parity')
    return dict(spec=spec, truth=truth, ends=ends, feats=features, post=post,
                k=k, rows=rows, event_x=event_x, samples=len(audio), sha=sha,
                native=native, legacy_S701=legacy, protected_S702=fixed)


def render(item, quality, selection, name):
    if name in ('native', 'legacy_S701', 'protected_S702'):
        return item[name]
    nt, rt = POLICIES[name]
    return render_candidates(item['ends'], item['post'], item['rows'], quality,
                              selection, item['samples'], native_threshold=nt,
                              rescue_threshold=rt, protect_native=nt is None)


def release_metrics(item, events, offset_model):
    onset, pairs = event_score(events, item['truth'], tolerance=.080)
    pack = dict(item, events=events, onset=onset, pairs=pairs,
                notes=note_instances(events, item['ends']),
                clock=native_rhythm_clock(events, item['ends']))
    hazard = hazard_predictions(pack, offset_model)
    positions = [choose_offset(pack, i, hazard[0][i], hazard[1][i], OFFSET_GATE,
                              'hazard', batch_hops=BATCH_HOPS,
                              persistence=OFFSET_PERSISTENCE)
                 for i in range(len(pack['notes']))]
    valid = correct = long_correct = long_eligible = 0
    matched = {}
    ons = [e for e in events if e['type'] == 'note_on']
    for index, gt_id in pairs:
        gt = item['truth'][gt_id]
        target = gt['time'] + gt['duration']
        if target >= item['ends'][-1] / SR:
            continue
        valid += 1
        hit = int(positions[index] < item['ends'][-1] / SR - 1e-7
                  and abs(positions[index] - target) <= .050)
        correct += hit
        long_eligible += int(gt['duration'] >= 1.)
        long_correct += int(gt['duration'] >= 1.) * hit
        matched[str(gt_id)] = dict(correct=hit, time=float(positions[index]),
                                  onset_key=[ons[index]['pitch'],
                                             ons[index]['decision_sample']])
    predicted, ground_truth = len(pack['notes']), len(item['truth'])
    return dict(correct=correct, eligible=valid, accuracy=correct/max(1,valid),
                full_note_f1=2*correct/max(1,predicted+ground_truth),
                predicted=predicted, ground_truth=ground_truth,
                long_correct=long_correct, long_eligible=long_eligible,
                NoteOn_TP=onset['TP'], NoteOn_FP=onset['FP'], NoteOn_FN=onset['FN'],
                matched=matched)


def evaluate(item, model, selection, variants, offset_model=None):
    quality = (model.predict_proba(item['event_x'])[:,1] if len(item['rows'])
               else np.empty(0, np.float32))
    scores, audits, releases, events_by_name = {}, {}, {}, {}
    for name in variants:
        events = render(item, quality, selection, name)
        events_by_name[name] = events
        scores[name] = native_metrics(events, item['truth'], item['ends'])
        audits[name] = paired_onsets(item['native'], events, item['truth'], event_score)
        if offset_model is not None:
            releases[name] = release_metrics(item, events, offset_model)
    common = {}
    if releases:
        old = releases['native']['matched']
        for name, score in releases.items():
            new = score['matched']
            shared = {i for i in old.keys() & new.keys()
                      if old[i]['onset_key'] == new[i]['onset_key']}
            lost = [int(i) for i in shared if old[i]['correct'] and not new[i]['correct']]
            gained = [int(i) for i in shared if new[i]['correct'] and not old[i]['correct']]
            common[name] = dict(same_event_common=len(shared), lost_correct=len(lost),
                                gained_correct=len(gained), lost_truth_ids=sorted(lost))
    return dict(name=item['spec']['name'], composition=item['spec']['composition'],
                sha256=item['sha'], scores=scores, audits=audits, release=releases,
                common_offsets=common, events=events_by_name, truth=item['truth'])


def summarize(rows, variants):
    scores = {n: aggregate([r['scores'] for r in rows], n) for n in variants}
    audit = {n: {k: sum(r['audits'][n][k] for r in rows)
                  for k in ('recovered','lost','retained_truth','added_events',
                            'removed_events','delta_tp','delta_fp')} for n in variants}
    texture = {kind: {n: aggregate([r['scores'] for r in rows
                                   if r['name'].endswith('_'+kind)], n)
                      for n in variants} for kind in ('solo','comp')}
    release = ({n: full_aggregate([r['release'] for r in rows], n) for n in variants}
               if rows and rows[0]['release'] else {})
    common = ({n: {k: sum(r['common_offsets'][n][k] for r in rows)
                   for k in ('same_event_common','lost_correct','gained_correct')}
               for n in variants} if release else {})
    return dict(scores=scores, paired=audit, texture=texture,
                release=release, common_offsets=common)


def choose_calibration(summary):
    scores = summary['scores']
    base = scores['native']
    candidates = [n for n in POLICIES if n != 'native']
    eligible = [n for n in candidates
                if scores[n]['onset_F1'] > base['onset_F1'] + .002
                and scores[n]['K_frames_correct'] >= base['K_frames_correct']
                and scores[n]['tp'] >= base['tp']
                and scores[n]['precision'] >= base['precision']]
    best = max(candidates, key=lambda n: scores[n]['onset_F1'])
    chosen = max(eligible, key=lambda n: scores[n]['onset_F1']) if eligible else 'native'
    return chosen, best, eligible


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--onset-checkpoint', type=Path, required=True)
    ap.add_argument('--offset-checkpoint', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('S702 refuses to overwrite results')
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    frozen = json.loads((ROOT/'selection_frozen.json').read_text())
    protocol = json.loads((ROOT/'protocol.json').read_text())
    assert digest(MODEL.read_bytes()) == frozen['model_sha256']
    assert digest((ROOT/'protocol.json').read_bytes()) == frozen['protocol_sha256']
    heads = load_heads(MODEL, expected_sha256=frozen['model_sha256'])
    selection = Selection(**heads.metadata['selection'])
    from causal_note.s672_live_offset import S672TrialNoteOff
    from scripts.s699_player_held_full_train_K import expanded_train_specs
    assert digest(args.onset_checkpoint.read_bytes()) == 'cc72cf81f477c6b734e253df5fb04d1789ab707dc160f06bdc5cf9ffdff2fb0f'
    assert digest(args.offset_checkpoint.read_bytes()) == 'ee5ae30917cba201c823b9721fc42f1052e047891a4ff16f32afbea8ddf9b80e'
    offset = S672TrialNoteOff.load(args.offset_checkpoint, heads)
    onset_model = joblib.load(args.onset_checkpoint)
    assert offset.gate == OFFSET_GATE and offset.persistence == OFFSET_PERSISTENCE
    train = [s for s in protocol['sources'] if s['corpus']=='GuitarSet' and s['split']=='train']
    dev = [s for s in protocol['sources'] if s['corpus']=='GuitarSet' and s['split']=='dev']
    comps = sorted({s['composition'] for s in train})
    assert len(comps)==9 and len(train)==36 and len(dev)==12
    fit_comps, cal_comps, held_comps = set(comps[:4]), {comps[4]}, set(comps[5:])
    report = dict(schema='S702_CAUSAL_EVENT_STATE_AND_QUALITY', promoted=False,
                  official_LIVE_changed=False, TEST_opened=False, DEV_previously_exposed=True,
                  native_sha256=frozen['model_sha256'],
                  S701_sha256=digest(args.onset_checkpoint.read_bytes()),
                  S675_sha256=digest(args.offset_checkpoint.read_bytes()),
                  fit_compositions=sorted(fit_comps), calibration_compositions=sorted(cal_comps),
                  held_compositions=sorted(held_comps), policies=POLICIES,
                  protocol_note='S701 reused TRAIN-five-composition head; new event head fits four. '
                    'CAL uses fifth TRAIN composition. HELD and DEV previously examined by research; '
                    'neither is an unopened final test. K uses native-release decoded intervals; '
                    'S675 is separately evaluated shadow replay, not certified microphone deployment.')

    def checkpoint():
        report['elapsed_s'] = round(time.perf_counter()-started,2)
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); (root/'data').mkdir()
        for key, filename in (('GuitarSet_audio','GuitarSet_audio.zip'),
                              ('GuitarSet_annotations','GuitarSet_annotations.zip')):
            src=protocol['guitar_archives'][key]
            download(src['url'],root/'data'/filename,src['size'],src['sha256'])
        with zipfile.ZipFile(root/'data/GuitarSet_audio.zip') as archive, \
             zipfile.ZipFile(root/'data/GuitarSet_annotations.zip') as annotations:
            expanded=expanded_train_specs(train,archive,annotations)
            fitting=[s for s in expanded if s['composition'] in fit_comps]
            calibration=[s for s in expanded if s['composition'] in cal_comps]
            held=[s for s in expanded if s['composition'] in held_comps]
            assert (len(fitting),len(calibration),len(held))==(64,16,64)
            xs,ys=[],[]
            for i,spec in enumerate(fitting,1):
                item=prepare(spec,root,archive,heads,selection,onset_model)
                _,pairs=event_score(item['rows'],item['truth'],tolerance=.080)
                y=np.zeros(len(item['rows']),np.uint8)
                y[[a for a,_ in pairs]]=1
                xs.append(item['event_x']); ys.append(y)
                print('S702 FIT',i,len(fitting),spec['name'],len(y),int(y.sum()),flush=True)
                del item
            x=np.concatenate(xs); y=np.concatenate(ys)
            assert len(y)>1000 and y.sum()>500 and (y==0).sum()>500
            model=HistGradientBoostingClassifier(max_iter=145,learning_rate=.055,
                    max_leaf_nodes=15,min_samples_leaf=35,l2_regularization=12.,
                    early_stopping=False,random_state=702)
            model.fit(x,y)
            report['FIT']=dict(rows=len(y),positive=int(y.sum()),negative=int((y==0).sum()),
                               features=x.shape[1],sources=len(fitting))
            joblib.dump(model,args.out/'s702_event_quality.joblib',compress=3)
            del x,y,xs,ys
            calrows=[]
            for i,spec in enumerate(calibration,1):
                item=prepare(spec,root,archive,heads,selection,onset_model)
                calrows.append(evaluate(item,model,selection,tuple(POLICIES)))
                print('S702 CAL',i,len(calibration),spec['name'],flush=True)
                del item
            report['CAL']=summarize(calrows,tuple(POLICIES))
            selected,best,eligible=choose_calibration(report['CAL'])
            candidate=selected if selected!='native' else best
            report.update(CAL_selected=selected,CAL_best_exploratory=best,
                          CAL_eligible=eligible,frozen_candidate=candidate)
            del calrows
            checkpoint()
            print('S702 FROZEN',selected,candidate,
                  json.dumps(report['CAL']['scores'][candidate]),flush=True)
            variants=('native','legacy_S701','protected_S702',candidate)
            for phase,specs in (('HELD',held),('DEV',dev)):
                rows=[]
                with (args.out/(phase.lower()+'_events.jsonl')).open('w') as event_file:
                    for i,spec in enumerate(specs,1):
                        item=prepare(spec,root,archive,heads,selection,onset_model)
                        row=evaluate(item,model,selection,variants,offset.model)
                        event_file.write(json.dumps(dict(name=row['name'],truth=row['truth'],
                                                        events=row.pop('events')))+'\n')
                        rows.append(row)
                        print('S702',phase,i,len(specs),spec['name'],
                              json.dumps({n:row['scores'][n] for n in ('native',candidate)}),flush=True)
                        del item
                report[phase]=summarize(rows,variants)
                (args.out/(phase.lower()+'_paired_audit.json')).write_text(json.dumps(rows,indent=2)+'\n')
                checkpoint()
    for phase in ('HELD','DEV'):
        s=report[phase]; base=s['scores']['native']; trial=s['scores'][candidate]
        report[phase+'_four_locks']=bool(
            trial['onset_F1']>base['onset_F1'] and trial['tp']>=base['tp']
            and trial['K_frames_correct']>=base['K_frames_correct']
            and safe_full(s['release']['native'],s['release'][candidate])
            and s['common_offsets'][candidate]['lost_correct']==0)
    report['four_locks_passed']=bool(selected!='native' and report['HELD_four_locks']
                                    and report['DEV_four_locks'])
    original=report['DEV']['scores']['native']
    assert (original['tp'],original['fp'],original['fn'])==(998,585,582)
    assert digest(MODEL.read_bytes())==frozen['model_sha256']
    checkpoint()
    print('S702 FINAL',json.dumps({k:report[k] for k in (
        'CAL_selected','frozen_candidate','HELD','DEV','four_locks_passed','promoted')}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
