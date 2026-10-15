"""Frozen S601 transfer. S607/S608 audio only, then S609 annotation scoring.

No fitting, source replacement, threshold search, time-shift optimization or
changes to the S602 contract. All corpora must finish freezing before scoring.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import importlib.metadata
import io
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
import numpy as np
from scripts.s585_attack_sustain_restrike import need, sha, save, metrics
from scripts.s592_s597_urmp_prospective import read, digest
from scripts.s602_s605_reserved_urmp import contract, CONTRACT, POLICIES
from scripts.s600_bp_native_development import decode
from scripts.s601_bp_candidate_gate import candidates, feature_matrix, emit, counts
from scripts.s591_score_state_note_predictions import match_ledger
from scripts.s277_original_freeze_dataset3_native_prelabel import decode_external_wav, pcm
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s400_evaluate_S399_MIDI_note_pitches_GuitarSet import midi_value

MANIFEST = Path('analysis/v273_s606_cross_corpus_preregistration.json')
CORPORA = ('GuitarSet', 'IDMT', 'MAESTRO')


def request(url, limit, expected_size=None, byte_range=None):
    need(url.startswith(('https://huggingface.co/', 'https://zenodo.org/')), 'Pinned public source only')
    headers = {'Accept-Encoding': 'identity', 'User-Agent': 'NOTE-S606-frozen-transfer/1.0'}
    if byte_range:
        a,b,total = byte_range
        headers['Range'] = f'bytes={a}-{b}'
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                if byte_range:
                    need(r.status == 206 and r.headers.get('Content-Range') == f'bytes {a}-{b}/{total}',
                         'Exact bounded HTTP Range required')
                else:
                    need(r.status == 200, 'Complete source GET required')
                blob = r.read(limit+1)
            need(len(blob) <= limit, 'Source exceeds registered bound')
            if expected_size is not None:
                need(len(blob) == expected_size, 'Source length changed')
            return blob
        except Exception as e:
            if attempt == 3:
                raise
            print('SOURCE RETRY', attempt+1, type(e).__name__, flush=True)
            time.sleep(2*(attempt+1))
    raise RuntimeError('unreachable')


def archive(source, cache):
    """An audio-only archive is read in full; mixed IDMT uses bounded ranges."""
    if source['access'] == 'range':
        def get(a,b):
            return request(source['url'], b-a+1, b-a+1, (a,b,source['size']))
        reader = HttpRangeZIP(source['url'], source['size'], fetch=get, window=4*1024*1024)
        return zipfile.ZipFile(reader), reader
    path = cache/source['cache_name']
    if not path.exists():
        path.write_bytes(request(source['url'], source['size'], source['size']))
    need(path.stat().st_size == source['size'], 'Pinned archive size')
    for name in ('sha256','md5'):
        if name in source:
            h = hashlib.new(name)
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024), b''):
                    h.update(chunk)
            need(h.hexdigest() == source[name], 'Pinned archive digest: '+name)
    return zipfile.ZipFile(path), None


def member(z, item):
    i = z.getinfo(item['path'])
    need(i.file_size == item['size'] and i.CRC == item['crc32'], 'Registered ZIP member identity')
    blob = z.read(i)  # zipfile independently verifies decompressed CRC32.
    need(len(blob) == item['size'], 'Truncated archive member')
    return blob


def standalone(item, base):
    blob = request(base+urllib.parse.quote(item['path'], safe='/'), item['size'], item['size'])
    if 'sha256' in item:
        need(digest(blob) == item['sha256'], 'Pinned LFS object SHA256')
    else:
        header = ('blob '+str(len(blob))+'\0').encode()
        need(hashlib.sha1(header+blob).hexdigest() == item['git_oid'], 'Pinned git object identity')
    return blob


def registered():
    c,_ = contract()
    m = read(MANIFEST)
    need(m['contract_sha256'] == sha(CONTRACT), 'Unchanged S602 selection')
    need(m['no_tuning'] and m['model_sha256'] == c['model_sha256'], 'Frozen S601 transfer only')
    need({k:len(v) for k,v in m['sources'].items()} == {'GuitarSet':60,'IDMT':5,'MAESTRO':12},
         'All registered sources required; no replacements')
    return c,m


def freeze(corpus, root):
    import joblib
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, run_inference
    c,m = registered()
    for package,version in c['runtime_versions'].items():
        need(importlib.metadata.version(package) == version, 'Frozen runtime: '+package)
    weights = {str(p.relative_to(ICASSP_2022_MODEL_PATH)):sha(p)
               for p in sorted(Path(ICASSP_2022_MODEL_PATH).rglob('*')) if p.is_file()}
    need(weights == c['basic_pitch_weight_hashes'], 'Frozen pretrained BP weights')
    out = root/corpus
    need(not out.exists(), 'Immutable corpus freeze')
    out.mkdir(parents=True)
    cache = root.parent/('S607_private_'+corpus)
    cache.mkdir(exist_ok=True)
    bp = Model(ICASSP_2022_MODEL_PATH)
    head = joblib.load(c['model_path'])
    pred = {name:{} for name in POLICIES}
    proof = []
    z = reader = None
    if corpus != 'MAESTRO':
        z,reader = archive(m['archives'][corpus+'_audio'], cache)
    for spec in m['sources'][corpus]:
        sid = spec['source_id']
        blob = member(z,spec['audio']) if z else standalone(spec['audio'],m['maestro_base'])
        signal,layout = decode_external_wav(blob)
        offset = m['offset_seconds'][corpus]
        clip = signal[int(offset*44100):int((offset+24)*44100)]
        seconds = len(clip)/44100
        need(seconds >= 2, 'Registered source is too short; fail without replacement')
        path = cache/(sid+'.wav')
        path.write_bytes(pcm(clip))
        raw = run_inference(str(path),bp)
        need(all(np.isfinite(v).all() for v in raw.values()), 'Finite BP output')
        np.savez_compressed(out/(sid+'.raw.npz'), **raw)
        pred[POLICIES[0]][sid] = decode(raw)
        pred[POLICIES[1]][sid] = decode(raw,**c['global_decoder'])
        ev = candidates(raw)
        X,names = feature_matrix(raw,ev)
        need(names == c['feature_names'], 'Unchanged 168D feature schema')
        prob = head.predict_proba(X)[:,1]
        pred[POLICIES[2]][sid] = emit(ev,prob,**c['learned_decoder'])
        # Scoring boundary only; never change candidate context or decoder.
        for name in POLICIES:
            pred[name][sid] = [x for x in pred[name][sid] if 0 <= x['time'] < seconds]
        save(out/(sid+'.candidates.json'), ev)
        np.savez_compressed(out/(sid+'.head.npz'), probability=prob)
        proof.append(dict(source_id=sid,audio_sha256=digest(blob),audio_bytes=len(blob),
            source_duration_seconds=len(signal)/44100,offset_seconds=offset,seconds=seconds,
            source_layout=layout,clip_sha256=sha(path),raw_sha256=sha(out/(sid+'.raw.npz')),
            candidate_sha256=sha(out/(sid+'.candidates.json')),probability_sha256=sha(out/(sid+'.head.npz')),
            counts={name:len(pred[name][sid]) for name in POLICIES}))
        print('S607 AUDIO ONLY',corpus,spec['label'],seconds,proof[-1]['counts'],flush=True)
        path.unlink()
    if z:
        z.close()
    save(out/'predictions.json',pred)
    save(out/'report.json',dict(status='S608_CORPUS_PREDICTIONS_FROZEN_BEFORE_ANNOTATIONS',
        corpus=corpus,source_count=len(proof),total_seconds=sum(x['seconds'] for x in proof),
        annotations_read=False,fit_or_threshold_selection=False,model_sha256=c['model_sha256'],
        manifest_sha256=sha(MANIFEST),contract_sha256=sha(CONTRACT),script_sha256=sha(__file__),
        prediction_sha256=sha(out/'predictions.json'),audio_proof=proof,policies=list(POLICIES),
        runtime_versions=c['runtime_versions'],basic_pitch_weight_hashes=weights,
        partial_archive_access=(dict(bytes=reader.network_bytes,requests=reader.range_requests,
            full_archive_MD5_NOT_recalculated=True) if reader else None),no_promotion=True))
    print('S608 FROZEN',corpus,sha(out/'predictions.json'),flush=True)


def annotations(blob, corpus):
    """Preserve individual NoteOn times; never use sounding K or onset groups."""
    events = []
    if corpus == 'GuitarSet':
        for ann in json.loads(blob)['annotations']:
            if ann.get('namespace') != 'note_midi':
                continue
            track = str(ann['annotation_metadata']['data_source'])
            need(track in ('0','1','2','3','4','5'), 'Physical guitar string annotation')
            for x in ann['data']:
                events.append(dict(time=float(x['time']),pitch=midi_value(x['value']),track=track))
    elif corpus == 'IDMT':
        for e in ET.fromstring(blob).iter():
            if e.tag.split('}')[-1].lower() != 'event':
                continue
            fields = {x.tag.split('}')[-1].lower():(x.text or '').strip() for x in e.iter()}
            # Fail on unscorable pitched events rather than silently omitting them.
            need(all(k in fields for k in ('onsetsec','offsetsec','pitch')), 'Complete IDMT event')
            p = float(fields['pitch'])
            need(p.is_integer() and float(fields['offsetsec']) > float(fields['onsetsec']), 'IDMT note schema')
            events.append(dict(time=float(fields['onsetsec']),pitch=int(p),track='guitar'))
    else:
        import mido
        mf = mido.MidiFile(file=io.BytesIO(blob))
        tempo = 500000
        t = 0.
        for msg in mido.merge_tracks(mf.tracks):
            t += mido.tick2second(msg.time,mf.ticks_per_beat,tempo)
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                events.append(dict(time=t,pitch=int(msg.note),track=str(msg.channel)))
    need(events and all(np.isfinite(x['time']) and x['time'] >= 0 and 0 <= x['pitch'] <= 127 for x in events),
         'Finite original note annotations')
    return sorted(events,key=lambda x:(x['time'],x['pitch'],x['track']))


def checked_freezes(root):
    c,m = registered()
    reports = {}
    for corpus in CORPORA:
        folder = root/corpus
        r = read(folder/'report.json')
        need(r['status'] == 'S608_CORPUS_PREDICTIONS_FROZEN_BEFORE_ANNOTATIONS'
             and not r['annotations_read'] and r['manifest_sha256'] == sha(MANIFEST)
             and r['contract_sha256'] == sha(CONTRACT) and r['script_sha256'] == sha(__file__)
             and r['prediction_sha256'] == sha(folder/'predictions.json'), 'All three prelabel freezes required')
        pred = read(folder/'predictions.json')
        ids = {s['source_id'] for s in m['sources'][corpus]}
        need(set(pred) == set(POLICIES) and all(set(v) == ids for v in pred.values()), 'All policies and sources')
        need({x['source_id'] for x in r['audio_proof']} == ids, 'All source proofs required')
        for item in r['audio_proof']:
            for suffix,key in (('.raw.npz','raw_sha256'),('.candidates.json','candidate_sha256'),('.head.npz','probability_sha256')):
                need(sha(folder/(item['source_id']+suffix)) == item[key], 'Frozen intermediate evidence')
        reports[corpus] = r
    return c,m,reports


def score(root,out):
    c,m,reports = checked_freezes(root)
    need(not out.exists(), 'Immutable first cross-corpus score')
    out.mkdir(parents=True)
    cache = out.parent/'S609_private_annotations'
    cache.mkdir(exist_ok=True)
    summaries = {}
    for corpus in CORPORA:
        folder = out/corpus
        folder.mkdir()
        pred = read(root/corpus/'predictions.json')
        proof = {x['source_id']:x for x in reports[corpus]['audio_proof']}
        truth = {}; scope = {}; original = {}; provenance = []; rapid = []
        z = reader = None
        if corpus != 'MAESTRO':
            z,reader = archive(m['archives'][corpus+'_annotations'],cache)
        for spec in m['sources'][corpus]:
            sid = spec['source_id']; seconds = proof[sid]['seconds']
            blob = member(z,spec['annotation']) if z else standalone(spec['annotation'],m['maestro_base'])
            events = annotations(blob,corpus)
            for x in events:
                x['time'] -= m['offset_seconds'][corpus]
            original[sid] = events
            excerpt = [x for x in events if 0 <= x['time'] < seconds]
            scored = sorted({(x['time'],x['pitch']) for x in excerpt if 40 <= x['pitch'] <= 88})
            truth[sid] = [dict(time=t,notes=[p]) for t,p in scored]
            scope[sid] = dict(label=spec['label'],seconds=seconds,raw_note_onsets=len(excerpt),
                out_of_range_notes=[x for x in excerpt if not 40 <= x['pitch'] <= 88],
                exact_unison_duplicates_removed=sum(40<=x['pitch']<=88 for x in excerpt)-len(scored),
                true_notes=len(scored),strata=spec['strata'])
            provenance.append(dict(source_id=sid,path=spec['annotation']['path'],sha256=digest(blob),bytes=len(blob)))
            for track,pitch in {(x['track'],x['pitch']) for x in excerpt if 40<=x['pitch']<=88}:
                ts = sorted(x['time'] for x in excerpt if x['track']==track and x['pitch']==pitch)
                for a,b in zip(ts,ts[1:]):
                    if 0 < b-a < .1:
                        rapid.append(dict(source_id=sid,track=track,pitch=pitch,first=a,second=b,delta_ms=(b-a)*1000))
        if z:
            z.close()
        scores = metrics(pred,truth,{sid:dict(composer=s['label']) for sid,s in scope.items()})
        checks = 0
        for name in POLICIES:
            for sid in truth:
                for tol in (.03,.05,.08):
                    q = counts(pred[name][sid],truth[sid],tol)
                    need(all(q[k]==scores[name]['per_song'][sid]['scores'][str(int(tol*1000))][k] for k in ('TP','FP','FN')),
                         'Independent maximum-cardinality count mismatch')
                    checks += 1
        comparison = {}; audit = []; misses = []; coverage = {}
        for name in POLICIES[1:]:
            summary = collections.Counter()
            for sid in truth:
                old,oldfp = match_ledger(pred[POLICIES[0]][sid],truth[sid])
                new,newfp = match_ledger(pred[name][sid],truth[sid])
                for change,values in (('gained',new-old),('lost',old-new)):
                    summary[change] += len(values)
                    audit.extend(dict(policy=name,source_id=sid,change=change,pitch=p,truth_index=j,truth_time=t)
                                 for p,j,t in sorted(values))
                oc = collections.Counter((x['pitch'],round(x['time'],9)) for x in oldfp)
                nc = collections.Counter((x['pitch'],round(x['time'],9)) for x in newfp)
                summary['FP_removed'] += sum((oc-nc).values())
                summary['FP_added'] += sum((nc-oc).values())
                summary['FP_common'] += sum((oc&nc).values())
            summary['TP_delta'] = scores[name]['fullstream']['80']['TP']-scores[POLICIES[0]]['fullstream']['80']['TP']
            summary['FP_delta'] = scores[name]['fullstream']['80']['FP']-scores[POLICIES[0]]['fullstream']['80']['FP']
            need(summary['gained']-summary['lost']==summary['TP_delta'] and summary['FP_added']-summary['FP_removed']==summary['FP_delta'],
                 'Paired gains/losses reconciliation')
            comparison[name] = dict(summary)
        for sid in truth:
            ev = read(root/corpus/(sid+'.candidates.json'))
            prob = np.load(root/corpus/(sid+'.head.npz'),allow_pickle=False)['probability']
            coverage[sid] = counts([dict(time=x['time'],notes=[x['pitch']]) for x in ev
                                   if x['time'] < proof[sid]['seconds']],truth[sid])
            matched,_ = match_ledger(pred[c['primary_policy']][sid],truth[sid])
            for pitch in range(40,89):
                for j,t in enumerate(sorted(x['time'] for x in truth[sid] if pitch in x['notes'])):
                    if (pitch,j,t) in matched:
                        continue
                    near = [i for i,x in enumerate(ev) if x['pitch']==pitch and abs(x['time']-t)<=.08]
                    misses.append(dict(source_id=sid,pitch=pitch,truth_time=t,eligible_candidates=len(near),
                        maximum_candidate_probability=max((float(prob[i]) for i in near),default=None)))
        for event in rapid:
            event['both_notes_matched_by_policy'] = {}
            for name in POLICIES:
                matched,_ = match_ledger(pred[name][event['source_id']],truth[event['source_id']])
                event['both_notes_matched_by_policy'][name] = [any(p==event['pitch'] and abs(t-v)<1e-9 for p,j,t in matched)
                                                              for v in (event['first'],event['second'])]
        strata = {}
        for key in sorted({k for x in scope.values() for k in x['strata']}):
            strata[key] = {}
            for value in sorted({x['strata'][key] for x in scope.values()}):
                ids = {sid for sid,x in scope.items() if x['strata'][key]==value}
                strata[key][value] = {name:counts([dict(x,time=x['time']+30*i) for i,sid in enumerate(sorted(ids)) for x in pred[name][sid]],
                    [dict(x,time=x['time']+30*i) for i,sid in enumerate(sorted(ids)) for x in truth[sid]]) for name in POLICIES}
        outputs = {'truth_after_freeze.json':truth,'annotations_after_freeze.json':original,
                   'individual_note_audit.json':audit,'primary_FN_diagnosis.json':misses}
        for name,value in outputs.items():
            save(folder/name,value)
        primary = scores[c['primary_policy']]['fullstream']['80']
        report = dict(status='S609_FIRST_FROZEN_CROSS_CORPUS_TEST_NO_TUNING',corpus=corpus,
            primary_policy=c['primary_policy'],primary_F1=primary['F1'],objective_F1_080_reached=primary['F1']>=.8,
            scores=scores,comparison_to_default_BP=comparison,source_scope=scope,strata=strata,
            source_count=len(truth),true_notes=sum(len(x) for x in truth.values()),total_seconds=reports[corpus]['total_seconds'],
            independent_matching_checks=checks,candidate_coverage=coverage,genuine_same_instrument_rapid_repeats=rapid,
            annotation_provenance=provenance,freeze_report_sha256=sha(root/corpus/'report.json'),
            prediction_sha256=sha(root/corpus/'predictions.json'),contract_sha256=sha(CONTRACT),
            manifest_sha256=sha(MANIFEST),audit_hashes={name:sha(folder/name) for name in outputs},
            no_claim_of_unseen_BP_pretraining=True,full_S80_ExactK_NOT_REPLAYED=True,
            Live_NOT_modified=True,NoteOff_NOT_validated=True,no_promotion=True)
        save(folder/'report.json',report)
        summaries[corpus] = dict(primary_F1=primary['F1'],scores={name:scores[name]['fullstream'] for name in POLICIES},
            comparison_to_default_BP=comparison,source_count=len(truth),true_notes=report['true_notes'],
            total_seconds=report['total_seconds'],report_sha256=sha(folder/'report.json'))
        for name in POLICIES:
            print('S609 FIRST TRANSFER',corpus,name,scores[name]['fullstream']['80'],flush=True)
    save(out/'report.json',dict(status='S609_ALL_THREE_CORPORA_SCORED_NO_SELECTION',corpora=summaries,
        all_corpora_reach_080=all(x['primary_F1']>=.8 for x in summaries.values()),
        manifest_sha256=sha(MANIFEST),model_sha256=c['model_sha256'],no_tuning=True,no_promotion=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage',choices=('freeze','score'))
    p.add_argument('--corpus',choices=CORPORA)
    p.add_argument('--frozen',type=Path,default=Path('model/S608'))
    p.add_argument('--out',type=Path,default=Path('model/S609'))
    a = p.parse_args()
    if a.stage == 'freeze':
        need(a.corpus is not None,'Choose registered corpus')
        freeze(a.corpus,a.frozen)
    else:
        score(a.frozen,a.out)
