"""Metrics from ONE serialized event stream, with no optimistic EOF credit."""
import numpy as np
from causal_note.strict_live import SR
from scripts.s613_train_strict_live import event_score

SCORES = ('onset_f1', 'exact_k', 'active_exact_k', 'offset_accuracy', 'full_note_f1', 'frame_pitch_f1')


def metric(events, truth, ends):
    onset, pairs = event_score(events, truth)
    intervals, active = [], {}
    pred = np.zeros((len(ends), 49), bool)
    maximum = 0
    for e in events:
        p = int(e['pitch'])
        if e['available_sample'] < e['decision_sample'] or abs(e['time']-e['available_sample']/SR)>1e-10:
            raise AssertionError('Backdated or unavailable emission')
        frame = int(np.searchsorted(ends, e['available_sample']))
        if e['type'] == 'note_on':
            if p in active:
                raise AssertionError('NoteOn without a paired previous NoteOff')
            active[p] = len(intervals)
            intervals.append(dict(on=e, frame=frame))
            maximum = max(maximum, len(active))
        elif p in active:
            row = intervals[active.pop(p)]
            row['off'] = e
            pred[row['frame']:frame, p-40] = True
    for p, i in active.items():
        pred[intervals[i]['frame']:, p-40] = True
    target = np.zeros_like(pred)
    times = ends/SR
    count_target = np.zeros(len(ends), np.int32)
    for n in truth:
        target[(times>=n['time']) & (times<n['time']+n['duration']), n['pitch']-40] = True
        count_target += (times>=n['time']) & (times<n['time']+n['duration'])
    pc, tc = pred.sum(1), count_target
    valid = correct = long_correct = 0
    matched, good_offsets, long_onsets = [], [], []
    latency = []
    for i, j in pairs:
        matched.append(j)
        n = truth[j]
        if n['duration'] >= 1.:
            long_onsets.append(j)
        latency.append((intervals[i]['on']['time']-n['time'])*1000)
        stop = n['time']+n['duration']
        if stop >= ends[-1]/SR:
            continue
        valid += 1
        off = intervals[i].get('off')
        if (off and off['reason'] not in ('stream_end','shutdown','stream_error')
                and abs(off['time']-stop) <= .050):
            correct += 1
            good_offsets.append(j)
            long_correct += n['duration'] >= 1.
    result = dict(tp=onset['TP'], fp=onset['FP'], fn=onset['FN'],
        frames=len(ends), k_correct=int(np.count_nonzero(np.minimum(pc,6)==np.minimum(tc,6))),
        active_frames=int(np.count_nonzero(tc)),
        active_k_correct=int(np.count_nonzero((np.minimum(pc,6)==np.minimum(tc,6)) & (tc>0))),
        frame_pitch_tp=int(np.count_nonzero(pred & target)),
        frame_pitch_fp=int(np.count_nonzero(pred & ~target)),
        frame_pitch_fn=int(np.count_nonzero(~pred & target)),
        offset_eligible=valid, offset_correct=correct, long_offset_correct=int(long_correct),
        max_active=maximum, matched_truth_ids=matched, correct_offset_truth_ids=good_offsets,
        long_onset_truth_ids=long_onsets, latency_ms=latency)
    result.update(ratios(result))
    return result


def ratios(r):
    den = max(1, 2*r['tp']+r['fp']+r['fn'])
    return dict(onset_f1=2*r['tp']/den, onset_precision=r['tp']/max(1,r['tp']+r['fp']),
        onset_recall=r['tp']/max(1,r['tp']+r['fn']), exact_k=r['k_correct']/max(1,r['frames']),
        active_exact_k=r['active_k_correct']/max(1,r['active_frames']),
        offset_accuracy=r['offset_correct']/max(1,r['offset_eligible']),
        full_note_f1=2*r['offset_correct']/den,
        frame_pitch_f1=2*r['frame_pitch_tp']/max(1,2*r['frame_pitch_tp']+r['frame_pitch_fp']+r['frame_pitch_fn']))


def aggregate(rows):
    keys = ('tp','fp','fn','frames','k_correct','active_frames','active_k_correct',
            'frame_pitch_tp','frame_pitch_fp','frame_pitch_fn','offset_eligible',
            'offset_correct','long_offset_correct')
    r = {k: sum(x[k] for x in rows) for k in keys}
    r.update(ratios(r))
    r['max_active'] = max(x['max_active'] for x in rows)
    delays = [d for x in rows for d in x['latency_ms']]
    r['latency_ms_percentiles_10_50_95'] = np.percentile(delays,[10,50,95]).tolist() if delays else []
    r['all_targets_above_80'] = all(r[s] > .80 for s in SCORES)
    return r


def target_rank(r):
    # Full note F1 normally limits the minimum. All four still need to pass.
    return (all(r[s]>.80 for s in SCORES), min(r[s] for s in SCORES),
            sum(r[s] for s in SCORES)/len(SCORES))
