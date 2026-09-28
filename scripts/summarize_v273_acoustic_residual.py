"""Cross-check the acoustic audit against its frozen prediction source."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rebuild_v273_sources import digest, write_json
from scripts.restore_v273_original_backup import validate_original_inventory


def describe(values):
    x = np.asarray(values, np.float64)
    return dict(rows=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p10=float(np.quantile(x, .1)), p90=float(np.quantile(x, .9)))


def summarize(root, previous, output):
    for source in (root, previous):
        validate_original_inventory(source)
    report = json.loads((root / 'report.json').read_text())
    assert report['status'] == 'completed'
    assert report['script_sha256'] == digest(Path(__file__).with_name('audit_v273_acoustic_residual.py'))
    assert report['config_sha256'] == digest(Path(__file__).parents[1] / 'analysis/v273-native-paired-config.json')
    with np.load(root / 'rows.npz', allow_pickle=False) as z:
        rows = {name: np.asarray(z[name]) for name in z.files}
    with np.load(previous / 'row-context.npz', allow_pickle=False) as z:
        old = {name: np.asarray(z[name]) for name in z.files}
    with np.load(previous / 'frames-31-uniform-probes.npz', allow_pickle=False) as z:
        probability = np.asarray(z['baseline'])
    for name in ('k', 'member', 'global_index'):
        np.testing.assert_array_equal(rows[name], old[name])
    np.testing.assert_array_equal(rows['start'], old['cluster_start_samples'])
    np.testing.assert_array_equal(rows['predicted'], probability.argmax(1))
    assert np.isfinite(probability).all() and probability.shape == (15279, 7)
    np.testing.assert_allclose(probability.sum(1), 1, atol=1e-5)
    k, pred = rows['k'], rows['predicted']
    low, over = k < 4, (k < 4) & (pred > k)
    assert low.sum() == 14744 and over.sum() == report['over_rows'] == 874
    assert len(np.unique(rows['global_index'])) == len(k)
    eligible, active = rows['eligible_births'], rows['max_simultaneous_notes']
    np.testing.assert_array_equal(rows['contested_births'], eligible - k)
    masks = {
        'local_eligible_equals_prediction': over & (eligible == pred),
        'local_competition_present': over & (eligible > k),
        'active_note_count_equals_prediction': over & (active == pred),
        'foreign_birth_visible': over & (old['foreign_births_31'] > 0),
        'no_note_annotation_overlap': over & (old['overlapping_notes_31'] == 0),
    }
    assert report['categories'] == {name: int(mask.sum()) for name, mask in masks.items()}
    np.testing.assert_array_equal(active == 0, old['overlapping_notes_31'] == 0)
    measurement_names = set(report['by_true_k']['0']['measurements'])
    for value in range(4):
        same = k == value
        expected = report['by_true_k'][str(value)]
        assert expected['rows'] == same.sum() and expected['over'] == (same & over).sum()
        assert expected['categories'] == {name: int((same & mask).sum()) for name, mask in masks.items()}
        for name in measurement_names:
            assert np.isfinite(rows[name]).all()
            for group, take in (('over', same & over), ('correct', same & (pred == k))):
                computed = describe(rows[name][take])
                for stat, val in computed.items():
                    np.testing.assert_allclose(val, expected['measurements'][name][group][stat], rtol=1e-12, atol=1e-12)
    for name in measurement_names:
        computed = describe(rows[name][masks['no_note_annotation_overlap']])
        for stat, val in computed.items():
            np.testing.assert_allclose(val, report['annotation_quiet_errors'][name][stat], rtol=1e-12, atol=1e-12)

    by_id = {int(index): i for i, index in enumerate(rows['global_index'])}
    traces = json.loads((root / 'overcount-traces.json').read_text())
    assert len(traces) == 874
    assert {t['global_index'] for t in traces} == set(rows['global_index'][over])
    no_foreign_annotations = np.zeros(len(k), bool)
    for trace in traces:
        row = by_id[trace['global_index']]
        assert trace['member'] == rows['member'][row]
        assert trace['start_sample'] == rows['start'][row]
        assert trace['true_k'] == k[row] and trace['predicted_k'] == pred[row]
        np.testing.assert_array_equal(trace['probability'], probability[row])
        start, events = trace['start_sample'], trace['events']
        left, right = start - 1308, start + 2788
        assert sum(e['own'] for e in events) == k[row]
        assert sum(e['distance_to_current_candidate'] <= 882 for e in events) == eligible[row]
        assert trace['local_eligible_births'] == eligible[row]
        assert trace['contested_births'] == rows['contested_births'][row]
        assert sum(left <= e['onset_sample'] < right and not e['own'] for e in events) == old['foreign_births_31'][row]
        assert sum(e['onset_sample'] < right and e['offset_sample'] > left for e in events) == old['overlapping_notes_31'][row]
        no_foreign_annotations[row] = all(e['own'] for e in events)
        for name, val in trace['audio'].items():
            np.testing.assert_allclose(val, rows[name][row], rtol=1e-12, atol=1e-12)
    with np.load(root / 'illustrative-clips.npz', allow_pickle=False) as clips:
        assert set(clips.files) == {str(e['global_index']) for e in report['examples']}
        for example in report['examples']:
            row = by_id[example['global_index']]
            clip = clips[str(example['global_index'])].astype(np.float64)
            assert clip.shape == (22050,)
            segment = clip[11025 - 1308:11025 + 2788]
            rms = 20 * np.log10(max(np.sqrt(np.mean(segment ** 2)), 1e-10))
            np.testing.assert_allclose(rms, rows['rms_dbfs'][row], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(np.max(np.abs(segment)), rows['peak_absolute'][row], rtol=0, atol=0)

    def exposure(same, flag):
        return {name: dict(rows=int((same & take).sum()),
                           over=int((same & take & over).sum()),
                           rate=float((same & take & over).sum() / (same & take).sum()))
                for name, take in (('present', flag), ('absent', ~flag))}

    by_k = {}
    for value in range(4):
        same = k == value
        by_k[str(value)] = dict(rows=int(same.sum()), over=int((same & over).sum()),
            correct=int((same & (pred == k)).sum()), under=int((same & (pred < k)).sum()),
            competing_eligibility=exposure(same, eligible > k),
            foreign_birth=exposure(same, old['foreign_births_31'] > 0),
            no_foreign_annotation_over=int((same & no_foreign_annotations).sum()),
            predicted_exceeds_max_active_over=int((same & over & (pred > active)).sum()),
            over_predictions=np.bincount(pred[same & over], minlength=7).tolist())
    summary = dict(status='completed_and_cross_checked', outer_fold=3,
        variant='frames-31-uniform', rows=len(k), low_k_rows=int(low.sum()),
        over=int(over.sum()), correct=int((low & (pred == k)).sum()),
        under=int((low & (pred < k)).sum()), by_true_k=by_k,
        categories=report['categories'], category_counts_overlap=True,
        no_foreign_annotation_over=int(no_foreign_annotations.sum()),
        no_foreign_annotation_positive_k_over=int((no_foreign_annotations & (k > 0)).sum()),
        predicted_exceeds_all_overlapping_annotations=int((over & (pred > old['overlapping_notes_31'])).sum()),
        predicted_exceeds_max_active=int((over & (pred > active)).sum()),
        prediction_and_row_source_verified=True, all_report_statistics_recomputed=True,
        all_error_traces_cross_checked=True, illustrative_clip_metrics_verified=True,
        raw_cache_replay_location='GitHub Actions run 36373588730; all 874 errors and 667 correct controls',
        audit_report_sha256=digest(root / 'report.json'), rows_sha256=digest(root / 'rows.npz'),
        traces_sha256=digest(root / 'overcount-traces.json'),
        previous_context_sha256=digest(previous / 'row-context.npz'),
        previous_probabilities_sha256=digest(previous / 'frames-31-uniform-probes.npz'),
        summarizer_sha256=digest(__file__), no_model_training=True, no_output_correction=True,
        no_relabelled_score=True, causal_status='descriptive residual audit; not a validated correction')
    write_json(output, summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'previous', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.previous, args.output)
    print(json.dumps({name: result[name] for name in ('status', 'low_k_rows', 'over', 'no_foreign_annotation_positive_k_over')}))
