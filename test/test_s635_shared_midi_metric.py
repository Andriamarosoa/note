"""S635 common same-audio causal matcher tests; no fabricated benchmark claims."""
from __future__ import annotations
import unittest
from causal_note.s635_shared_midi_metric import (
    causal_score,extract_reference,extract_events,compare_manifest,event_time,
    DEADLINES_MS
)


class CommonMetricTests(unittest.TestCase):
    def test_future_reference_never_counts_as_detected_note(self):
        refs=extract_reference([dict(pitch=52,start_s=.100,end_s=.400)],1.)
        predictions=extract_events([dict(type="note_on",pitch=52,time_s=.099)],1.)
        v=causal_score(refs,predictions,80)
        self.assertEqual((v["TP"],v["FP"],v["FN"]),(0,1,1))

    def test_retrigger_matches_latest_and_not_old_onset(self):
        refs=extract_reference([
            dict(pitch=52,start_s=.100,end_s=.12),
            dict(pitch=52,start_s=.180,end_s=.30)],1.)
        events=extract_events([dict(type="note_on",pitch=52,time_s=.181)],1.)
        r=causal_score(refs,events,80)
        self.assertEqual(r["TP"],1)
        self.assertEqual(r["FN"],1)
        self.assertEqual(r["matches"][0][0],1)

    def test_duplicate_noteons_are_false_positives(self):
        refs=extract_reference([dict(pitch=52,start_s=.1,end_s=.2)],1.)
        events=extract_events([
            dict(type="note_on",pitch=52,time_s=.13),
            dict(type="note_on",pitch=52,time_s=.14)],1.)
        r=causal_score(refs,events,80)
        self.assertEqual((r["TP"],r["FP"],r["FN"]),(1,1,0))

    def test_deadline_recomputed_not_loosened_posthoc(self):
        refs=extract_reference([dict(pitch=52,start_s=.1,end_s=.2)],1.)
        prediction=extract_events([dict(type="note_on",pitch=52,time_s=.145)],1.)
        self.assertEqual(causal_score(refs,prediction,35)["TP"],0)
        self.assertEqual(causal_score(refs,prediction,46)["TP"],1)

    def test_actual_output_sample_cannot_be_backdated(self):
        self.assertAlmostEqual(event_time(dict(type="note_on",pitch=52,
                      available_sample=4410,decision_sample=4400,time=.1)),.1)
        with self.assertRaises(ValueError):
            event_time(dict(available_sample=4400,decision_sample=4410))
        with self.assertRaises(ValueError):
            event_time(dict(available_sample=4410,time=.05))

    def test_only_identical_audio_and_common_pitch_range(self):
        sha="c"*64
        c=dict(clip_id="identical",
            corpus="GuitarSet",audio_sha256=sha,duration_s=1.,
            reference=[dict(pitch=52,start_s=.1,end_s=.4),
                       dict(pitch=85,start_s=.3,end_s=.5)],
            note=dict(audio_sha256=sha,audio_sample_rate=44100,
                events=[dict(type="note_on",pitch=52,time_s=.12),
                        dict(type="note_on",pitch=85,time_s=.32)]),
            midi=dict(audio_sha256=sha,audio_sample_rate=44100,
                events=[dict(type="note_on",pitch=52,time_s=.13)]))
        result=compare_manifest(dict(protocol="NOTE_MIDI_SHARED_ONSET_V1",clips=[c]))
        self.assertEqual(result["common_pitch_range"],[40,76])
        self.assertEqual(result["by_clip"]["identical"]["reference_notes"],1)
        self.assertEqual(result["by_clip"]["identical"]["reference_notes_outside_common_range"],1)
        self.assertEqual(result["by_clip"]["identical"]["ignored_predictions_outside_common_range"]["note"],1)
        self.assertEqual(result["overall"]["note"]["80"]["F1"],1.)
        self.assertEqual(result["overall"]["midi"]["80"]["F1"],1.)
        c["midi"]["audio_sha256"]="a"*64
        with self.assertRaisesRegex(ValueError,"identical"):
            compare_manifest(dict(protocol="NOTE_MIDI_SHARED_ONSET_V1",clips=[c]))

    def test_no_sources_or_patched_pretrained_scores(self):
        with self.assertRaises(ValueError):
            compare_manifest(dict(protocol="NOTE_MIDI_SHARED_ONSET_V1",clips=[]))
        with self.assertRaises(ValueError):
            compare_manifest(dict(protocol="historical_scores_only",clips=[]))
        self.assertEqual(len(DEADLINES_MS),7)


if __name__=="__main__":
    unittest.main()
