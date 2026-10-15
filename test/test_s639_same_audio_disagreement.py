"""S639 register and matched-truth overlap are auditable without training."""
from __future__ import annotations
import unittest
from causal_note.s635_shared_midi_metric import compare_manifest
from scripts.s639_audit_same_audio_disagreement import audit


class SameAudioDisagreementTests(unittest.TestCase):
    def test_note_only_midi_only_both_and_missed_accounting(self):
        sha="b"*64
        labels=[dict(pitch=p,start_s=t,end_s=t+.25)
                for p,t in ((40,.1),(52,.3),(64,.5),(76,.7))]
        a=dict(audio_sha256=sha,audio_sample_rate=44100,
            events=[dict(type="note_on",pitch=40,time_s=.11),
                    dict(type="note_on",pitch=52,time_s=.31),
                    dict(type="note_on",pitch=68,time_s=.9)])
        b=dict(audio_sha256=sha,audio_sample_rate=44100,
            events=[dict(type="note_on",pitch=40,time_s=.12),
                    dict(type="note_on",pitch=64,time_s=.52)])
        manifest=dict(protocol="NOTE_MIDI_SHARED_ONSET_V1",clips=[
            dict(clip_id="c",corpus="GuitarSet",duration_s=1.,
                 audio_sha256=sha,reference=labels,note=a,midi=b)])
        score=compare_manifest(manifest)
        audited=audit(manifest,score)
        all=audited["matched_reference_outcomes"]["all"]
        self.assertEqual(all["both"],1)
        self.assertEqual(all["NOTE_only"],1)
        self.assertEqual(all["MIDI_only"],1)
        self.assertEqual(all["neither"],1)
        self.assertEqual(audited["F1_by_register"]["note"]["middle"]["FN"],1)
        self.assertEqual(audited["F1_by_register"]["note"]["high"]["FP"],1)
    def test_changed_score_report_is_not_accepted(self):
        sha="a"*64
        manifest=dict(protocol="NOTE_MIDI_SHARED_ONSET_V1",clips=[
            dict(clip_id="c",duration_s=1.,audio_sha256=sha,
                 reference=[dict(pitch=44,start_s=.3,end_s=.5)],
                 note=dict(audio_sha256=sha,audio_sample_rate=44100,events=[]),
                 midi=dict(audio_sha256=sha,audio_sample_rate=44100,events=[]))])
        score=compare_manifest(manifest)
        score["overall"]["note"]["80"]["FN"]=0
        with self.assertRaises(AssertionError):
            audit(manifest,score)


if __name__=="__main__":
    unittest.main()
