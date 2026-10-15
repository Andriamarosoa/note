"""Recent NOTE fidelity, bounded state and causal-normalization contracts."""
import unittest
from unittest.mock import patch
import numpy as np

from causal_note.recent_native import RecentFeatureState, RecentSelector, POLICIES
from causal_note.recent_audio import RecentAudioStream, RecentEvent, reference_profile, FFT, HOP, SR
from scripts.live_recent_notes import replay_acoustics, replay_events
from scripts.s569_train_real_23_source_52D_temporal_restrike_note_heads import temporal52
from scripts.s585_attack_sustain_restrike import emit as state_emit
from scripts.s536_onset_burst_adaptive_restrike_dedup_loops import emit as global_emit
from scripts.s515_freeze_real_WAV_49_pitch_frame_onset_harmonic_experts import features


def row(t, i, x=None, births=()):
    if x is None:
        x = np.random.default_rng(i).uniform(0, 2, (49, 11)).astype(np.float32)
    return dict(source_recording_sha="source", V88_original_event_id=f"event:{i:05d}",
                original_V88_candidate_start_s=t, original_V88_K_soft_prior_ONLY=0,
                old_basicpitch_birth_pitches=list(births), per_midi_features_49x11=x.tolist())


class RecentNativeTests(unittest.TestCase):
    def test_temporal_features_equal_original_at_gaps_and_tied_times(self):
        rows = [row(t, i) for i,t in enumerate((0., .01, .08, .2, .55, .901, 1., 1.))]
        stream = RecentFeatureState()
        np.testing.assert_array_equal(np.stack([stream.push(r) for r in rows]), temporal52(rows))
        self.assertEqual(len(stream.history), 2)
        with self.assertRaises(ValueError): stream.push(rows[-1])
        with self.assertRaises(ValueError): stream.push(dict(row(2., 9), source_recording_sha="other"))

    def test_original_selectors_keep_BP_K0_and_harmonic_restrike_rules(self):
        x = np.ones((49, 11)); x[12, [3, 5]] = .01
        rows = [row(t, i, x, (52,) if i == 2 else ())
                for i,t in enumerate((.1, .15, .18, .25, .4, .45001))]
        gp = np.zeros((len(rows), 49)); gp[:, [0, 12, 30]] = .8
        sp = np.zeros((len(rows), 49, 4)); sp[:, :, 0] = 1
        sp[:, [0, 12, 30], 0] = 0; sp[:, [0, 12, 30], 2] = .2; sp[:, [0, 12, 30], 3] = .8
        for name in POLICIES:
            s = RecentSelector(name); result=[]
            for i,r in enumerate(rows):
                notes=s.select(r,gp[i],sp[i])
                if notes: result.append(dict(time=r['original_V88_candidate_start_s'],notes=notes))
            if name == POLICIES[0]:
                expected, counts=global_emit(list(zip(rows,[dict(raw_only_49_pitch_prob=g) for g in gp])),
                                             dict(dedup_ms=100,harmonic_ratio=.8,rearticulation_ratio=0))
            else:
                expected, counts=state_emit(rows,sp,name==POLICIES[2])
            self.assertEqual(result,expected); self.assertEqual(dict(s.counts),counts)
            self.assertLessEqual(len(s.last),49)
            self.assertIn(dict(time=.18,notes=[52]),result)

    def test_reference_acoustics_match_original_and_block_sizes(self):
        t=np.arange(3*SR)/SR
        wave=(.15*np.sin(2*np.pi*164.81*t)*np.maximum(0,np.sin(2*np.pi*3*t))).astype(np.float32)
        rows=[row(v,i) for i,v in enumerate((0., .09, .3, .5, 1., 1.5, 2.5, 2.96))]
        notes=[dict(pitch_MIDI=52,predicted_onset_s=.3,predicted_offset_s=1.5,pretrained_note_score=.9)]
        ev=replay_events(rows,notes)
        oracle=np.asarray(features(wave,[dict(event_time_seconds=e.time,bp_note_pitches=e.births,
            bp_active_pitches_at_birth_plus30=e.active) for e in ev],notes)).round(6)
        a=list(replay_acoustics(wave,rows,notes,normalization='reference'))
        b=list(replay_acoustics(wave,rows,notes,normalization='reference',blocks=(127,997,4096)))
        actual=np.asarray([q['row']['per_midi_features_49x11'] for q in a])
        np.testing.assert_allclose(actual,oracle,atol=2e-6,rtol=0)
        np.testing.assert_array_equal(actual,np.asarray([q['row']['per_midi_features_49x11'] for q in b]))
        for packet in a:
            self.assertGreaterEqual(packet['available_sample'],packet['audio_context_end_sample'])
        self.assertTrue(a[-1]['end_of_source'])

    def test_rolling_prefix_cannot_change_when_future_changes(self):
        rng=np.random.default_rng(612);wave=rng.normal(0,.05,3*SR).astype(np.float32)
        changed=wave.copy();changed[2*SR:]*=8
        rows=[row(t,i) for i,t in enumerate((.1,.3,.6,1.,1.7))]
        a=list(replay_acoustics(wave,rows,[]));b=list(replay_acoustics(changed,rows,[],blocks=(997,)))
        self.assertEqual([p['row'] for p in a],[p['row'] for p in b])

    def test_no_future_padding_and_explicit_failure_on_gap_and_late_producer(self):
        s=RecentAudioStream();e=RecentEvent('source','a',0.,0,(),(),(0.,)*49,1323)
        out=s.push(np.zeros(FFT,np.float32),start_sample=0,events=[e]);self.assertEqual(out,[])
        out=s.push(np.zeros(2*HOP,np.float32),start_sample=FFT)
        self.assertEqual(len(out),1);self.assertEqual(out[0]['audio_context_end_sample'],FFT+2*HOP)
        with self.assertRaises(ValueError):s.push(np.zeros(1,np.float32),start_sample=0)
        with self.assertRaises(ValueError):RecentEvent('source','bad',1.,0,(),(),(0.,)*49,44100)
        for _ in range(6):s.push(np.zeros(SR,np.float32),start_sample=s.received)
        late=RecentEvent('source','b',.2,0,(),(),(0.,)*49,10143)
        with self.assertRaises(BufferError):s.push(np.empty(0,np.float32),start_sample=s.received,events=[late])
        self.assertLess(len(s.tail),FFT);self.assertLessEqual(len(s.frames),512)
        s.finish()
        with self.assertRaises(ValueError):s.push(np.empty(0,np.float32),start_sample=s.received)

    def test_router_exposes_recent_native_mode_without_changing_old_mode(self):
        from scripts.note_runtime import main
        with patch('scripts.live_recent_notes.main') as recent:
            main(['--mode','recent-notes','--status'])
            recent.assert_called_once_with(['--status'])


if __name__ == '__main__':
    unittest.main()
