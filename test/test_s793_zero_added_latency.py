"""S793 fail-closed tests of events, scheduling, runtime timing and S792."""
import unittest
from causal_note.s793_zero_added_latency import (
    LatencyViolation,mark_reference,SchedulingContract,veto_added_event_delay,
    veto_runtime_regression,veto_metric_regression,veto_promotion,SCORES)
from scripts.s793_zero_added_latency_gate import assess_s792

def source():
    return mark_reference([dict(type='note_on',pitch=40,decision_sample=1024,available_sample=1024),
                           dict(type='note_off',pitch=40,decision_sample=2048,available_sample=2048)])
def measured():
    d=dict(hardware_id='pc',audio_device_id='device15',driver_mode='WASAPI',
        sample_rate_hz=44100,block_size_samples=256,hop_size_samples=128,
        workload_sha256='same_input',paired_repeated_trials=True)
    for channel in ('capture_to_noteon','capture_to_noteoff','total_audio_to_midi','cpu_inference'):
        for q in ('p50_ms','p95_ms','max_ms'):d[f'{channel}_{q}']=3.5
    return d
def metrics():return {s:.75 for s in SCORES}

class StrictZeroAdditionalLatencyTests(unittest.TestCase):
    def test_identity_succeeds_when_all_four_proofs_exist(self):
        ref=source()
        self.assertTrue(veto_promotion(contract=SchedulingContract(),
            reference=ref,candidate=[dict(e) for e in ref],
            baseline_runtime=measured(),candidate_runtime=measured(),
            baseline_metrics=metrics(),candidate_metrics=metrics())['permitted'])

    def test_any_late_noteoff_is_rejected(self):
        ref=source()
        for delay in (1,256,1024,2048,4096):
            cand=[dict(e) for e in ref]
            cand[1]['available_sample']+=delay;cand[1]['decision_sample']+=delay
            with self.assertRaises(LatencyViolation):veto_added_event_delay(ref,cand)

    def test_any_late_noteon_is_rejected(self):
        ref=source();cand=[dict(e) for e in ref]
        cand[0]['available_sample']+=1
        with self.assertRaises(LatencyViolation):veto_added_event_delay(ref,cand)

    def test_earlier_release_allowed(self):
        ref=source();cand=[dict(e) for e in ref]
        cand[1]['available_sample']-=256;cand[1]['decision_sample']-=256
        self.assertEqual(veto_added_event_delay(ref,cand)['max_added_event_samples'],0)

    def test_missing_event_lineage_and_delayed_new_note_fail(self):
        ref=source()
        with self.assertRaises(LatencyViolation):veto_added_event_delay(ref,ref[:1])
        extra=dict(type='note_on',pitch=52,decision_sample=1536,
                   available_sample=1536,first_eligible_sample=1280)
        with self.assertRaises(LatencyViolation):
            veto_added_event_delay(ref,[ref[0],extra,ref[1]])
        extra['first_eligible_sample']=1536
        self.assertEqual(veto_added_event_delay(ref,[ref[0],extra,ref[1]])['candidate_events_checked'],3)

    def test_any_added_lookahead_queue_confirmation_hold_or_hop_is_vetoed(self):
        for field in vars(SchedulingContract()):
            with self.assertRaises(LatencyViolation):
                SchedulingContract(**{field:1}).validate()

    def test_missing_real_hardware_measurement_blocks_promotion(self):
        ref=measured();trial=measured();trial.pop('cpu_inference_p95_ms')
        with self.assertRaises(LatencyViolation):veto_runtime_regression(ref,trial)
        trial=measured();trial['hardware_id']='another_pc'
        with self.assertRaises(LatencyViolation):veto_runtime_regression(ref,trial)

    def test_any_extra_cpu_or_noteon_noteoff_delay_fails(self):
        for key in ('capture_to_noteon_p95_ms','capture_to_noteoff_p50_ms',
                    'total_audio_to_midi_max_ms','cpu_inference_p95_ms'):
            base=measured();trial=measured();trial[key]+=.0001
            with self.assertRaises(LatencyViolation):veto_runtime_regression(base,trial)

    def test_score_gain_cannot_override_latency_veto(self):
        baseline=metrics();better=metrics();better['frame_pitch_f1']=.99
        ref=source();late=[dict(e) for e in ref];late[1]['available_sample']+=1024
        with self.assertRaises(LatencyViolation):
            veto_promotion(contract=SchedulingContract(),reference=ref,
                candidate=late,baseline_runtime=measured(),candidate_runtime=measured(),
                baseline_metrics=baseline,candidate_metrics=better)
        with self.assertRaises(LatencyViolation):
            veto_metric_regression(baseline,dict(baseline,offset_accuracy=.749))

class HistoricalS792Tests(unittest.TestCase):
    @staticmethod
    def report():
        values=metrics()
        variants={str(k):dict(eligible=k==0,event_changes=dict(held=20 if k else 0))
                  for k in (0,1,2,4)}
        return dict(schema='S792_FRAMEPITCH_S790_TRANSFER_REAL_MIDI256',complete=True,
          fit_done=80,cal_done=32,CAL=dict(official_source_parity=True,
          baseline=dict(values),trial=dict(values)),FIT=dict(selected_hold=0,variants=variants),selected_hold=0)
    def test_s792_1_2_4_blocks_rejected(self):
        d=assess_s792(self.report())
        self.assertEqual(d['rejected_hold_blocks'],[1,2,4])
        self.assertFalse(d['production_promotion_allowed'])
    def test_no_hidden_1_block_approval(self):
        d=self.report();d['FIT']['variants']['1']['eligible']=True
        with self.assertRaises(LatencyViolation):assess_s792(d)
    def test_no_unreported_baseline_change(self):
        d=self.report();d['CAL']['trial']['frame_pitch_f1']+=.01
        with self.assertRaises(LatencyViolation):assess_s792(d)
    def test_hardware_remains_unmeasured(self):
        self.assertFalse(assess_s792(self.report())['physical_microphone_latency_measured'])

if __name__=='__main__':unittest.main()
