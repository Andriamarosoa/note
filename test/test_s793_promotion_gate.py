"""S793 ensure the real future-series promotion validator invokes the veto."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.verify_series_reports import validate_series,ROLES,METRICS
from causal_note.s793_zero_added_latency import mark_reference,SCORES

class MandatoryFuturePromotionGateTests(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        folder=root/"series"/"S794";folder.mkdir(parents=True)
        text=" ".join(METRICS)+"\n"+(
            "https://github.com/Andriamarosoa/note/actions/runs/39328341071\n")
        text+="\n".join(f"| {role} | PASS | CI verified |" for role in ROLES)
        (folder/"resultat.md").write_text(text)
        return root,folder

    def metadata(self):
        return dict(serie="S794",statut="termine",type="experience",
            reference="S708",branche="codex/note-s794-latency-test",
            run_id="39328341071",commit_sha="a"*40,promotion="true")

    def proof(self):
        ref=mark_reference([dict(type="note_on",pitch=40,
            decision_sample=1024,available_sample=1024),
            dict(type="note_off",pitch=40,
            decision_sample=2048,available_sample=2048)])
        hardware=dict(hardware_id="synthetic_test",audio_device_id="synthetic",
            driver_mode="synthetic",sample_rate_hz=44100,block_size_samples=256,
            hop_size_samples=256,workload_sha256="controlled",paired_repeated_trials=True)
        for channel in ("capture_to_noteon","capture_to_noteoff",
                        "total_audio_to_midi","cpu_inference"):
            for metric in ("p50_ms","p95_ms","max_ms"):
                hardware[f"{channel}_{metric}"]=2.0
        return dict(scheduling=dict(),reference_events=ref,
            candidate_events=[dict(e) for e in ref],
            baseline_runtime=hardware,candidate_runtime=dict(hardware),
            baseline_metrics={k:.5 for k in SCORES},
            candidate_metrics={k:.5 for k in SCORES})

    def check(self,root):
        with patch("scripts.verify_series_reports.parse_document",
                   side_effect=lambda *args:self.metadata()):
            return validate_series(root,"S794")

    def test_missing_hardware_proof_blocks_future_promotion(self):
        root,_=self.fixture()
        with self.assertRaisesRegex(ValueError,"missing mandatory S793"):
            self.check(root)

    def test_forged_late_noteoff_blocks_future_promotion(self):
        root,folder=self.fixture()
        proof=self.proof();proof["candidate_events"][1]["available_sample"]+=1
        (folder/"latency_evidence.json").write_text(json.dumps(proof))
        with self.assertRaisesRegex(ValueError,"S793 zero-added-latency"):
            self.check(root)

    def test_forged_extra_cpu_latency_blocks_future_promotion(self):
        root,folder=self.fixture()
        proof=self.proof();proof["candidate_runtime"]["cpu_inference_p95_ms"]+=.01
        (folder/"latency_evidence.json").write_text(json.dumps(proof))
        with self.assertRaisesRegex(ValueError,"S793 zero-added-latency"):
            self.check(root)

    def test_structurally_valid_synthetic_proof_only_proves_gate(self):
        root,folder=self.fixture()
        proof=self.proof()
        (folder/"latency_evidence.json").write_text(json.dumps(proof))
        self.assertEqual(self.check(root)["promotion"],"true")
        # This is NOT proof of actual measured microphone latency.
        self.assertEqual(proof["baseline_runtime"]["hardware_id"],"synthetic_test")

if __name__=="__main__":unittest.main()
