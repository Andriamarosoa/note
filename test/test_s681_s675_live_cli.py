"""S681 retains S630 original NOTE Onsets, supports latest S675 CLI."""
import io,unittest,tempfile
from contextlib import redirect_stdout
from pathlib import Path
from contextlib import redirect_stderr
from scripts.live_strict_notes import main

class S681CommandTests(unittest.TestCase):
    def test_latest_checkpoint_must_exist(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main(["--status","--s675-hazard-model","does-not-exist.joblib",
                      "--inference-batch-hops","4"])
        self.assertEqual(result.exception.code,2)

    def test_latest_model_rejects_lookahead_and_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp)/"s675_hazard.joblib"
            model.write_bytes(b"not loaded during guard test")
            for extra in (
                ["--inference-batch-hops","4","--lookahead-ms","5"],
                ["--inference-batch-hops","4","--s667-release-frames","3"],
                ["--inference-batch-hops","4","--dual-hop-mode","shadow"],
            ):
                with self.subTest(extra=extra),redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as result:
                        main(["--status","--s675-hazard-model",str(model),*extra])
                self.assertEqual(result.exception.code,2)

    def test_batch_one_and_four_are_valid_research_configs(self):
        # --status exits after argument validation, before loading joblib.
        # This is a CLI guard test; real model routing tested separately.
        with tempfile.TemporaryDirectory() as tmp:
            for family in ("s672", "s675"):
                model=Path(tmp)/(family+"_hazard.joblib")
                model.write_bytes(b"placeholder")
                for batch in ("1", "4"):
                    with self.subTest(family=family,batch=batch):
                        with redirect_stdout(io.StringIO()) as visible:
                            main(["--status",
                                  "--"+family+"-hazard-model",str(model),
                                  "--inference-batch-hops",batch])
                        self.assertIn("EXPERIMENTAL_AUTONOMOUS_CAUSAL_NOTE_LIVE",
                                      visible.getvalue())

    def test_cannot_load_two_versions_simultaneously(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/"s672_hazard.joblib";old.write_bytes(b"x")
            new=Path(tmp)/"s675_hazard.joblib";new.write_bytes(b"x")
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    main(["--status","--s675-hazard-model",str(new),
                          "--s672-hazard-model",str(old),"--inference-batch-hops","4"])
            self.assertEqual(result.exception.code,2)
if __name__=="__main__":unittest.main()
