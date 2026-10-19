import unittest
from scripts.s780_k_tail_domain_shift_audit import audit

def metrics(n,acc5,acc6,frames):
    per=[]
    for k in range(7):
        support=n if k in (5,6) else 1
        correct=acc5 if k==5 else acc6 if k==6 else 1
        per.append(dict(k=k,n=support,correct=correct,
            accuracy=correct/support,under=support-correct,
            major_under=0,over=0))
    return dict(exact_accuracy=.7,frames=frames,per_k=per)

class S780Tests(unittest.TestCase):
    def fixture(self):
        train=metrics(100,90,80,1000)
        cal=metrics(10,3,1,100)
        base=metrics(10,2,0,100)
        return dict(schema="S779_CLASS_AWARE_EXACTK_TAIL_SPECIALIST_FIT80",
            complete=True,promoted=False,final_MIDI_changed=False,
            fit_recordings_done=80,cal_recordings_done=32,
            decision_clock_samples=1024,no_CAL_training_or_selection=True,
            selected_tail_weight=4.0,
            FIT=dict(trial_ood={"4.0":train},baseline=train,
                     folds={"BN1":{},"Jazz1":{}}),
            CAL=dict(selected=cal,baseline=base,
                     comp=dict(cal,frames=60),
                     solo=dict(cal,frames=40)))
    def test_tail_recall_gap_and_no_promotion(self):
        d=audit(self.fixture())
        self.assertEqual(d["tail"]["6"]["fit_support"],100)
        self.assertEqual(d["tail"]["6"]["cal_support"],10)
        self.assertEqual(d["tail"]["6"]["recall_drop_points"],70.0)
        self.assertFalse(d["promoted"])
        self.assertEqual(d["evaluation_cadence_samples"],1024)
    def test_fit_cal_contract_must_be_frozen(self):
        r=self.fixture();r["no_CAL_training_or_selection"]=False
        with self.assertRaises(AssertionError):audit(r)
    def test_no_incorrect_schema(self):
        r=self.fixture();r["schema"]="incompatible"
        with self.assertRaises(ValueError):audit(r)

if __name__=="__main__":
    unittest.main()
