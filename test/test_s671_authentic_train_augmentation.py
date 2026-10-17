"""Safety checks for additional authentic GuitarSet training players."""
import unittest,io,zipfile
from scripts.s671_causal_hazard_more_players import expanded_train_specs

class S671AugmentationTests(unittest.TestCase):
    def test_group_and_player_constraints(self):
        originals=[dict(name="01_BN1-129-Eb_comp",composition="BN1",
                        audio="01_BN1-129-Eb_comp_mic.wav",
                        annotation="01_BN1-129-Eb_comp.jams")]
        bio=io.BytesIO();ann=io.BytesIO()
        with zipfile.ZipFile(bio,"w") as za,zipfile.ZipFile(ann,"w") as z:
            for player in ("00","01","02","03","04","05"):
                name=player+"_BN1-129-Eb_comp"
                za.writestr(name+"_mic.wav",b"123")
                z.writestr(name+".jams",b"123")
        bio.seek(0);ann.seek(0)
        with zipfile.ZipFile(bio) as a,zipfile.ZipFile(ann) as b:
            out=expanded_train_specs(originals,a,b)
        self.assertEqual(len(out),4)
        self.assertTrue(all(x["composition"]=="BN1" for x in out))
        self.assertEqual([x["name"][:2] for x in out],["00","01","02","03"])
    def test_missing_original_fails(self):
        original=[dict(name="00_BN1_comp",composition="BN1",audio="a",annotation="b")]
        a=io.BytesIO();b=io.BytesIO()
        with zipfile.ZipFile(a,"w"),zipfile.ZipFile(b,"w"):
            pass
        a.seek(0);b.seek(0)
        with zipfile.ZipFile(a) as z1,zipfile.ZipFile(b) as z2:
            with self.assertRaises(AssertionError):
                expanded_train_specs(original,z1,z2)
if __name__=="__main__":
    unittest.main()
