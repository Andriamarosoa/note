"""S698 strictly retain composition split and prohibit unseen DEV/TEST players."""
import io,unittest,zipfile
from scripts.s698_multiplayer_K_shadow import expanded_train_specs,TRAIN_ROW_STRIDE
class S698KTrainTests(unittest.TestCase):
    def test_only_authentic_train_compositions_and_players(self):
        original=[dict(name="01_BN1-111-C_comp",composition="BN1",
                       audio="01_BN1-111-C_comp_mic.wav",
                       annotation="01_BN1-111-C_comp.jams")]
        wav=io.BytesIO();annotations=io.BytesIO()
        with zipfile.ZipFile(wav,"w") as za,zipfile.ZipFile(annotations,"w") as zb:
            for i in range(6):
                label=f"{i:02d}_BN1-111-C_comp"
                za.writestr(label+"_mic.wav",b"WAV")
                zb.writestr(label+".jams",b"JSON")
        wav.seek(0);annotations.seek(0)
        with zipfile.ZipFile(wav) as za,zipfile.ZipFile(annotations) as zb:
            produced=expanded_train_specs(original,za,zb)
        self.assertEqual(len(produced),4)
        self.assertTrue(all(row["name"][:2] in ("00","01","02","03") for row in produced))
        self.assertTrue(all(row["composition"]=="BN1" for row in produced))
        self.assertEqual(TRAIN_ROW_STRIDE,3)
if __name__=="__main__":unittest.main()
