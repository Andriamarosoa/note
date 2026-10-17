"""S699 strict player 03 held and all nine TRAIN compositions retained."""
import io,unittest,zipfile
from scripts.s699_player_held_full_train_K import expanded_train_specs
class S699SplitTests(unittest.TestCase):
    def test_player_zero_to_two_train_three_unseen(self):
        original=[dict(name="01_Rock3-110-C_comp",composition="Rock3",
                       audio="01_Rock3-110-C_comp_mic.wav",
                       annotation="01_Rock3-110-C_comp.jams")]
        wav=io.BytesIO();anno=io.BytesIO()
        with zipfile.ZipFile(wav,"w") as z1,zipfile.ZipFile(anno,"w") as z2:
            for player in range(6):
                stem=f"{player:02d}_Rock3-110-C_comp"
                z1.writestr(stem+"_mic.wav",b"original")
                z2.writestr(stem+".jams",b"original")
        wav.seek(0);anno.seek(0)
        with zipfile.ZipFile(wav) as z1,zipfile.ZipFile(anno) as z2:
            alltracks=expanded_train_specs(original,z1,z2)
        training=[row for row in alltracks if row["name"][:2] in ("00","01","02")]
        held=[row for row in alltracks if row["name"][:2]=="03"]
        self.assertEqual(len(training),3)
        self.assertEqual(len(held),1)
        self.assertEqual({row["name"][:2] for row in training},{"00","01","02"})
        self.assertEqual(held[0]["name"][:2],"03")
        self.assertEqual({row["composition"] for row in training+held},{"Rock3"})
if __name__=="__main__":unittest.main()
