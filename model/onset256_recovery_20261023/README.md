# Récupération Onset HGB — expérience du 23 octobre 2026

Cinq checkpoints de recherche, choisis avec quatre compositions internes
par pli, puis sauvegardés avant le calcul du score OUTER.
Les SHA256, la politique et le seuil figurent dans chaque fichier JSON.
`manifest.json` conserve les empreintes des dix fichiers.

Ces poids remplacent uniquement FNRecovery. Le modèle principal, S701,
ErrorPrediction et la tête de sélection restent ceux du dossier
`model/onset256_research_39915038301/`.

Exemple de chargement du pli BN1 depuis la racine du dépôt :

```python
import json
from pathlib import Path
from onset256.live_event_stream import load_research_stream, LowLatencyOnsetStream
from onset256.fn_recovery import load_fn_recovery

base = load_research_stream(
    "model/onset256_research_39915038301", fold="BN1",
    native_path="model/live/strict_note/NOTE_strict_live.npz")
path = Path("model/onset256_recovery_20261023/recovery_BN1.joblib")
meta = json.loads(path.with_suffix(".json").read_text())
recovery = load_fn_recovery(path, path.with_suffix(".json"),
                           expected_sha256=meta["checkpoint_sha256"])
stream = LowLatencyOnsetStream(
    base.stream.native, base.stream.s701, base.risk, recovery, base.head)
# stream.push(bloc_float32_de_256, start_sample=0), puis 256, 512, etc.
```

Le pli est un choix explicite du protocole de recherche ; ces poids ne sont
pas un checkpoint unique de production. La sortie reste privée à Onset.
