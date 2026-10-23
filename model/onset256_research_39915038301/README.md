# Poids de recherche Onset PCM256 — run 39915038301

Ce dossier conserve les cinq modèles finaux choisis avant leur replay OUTER,
les cinq experts ErrorPrediction, les cinq experts FNRecovery et S701.
Le modèle S613 natif reste dans `model/live/strict_note/NOTE_strict_live.npz`.

Les fichiers sont copiés sans modification depuis les artefacts des runs
indiqués dans `manifest.json`. Les métadonnées originales et les SHA256 sont
conservés. Rock3 a retenu le contrôle MLP69 ; les quatre autres plis ont
retenu les nouveaux HGB à 7 feuilles.

Installer `requirements-onset256.txt`, puis depuis la racine du dépôt :

```python
from onset256.live_event_stream import load_research_stream

stream = load_research_stream(
    "model/onset256_research_39915038301",
    fold="BN1",
    native_path="model/live/strict_note/NOTE_strict_live.npz",
)
# La construction et la compilation précèdent la réception audio.
decision = stream.push(audio_float32_de_256_echantillons, start_sample=0)
# Continuer avec start_sample=256, puis 512, etc.
```

Le pli est un paramètre explicite du protocole de recherche. Il ne doit pas
être deviné depuis l'audio ou le nom d'un fichier pour annoncer un score
de déploiement. Cet ensemble n'est pas un checkpoint unique de production.
La sortie contient uniquement les intentions Onset privées, sans MIDI.
