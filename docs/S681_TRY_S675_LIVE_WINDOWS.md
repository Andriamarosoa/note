# S681 — essayer S675 NoteOff sur le microphone Windows

## Résultat de recherche, pas score LIVE

S675 : 773 / 990 fins ±50 ms = **78,08 % sur GuitarSet DEV**, en
arrondissant les décisions à la disponibilité du microbatch ×4.
La politique « hazard_0.60_persist_1 » a été sélectionnée sur
compositions TRAIN avant ouverture de DEV. Les NoteOn et le K
prédits du modèle natif sont conservés.

Le S681 est un portage EXPÉRIMENTAL sur le moteur causal S630 :
enregistrement de nouvelles fins de notes à partir d'un modèle
GuitarSet S675 75D, et émission réelle sans retour dans le temps.
Le repli après une fermeture native sans signal S675 attend
24 trames supplémentaires. Cela diffère du protocole offline ;
**78,08 % ne constitue donc pas une précision mesurée sur microphone**.

## Installation

Rester à l'écart de la branche de référence pendant les essais.
En PowerShell :

```powershell
cd C:\Users\user\Desktop\note
git fetch origin
git switch codex/note-s681-s675-live-trial
git pull --ff-only origin codex/note-s681-s675-live-trial
python -m pip install -r requirements-live-strict.txt
python -m pip install sounddevice==0.5.3
```

Télécharger l'artifact « S681_LATEST_S675_TRAINED_NOTEOFF_WINDOWS_TRIAL »
depuis le run GitHub Actions ci-dessous, APRÈS réussite. Décompresser
le ZIP dans le dossier `s681_model` du dépôt. Il doit contenir
`s675_hazard.joblib` et `s675_checkpoint.json`, dont les SHA sont
vérifiés au chargement. **Ne pas charger d'autres fichiers joblib** :
joblib est un format de sérialisation basé sur pickle.

Exécution CI et artefact :
https://github.com/Andriamarosoa/note/actions/workflows/note-s681-s675-live-trial.yml

## WAV mono PCM16 44,1 kHz

Même audio, mêmes NoteOn attendus, S675 modifie uniquement NoteOff :

```powershell
python -m scripts.live_strict_notes --wav guitare.wav --hop-size 256 --block-size 256 --inference-batch-hops 4 --out original.jsonl
python -m scripts.live_strict_notes --wav guitare.wav --hop-size 256 --block-size 256 --inference-batch-hops 4 --s675-hazard-model s681_model/s675_hazard.joblib --out s675.jsonl
```

## Microphone

Actualiser l'indice de l'entrée avec `--list-devices`. Exemple avec
l'indice historique 15 uniquement si c'est toujours la bonne entrée :

```powershell
python -m scripts.live_strict_notes --list-devices
python -m scripts.live_strict_notes --mic --device 15 --auto-level --hop-size 256 --block-size 256 --inference-batch-hops 4 --s675-hazard-model s681_model/s675_hazard.joblib --out essai_s675.jsonl
```

Pour comparer, relancer sans `--s675-hazard-model` et changer
le nom de sortie. Les tests utilisent un WAV simulé et de vrais
Poids GitHub, **pas un microphone iRig Windows réel**.

Lancer explicitement le mode expérimental ; la CLI native sans cette
option utilise la référence S630 inchangée.
