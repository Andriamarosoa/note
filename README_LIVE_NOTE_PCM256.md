# Live microphone NOTE Onset PCM256 — Windows

Ce fichier concerne **uniquement** le dépôt `Andriamarosoa/note` et ses têtes Onset. Il ne faut pas utiliser les commandes ni les artefacts de `Andriamarosoa/midi`.

## Statut du modèle

La commande lance un vrai flux microphone, 44100 Hz, mono, un bloc de **256 nouveaux échantillons par décision** et un événement **Onset / NoteOn seulement**. Le flux produit des horodatages à la frontière courante, sans confirmation future, ni attente logicielle ajoutée, ni files d'attaques.

- `--variant recovery-hgb` (défaut) : version de récupération Onset **68,370987 % F1** sur GuitarSet FIT80 historique, utilisant S613, S701, ErrorPrediction, EventConfidence et les cinq modèles FNRecovery révisés, avec un **seul pli** explicitement choisi. Ce chiffre n'est pas une prédiction de la performance de ton microphone.
- `--variant research-original` : référence archivée antérieure, utile comme témoin.
- Le modèle acoustique direct à 69,206204 % et le réentraînement à 69,258433 % ne sont **pas** présents ici comme checkpoints complets de déploiement.
- L'arbitre HGB du run #39957354552 a **régressé** de 69,258433 à 69,248976 %. Il est rejeté et **n'est pas activé**.

Ces modèles sont des plis de **recherche** indépendants, pas un poids unique de production. On ne peut pas annoncer le F1 à partir de tests réalisés sur ce même microphone sans annotations et évaluation.

**Important** : cette commande **n'émet pas de MIDI** et n'a pas de NoteOff ; elle affiche les NoteOn détectés dans PowerShell et peut les écrire en JSONL. Elle ne remplace pas les cinq autres caractéristiques, qui restent inchangées.

## Installation (PowerShell, racine de NOTE)

Pour une archive ZIP GitHub, **retélécharger la branche après ajout de ce fichier**. Dans un clone Git, utiliser `git pull`. Le dossier doit contenir `scripts/live_note_onset256.py` ainsi que les sous-dossiers `model/onset256_research_39915038301`, `model/onset256_recovery_20261023` et `model/live/strict_note`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-onset256-cache.txt
.\.venv\Scripts\python.exe -m pip install sounddevice
$env:PYTHONPATH = (Get-Location).Path + "\src"
```

**Ne pas lancer `pip install -e .`** : NOTE ne possède pas de `pyproject.toml` ou `setup.py`.

## Démarrage

D'abord repérer l'entrée de l'iRig / de la guitare :

```powershell
.\.venv\Scripts\python.exe -m scripts.live_note_onset256 --list-devices
```

Puis sélectionner son **index d'entrée** (exemple `15`, à vérifier sur la machine). La famille de pli `BN1` est un exemple explicitement sélectionné pour la recherche, pas le meilleur modèle garanti pour de la guitare réelle :

```powershell
.\.venv\Scripts\python.exe -m scripts.live_note_onset256 `
  --fold BN1 `
  --variant recovery-hgb `
  --input-device 15 `
  --duration-s 60 `
  --jsonl runs\onset_live_60s.jsonl
```

Sans `--duration-s`, le Live reste actif jusqu'à Ctrl+C. Sans `--input-device`, PortAudio choisit le périphérique Windows par défaut. Choisir une entrée **fonctionnelle**, notamment External Microphone (Realtek Audio), pas un pin WDM-KS qui donne du silence.

Le diagnostic console affiche les événements `ONSET` avec `sample`, `time_s`, `pitch` MIDI, `confidence`, `reason` et un niveau `INPUT_METER` en dBFS environ toutes les secondes. E2 = MIDI 40. Une entrée silencieuse (`peak_1s` sous ~−60 dBFS durant les attaques) est un problème de capture/gain, pas automatiquement un défaut du modèle.

**Attention latence** : 256/44100 = 5,805 ms est l'intervalle entre décisions, **pas** la latence matérielle ni la latence CPU garantie. Le script affiche `worst_compute` ; s'il dépasse 5,805 ms, le système ne peut pas soutenir la cadence temps réel. La latence WASAPI/MME et la latence instrument doivent être mesurées séparément. Aucun bloc ne doit être inventé ni sauté silencieusement : le script s'arrête si PortAudio signale un overflow.

## Diagnostic sans microphone

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Contraintes préservées

- PCM256 exact : 44,1 kHz, 256 échantillons mono, aucun futur audio et aucune attente par sélection ;
- seul l'axe Onset est exécuté, pas d'autre tête ni d'écriture MIDI ;
- aucun checkpoint d'arbitre régressif n'est promu ;
- après une faible progression ou une régression, inspecter les TP/FN/FP événement par événement avant d'émettre une nouvelle hypothèse.
