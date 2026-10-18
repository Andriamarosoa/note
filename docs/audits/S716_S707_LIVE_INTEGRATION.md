# S716/S717 — spécialiste Offset causal signé pour S708 LIVE

**État** : implémentation S716 et tests techniques validés en CI ; validation indépendante des scores sur vraie guitare par S717, sans promotion automatique.

## Ce qui change

L'ancien mode `--s708-model` continue à utiliser l'Offset du sélecteur
S708. S716 ajoute **uniquement sur demande**
`--s708-s707-offset-model <s675_hazard.joblib>`. Il charge le vrai
checkpoint S675 signé via les vérifications existantes
`S672TrialNoteOff.load()` (modèle 75D, provenance, SHA256, modèle S613
original). Il applique la politique de recherche S707 réellement choisie
par CAL dans le rapport S708 : `gate=.80`, `persistence=1`,
`grace=24` hops natifs.

Le moteur `S708StrictLiveStream` conserve ses têtes, ses probabilités,
son Exact-K et son `GuidedSelector` figés. Un gestionnaire
`S716S707OnlineRelease` construit les **mêmes 75 caractéristiques**
qu'en évaluation S707, avec une mémoire bornée de 65 frames natives,
et décide les NoteOff uniquement aux instants réellement reçus.

L'état de notes effectivement émises est **unique** : un NoteOn
doit être apparié à un NoteOff, aucune hauteur ne peut être active en
double, jamais plus de six notes actives. Si un NoteOff natif est
observé, il devient le début d'un délai de grâce causal, et non
un crédit de libération rétrodatée. Les nouvelles attaques réarmées
ferment l'ancienne instance de même hauteur en premier.

**Compromis majeur** : une fin de note maintenue plus longtemps par S707
peut occuper un des six slots et empêcher une nouvelle attaque,
ce qui n'est pas forcément le comportement de l'évaluation S708
hors ligne. Les six scores et la latence doivent donc être
revalidés indépendamment, et peuvent être moins bons.

## Tester sur Windows PowerShell

À la racine du dépôt :

```powershell
git fetch origin codex/note-s713-independent-code-audit-agents
git switch codex/note-s713-independent-code-audit-agents
.\.venv-live\Scripts\Activate.ps1
$env:PYTHONPATH = ".;src"
```

Les fichiers doivent être dans les dossiers :

```text
note/
  s708_model/
    s708_selected.pt
    report.json
  s675_model/
    s675_hazard.joblib
    s675_checkpoint.json
```

Télécharger le S675 original depuis le workflow
https://github.com/Andriamarosoa/note/actions/runs/39059008484
(nom d'artefact `S681_LATEST_S675_TRAINED_NOTEOFF_WINDOWS_TRIAL`)
et extraire les deux fichiers précisés. S708 reste l'artefact du
workflow https://github.com/Andriamarosoa/note/actions/runs/39094682419 .

### Vérifier le checkpoint et la configuration

```powershell
python -m scripts.live_strict_notes --status --s708-model ".\s708_model\s708_selected.pt" --s708-s707-offset-model ".\s675_model\s675_hazard.joblib"
```

### Test WAV

```powershell
python -m scripts.live_strict_notes --wav ".\test_guitare.wav" --hop-size 256 --block-size 256 --s708-model ".\s708_model\s708_selected.pt" --s708-s707-offset-model ".\s675_model\s675_hazard.joblib" --out ".\s716_test_guitare.jsonl"
```

### Test microphone (vérifier --list-devices)

```powershell
python -m scripts.live_strict_notes --mic --device 15 --hop-size 256 --block-size 256 --s708-temporal-engine incremental --s708-model ".\s708_model\s708_selected.pt" --s708-s707-offset-model ".\s675_model\s675_hazard.joblib" --soundfont ".\SalamanderGrandPiano-V3+20200602.sf2" --out ".\s716_micro.jsonl"
```

Sans SoundFont, enlever `--soundfont`. Le checkpoint doit provenir
du workflow épinglé, car `joblib` est un format de sérialisation
qui peut exécuter du code au chargement.

### Comparer avant/après sur les mêmes notes

Faire un contrôle séparé **sans** l'option
`--s708-s707-offset-model`, sur le même WAV et dans un autre JSONL.
Ne pas déclarer d'amélioration de précision à partir des exemples
sans annotation.

## Preuves

- [S716 CI](https://github.com/Andriamarosoa/note/actions/runs/39110131614) : tests de l'ABI acoustique 75D et horloge native,
  signaux causal/réattaque/fallback, vrai S708 + S675 dans CLI,
  cycle valide NoteOn–NoteOff avec au plus six actives.
- [S717 comparateur](https://github.com/Andriamarosoa/note/actions/workflows/note-s717-real-guitarset-s716.yml) : HELD 32 pistes et DEV 12 pistes,
  modèle de tête S708 identique pour les variantes, émission source figée,
  six métriques réellement recomptées. Il refuse la comparaison si
  l'ancien score S708 hors ligne ne se reproduit pas.
- Aucun TEST externe ouvert ni promotion, aucun changement au
  checkpoint natif, aucun F1 iRig mesuré.

## Réserves

L'equalité des 75 entrées au moment de l'évaluation a été testée, mais
la séquence de décisions de l'Offset ne peut pas être déclarée
égale à l'évaluation hors ligne sans S717.
Le 23,22 ms des blocs S708 n'est pas la latence bout en bout ;
l'inférence hazard ajoute du temps CPU à mesurer sur Windows.
