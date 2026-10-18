# S712 — essai microphone LIVE de la vraie sélection S708 (non promue)

## Statut et limites

S708 est un candidat **de recherche**, distinct de la référence S613/S630 LIVE.
Sur GuitarSet DEV, sa chaîne complète **avec l'Offset S707 de recherche**
obtient Onset F1 68,752%, Exact-K décodé 58,973%, F1 notes complètes
50,310%. Ce ne sont **pas** des scores du microphone physique.

S712 permet de brancher le véritable `s708_selected.pt` (S706 epoch10,
politique S708 figée sur CAL) au micro de `scripts.live_strict_notes` dans
un mode opt-in. La tête acoustique conserve les 126 caractéristiques NOTE,
les 39 caractéristiques Onset causales, 27 éléments d'historique, et les
96 caractéristiques spectrales S706 (288 au total). PyTorch CPU infère
l'Onset, le maintien, la libération et K par bloc de 4×256 échantillons,
puis la sélection native S708 produit les événements. Aucune trame future
ou annotation n'est lue lors de l'inférence, aucune émission n'est
antidatée, aucune référence n'est remplacée.

**Limite déterminante :** l'Offset livré par cet essai microphone est le
décodeur joint de S708. Le réglage de hazard S707 utilisé pour évaluer
les 73,73% d'Offset du benchmark S708 n'a pas encore été branché comme
gestionnaire d'Offset dans cette variante LIVE. Le chiffre 50,31% de F1
notes complètes ne peut donc pas être transféré au test microphone.
Les performances CPU, latences de pilote, fausses attaques et notes graves
doivent être mesurées sur Windows.

## Installation — Windows PowerShell

Depuis le dossier existant `note` :

```powershell
git fetch origin codex/note-s712-s708-strict-live-trial
git switch codex/note-s712-s708-strict-live-trial
.\.venv-live\Scripts\Activate.ps1
$env:PYTHONPATH = ".;src"
python -m pip install -r requirements-live-strict.txt -r requirements-live-devices.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Le checkpoint n'est **pas** inventé ni reconstruit : il faut télécharger
l'artefact exact du workflow GitHub
https://github.com/Andriamarosoa/note/actions/runs/39094682419 ,
nom **S708_K_GUIDED_S706_JOINT_CAUSAL_LOOPS**, en conservant
`s708_selected.pt` et `report.json` dans le même dossier.
Avec la CLI GitHub authentifiée :

```powershell
New-Item -ItemType Directory -Force .\s708_model | Out-Null
gh run download 39094682419 --repo Andriamarosoa/note --name S708_K_GUIDED_S706_JOINT_CAUSAL_LOOPS --dir .\s708_model
Test-Path .\s708_model\s708_selected.pt
Test-Path .\s708_model\report.json
```

La chaîne valide le SHA256 du vrai checkpoint par rapport au rapport
immuable avant de produire la première note. Le téléchargement peut aussi
se faire à la main depuis la page Actions, puis extraction au même dossier.

## Essai d'abord sur un WAV

Le WAV doit être mono, PCM16, 44,1 kHz. Le fichier JSONL de sortie
doit être nouveau, sinon l'outil refuse de l'écraser.

```powershell
python -m scripts.live_strict_notes --wav .\test_guitare.wav --block-size 256 --hop-size 256 --s708-model .\s708_model\s708_selected.pt --out .\s708_wav.jsonl
```

## Essai micro / iRig

Commencer par trouver le numéro du micro `External Microphone (Realtek Audio)` :

```powershell
python -m scripts.live_strict_notes --list-devices
```

Exemple si le périphérique d'entrée actuel est bien `15` :

```powershell
python -m scripts.live_strict_notes --mic --device 15 --block-size 256 --hop-size 256 --s708-model .\s708_model\s708_selected.pt --soundfont ".\SalamanderGrandPiano-V3+20200602.sf2" --out .\s708_micro.jsonl
```

Sans SoundFont, retirer `--soundfont ...`; les NoteOn/NoteOff restent
enregistrés en JSONL. Éviter dans ce premier essai `--auto-level`,
`--s675-hazard-model`, `--inference-batch-hops 4`, lookahead et dual
hop, pour isoler exactement la tête S708. Le CLI rejette ces combinaisons.

Pour évaluer les faux positifs, jouer dans cet ordre : six cordes
à vide séparément, accord E2+B2+E3, répétition rapide d'une même note,
accord tenu jusqu'au relâchement. Conserver les événements JSONL,
le diagnostic de file micro et le CPU. Comparer au LIVE natif à
microphone/WAV comparable, sans attribuer 68,75% au microphone.

## Latence et sûreté

Le pas de décision S708 est 1 024/44 100 ≈ **23,22 ms**,
indépendamment du temps de capture, de la durée des fenêtres FFT,
du temps CPU et du synthétiseur. Avec un callback de 256 échantillons,
une attaque peut attendre plusieurs blocs reçus avant la prochaine
décision. Aucun résultat de latence matérielle n'est revendiqué.

L'essai s'arrête plutôt que retarder silencieusement l'audio si la file
est débordée ; il libère les notes à l'arrêt ou à l'erreur.
Aucune écriture de référence, migration S80, promotion ou merge automatique.

## Vérifications CI

Les tests `test.test_s712_s708_live` couvrent :

- parité des **288 caractéristiques** entre la version offline et le vrai
  flux construit en pas micro de 256 échantillons ;
- invariance des événements aux tailles de callback (128/256/333/2048) ;
- invariance des émissions passées quand seul l'audio futur est modifié ;
- rejet des gaps et échantillons non finis ;
- chargement SHA verrouillé du vrai artifact S708, CLI et WAV smoke.

Workflow : https://github.com/Andriamarosoa/note/actions/runs/39103985126

La réussite des tests CI ne garantit **pas** la cadence temps réel ni
l'absence de glitches sur Windows : le premier test physique doit mesurer
la file d'entrée, le temps CPU et l'audibilité.
