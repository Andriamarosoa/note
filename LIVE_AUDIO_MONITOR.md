# NOTE LIVE — sortie audio de monitoring (prototype isolé)

Cette branche ajoute au chemin `scripts.live_strict_notes` un monitoring optionnel **du signal brut du microphone**, distinct de la synthèse de notes MIDI. La reconnaissance, les têtes, les sélections et les poids NOTE sont inchangés.

Sous Windows PowerShell, depuis la racine du dépôt et après installation des dépendances décrites dans LIVE.md :

```powershell
$env:PYTHONPATH = ".;src"
python -m scripts.live_strict_notes --list-devices
python -m scripts.live_strict_notes --mic --monitor-input --monitor-gain 0.3
```

Pour cibler une sortie WASAPI précise, relever l'index du périphérique WASAPI affiché par `--list-devices`, puis ajouter `--output-device INDEX`. L'entrée se choisit séparément par `--device INDEX`.

**Attention : utiliser un casque.** La boucle microphone → enceintes peut créer du Larsen. Le monitoring est désactivé par défaut. Une file bornée indépendante évite que la sortie audio ne bloque la reconnaissance : lorsqu'elle est vide, le programme émet du silence. Le son entendu est l'entrée microphone, **pas** une note resynthétisée.

Mise à jour : la sélection automatique du dépôt `midi` est maintenant portée dans `scripts/live_audio_io.py` : test des périphériques, préférence pour la plus faible latence négociée stable, entrée sans WDM-KS automatique, sortie avec WDM-KS/WASAPI. L'usage du périphérique explicitement sélectionné reste prioritaire. Cette adaptation n'a pas encore été testée sur le matériel Windows de l'utilisateur.

Validation à faire : tests matériels entrée/sortie, arrêt propre, niveaux et clipping, choix WASAPI vs MME, latence entrée→sortie (médiane/p95), overflows sous charge, vérification d'identité des prédictions avec/sans monitoring.

## Sortie piano (nouveau)

La demande principale est d'entendre **les notes prédites jouées au piano**, pas l'audio brut du microphone. Une sortie `--soundfont` optionnelle synthétise les événements NOTE `note_on` et `note_off` en stéréo via FluidSynth et PortAudio ; le programme sélectionné est General MIDI 0 (Acoustic Grand Piano). Les prédictions du moteur NOTE sont inchangées.

Dans un venv Windows PowerShell, installer les dépendances Python :

```powershell
python -m pip install pyfluidsynth==1.3.6
$env:PYTHONPATH = ".;src"
python -m scripts.live_strict_notes --list-devices
python -m scripts.live_strict_notes --mic --soundfont "C:\\SoundFonts\\piano.sf2"
```

Le paquet Python `pyfluidsynth` dépend également de la bibliothèque native FluidSynth installée sur Windows et disponible dans PATH. La SoundFont indiquée doit réellement exister ; il ne s'agit pas d'un fichier livré par NOTE. On peut cibler une sortie WASAPI, si disponible sur le PC, en ajoutant `--piano-output-device INDEX` où INDEX provient de `--list-devices`. Sans cet argument, NOTE teste automatiquement WDM-KS, WASAPI et le périphérique système et choisit le candidat stable ayant la latence négociée la plus faible.

`--soundfont` et `--monitor-input` s'excluent volontairement afin d'éviter le mélange involontaire entre piano synthétisé et microphone. La sortie piano est facultative ; la détection reste utilisable en JSONL et en MIDI. **Aucune validation sonore réelle sous Windows ni mesure de latence microphone→piano n'a encore été effectuée.**

## Mode recommandé : piano automatiquement routé

Le venv de NOTE doit être actif dans PowerShell. Depuis la racine du dépôt :

```powershell
git fetch origin
git switch codex/note-live-audio-monitor-wasapi
git pull --ff-only
python -m pip install -r requirements-live-strict.txt -r requirements-live-devices.txt
$env:PYTHONPATH = ".;src"
python -m unittest -v test.test_live_audio_selection
python -m scripts.live_strict_notes --list-devices
python -m scripts.live_strict_notes --mic --soundfont "C:\Users\user\Documents\Codex\guitarRig\.venv309\Lib\site-packages\pretty_midi\TimGM6mb.sf2"
```

Si `python -m pip install` n'a installé que le package Python FluidSynth, la bibliothèque Windows native FluidSynth doit également être installée et accessible dans PATH, sinon le lancement renverra une erreur explicite.

À l'ouverture, le programme affiche maintenant sur stderr la sélection automatique des entrées/sorties, le nom de l'API et les latences négociées, en conservant la sortie JSONL sur stdout. En cas de panne de synthèse répétée ou d'arrêt du flux de sortie, le LIVE remonte une erreur plutôt que de rester silencieux.

Cette sortie piano est une **synthèse des notes prédites** par le moteur NOTE, non un monitoring de la guitare d'origine, non une modification des poids ou des têtes, et non un nouveau score de F1. Le port MIDI externe reste optionnel et s'active séparément avec `--midi-port`. Les tests unitaires du sélecteur sont fournis dans `test/test_live_audio_selection.py`, mais le test matériel Windows et la mesure de latence acoustique restent à effectuer.

## Diagnostic ciblé iRig : cordes graves absentes

Comparer le même signal *avant tout gain et tout modèle*. Le script `scripts/diagnose_low_strings.py` enregistre le premier canal du périphérique effectivement ouvert par le sélecteur audio NOTE, sans le normaliser, et affiche le nom de l'entrée, RMS dBFS, crêtes, clipping, énergie à la fondamentale et aux harmoniques et les pics spectraux principaux. C'est un diagnostic de capture et non une validation de note reconnue.

Depuis le venv actif sous PowerShell :

```powershell
$env:PYTHONPATH = ".;src"
python -m unittest -v test.test_low_string_diagnostics
python -m scripts.live_strict_notes --list-devices
python -m scripts.diagnose_low_strings --mic --note E2 --duration-s 4 --record-wav "irig_E2.wav"
python -m scripts.diagnose_low_strings --wav "irig_E2.wav" --note E2
python -m scripts.live_strict_notes --wav "irig_E2.wav" --out "irig_E2_notes.jsonl"
```

Au début du diagnostic, jouer uniquement la corde Mi grave E2, puis enregistrer la corde La A2 en remplaçant E2 par A2 et le nom de fichier. Si l'entrée choisie n'est pas l'iRig, passer `--device INDEX` d'après la liste; la sélection automatique suit normalement l'entrée Windows par défaut. D2/Drop D (MIDI38) est en-dessous de la limite MIDI40 du modèle NOTE strict actuel.

Pour tester **seulement le piano**, sans guitare ni modèle, avec une SoundFont existante :

```powershell
python -m scripts.diagnose_low_strings --test-piano --note E2 --soundfont "C:\Users\user\Documents\Codex\guitarRig\.venv309\Lib\site-packages\pretty_midi\TimGM6mb.sf2"
```

Interprétation : si E2 ne s'entend pas en test piano direct, examiner d'abord la sortie audio, le périphérique d'écoute ou la SoundFont. Si le piano joue E2 mais que son énergie est très faible ou absente dans le WAV brut, examiner capture iRig, gain et traitements Windows. Si le WAV a une harmonique grave visible mais aucune note E2 dans `irig_E2_notes.jsonl`, examiner le modèle, les caractéristiques et les seuils de sélection de NOTE.

Pour la comparaison avec `midi`, réutiliser **exactement ce WAV brut** dans `guitar-midi-poly-transcribe` avec le bundle polyphonique adéquat, une fois avec `--no-auto-level` puis avec `--auto-level` ; ne pas comparer deux interprétations différentes de la guitare. Les différences de note ou de gain ne prouvent rien sans inspecter les rapports.

Les tests de cette branche sont à exécuter dans le venv ; aucun test d'iRig physique ou mesure réelle par corde n'a encore été effectué. Ne pas changer le modèle ou amplifier chaque note à l'aveugle.

## Gain automatique causal adapté de `midi` (expérimental, opt-in)

Le fichier `scripts/live_input_level.py` reprend le contrôleur `CausalModelInputLeveler` du dépôt `Andriamarosoa/midi/src/polyphonic/input_level.py`, à gain borné +18 dB et gestion de la décroissance des notes. `note` strict ne dispose **pas** de la calibration d'activité du LIVE `midi` : un adaptateur spécifique exige les seuils bruts peak -55 dBFS et RMS -65 dBFS. Ce n'est donc pas une copie complète du pipeline MIDI. Sans `--auto-level`, les données transmises au modèle restent inchangées.

Exemple dans le venv actif, à partir du dépôt NOTE :

```powershell
git fetch origin
git switch codex/note-live-audio-monitor-wasapi
git pull --ff-only
$env:PYTHONPATH = ".;src"
python -m unittest -v test.test_live_input_level
python -m scripts.live_strict_notes --mic --device "WASAPI:External Microphone" --auto-level --soundfont "C:\Users\user\Documents\Codex\guitarRig\.venv309\Lib\site-packages\pretty_midi\TimGM6mb.sf2"
```

Comparer **le même** WAV sans et avec auto-level (noms de sortie distincts) :

```powershell
python -m scripts.live_strict_notes --wav "irig_E2_wasapi.wav" --out "e2_base.jsonl"
python -m scripts.live_strict_notes --wav "irig_E2_wasapi.wav" --auto-level --out "e2_autogain.jsonl"
Get-Content e2_base.jsonl | Select-String '"type": "note_on"'
Get-Content e2_autogain.jsonl | Select-String '"type": "note_on"'
```

Réglages : `--auto-level-max-db 18` (0–24), `--auto-level-peak-gate-dbfs -55` et `--auto-level-rms-gate-dbfs -65`. Ne pas monter les seuils/gain sans examiner les notes fantômes. Le contrôleur n'applique aucune normalisation individuelle aux queues de notes. Les diagnostics du contrôleur sont imprimés sur stderr lors de l'arrêt propre.

**Limitation majeure pour cet iRig** : le signal E2 enregistré à RMS ≈ -116.9 dBFS est quasi nul, et l'export PCM16 peut quantifier les faibles composantes à zéro. Un maximum +18 dB ne reconstitue pas une fondamentale manquante ; le gain vise plutôt les signaux faibles mais encore exploitables. Aucune amélioration F1 ou résultat matériel Windows n'est établi. Valider les faux positifs et régressions sur WAV tenus à l'écart avant de promouvoir cette option.

## Audit d'une corde grave perçue comme accord (E2 -> faux harmoniques)

Sur une prise **MME**, E2 enregistrée à peak -6.5 dBFS / RMS -23.4 dBFS est bien présente spectralement. Le LIVE sans gain a émis MIDI 52 (E3), 59 (B3), 64 (E4) et 68 (G#4), coïncidant respectivement avec les harmoniques 2, 3, 4 et 5 de E2, plus MIDI 61 (C#4). Il n'a pas émis E2 MIDI 40. Ce cas prouve une confusion sur une prise guitaristique réelle, mais ne suffit pas pour une statistique F1.

Un défaut observable : la configuration native comporte `count_confidence=1.1`. La sortie de la tête K est une softmax (valeurs entre 0 et 1), donc la sélection actuelle n'utilise **jamais** le budget K pour limiter les notes. Cette valeur peut être issue d'une ablation sélectionnée par validation ; elle ne doit pas être changée sur les jeux d'évaluation sans audit de régressions.

```powershell
git pull --ff-only
python -m unittest -v test.test_live_count_veto
python -m scripts.audit_live_pitch_count --wav "irig_E2_wasapi.wav" --expected-pitch 40 --report "audit_e2.json"
$a = Get-Content "audit_e2.json" -Raw | ConvertFrom-Json
$a.note_on_by_selection | ConvertTo-Json -Depth 5
$a.first_original_attack_frame | Select-Object time_s,predicted_k,k_confidence,k_probabilities_0_through_6,expected | Format-List
$a.first_original_attack_frame.pitch_ranking | Select-Object -First 8 pitch,name,onset,sustain
```

Trois rejouages causaux sont effectués **sur les mêmes sorties des mêmes têtes** :
- `original`: configuration gelée, sans veto K,
- `count_gate_050`: veto fondé sur la vraie distribution K, si sa probabilité maximale >= .5,
- `mono_budget_1`: borne artificielle à 1 note, simple témoin de diagnostic incompatible avec la polyphonie.

Ne jamais promouvoir directement le mono budget ou le count_gate_050 à partir de la seule corde E2. Il faut également vérifier les vrais accords K2–K6, les régressions, et le comportement LIVE strict. Le code d'audit n'ajoute aucun délai ni aucune fonction à l'architecture de production.
