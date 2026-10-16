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
