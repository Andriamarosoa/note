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

Différences avec `midi` : cette variante ne recopie pas encore la découverte WASAPI/WDM-KS, le benchmark automatique des périphériques, ni la sortie FluidSynth SF2. `sounddevice` passe par le backend correspondant au périphérique choisi. La latence et l'audio doivent être validés sur Windows réel avant promotion.

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

Le paquet Python `pyfluidsynth` dépend également de la bibliothèque native FluidSynth installée sur Windows et disponible dans PATH. La SoundFont indiquée doit réellement exister ; il ne s'agit pas d'un fichier livré par NOTE. On peut cibler une sortie WASAPI, si disponible sur le PC, en ajoutant `--piano-output-device INDEX` où INDEX provient de `--list-devices` ; sinon sounddevice ouvre la sortie par défaut.

`--soundfont` et `--monitor-input` s'excluent volontairement afin d'éviter le mélange involontaire entre piano synthétisé et microphone. La sortie piano est facultative ; la détection reste utilisable en JSONL et en MIDI. **Aucune validation sonore réelle sous Windows ni mesure de latence microphone→piano n'a encore été effectuée.**
