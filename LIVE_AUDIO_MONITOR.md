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
