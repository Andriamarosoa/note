# NOTE Live — première intégration expérimentale S610

Cette version transforme un flux microphone en notes nommées (`E3`, `G3`, etc.) et peut envoyer les événements NoteOn/NoteOff vers un port MIDI. Elle fonctionne séparément des anciens détecteurs Live et des modèles de recherche de NOTE.

Le moteur utilisé ici est **le modèle causal polyphonique v2.2.1 de `midi`**, copié avec ses poids et ses seuils figés au commit `536536f54290fdc4f63db28aa063d2e324121642`. Il couvre **E2 à E5, MIDI 40–76, jusqu'à six notes simultanées**. La provenance et les empreintes sont dans [PROVENANCE.json](src/causal_note/live_reference/PROVENANCE.json).

**Les scores de S601 sur GuitarSet, IDMT et MAESTRO restent des résultats hors ligne. Ils ne mesurent pas cette version Live.** S601 emploie des informations futures et n'a pas été branché sur ce flux. S80/Exact-K et le Live historique conservent leurs fichiers et leurs modèles.

## Démarrer sous Windows

Depuis la racine de cette branche du dépôt, avec Python 3.11 :

```powershell
py -3.11 -m venv .venv-live
.venv-live\Scripts\python -m pip install -r requirements-live.txt
.venv-live\Scripts\python scripts\live_notes.py --list-devices
.venv-live\Scripts\python scripts\live_notes.py --microphone --report logs\live_session.json
```

Le lanceur `START_NOTE_LIVE.bat` utilise ce même environnement. Aucun entraînement ni téléchargement de poids supplémentaire n'est nécessaire : le petit modèle TFLite est inclus dans `model/live/midi_v221/`.

Rester silencieux pendant la calibration, entre une et trois secondes, puis jouer après le message **« Calibration terminée. Jouez. »**. Le terminal affiche les naissances, les fins et les notes actives. `Ctrl+C` ferme les notes et le port MIDI.

Pour choisir un microphone, ajouter `--device 2` avec l'index retourné par `--list-devices`. L'entrée doit accepter **44 100 Hz, mono, blocs de 256 échantillons**. Cette première intégration ne convertit pas automatiquement un périphérique exclusivement 48 kHz. Si le niveau est trop faible, essayer un gain manuel explicite, par exemple `--gain 2` ; le rapport compte l'écrêtage. L'amplification automatique reste désactivée, comme dans la référence `midi`.

Sous Linux/macOS, créer l'environnement avec `python3.11 -m venv .venv-live`, puis utiliser `.venv-live/bin/python` dans les mêmes commandes. Sur Linux, le système doit fournir la bibliothèque PortAudio pour la capture ; une absence est signalée au démarrage. Le rejeu WAV ne demande aucun périphérique audio.

## Envoyer les notes vers un instrument MIDI

```powershell
.venv-live\Scripts\python scripts\live_notes.py --list-midi-ports
.venv-live\Scripts\python scripts\live_notes.py --microphone --midi-port "NOM EXACT DU PORT" --report logs\live_midi.json
```

Choisir un port déjà disponible et le connecter à l'instrument ou au logiciel souhaité. `--channel 1` est la valeur par défaut. L'application produit des événements MIDI ; le son de l'instrument dépend de la destination choisie.

Pour obtenir un flux exploitable par une autre application :

```powershell
.venv-live\Scripts\python scripts\live_notes.py --microphone --format jsonl --events logs\live_events.jsonl --report logs\live_session.json
```

Les événements indiquent la hauteur, le nom, la vélocité, un identifiant de note stable et le motif (`model_onset`, `retrigger`, `release`, etc.). Leur temps est **l'instant de décision à la fin du bloc reçu**, sans antidatation. Ce flux n'invente pas de valeur `s80_K` et n'utilise pas le protocole S424 comme s'il provenait de S80.

## Rejouer et diagnostiquer

```powershell
.venv-live\Scripts\python scripts\live_notes.py --wav guitare.wav --realtime --events logs\replay_events.jsonl --report logs\replay.json
```

Le WAV doit être PCM16, 44,1 kHz, mono ou stéréo. La stéréo est moyennée. Sans `--realtime`, le même moteur traite les blocs aussi vite que possible. Le rejeu calibre le bruit sur un silence synthétique ; une séance microphone calibre sur le bruit effectivement capturé. Les deux calibrations ne sont pas équivalentes dans une pièce bruyante.

Pour exercer la récupération après une pause de calcul de 200 ms :

```powershell
.venv-live\Scripts\python scripts\live_notes.py --wav guitare.wav --realtime --inject-stall-ms 200 --inject-stall-at-s 2 --report logs\overload.json
```

Un petit retard peut supprimer au plus deux inférences sur 32 blocs tout en conservant les échantillons et la date de l'attaque. Au-delà de huit blocs en attente, les blocs anciens sont abandonnés, les notes ouvertes sont fermées et le contexte audio repart proprement. La file de capture est limitée à 16 blocs. Une coupure n'est donc pas assemblée artificiellement en un signal continu. Une surcharge peut faire perdre des notes : elle est comptée dans le rapport.

Le callback de capture n'exécute ni réseau ni sortie MIDI. L'audio passé est limité à 4 096 échantillons, **sans audio futur** ; cette fenêtre de 92,9 ms décrit le contexte, pas une attente obligatoire de 92,9 ms. Le premier calcul peut commencer avec 512 échantillons après calibration. Un bloc dure 5,80 ms, ce qui ne constitue pas une mesure de latence totale.

## Lire les mesures

Le rapport distingue le réseau, le traitement avec/sans sortie, l'attente depuis le callback, les blocs perdus, les inférences sautées et les notes encore ouvertes. Comptes, moyennes et maxima portent sur toute la séance ; les percentiles portent sur les **4 096 dernières observations** pour garder une mémoire bornée. Une erreur microphone, réseau ou sortie MIDI termine la séance en tentant de fermer chaque note puis en envoyant sustain off, all sound off et all notes off sur le canal choisi.

Les mesures de rejeu excluent la conversion analogique du microphone, le délai entre attaque acoustique et détection, ainsi que le synthétiseur et sa sortie audio. **La latence physique et la qualité musicale en microphone restent à mesurer sur le poste de jeu.** Les tests logiciels ne remplacent pas cet essai.

Les résultats d'intégration, les premiers essais conservés et les limites sont dans [le rapport S610](analysis/README_V273_S610_LIVE.md). Pour reproduire les contrats :

```text
python -m unittest test.test_s610_live test.test_live_k_midi_bridge test.test_detect_live test.test_pipeline -q
python scripts/s610_smoke_live.py --output logs/live_smoke.json
```

Ces commandes nécessitent que `PYTHONPATH` inclue la racine du dépôt et `src` (`.;src` sous Windows, `.:src` sous Linux/macOS). Les commandes de lancement `scripts/live_notes.py` et `scripts/s610_smoke_live.py` ajoutent elles-mêmes les chemins nécessaires.
