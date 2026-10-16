# S635 — trois fenêtres depuis le microphone, sans fichier WAV

Ce prototype prend directement le microphone physique à 44,1 kHz pendant
quelques secondes, sans SoundFont ni fichier audio préalable, et vérifie
successivement les vues `×2`, `×1`, `÷2` de S634.

## Commande Windows PowerShell

Depuis le dossier du dépôt `note` :

```powershell
git fetch origin
git switch codex/s635-three-view-microphone-no-wav
python -m scripts.s635_three_view_mic --mic --seconds 8 --block-size 1024 --out three_views_mic.json
```

Si le PC propose plusieurs entrées audio :

```powershell
python -m scripts.s635_three_view_mic --list-devices
python -m scripts.s635_three_view_mic --mic --device "WASAPI:Nom de votre micro" --seconds 8 --out three_views_mic.json
```

Si `git switch` refuse de changer de branche, vérifier `git status --short`;
ne jamais écraser un travail local non commité.

Le microphone est lu dans une mémoire bornée par `--seconds` (1–120 s),
en blocs PortAudio. Il est fermé AVANT le traitement des trois vues. Aucune
FFT ni inférence ONNX ne tourne dans le callback, et aucun WAV n'est créé.

La console et le JSON rapportent les propositions d'attaques par fenêtre,
les notes MIDI rétablies, le K original et le temps de traitement.
Ces propositions NE SONT PAS des mesures de vérité (pas de F1 sans annotations).

**Ce diagnostic n'est pas encore du LIVE strict** : pour ne pas accroître
les précédents débordements `input_queue_full`, il sépare expressément
la capture physique et l'analyse lourde. La branche principale du
`scripts.live_strict_notes`, ses notes SoundFont, ses checkpoints S614/S80
et les modes opt-in S633 ne sont pas modifiés.

Validation GitHub simulée (Windows/Linux, aucun matériel audio réel en CI) :
https://github.com/Andriamarosoa/note/actions/runs/38912731866
