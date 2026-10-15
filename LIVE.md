# NOTE Live — portage de notre architecture, S611

**Le moteur de reconnaissance importé de `midi` dans S610 est retiré.** La direction demandée est d'améliorer les têtes et les sélections de NOTE. S611 porte des composants natifs vers le flux continu ; il ne livre pas encore un microphone → S80 → notes complet.

## Composants portés

| Composant NOTE | Travail S611 | Limite actuelle |
|---|---|---|
| Extraction S97 | Même FFT, dictionnaire harmonique, 58 caractéristiques et trajectoire 42×49, historique borné | Attend 160 ms d'audio après l'événement |
| Entrées du comptage | Flow 245D, morphologie 490D, physique 294D, géométries 3D et PR16 46D reproduites | La géométrie 46D attend la certification des événements jusqu'à +240 ms |
| S402, S416, S428 | Douze modèles entraînés d'origine chargés une fois | Chemin S430, aucune promotion comme meilleur modèle Live |
| Sélections S419 puis S428 | Votes et règles par K identiques à S430, empreintes vérifiées | K doit encore être calculé par S80/S397 en amont |
| Noms de notes | Exactement K nouvelles hauteurs MIDI 40–88 ; K0 reste vide | Pas d'inférence des fins de notes dans cet adaptateur |

Les producteurs V84→V88, les familles de têtes S5/S18/S35/S45/S46, les sélections S69→S80 et la protection S397 sont conservés. Leur ordonnanceur complet doit encore être extrait des scripts de recherche et raccordé au microphone. Les têtes S569/S578 et S601 restent des références/challengers à adapter et comparer ; elles ne sont pas remplacées ou promues par S611.

Le Live historique et `LockedKLiveMidi` restent accessibles via `scripts.note_runtime`. K0 ne ferme pas les notes déjà actives. Les vrais NoteOff demandent le raccordement et la validation des têtes de fin de note.

## Installation

Depuis la racine du dépôt, avec Python 3.11 ou 3.12 :

```powershell
py -3.11 -m venv .venv-live
.venv-live\Scripts\python -m pip install -r requirements-live.txt
.venv-live\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
$env:PYTHONPATH = ".;src"
.venv-live\Scripts\python -m scripts.live_native_notes --status
```

Sous Linux/macOS : `python3 -m venv .venv-live`, `.venv-live/bin/python` et `export PYTHONPATH=.:src`. Les douze checkpoints natifs et les deux rapports de sélection sont inclus dans `model/live/native_note`, copiés sans modification et vérifiés par SHA256.

## Rejeu avec les décisions originales de S80

```text
python -m scripts.live_native_notes --wav guitare.wav --decisions original_S80_K_native_join.jsonl --out logs/native_replay
```

Le WAV doit être mono PCM16 à 44,1 kHz. Le JSONL est celui du raccordement natif S433 : `row_id`, `start_sample`, `end_sample`, `s80_K`. Il faut des prédictions S80 authentiques ; des annotations ou K inventés ne constituent pas ce frontal.

Le rejeu utilise des blocs de 256 échantillons par défaut. `notes.jsonl` conserve l'identité et l'attaque acoustique de l'événement, le K d'entrée, la tête sélectionnée et l'instant de disponibilité de son contexte audio. **Cet instant ne mesure ni le calcul de S80 en amont, ni la latence physique microphone/MIDI.** Un bloc de 5,80 ms n'efface pas l'attente de 160 ms.

L'adaptateur refuse trous audio, groupes qui se chevauchent, débordement de file, événement trop tardif pour l'historique conservé et fin de WAV sans contexte suffisant. Il ne remplit pas le futur par du silence et ne change pas de moteur en cas d'erreur. Lors du futur raccordement microphone, l'inférence doit tourner hors du callback audio.

## Vérification

```text
python -m unittest -v test.test_native_stream test.test_live_k_midi_bridge test.test_note_runtime_modes test.test_v273_harmonic_trajectory
python -m scripts.s611_fetch_native_fixture --out data/S611/original_excerpt.wav
python -m scripts.s611_validate_native_stream --wav data/S611/original_excerpt.wav --decisions analysis/evidence/v273-s611-native-stream/original_S434_S80_decisions.jsonl --legacy-notes analysis/evidence/v273-s611-native-stream/original_S434_notes.jsonl --native-events analysis/evidence/v273-s611-native-stream/native_events_full_recording.jsonl --out logs/native_validation
```

Sur les 10 secondes de GuitarSet MIC déjà utilisées par S434 : **30 événements et 22 notes prédites** identiques au S430 d'origine et à l'archive S434, en blocs de 256 et 997 échantillons. Les caractéristiques acoustiques et géométries de **109 événements natifs complets** sont identiques bit à bit au calcul d'origine.

C'est un contrôle du portage : extrait connu, K précalculé sur le mix pickup, audio MIC différent, chevauchement avec le développement GuitarSet. **Aucun nouveau score de précision Live, IDMT ou multi-guitares n'est revendiqué.** Les scores hors ligne S601 restent inchangés.

[Rapport S611](analysis/README_V273_S611_NATIVE_STREAM.md) · [Preuves JSON](analysis/evidence/v273-s611-native-stream/report.json). Le rapport S610 est conservé comme historique d'une intégration retirée ; son code reste dans le commit `e850198`.
