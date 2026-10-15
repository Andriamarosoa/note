# NOTE Live — chaîne autonome native S613–S614

Le mode `strict-live` relie le PCM continu aux têtes natives de NOTE, à leur sélection et aux événements NoteOn/NoteOff. Il accepte un microphone, du PCM sur stdin ou un WAV. **Aucun fichier d’événements V88, aucune observation Basic Pitch et aucun K précalculé ne sont demandés.** Les variantes sont entraînées pour ce chemin causal ; ce ne sont pas les anciens poids S578 privés de leur contexte.

**S80/S397 reste intact.** Sa chaîne complète et ses scores Exact-K ne sont pas ceux de cette branche. La nouvelle tête K estime les hauteurs actuellement actives dans MIDI40–88 ; le K historique de S80 compte les nouvelles notes d’un événement. Ces deux quantités ne doivent pas être confondues. Les méthodes natives conservées ici sont les représentations harmoniques, la compétition entre fondamentales, les différences temporelles, les têtes conditionnées par le comptage et les sélections. Aucun moteur ni poids de `midi` n’est importé.

## Utilisation autonome

Python 3.11/3.12, depuis la racine du dépôt :

```bash
python -m pip install -r requirements-live-strict.txt -r requirements-live-devices.txt
export PYTHONPATH=.:src
python -m scripts.note_runtime --mode strict-live --status
python -m scripts.live_strict_notes --list-devices
python -m scripts.live_strict_notes --list-midi-ports
python -m scripts.live_strict_notes --mic --midi-port "Nom exact du port" --out live_events.jsonl
```

Sous PowerShell, remplacer la ligne `export` par `$env:PYTHONPATH = ".;src"`. `--device` sélectionne le microphone par nom ou index. Sans `--midi-port`, les événements sortent en JSONL. PortAudio et le backend MIDI doivent être disponibles sur la machine. Les modèles sont inclus dans `model/live/strict_note` ; l’inférence ne nécessite ni PyTorch ni TensorFlow.

```bash
python -m scripts.live_strict_notes --wav guitare.wav --out notes.jsonl
python -m scripts.live_strict_notes --stdin --out notes.jsonl
```

Le WAV doit être mono PCM16 à 44,1 kHz ; stdin reçoit du PCM16 signé little endian sans en-tête. Le fichier de sortie doit être nouveau. Les messages indiquent le pitch MIDI, le type, la cause et les instants de décision et de disponibilité de l’audio. **Ces instants ne sont jamais antidatés à une attaque reconstruite.** Ils ne comprennent pas la latence physique du pilote ou du synthétiseur.

## Contrat causal et temps réel

- Pas de 256 échantillons, soit 5,805 ms ; fenêtres FFT de 2048, 4096 et 8192 échantillons terminant à l’instant disponible. Le complément au démarrage est uniquement à gauche.
- Aucun centrage qui lise le futur, aucune normalisation sur le morceau entier, aucune attente fixe de 110 ms après une attaque. L’information acoustique doit néanmoins devenir assez fiable pour reconnaître la note : **zéro lookahead ne signifie pas zéro latence de reconnaissance**.
- Trois états appris par hauteur : attaque, maintien et relâchement. La sélection suit les notes actives, traite les réattaques et ferme les notes à l’arrêt. Les NoteOff de fermeture ne sont pas comptés comme des fins acoustiques correctes.
- L’inférence est hors du callback. La file microphone contient au maximum deux blocs ; un débordement ou une décision trop tardive arrête le flux et libère les notes, au lieu d’accumuler du retard. Ce programme Python ne garantit pas les délais d’un système d’exploitation temps réel.
- Mémoire temporelle bornée : 8192 échantillons audio, 32 trames de caractéristiques et 49 états de hauteur, avec au plus six notes actives.

La validation utilise le vrai PCM alimenté bloc par bloc, les mutations du futur, plusieurs tailles de blocs et la conservation des sources historiques. Le raccordement matériel microphone/MIDI n’a pas pu être essayé dans l’environnement de calcul : PortAudio et le séquenceur ALSA n’y sont pas disponibles. Les résultats de calcul ne sont donc pas une mesure de latence acoustique bout en bout.

[Rapport, boucles et test réservé S613–S614](analysis/README_V273_S613_S614_STRICT_LIVE.md).

---

# Historique : adaptateurs natifs S612

**Le chemin récent S511/S515 → S569 → S578/S585 est maintenant porté par blocs**, avec les sélections de NOTE et leurs poids d'origine. S80/S397, ses familles de têtes et ses sélections restent conservés. `recent-notes` s'ajoute au chemin Exact-K strict `native-notes` de S611 et au Live historique.

```text
python -m scripts.note_runtime --mode recent-notes --status
python -m scripts.s612_restore_artifacts --out data/S612 --models-only
python -m scripts.live_recent_notes --wav source.wav --events original_V88_source.jsonl --bp original_BP_notes.jsonl --models data/S612 --normalization rolling3s --out logs/recent_notes
```

La restauration nécessite `gh` ; les dépendances Python sont celles ci-dessous. Ce rejeu attend encore les **événements V88 et les observations BP originaux en entrée**. Ce n'est pas une capture microphone autonome. Le mode `reference` reproduit la normalisation sur le morceau entier ; `rolling3s` utilise uniquement l'historique disponible et reste expérimental. Les fins de notes ne sont pas inférées par cet adaptateur.

Vérification : **1 698 événements, sept enregistrements, aucune décision de note changée** dans les politiques figées ; audio réel URMP de cinq ensembles, 120 s, blocs 256/997 ; **36 tests réussis**. Avec les producteurs précalculés, S585 strict régresse de **64,78 % à 64,25 %** lorsque la normalisation utilise uniquement trois secondes passées. Aucun score Live à 80 % n'est annoncé.

Le contexte audio demande environ **108–123 ms** en régime établi, puis s'ajoutent les coûts des producteurs et du calcul. [Rapport S612, architectures, commandes de validation et mesures détaillées](analysis/README_V273_S612_RECENT_NATIVE_STREAM.md).

## Chemin Exact-K natif conservé — S611

**Le moteur de reconnaissance importé de `midi` dans S610 est retiré.** La direction demandée est d'améliorer les têtes et les sélections de NOTE. S611 porte des composants natifs vers le flux continu ; il ne livre pas encore un microphone → S80 → notes complet.

## Composants portés

| Composant NOTE | Travail S611 | Limite actuelle |
|---|---|---|
| Extraction S97 | Même FFT, dictionnaire harmonique, 58 caractéristiques et trajectoire 42×49, historique borné | Attend 160 ms d'audio après l'événement |
| Entrées du comptage | Flow 245D, morphologie 490D, physique 294D, géométries 3D et PR16 46D reproduites | La géométrie 46D attend la certification des événements jusqu'à +240 ms |
| S402, S416, S428 | Douze modèles entraînés d'origine chargés une fois | Chemin S430, aucune promotion comme meilleur modèle Live |
| Sélections S419 puis S428 | Votes et règles par K identiques à S430, empreintes vérifiées | K doit encore être calculé par S80/S397 en amont |
| Noms de notes | Exactement K nouvelles hauteurs MIDI 40–88 ; K0 reste vide | Pas d'inférence des fins de notes dans cet adaptateur |

Les producteurs V84→V88, les familles de têtes S5/S18/S35/S45/S46, les sélections S69→S80 et la protection S397 sont conservés. Leur ordonnanceur complet doit encore être extrait des scripts de recherche et raccordé au microphone. S612 porte désormais les têtes S569/S578/S585 dans l'adaptateur récent décrit ci-dessus ; S601 reste une référence hors ligne à adapter séparément.

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
