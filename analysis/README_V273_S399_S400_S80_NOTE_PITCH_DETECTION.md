# S399–S400 — Ajout de l'identification de notes à S80

Date : 13 octobre 2026.

## Référence inchangée pour le nombre de notes

Le modèle de comptage actif reste `S80_S397_K5_GUARDED`. Il a été évalué sur GuitarSet avec recoupement d'entraînement : **53 529/59 309 = 90,2544 % Exact-K global** ; **6 054/7 385 = 81,9770 % Exact-K polyphonique**. Ces résultats ne sont pas indépendants.

## Nouvelle tête d'identification de hauteur — S399

Le **nouveau module S399** convertit la trajectoire réelle `sequence42x49` déjà extraite de l'audio d'origine en 49 scores de candidats MIDI **40..88 (E2..E6)**.

- Pré-attaque, attaque et suite immédiate sont extraites des **42 trames originales** avec positions temporelles fixes.
- Score par hauteur : apparition relative après attaque, importance spectrale relative, persistance. Poids **prédéfinis et non appris sur les annotations de test**.
- Les notes sont triées de manière déterministe ; **S80 impose le nombre K de notes émises**. Pour K0, liste vide. Pour K1..K6, exactement K hauteurs candidates.
- Chaque sortie fournit : `row_id`, `s80_K`, `midi_pitch_slots` (6 positions remplies par MIDI 40..88 ou -1), `note_candidate_scores` (49), `chosen_note_scores` (6). Les noms lisibles (C3, E3, G3) découlent du numéro MIDI.
- Une sélection de pics harmoniques, même conditionnée par K, **n'établit PAS** que les notes sont toutes réellement présentes. Un même pitch sur deux cordes ne peut pas être répété par cette première version ; le vocabulaire MIDI 40..88 est limité.

**Aucun poids de S80 ni aucune décision Exact-K n'est modifié.**

Source : `scripts/s399_s80_exactK_constrained_midi_pitch_decoder.py`.

## Évaluation séparée des véritables notes — S400

Source : `scripts/s400_evaluate_S399_MIDI_note_pitches_GuitarSet.py`.

La démarche suit cet ordre : prédire sans annotations → figer et hacher la sortie → télécharger les annotations officielles JAMS `note_midi` → aligner chaque note à un événement native S80 (tolérance originale 20 ms, une attribution maximale par attaque) → comparer **l'ensemble multiséquence des notes MIDI**, pas seulement le nombre K.

Les MIDI fractionnaires du GuitarSet sont quantifiés au demi-ton le plus proche **uniquement pour l'évaluation tempérée** ; l'uniformité de l'accordage n'est pas supposée. Les groupes dont les annotations ne correspondent pas au K de référence, dépassent six notes ou sortent du vocabulaire autorisé sont explicitement comptés comme non évaluables pour l'audit des hauteurs.

Les résultats doivent présenter séparément :
1. Exact-K de S80, inchangé ;
2. **Exact accord** : toutes les hauteurs MIDI exactes, avec doublons de hauteur pris en compte ;
3. Précision, rappel et F1 **des hauteurs**, pour K1–K6 et K2–K6 ;
4. **Contrôle oracle-K** : meilleures K vraies hauteurs selon les mêmes scores, afin de séparer défauts de comptage et défauts d'identification ;
5. Par classe K0–K6, contrôle des annotations non alignées et des notes hors vocabulaire.

L'ancienne reconstruction S80 a des recoupements d'entraînement sur GuitarSet, ce qui interdit de déclarer les résultats futurs indépendants. Le décodeur de notes lui-même est *zéro-shot* ; la validation suivante devra inclure des morceaux réellement nouveaux.

Workflow : [S399–S400](https://github.com/Andriamarosoa/note/actions/workflows/v273-s399-s400-midi-note-identification.yml).

## Règle de promotion

Le module de notes n'est **pas** déclaré champion avant de disposer d'un score de hauteur fiable. Le système de comptage S80 reste la référence active ; ajouter des hauteurs candidates à sa sortie ne garantit pas une vraie transcription MIDI.

## Résultat vérifié — run S399/S400

[GitHub Actions #38354784594](https://github.com/Andriamarosoa/note/actions/runs/38354784594) : **success**. Artefact `11792925560` contient les 59 309 prédictions de notes MIDI, leur score et le rapport de validation.

Les 190 JAMS de GuitarSet ont été retrouvés par identité exacte de piste. Les 59 309 groupes ont des comptes de notes annotées cohérents avec le vrai K original (**0 divergence de cardinalité**). 34 954 notes originales lues ; **1 788 attaques hors de la fenêtre de rattachement de 20 ms**, qui ne doivent pas être comptées par erreur comme de nouvelles notes présentes dans les événements. Le ZIP JAMS du miroir a été vérifié contre le MD5 officiel `b39b78e63d3446f2e54ddb7a54df9b10`, les résultats de calcul étant scellés avant l'ouverture des annotations.

| Vrai K | Groupes | Exact-K S80 corrigé | Toutes hauteurs exactes S399 |
|---|---:|---:|---:|
| K0 | 39 652 | 96,92 % | 96,92 % |
| K1 | 12 272 | 73,69 % | **62,04 %** (7 614) |
| K2 | 3 445 | 87,05 % | **20,17 %** (695) |
| K3 | 2 289 | 82,87 % | **21,06 %** (482) |
| K4 | 1 207 | 66,78 % | **7,95 %** (96) |
| K5 | 355 | 84,79 % | **4,51 %** (16) |
| K6 | 89 | 57,30 % | **3,37 %** (3) |

Mesures agrégées :
- **Exact-K poly (K2..K6)** : 6 054/7 385 = **81,98 %** — inchangé.
- **Exact accord MIDI complet poly** : 1 292/7 385 = **17,49 %**.
- **F1 micro par hauteur poly** : **58,66 %** ; précision **59,85 %**, rappel **57,52 %**.
- **Accord complet avec vrai K injecté seulement à l'évaluation (plafond de décodeur fixé)** : 1 526/7 385 = **20,66 %**.
- **Exact hauteur mono K1** : 7 614/12 272 = **62,04 %**.
- **Accord complet toutes classes K0..K6** : 47 338/59 309 = 79,82 %, **trompeur si cité sans préciser que K0 représente 39 652 événements**.
- **Vrais accords avec doublon de même hauteur sur deux cordes** : 100, que la version unique-pitch S399 ne peut représenter.

## Verdict technique

**Un premier vrai décodage des hauteurs MIDI a été ajouté au système S80 et mesuré sur des annotations réelles**, mais il **ne constitue pas encore un modèle de transcription polyphonique performant**. L'augmentation théorique de 17,49 % à seulement 20,66 % lorsqu'on lui donne le K réel prouve que l'étape de **choix des hauteurs** est désormais le principal défaut de ce module ; corriger davantage Exact-K seul n'y suffira pas.

Le S399 actuel est **non entraîné**, simple classement des 49 activations harmoniques issues des WAV. Les harmoniques confondues, notes proches/octaves et doublons de hauteur ne sont pas résolus. La prochaine architecture à tester devrait conserver la branche S80 de cardinalité figée, puis apprendre une tête **multi-label MIDI**, idéalement **conditionnée par corde/voix** et exploitant le temps réel du signal, en séparant les compositions d'entraînement et de test. Ne pas adapter les poids ou les seuils de S399 sur GuitarSet avant de présenter un nouveau lot indépendant.

**Pas de modification ni promotion du modèle de cardinalité ; S399 ajouté en tant que première tête de notes expérimentale et mesurée.**

## S401 — Interface d'inférence générique pour d'autres audio

[Workflow S401](https://github.com/Andriamarosoa/note/actions/runs/38355190070) : **self-tests réussis**. `scripts/s401_midi_pitch_head_on_any_s80_audio.py` accepte deux fichiers NPZ alignés, même `row_id` : sortie acoustique originale S97 `sequence42x49` et sortie de comptage S80 `s80_K`. Il écrit `notes_midi_predictions.npz`, `notes.csv` et un rapport SHA, avec **exactement K noms de notes** pour chaque événement.

Exemple :

```bash
python -m scripts.s401_midi_pitch_head_on_any_s80_audio \
  --audio model/S97/original_s45_audio_only.npz \
  --s80 model/S80/predictions.npz \
  --out model/MIDI_notes
```

**Important** : S401 ne calcule pas lui-même la sortie `s80_K` ni ne porte automatiquement la protection S70/K5 de S397 vers de nouveaux WAV. La chaîne S80 complète doit produire la décision K correcte en amont.
