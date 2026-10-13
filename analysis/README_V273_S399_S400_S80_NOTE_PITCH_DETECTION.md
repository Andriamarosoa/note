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
