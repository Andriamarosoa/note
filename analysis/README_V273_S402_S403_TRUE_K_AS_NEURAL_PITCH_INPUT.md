# S402–S403 — Le K de S80 est une **entrée** du réseau de notes MIDI

**13 octobre 2026.** Nouvelle référence de développement des hauteurs : `S402_K_CONDITIONED_352D_MLP`. La référence de comptage `S80_S397_K5_GUARDED` ne change pas.

## Modification de conception demandée

Avant, S399 effectuait `signal -> 49 scores identiques quel que soit K -> top-K`. Le K n'influençait **pas** les probabilités musicales.

Après, S402 effectue :

```
Signal: 42 × 49 activations MIDI réelles
     + K_PRED=S80 (7D one-hot + deux scalaires K et K²)
     + interactions directes K × activations de notes
               │
    Entrée commune 352D
               │
    Réseau MLP entraîné (128 -> 64 -> 49 probabilités MIDI)
               │
    Sélection finale de K notes distinctes
```

**K modifie maintenant le calcul des probabilités à l'intérieur du réseau, avant sélection.** Le réseau n'a jamais accès au K réel pendant une prédiction. L'apprentissage est supervisé par les notes `note_midi` des JAMS originaux, mais l'entrée K reste **la prédiction S80** et non l'étiquette.

## Protocole et statut

[Run S402](https://github.com/Andriamarosoa/note/actions/runs/38361853607) réussi, quatre MLP entraînés, artefact GitHub `11795501155`, 59 309 groupes et **38 compositions distinctes**. Pour chacun des quatre folds, toutes les prises du même morceau, quel que soit le guitariste, sont soit entraînement, soit validation, jamais les deux. Les intersections de compositions ont été contrôlées à **0 dans tous les folds**. Toutes les prédictions de notes sont obtenues hors entraînement de cette tête S402.

**Limite de crédibilité** : ceci reste GuitarSet, déjà utilisé de nombreuses fois pour le développement des architectures. Le calcul K de S80 S397 lui-même contient un **recoupement avec ses données d'entraînement**, et aucune validation sur un nouveau corpus de compositions inédites n'est encore disponible. Les résultats ne sont donc ni une estimation indépendante de l'ensemble S80+S402, ni une promotion officielle.

## Résultats appariés avec EXACTEMENT le même S80-K

| Vrai K | S399 accord entièrement exact | S402 accord entièrement exact | Corrections S402 | Régressions S402 |
|---|---:|---:|---:|---:|
| K0 (39 652) | 38 432 (96,92 %) | 38 432 (96,92 %) | 0 | 0 |
| K1 (12 272) | 7 614 (62,04 %) | **8 029 (65,42 %)** | 665 | 250 |
| K2 (3 445) | 695 (20,17 %) | **1 338 (38,84 %)** | 799 | 156 |
| K3 (2 289) | 482 (21,06 %) | **1 009 (44,08 %)** | 626 | 99 |
| K4 (1 207) | 96 (7,95 %) | **266 (22,04 %)** | 219 | 49 |
| K5 (355) | 16 (4,51 %) | **88 (24,79 %)** | 80 | 8 |
| K6 (89) | 3 (3,37 %) | **14 (15,73 %)** | 12 | 1 |

**Poly K2–K6** : 1 292/7 385 = **17,49 %** (S399) contre **2 715/7 385 = 36,76 %** (S402), gain **+1 423 accords**, avec **1 736 corrections / 313 régressions**, et F1 micro MIDI **58,66 % -> 74,71 %**.

**Global K0–K6** : 47 338 -> 49 176 accords entièrement exacts, +1 838. Le taux global inclut une très grande proportion de K0 et ne doit pas cacher le score polyphonique.

La précision hauteurs globale par note n'est pas identique à l'Exact-K0–K6.

## Ablation pour prouver l'entrée K

Même MLP appris, même audio, même K utilisé pour le nombre de sorties. Seule la valeur K **injectée à l'entrée du MLP** est remplacée par 0 avant de calculer les probabilités.

- S402 K neutralisé au réseau : **2 546/7 385 = 34,48 %** de vrais accords polyphoniques complets.
- S402 K correctement injecté : **2 715/7 385 = 36,76 %**, soit **+169 accords polyphoniques exacts** grâce à l'information conditionnelle K.
- L'entrée K change effectivement les notes choisies dans respectivement **1 301, 580, 554 et 1 332** événements des quatre folds.

Ce protocole démontre que K n'est plus une simple consigne `top-K`. L'ablation est elle-même une analyse de développement : sans évaluation supplémentaire sur nouveaux morceaux, le chiffre +169 n'est pas un gain de généralisation vérifié.

## S403 — réutiliser les poids pour les nouveaux audio

Le runtime `scripts/s403_runtime_K_first_MIDI_notes_on_new_S80_audio.py` consomme :
- un NPZ audio natif `row_id + sequence42x49`, calculé à partir du signal sans annotation ;
- une prédiction alignée de S80 `row_id + s80_K` ;
- les quatre checkpoints S402 `S402_pitch_head_held_song_fold{0..3}.joblib`.

Il calcule `feature_bank(audio,K)` **avant de lancer le réseau**, agrège les probabilités des quatre modèles, puis produit `notes.csv` avec les hauteurs MIDI et les noms des notes, ainsi que `S403_S80_K_CONDITIONED_note_predictions.npz`.

Sur de *nouveaux morceaux*, les quatre modèles peuvent être utilisés en ensemble, mais l'évaluation de cet ensemble sur les morceaux GuitarSet déjà entraînés serait contaminée. Le score S402 ci-dessus provient exclusivement des **prédictions composition-heldout** et non de cet ensemble.

S403 **ne calcule pas le K lui-même** et ne transporte pas automatiquement les poids/modifications S397 vers un nouveau WAV : S80 doit avoir déjà produit K en amont.

**Décision** : S402 devient la meilleure tête de hauteurs *de développement* ; S399 reste témoin historique et S80 S397 reste la référence de comptage. Promotion indépendante bloquée tant que le système n'est pas testé sur des compositions sans recoupement d'entraînement de S80.
