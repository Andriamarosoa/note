# Onset — boucle expérimentale PCM256 : harmoniques, fausses réattaques, apprentissage des erreurs

## Objectif de la série

Cette série conserve les **trois têtes propriétaires d'Onset** : `MainAttackHead` (S613, poids natifs 256), `RescueAttackHead` (S701, poids natifs 256) et `ErrorPredictionHead` (un **vrai MLP** entraîné sur des exemples corrects/incorrects). Elle ne mélange jamais les cinq autres axes, ne modifie aucun autre décodeur et ne publie aucune décision dans le MIDI officiel.

L'audit scientifique antérieur a mesuré sur GuitarSet FIT80 :

| Modèle et source | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| S613 natif véritable 256 | 8099 | 4600 | 3168 | 67,5874155 % |
| S701 shadow 256 | 8113 | 4603 | 3154 | 67,6562565 % |
| Troisième tête ErrorPrediction MLP 63D, vrai replay | 8090 | 4553 | 3177 | 67,6704308 % |
| S708 historique **entraîné à 1024**, référence non exécutable | 8528 | 3489 | 2739 | 73,2520186 % |

Le réseau MLP original conteste seulement 165/12699 propositions (124 FP, 41 TP), ce qui ne couvre que 2,7 % des 4600 faux positifs, même si 75,15 % des contestations sont justifiées par l'annotation. Sur 40 pistes `comp`, 3685 faux positifs ; sur 40 `solo`, 915. La validation [run #39729646885](https://github.com/Andriamarosoa/note/actions/runs/39729646885) est **techniquement verte** et montre +0,0830 point F1 ; insuffisant pour la cible 85 %.

## Trois directions EXPÉRIMENTALES déjà implémentées

1. **Harmoniques** : nouvelle représentation 69D = 63D S613/S701 acoustiques causales et fondamentales/partiels (−12/−19/−24/−28) + 6D de l'état Onset *avant* tout NoteOn. Spécialiste `harmonic` : entraîne un MLP sur des exemples difficiles de faux positifs lors d'accords avec une fondamentale inférieure plus forte. Ce n'est **pas un veto fixe de parent harmonique** : de vrais accords peuvent réunir parent et hauteur supérieure.
2. **Fausses réattaques** : le MLP observe `active`, `armed`, `low/release_steps`, `high/confirmation_steps`, âge depuis le dernier NoteOn et fraction du temps réfractaire écoulée. Ces 6D sont capturées **avant** toute décision et calculées à partir de l'état privé de cette tête Onset. Spécialiste `restrike` : sur-échantillonne uniquement les faux positifs qui surviennent pendant une note active/récemment attaquée. **Aucune prolongation de release, de délai de confirmation ou de lookahead n'est ajoutée.**
3. **Amélioration de la tête d'erreur** : véritable entraînement `StandardScaler+MLPClassifier(64,32)` pour les trois variantes `balanced`, `harmonic`, `restrike`. Les biais d'entraînement des FP sont obtenus par échantillonnage répété déterministe sur TRAIN ; CAL et OUTER ne sont jamais ajoutés à l'entraînement. Chaque modèle a ses propres poids réellement appris à 256.

La sélection du spécialiste et du seuil de doute est faite **uniquement sur CAL**. Chaque fold utilise 3 compositions TRAIN, 1 CAL et 1 OUTER. Les cinq checkpoints OOF sont exportés avec SHA256 + provenance TRAIN/CAL/OUTER, puis revérifiés dans un **second replay de chaque échantillon WAV PCM256** : 80 pistes, **306536 frontières**, 1 appel par tête par 256 nouvelles valeurs mono 44,1 kHz, annotation uniquement *après* l'inférence de toute la piste.

Les 3 têtes restent indépendantes : un même frontal acoustique de lecture peut être partagé dans Onset, mais le MLP a des poids indépendants. La probabilité d'erreur est calculée **avant** les décodeurs, constitue un `HeadOpinions.error_probability` et un `Decision.head_doubts`, et alimente uniquement la **sélection expérimentale Onset**. Les sorties S613 et S701 demeurent strictement inchangées.

## Fichiers

- `src/onset256/error_evidence69.py` — extraction stricte 69D et chargement d'un MLP 69D SHA-pinné, propre à Onset.
- `src/onset256/runtime.py`, `src/onset256/heads.py` — capture de l'état pré-décision et prédiction MLP **avant toute émission** d'un Onset.
- `scripts/onset_harmonic_forensics256.py` — reconstitution réelle des événements/étiquettes TP/FP après le passage causal ; journaux `events.jsonl` comportant désormais `hard_predecision_vector` 69D en plus du 63D historique.
- `scripts/train_onset_hardnegative69.py` — **15 modèles réellement entraînés (3 familles × 5 plis)**, sélection CAL puis 5 checkpoints retenus.
- `scripts/replay_hardnegative69.py` — second replay sur WAV réels, audit `challenged_tp` et `challenged_fp`, F1 de la sélection causale et tous les écarts par piste.
- `tests/test_onset_hardnegative69.py` — état Onset capturé **avant**, 69D ABI, fragmentation, invariance de la tête Main et échantillonnage exclusivement TRAIN.
- `.github/workflows/onset-hardnegative-256.yml` — CI tests techniques, sources audio et checkpoints SHA, apprentissage, replay causal, artefacts scientifiques.

## Interdictions conservées

- **Aucun poids entraîné à 1024** (S708 reste un témoin historique, non importé).
- Aucune trame future, attente volontaire, temporisation de l'attaque, filtre K externe, extension d'Offset ou modification des autres axes.
- **Ne pas promouvoir** une politique si elle régresse le F1 Onset sur les 80 pistes dans la comparaison contrôlée. Le CI conserve l'artefact même en cas d'échec scientifique.
- Les anciens poids S613 et S701 ont déjà vu respectivement **20 et 48 pistes FIT80** durant leur entraînement. Le nouvel OOF ne constitue donc **PAS un test indépendant du pipeline complet**. Un ensemble réellement nouveau est nécessaire avant toute promotion industrielle.
- Une décision par 256 samples = une **cadence d'actualisation de 5,805 ms**, ce n'est **pas une latence totale mesurée**. Le coût CPU/ASIO/WASAPI reste à mesurer.

## Statut

Code expérimental prêt ; les **nouveaux scores seront uniquement ceux des rapports GitHub terminés**. Aucun gain n'est affirmé avant apprentissage et replay. Toujours archiver séparément les exemples d'erreurs corrigées et les régressions.
