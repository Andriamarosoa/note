---
serie: S814
statut: termine
type: experience
reference: S708
branche: codex/note-s814-activek-private-midi256
run_id: 39535784658
commit_sha: 36a9011618af40e16593462c99be2a0757671d87
promotion: false
---
# S814 — Enseignant Active Exact-K indépendant MIDI256

## Objectif

Faire progresser Active Exact-K avec deux têtes entraînées indépendamment sans modification de Frame Pitch, Exact-K, Onset, Offset ni Full Note. Objectif strictement supérieur à 85 % sur chaque enseignant avant la distillation, et résolution MIDI256 à 44 100 Hz (5,805 ms).

## Hypothèse

Le sous-comptage / surcomptage du MIDI émis ne doit pas être corrigé en manipulant les sorties historiques. Un prédicteur direct de K0–K6, confronté à un modèle de régression de cardinalité, pourrait apprendre les superpositions depuis les seules observations acoustiques causales de S813.

## Modifications

Ajout d'un nouvel enseignant Active-K à deux modèles privés HistGradientBoosting, agrégation acoustique de 220 dimensions à chaque décision, sélection privée parmi trois stratégies et trois biais fixés, apprentissage stratifié à poids de population, aucun post-traitement temporel ni anticipation. Les entraînements S708/S707 et les autres têtes restent inchangés.

## Protocole

FIT80 GuitarSet original uniquement; données cache S813 fraîches à 256 échantillons, audio PCM signé. Cinq partitions composition-disjointes, 48 morceaux apprentissage, 16 calibration interne, 16 évaluation externe par partition. Calibrage sans score de la composition d'évaluation. Métrique Active-K sur les trames réellement actives uniquement; K0 reste un état de prédiction, sans oracle en inférence. Parité de la référence S704 MIDI256 requise. FIT80 exploré historiquement ne constitue pas une validation indépendante.

## Audits obligatoires

Tests de causalité, changements du futur, découpage d'entrée, isolation de l'apprentissage, priors d'échantillonnage, vraies métriques Active-K et identité historique. Validation physique iRig et six audits musicaux sur MIDI unifié requis avant toute promotion. Aucune prétention sur les agents NVIDIA ou les six assistants personnels.

## Statut

Architecture, entraîneur et tests préparés sur branche S814. Mesures GuitarSet réelles et preuve de latence physique non encore obtenues au moment de ce rapport initial. Unification et promotion interdites.


## Résultat réel et clôture scientifique

Workflow [#39535784658](https://github.com/Andriamarosoa/note/actions/runs/39535784658) : SUCCESS pour restauration SHA, extraction MIDI256 des 80 enregistrements et entraînement des cinq plis. Score Active Exact-K réel de l'enseignant indépendant **54,67280914 %** (136360/249411), contre H0 FIT80 **57,01953803 %** (142213/249411), différence **−2,34672889 points**. Sous-comptages S814 74291 vs H0 80654, surcomptages S814 38760 vs H0 26544. Candidat NON promu, l'objectif >85 % non atteint. Aucune modification LIVE, zéro validation de latence physique. Cette série se termine comme expérience négative documentée.
