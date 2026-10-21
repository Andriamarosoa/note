---
serie: S821
statut: en_cours
type: experience
reference: S708
branche: codex/note-s821-nvidia-ss1-k4-guard
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S821 — Revue NVIDIA approfondie et nouveau garde-fou causal contre K4→K5/K6

## Objectif

Faire trois vrais modèles NVIDIA auditer les 665 faux changements vrai K4 vers K5/K6 sur SS1, puis entraîner un vérificateur privé avec hypothèse contrôlable, sans toucher S708 ou LIVE. Tout gain doit être démontré par cinq partitions composition-excluses avant toute éventuelle évaluation externe.

## Hypothèse

Dans S820, la règle max_high@0.70 corrige 355 frames SS1 mais détériore 913, surtout les GT K4 (−555), alors qu'elle améliore GT K5 (+152) et K6 (+17). Les indices globaux harmonique/fondamentale sont très similaires pour les bons et mauvais changements. Un signal par hauteur et un vérificateur appris exclusivement sur TRAIN pourraient mieux séparer vrai K4 d'un accord de 5/6 notes ; ce n'est pas présumé vrai sans ablation.

## Modifications

Nouveau briefing neutre S821 à trois modèles NVIDIA NIM indépendants ; quatre contrôles de cohérence arithmétique, code, latence/causalité et fuites ; chaque modèle livre son propre audit. Une architecture SHADOW Active-K indépendante sera ensuite testée, jamais choisie a posteriori sur les scores outer. Les modèles S815/S817 restent gelés comme comparateurs, sans nouvelle dépendance à Frame Pitch ou aux autres têtes.

## Protocole

GuitarSet FIT80 réel et SHA signé, 80 WAV, 5 plis 48/16/16 disjoints par composition, 306536 trames MIDI256. Préserver le même compteur 249411 frames Active-K et les comptes S815/S817/S820 à l'identique. Apprentissage d'une tête conditionnelle au vrai K4/K5/K6 sur uniquement TRAIN, poids propres, décision sur inner-CAL seulement, évaluation externe distincte nécessaire au-delà de FIT80 déjà exploré.

## Audits obligatoires

Frais NVIDIA et appel réel vérifiés par logs/artefacts ; auditions indépendantes aveugles, revue de contradictions et hypothèses falsifiables. Tests DSP et absence de données futures, de hold, de poids inter-caractéristiques, de GT en inférence et de fuite des compositions. Conditions de succès : score Active-K OOF et chaque K sans dégradation, puis contrôle six axes, mesure LIVE réelle et validation source inédite avant promotion.

## Statut

En cours. Aucun nouveau score S821, aucune promotion, référence S708 et LIVE intacts.
