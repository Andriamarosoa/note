---
serie: S813
statut: en_cours
type: experience
reference: S708
branche: codex/note-s813-independent-teachers85-midi256
run_id: pending
commit_sha: pending
promotion: false
---
# S813 — Enseignants indépendants >85 %, acoustique MIDI256

## Objectif

Consigne utilisateur du 20 octobre 2026 : dépasser strictement 85 % pour chacun des six enseignants, puis tester leur unification/distillation. Préserver une résolution de 256 échantillons à 44 100 Hz, soit 5,804988662 ms. Les poids, sélections, sorties et états de chaque enseignant doivent être privés. Les six scores du MIDI unifié seront évalués après cette première phase.

## Hypothèse

Les traces S800 comportent 76 600 observations acoustiques à 1024 échantillons, évaluées sur 306 536 frames à 256. Leur représentation supervisée S708 a déjà vu FIT80. Une extraction DSP réellement à 256, sans postérieurs supervisés historiques, permet un essai propre d'enseignant Frame Pitch sur des compositions exclues de son apprentissage.

## Modifications

- Contrat strict >0,85 pour les six axes, même cohorte de validation, horloges feature/decision/metric explicitement256; proxies, données FIT seules, état partagé et ensemble incomplet refusés.
- Extraction du PCM original à44100 Hz : 126 caractéristiques natives et 96 spectrales, uniquement courant/passé, sans interpolation de postérieurs à 1024.
- Premier enseignant Frame Pitch : deux têtes privées HistGradientBoosting (générale et graves/harmoniques), trois sélections et trois seuils prédéfinis, sans maintien ou anticipation ajouté.
- Checkpoints, décisions booléennes exactes, annotations originales signées, empreintes et scripts de recalcul conservés dans evidence/.
- Aucun changement des fichiers historiques S708/S707, du modèle LIVE ni des cinq autres enseignants.

## Protocole

GuitarSet FIT80, cinq compositions. Pour chaque pli :48 morceaux de trois compositions pour apprendre,16 d'une quatrième pour calibrer et16 de la composition exclue pour évaluer. La calibration est la composition suivante dans l'ordre BN1/Funk2/Jazz1/Rock3/SS1. Réglages fixés avant les scores externes, aucun réentraînement après calibration. Aucun accès CAL/HELD/TEST du projet.

FIT80 a été historiquement exploré : cette expérience reste recherche, pas certification indépendante. H0 est re-noté sur exactement les mêmes annotations et la même grille, mais sa représentation supervisée historique a vu FIT80 : les protocoles d'apprentissage ne sont pas équivalents.

## Audits obligatoires

Six sous-agents A–F réellement consultés : architecture/LIVE, DSP, apprentissage, métriques, CI, contre-revue. Ils ne sont pas les six configurations Work personnelles.41 tests ciblés passent localement. Trois avis NVIDIA non exécutés :UNKNOWN. Pas de consensus9/9 ni de validation physique Windows/iRig.

## Statut

Extraction, entraînement et recalcul local terminés. Frame Pitch =77,7132 %, objectif >85 % non atteint. La vérification GitHub des décisions sauvegardées reste à terminer; elle ne réentraîne pas les modèles. Unification et promotion bloquées. Les cinq autres enseignants restent à construire/entraîner dans ce nouveau protocole.
