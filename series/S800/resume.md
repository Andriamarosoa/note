---
serie: S800
statut: en_cours
type: experience
reference: S708
branche: codex/note-s800-fit-oof-replay-traces
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S800 — récupération des cinq probabilités OOF par fragment/hauteur

## Objectif

Recuperer les probabilites reelles des cinq têtes S795 pour les 80 enregistrements GuitarSet FIT, sans ouvrir CAL32, HELD ou TEST. Préserver l'architecture S708 et les décisions MIDI256 source et préparer l'évaluation scientifique du routeur dynamique S797 sans régression des six scores (Onset F1, Exact-K, Active Exact-K, Offset, Full Note F1, Frame Pitch F1). L'objectif 80 % n'est pas encore mesuré par cette instrumentation.

## Hypothèse

Les archives S795 contiennent les scores agrégés FIT80 mais pas les prédictions OOF, les 23 caractéristiques et l'horloge native par fragment/hauteur. Une reconstruction STRICTEMENT IDENTIQUE des modèles par composition exclue, avec signatures SHA256, doit reproduire les comptes source FIT historiques et livrer l'information nécessaire pour ablater les décisions séparées Onset/Offset et les veto harmoniques sans fuite de CAL.

## Modifications

Un écrivain de snapshots non intrusif \`scripts/s800_oof_trace.py\` archive par piste : les caractéristiques f16 [T,49,23], cinq probabilités float32 [T,49], horloges d'inférence 1024 et métrique MIDI256, notes sources, vérités FIT uniquement, masques des cibles, empreintes de fichiers, composition exclue et statistiques source. La commande \`scripts/s800_fit_oof_replay.py\` réutilise les fonctions originales S795 pour les 80 enregistrements et les cinq plis de composition. Elle ne modifie aucun seuil, tête, modèle, décodeur, NoteOn ou NoteOff. La référence archivée est S795 run 39340511467 ; artefacts officiels S708 et offset inchangés.

## Protocole

Téléchargement et vérification cryptographique des archives gelées ; S795 FIT80 refait, puisqu'aucun modèle OOF exploitable n'était sauvegardé ; affichage et sauvegarde des 80 partitions ; validation des **12 comptes bruts FIT** et six métriques contre l'archive S795 ; aucune ouverture de la CAL pour sélection et aucune simulation de gain. Les six audits systématiques ne peuvent passer sans les partitions et la validation stricte. Aucune sortie n'est injectée dans le système LIVE.

## Audits obligatoires

Données : exclusion stricte des compositions et CAL. Acoustique : cinq cartes complètes par fragment. Architecture : ABI et horloges 1024/256. Apprentissage : entraînement quatre compositions hors pli. Métriques : exactitude de tous les comptes FIT archivés sans tolérance. LIVE : aucune modification d'émission, aucune promesse non mesurée de latence physique. Consensus final fail-closed.

## Statut

En cours d'exécution CI S800. Les tests synthétiques d'export et la validation des signatures ne sont pas des scores musicaux ; les 80 traces réelles doivent être reproduites avant que S800 soit considérée terminée.
