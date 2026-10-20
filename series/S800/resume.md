---
serie: S800
statut: termine
type: experience
reference: S708
branche: codex/note-s800-fit-oof-replay-traces
run_id: 39354212035
commit_sha: 37b750a22c7275231ba35a6c1a347be28f09b3ea
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

**Terminée — instrumentation vérifiée, non promue.** La reconstruction réelle FIT80 et les checksums sont PASS dans le [run #39352880035](https://github.com/Andriamarosoa/note/actions/runs/39352880035). Les six audits indépendants et le consensus sont PASS dans le [run de réaudit #39354212035](https://github.com/Andriamarosoa/note/actions/runs/39354212035). Aucun changement aux six scores CAL32 ni à la chaîne LIVE.

## Preuves de reproduction

- [Run de reconstruction S800 #39352880035](https://github.com/Andriamarosoa/note/actions/runs/39352880035), commit scientifique `ce678c18c832e259f29386366973c41d8380357d` : `fit80_replay` SUCCESS, 80/80 pistes OOF sauvegardées, cinq plis de composition de 16 pistes chacun, 306536 frames, 189124 Exact-K corrects, tous les douze comptes FIT reproduits. Le job de vérification indépendante des archives (`S800_TRACE_AUDIT_PASS`) est également PASS. Le run global était FAILURE uniquement parce que les jobs d'audit Python ne trouvaient pas `joblib`.
- [Run de réaudit S800 #39354212035](https://github.com/Andriamarosoa/note/actions/runs/39354212035), commit `37b750a22c7275231ba35a6c1a347be28f09b3ea` : six rôles auditeurs **PASS**, consensus strict **PASS**. Il s'agit d'un examen des preuves gelées déjà calculées, sans refaire l'entraînement GuitarSet.
- Archive FIT OOF chiffrée par empreintes SHA256, 80 traces avec cinq cartes de probabilités par fragment/hauteur. Aucun accès CAL32/HELD/TEST et aucune promotion.
