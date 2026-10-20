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

# S800 — traces OOF GuitarSet FIT80 et reproduction signée S795

## Exécution

Workflow S800 terminé : test ABI PASS, reconstruction FIT80 authentique PASS, 80 traces par piste/hauteur validées, puis six audits indépendants PASS après reprise correcte des archives et consensus PASS. Échantillons CAL32 non utilisés pour choisir une politique. La reconstruction complète peut demander longtemps car elle recalcule les modèles et les caractéristiques sans disposer des banques S795 anciennes.

## Métriques

| Métrique | S708 officiel CAL32 | S800 |
|---|---:|---|
| Onset F1 | 74,2639 % | non mesuré |
| Exact-K | 62,0455 % | non mesuré |
| Active Exact-K | 56,2762 % | non mesuré |
| Offset accuracy | 77,3292 % | non mesuré |
| Full Note F1 | 57,1319 % | non mesuré |
| Frame Pitch F1 | 76,4626 % | non mesuré |

Comptes FIT historiques archivés S795, contrôle attendu : frames=306536 ; k_correct=189124 ; active_frames=249411 ; active_k_correct=142213 ; Onset TP=8528, FP=3489, FN=2739 ; Frame Pitch TP=414084, FP=76787, FN=146984 ; Offset éligibles=8438, corrects=5897. Ce sont des nombres de référence FIT, pas un score CAL nouveau.

## Comparaison à S708

Aucun changement aux poids, au processeur causal, aux décisions NOTE, aux voix actives ou à la latence LIVE. La signature S708 et ses six scores CAL restent la référence. Reproduire FIT80 ne constitue ni une nouvelle victoire Exact-K ni une promotion.

## Audits

| Rôle | Statut | Preuve |
|---|---|---|
| Données | PASS | [Réeaudit S800 #39354212035](https://github.com/Andriamarosoa/note/actions/runs/39354212035) |
| Acoustique | PASS | [Réeaudit S800 #39354212035](https://github.com/Andriamarosoa/note/actions/runs/39354212035) |
| Architecture | PASS | [Réeaudit S800 #39354212035](https://github.com/Andriamarosoa/note/actions/runs/39354212035) |
| Apprentissage | PASS | [Réeaudit S800 #39354212035](https://github.com/Andriamarosoa/note/actions/runs/39354212035) |
| Métriques | PASS | [Réeaudit S800 #39354212035](https://github.com/Andriamarosoa/note/actions/runs/39354212035) |
| LIVE | PASS | [Réeaudit S800 #39354212035](https://github.com/Andriamarosoa/note/actions/runs/39354212035) |

## Verdict

Pas de promotion. **S800 instrumentation terminée, sans promotion.** FIT80 GuitarSet a été reproduit et les douze comptes S795 correspondent exactement ; les six audits sur preuves gelées et leur consensus sont tous PASS. Le run scientifique initial échouait dans les auditeurs pour une dépendance Python `joblib`, pas dans la reconstruction ; le réaudit a été effectué sans répéter l'entraînement. L'artefact réel des cinq têtes est disponible via GitHub Actions.

## Prochaine action

Les preuves sont complètes. S801 peut maintenant utiliser les 80 traces OOF FIT80 pour tester réellement les ablations H4/H3/H2/H5 et mesurer les six métriques appariées, sans CAL/HELD/TEST pour la sélection. Le 7e conseiller GLM-5.3-Flash reste une aide consultative sans autorité sur ces mesures. Ne pas utiliser CAL pour sélectionner, et ne jamais ajouter un délai NoteOff.

## Preuves de reproduction

- [Run de reconstruction S800 #39352880035](https://github.com/Andriamarosoa/note/actions/runs/39352880035), commit scientifique `ce678c18c832e259f29386366973c41d8380357d` : `fit80_replay` SUCCESS, 80/80 pistes OOF sauvegardées, cinq plis de composition de 16 pistes chacun, 306536 frames, 189124 Exact-K corrects, tous les douze comptes FIT reproduits. Le job de vérification indépendante des archives (`S800_TRACE_AUDIT_PASS`) est également PASS. Le run global était FAILURE uniquement parce que les jobs d'audit Python ne trouvaient pas `joblib`.
- [Run de réaudit S800 #39354212035](https://github.com/Andriamarosoa/note/actions/runs/39354212035), commit `37b750a22c7275231ba35a6c1a347be28f09b3ea` : six rôles auditeurs **PASS**, consensus strict **PASS**. Il s'agit d'un examen des preuves gelées déjà calculées, sans refaire l'entraînement GuitarSet.
- Archive FIT OOF chiffrée par empreintes SHA256, 80 traces avec cinq cartes de probabilités par fragment/hauteur. Aucun accès CAL32/HELD/TEST et aucune promotion.
