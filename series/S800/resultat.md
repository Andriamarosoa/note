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

# S800 — traces OOF GuitarSet FIT80 et reproduction signée S795

## Exécution

Workflow GitHub S800 en cours/en attente : test ABI, reconstruction FIT80 avec le code S795 gelé, 80 partitions trace par piste/hauteur, puis six audits indépendants et consensus. Échantillons CAL32 non utilisés pour choisir une politique. La reconstruction complète peut demander longtemps car elle recalcule les modèles et les caractéristiques sans disposer des banques S795 anciennes.

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
| Données | UNKNOWN | Run S800 à vérifier |
| Acoustique | UNKNOWN | Run S800 à vérifier |
| Architecture | UNKNOWN | Run S800 à vérifier |
| Apprentissage | UNKNOWN | Run S800 à vérifier |
| Métriques | UNKNOWN | Run S800 à vérifier |
| LIVE | UNKNOWN | Run S800 à vérifier |

## Verdict

Pas de promotion. Résultat en attente de la reprise exacte des 80 enregistrements et du consensus des six audits. Tout écart avec les 12 comptes signés S795 est un rejet, jamais une tolérance silencieuse.

## Prochaine action

Après le retour de l'artefact complet : auditer la reproductibilité source par piste, les harmoniques et le routage fragmentaire FIT, puis lancer les ablations indépendantes H4/H3/H2/H5 avec métriques appariées. Le 7e conseiller GLM-5.3-Flash reste une aide consultative sans autorité sur ces mesures. Ne pas utiliser CAL pour sélectionner, et ne jamais ajouter un délai NoteOff.
