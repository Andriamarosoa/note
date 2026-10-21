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
# S814 — Résultats provisoires Active Exact-K indépendant

## Exécution

**Exécution scientifique réelle achevée** : [GitHub Actions #39535784658](https://github.com/Andriamarosoa/note/actions/runs/39535784658) SUCCESS. PCM original SHA vérifié, 80/80 pistes, 306536 grilles à 256 échantillons, cinq splits composition-disjoints et checkpoints privés. GitHub conserve report.json et tous les modèles sous l'artefact S814_ACTIVE_K_PRIVATE_MIDI256_FIT80_RESULTS (ID 12317845554). GitHub SUCCESS signifie preuve d'exécution, pas succès scientifique.

## Métriques

| Caractéristique | Référence S708 officielle CAL32 | Enseignant S814 FIT80 |
|---|---:|---|
| Onset F1 | 74,2639 % | non modifié / non mesuré |
| Exact-K | 62,0455 % | non modifié / non mesuré |
| Active Exact-K | 56,2762 % | **54,67280914 %** FIT80, en baisse vs H0 FIT80 57,01953803 % |
| Offset accuracy | 77,3292 % | non modifié / non mesuré |
| Full Note F1 | 57,1319 % | non modifié / non mesuré |
| Frame Pitch F1 | 76,4626 % | S813 expérimental 77,7132 % FIT80, non promu |

H0 FIT80 : 142213/249411 trames correctes (57,01953803 %) ; nouveau modèle 136360/249411 (54,67280914 %), soit **−2,34672889 points**. Sous-comptages 80654 → 74291 (−6363), surcomptages 26544 → 38760 (+12216). Folds : BN1 57,3142 %, Funk2 56,2230 %, Jazz1 55,5712 %, Rock3 53,5790 %, SS1 51,5650 %. Le score S810 shadow à 1024 n'est pas comparable à MIDI256.

## Comparaison à S708

Aucun événement MIDI S708/S707 modifié, pas de refonte du décodeur ni d'unification. Les 220 variables proviennent des 222 variables acoustiques originales recalculées à 256 par S813. Le résultat expérimental, qu'il soit supérieur ou inférieur à H0, demeure SHADOW et n'altère pas les cinq autres caractéristiques.

## Audits

| Axe | Statut | Limite |
|---|---|---|
| Données | PASS | Audio / manifeste / trames S800 SHA vérifiés dans #39535784658, FIT80 historiquement exploré |
| Acoustique | PASS | Entrée acoustique native 256 imposée par contrat de forme et de causalité |
| Architecture | PASS | Deux poids distincts et sélecteur privé, aucune entrée d'un enseignant voisin |
| Apprentissage | FAIL | Cinq plis réellement entraînés, précision 54,67 % en régression et objectif >85 % manqué |
| Métriques | PASS | Parité S704 historique 142213 / 249411 sur les vraies trames actives, audit comptages par classe sauvegardé |
| LIVE | UNKNOWN | Pas d'essai microphone, latence matérielle et CPU total inconnus |

## Verdict

**REJET SCIENTIFIQUE** du candidat S814 : 54,67 % < 57,02 % référence FIT80. Objectif >85 % non atteint, référence S708 intacte. Le run GitHub #39535784658 a PASS techniquement, sans promotion.

## Prochaine action

Tester une nouvelle architecture de sélection causalement indépendante pénalisant les surcomptages sur composition de calibration UNIQUEMENT, et conserver l'évaluation composition-exclue intacte. Examiner les décisions K2–K6, sans toucher au flux MIDI historique.
