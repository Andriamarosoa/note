---
serie: S809
statut: termine
type: infrastructure
reference: S708
branche: codex/note-s809-six-isolated-axis-pipelines
run_id: 39384080646
commit_sha: c3970969558520c7e0941c1d85b8251d2453eae3
promotion: false
---
# S809 — résultats de séparation stricte, six axes indépendants

## Exécution

[Run scientifique S809 #39384080646](https://github.com/Andriamarosoa/note/actions/runs/39384080646) **COMPLETED SUCCESS**, commit scientifique `c3970969558520c7e0941c1d85b8251d2453eae3`. Test syntaxique, anciens tests S763 et nouveaux tests S809 : **23/23 PASS**. Évaluation signée de **80 vraies pistes GuitarSet FIT80 OOF**, dont les 5 compositions BN1, Funk2, Jazz1, Rock3 et SS1. Contrôle SHA256 de chaque trace NPZ/JSON et reproduction stricte des 12 compteurs historiques S708 pour chaque piste. Les six décodeurs privés ont traité chaque piste entière, chacun avec ses propres têtes/états/sélecteurs, sans arbitre MIDI intervenant.

Pour chacune des 80 pistes, six contre-exemples modifient **uniquement un sélecteur** sur les 64 premières fenêtres disponibles (à 1024 échantillons par décision) et exigent la stricte égalité des cinq autres sorties et sélections. Résultats : **30 relations de non-interférence sur 30 PASS**, aucun décodeur étranger ne change. Les tests unitaires couvrent également perturbation hostile de chaque tête et modification de chaque décodeur. Les six audits réels ont réussi, suivis du consensus 6/6 sur les artefacts scientifiques.

### Effet effectivement mesuré des perturbations

| Axe modifié | Cellules passées au signal alternatif | Décisions sur cet axe modifiées | Décisions modifiées sur les cinq autres |
| --- | ---: | ---: | ---: |
| Exact-K | 58 016 | 1 184 | **0** |
| Active Exact-K | 8 078 | 1 341 | **0** |
| Onset | 3 639 | 477 | **0** |
| Offset | 2 850 | 225 | **0** |
| Frame Pitch | 7 789 | 235 | **0** |
| Full Note | 2 785 | 857 | **0** |

**Ce tableau mesure des changements de décisions SHADOW, non des erreurs corrigées ni des gains de F1.** Les sorties officielles S708 sont demeurées strictement identiques.

## Métriques

Les six valeurs **officielles S708 sur CAL32** demeurent protégées et inchangées. Ce tableau ne mesure PAS la performance des décodeurs SHADOW S809 :

| Métrique | S708 officiel CAL32 | Nouveau modèle S809 |
| --- | ---: | --- |
| Onset F1 | 74,2639 % | NON MESURÉ |
| Exact-K | 62,0455 % | NON MESURÉ |
| Active Exact-K | 56,2762 % | NON MESURÉ |
| Offset accuracy | 77,3292 % | NON MESURÉ |
| Full Note F1 | 57,1319 % | NON MESURÉ |
| Frame Pitch F1 | 76,4626 % | NON MESURÉ |

Le run #39384080646 a reproduit sur FIT80 le seul flux MIDI S708 d'origine : Onset 73,2520 %, Exact-K 61,6972 %, Active Exact-K 57,0195 %, Offset 69,8862 %, Full Note F1 50,6528 % et Frame Pitch F1 78,7278 %. Les évaluations mono-axe de S809 ne constituent pas les six ratios du MIDI final.

## Comparaison à S708

Aucun fichier S708, S707, MIDI de production ou code iRig modifié. L'indépendance est validée sur six décodeurs shadow ; elle ne permet pas de prédire l'effet d'un arbitrage qui assemblerait leurs six sorties. Aucune nouvelle latence algorithmique de confirmation ou de hold ; le coût CPU d'un tel système n'a pas encore été mesuré physiquement.

## Audits

| Audit | Statut | Preuve |
| --- | --- | --- |
| Données | PASS | #39384080646 : 80/80 traces signées, 12 compteurs par piste identiques |
| Acoustique | PASS | #39384080646 : trames actuelles/passées seulement, proxy argmax K déclaré |
| Architecture | PASS | #39384080646 : 30/30 comparaisons d'axes identiques, 6 sélecteurs distincts |
| Apprentissage | PASS | #39384080646 : aucun réseau indépendant entraîné prétendu |
| Métriques | PASS | #39384080646 : six scores S704 source séparés des diagnostics shadow |
| LIVE | PASS | #39384080646 : zéro hold/lookahead, production intacte ; latence CPU/iRig non mesurée |

Les six auditeurs sont CI déterministes, distincts des six agents personnalisés Work. Les modèles NVIDIA ne sont pas lancés. Consensus scientifique des six auditeurs CI **PASS 6/6**. Les trois NVIDIA n'ont pas été lancés ; aucun 9/9 revendiqué.

## Verdict

**S809 terminée comme architecture SHADOW : 80/80 traces réussies, 23 tests adversariaux PASS, 6/6 audits PASS et non-interférence des 30 couples d'axes mesurée.** Aucune régression des sorties des cinq autres chaînes lorsqu'un axe change. Les douze têtes sont des signaux déterministes, non des modèles entraînés ; cette réussite ne constitue pas une amélioration musicale. S708 officielle et LIVE intactes.

## Prochaine action

Entraîner chaque axe avec ses propres poids et sa propre fonction de perte, en préservant les six sélecteurs et six décodeurs déjà séparés. Ensuite seulement soumettre chaque adaptation à un replay FIT80 puis au contrôle indépendant CAL32 des **six** métriques du MIDI final ; interdire les régressions et toute latence supplémentaire. La séparation des décisions SHADOW est établie, pas leur performance musicale ni leur coût CPU iRig.
