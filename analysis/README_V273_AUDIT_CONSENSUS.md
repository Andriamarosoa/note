# V27.3 — Audit des erreurs restantes du consensus

**Audit terminé le 8 octobre 2026, sur les prédictions figées du bras `pooled_ce`. Aucun réentraînement ni changement de décision.**

Le modèle obtient +60 net, mais prévoit +383,2. L’audit localise trois limites : il surestime surtout la réussite des changements retenus ; il laisse passer de nombreuses occasions dont le bon K est déjà disponible ; une autre partie des erreurs n’a aucun bon verdict dans le catalogue. Les sorties seules ne permettent pas d’attribuer ces limites à un signal acoustique physique précis.

[Résultats du modèle](README_V273_RESULTATS_CONSENSUS.md) · [PR #9](https://github.com/Andriamarosoa/note/pull/9) · [Run source terminé](https://github.com/Andriamarosoa/note/actions/runs/37781750476)

## D’où viennent les 323 justesses nettes surestimées ?

Sur 1 800 changements : **582 corrections, 522 régressions, 696 changements faux→faux**. Le net observé est +60.

| Issue | Nombre attendu par le réseau | Nombre observé | Écart contribuant à la surestimation |
|---|---:|---:|---:|
| Corrections | 832.79 | 582 | +250.79 |
| Régressions | 449.59 | 522 | +72.41 |
| Gain net | 383.20 | +60 | +323.20 |

**77.6 % de cet écart vient des corrections surestimées**, contre 22,4 % des régressions sous-estimées. Ce sont des sommes de probabilités sur les décisions retenues, pas des promesses par fragment.

La calibration globale masque une partie du problème de sélection :

| Probabilité auditée | Population | Moyenne prédite | Fréquence observée |
|---|---|---:|---:|
| Référence juste | 7 493 éligibles | 34.80 % | 33.56 % |
| Référence juste | 1 800 changements | 24.98 % | 29.00 % |
| Changement correct | 1 800 changements | 46.27 % | 32.33 % |
| Destination choisie juste, référence fausse | 1 278 changements avec référence fausse | 61.96 % | 45.54 % |
| Aucun groupe changeant juste, référence fausse | 4 978 références fausses | 69.32 % | 67.38 % |
| Aucun groupe changeant juste, référence fausse | 1 278 changements avec référence fausse | 33.33 % | 44.91 % |

La probabilité de justesse de la référence reste identique à celle du modèle global précédent. Sur les changements retenus, le réseau sous-estime le risque d’abîmer la référence et surestime la justesse de l’alternative. La présence d’une issue OTHER et d’un total de probabilités égal à 1 n’a pas suffi à calibrer ces choix.

Le gain appris sépare faiblement les corrections des régressions : **AUC 0,5472**, calculée sur 582 corrections et 522 régressions, en excluant les 696 changements neutres. Cette mesure descriptive n’est pas une validation indépendante.

## Où se situent les régressions K3/K4 ?

Les vrais K3 perdent 223 justesses et en gagnent 217 (net −6). Les vrais K4 en perdent 126 et en gagnent 76 (net −50). Ensemble, ils représentent **349 des 522 régressions**. Les 173 autres touchent K2.

Dans le tableau suivant, la transition est **K prédit initialement → K prédit après sélection**, pas le vrai K. Une transition peut donc corriger un événement et en détériorer un autre.

| Transition | Changements | Corrections | Régressions | Neutres | Net réel | Gain annoncé |
|---|---:|---:|---:|---:|---:|---:|
| K2→K3 | 424 | 123 | 160 | 141 | -37 | 71.59 |
| K2→K4 | 38 | 5 | 13 | 20 | -8 | 7.55 |
| K2→K5 | 3 | 1 | 0 | 2 | +1 | 0.22 |
| K3→K2 | 643 | 219 | 169 | 255 | +50 | 143.24 |
| K3→K4 | 177 | 71 | 52 | 54 | +19 | 36.22 |
| K3→K5 | 21 | 8 | 2 | 11 | +6 | 5.19 |
| K4→K2 | 177 | 55 | 26 | 96 | +29 | 53.99 |
| K4→K3 | 293 | 94 | 95 | 104 | -1 | 56.41 |
| K4→K5 | 24 | 6 | 5 | 13 | +1 | 8.79 |

**K3→K2 a le plus de régressions brutes (169), mais reste positif (+50)** grâce à 219 corrections. **K2→K3 est le principal déficit net : −37**. Interdire une transition uniquement parce qu’elle comporte beaucoup de régressions peut donc supprimer davantage de corrections. Ces bilans ne sont pas utilisés pour créer des interdictions après coup.

## Pourquoi seules 582 des 1 624 occasions sont-elles récupérées ?

Parmi les 4 978 références fausses dans le périmètre éligible, 1 624 disposent d’au moins une proposition correcte. Leur répartition est exacte :

| Décision ou blocage | Événements |
|---|---:|
| Bon K sélectionné | 582 |
| KEEP avec probabilité de référence juste ≥0,5 | 265 |
| KEEP malgré une probabilité de référence juste <0,5 | 655 |
| Changement vers un mauvais K alors que le bon est disponible | 122 |

Le seuil 0,5 du premier cas découle de la formule de gain : si `r≥0,5`, `(1-r)q-r≤0` pour toute probabilité `q≤1`. Il s’agit d’un blocage analytique du changement dans cette architecture. Les 655 autres abstentions surviennent sans ce blocage absolu, mais aucune alternative n’obtient un gain positif. Au total, **920 occasions restent inchangées et 122 partent vers le mauvais K**.

Les 3 354 autres erreurs initiales éligibles n’ont aucune proposition correcte : 2 041 sont des vrais K0/K1, qui ne sont pas des destinations autorisées, et **1 313 sont des erreurs polyphoniques**. Aucun meilleur classement des mêmes propositions ne peut corriger ces 1 313 cas.

| Vrai K | Erreurs initiales éligibles | Bon K proposé | Bon K absent | Corrections obtenues |
|---|---:|---:|---:|---:|
| K2 | 852 | 636 | 216 | 274 |
| K3 | 981 | 715 | 266 | 217 |
| K4 | 696 | 227 | 469 | 76 |
| K5 | 325 | 46 | 279 | 15 |
| K6 | 83 | 0 | 83 | 0 |

La couverture est particulièrement limitée en K4 : le bon K n’est disponible que pour **227 des 696 erreurs initiales éligibles** (32,6 %). Les 76 corrections représentent 33,5 % de ces 227 occasions. Il existe donc à la fois une limite du catalogue et une limite du choix à l’intérieur du catalogue. Les événements initialement prédits K0/K1/K5/K6 ne sont pas modifiables dans cette expérience et ne figurent pas dans ce tableau.

## Les historiques reconnaissent-ils les fragments à risque ?

Le modèle audité utilise les **historiques globaux** du groupe, de sa classe initiale et de sa destination. Les taux des voisins locaux sont exportés pour diagnostic, mais leurs colonnes 58/59/60/62/63 sont mises à zéro avant l’entrée du réseau. Il n’exploite donc pas directement ces taux de similarité aux corrections/régressions voisines. Le contexte acoustique et les votes restent des entrées apprises.

Pour ne pas confondre le représentant du K avec une contribution causale, cet audit moyenne les taux historiques de **tous les groupes proposant la destination choisie**. Ce résumé descriptif n’est pas une décomposition du réseau non linéaire.

| Résumé historique | AUC corrections / régressions | Gain historique négatif parmi les corrections | Parmi les régressions |
|---|---:|---:|---:|
| Global utilisé | 0.5515 | 359 / 582 | 348 / 522 |
| Local non utilisé | 0.5390 | 272 / 582 | 279 / 522 |

Un veto rétrospectif sur le signe négatif du résumé global aurait supprimé 359 corrections pour éviter 348 régressions (−11). Le même diagnostic sur le local aurait supprimé 272 corrections pour éviter 279 régressions (+7). Ces calculs sur les mêmes données ne valident aucune nouvelle règle ; ils montrent que ces résumés ne séparent pas nettement les bons et mauvais changements. La question de la similarité apprise reste ouverte.

Exemple le plus confiant parmi les régressions : fragment **66074**, `04_Jazz2-110-Bb_comp.jams`, **22,899 s**, fold 2. Le vrai K et la référence valent **4**, mais le réseau choisit **2**. Il attribue **5,03 %** de probabilité à la justesse de la référence et **93,33 %** à la correction proposée, soit un gain annoncé de **+0,883**. Les résumés historiques global et local sont tous deux positifs. Ce cas illustre une erreur très confiante qui n’est pas signalée par leur seul signe ; les 522 régressions sont exportées pour éviter de s’en tenir à cet exemple extrême.

## Pourquoi le nouveau modèle ne gagne-t-il que trois événements ?

Face au précédent global, ses 675 corrections se répartissent en **504 conservées et 171 perdues**. Il crée 78 nouvelles corrections. Il répare 171 anciennes régressions mais en crée 75 nouvelles ; 447 régressions persistent.

```text
171 régressions réparées + 78 nouvelles corrections
− 171 corrections perdues − 75 nouvelles régressions = +3
```

Les 246 justesses perdues dans la comparaison appariée correspondent donc à 171 anciennes corrections et 75 nouvelles régressions. Les 249 justesses gagnées correspondent à 171 régressions réparées et 78 nouvelles corrections. Confondre « corrections perdues » avec toutes les « justesses perdues » masquerait ce compromis.

Les gains annoncés restent trop élevés, y compris hors des décisions proches de zéro :

| Gain prédit par changement | Événements | Corrections | Régressions | Net réel | Gain total annoncé |
|---|---:|---:|---:|---:|---:|
| [0.00, 0.05[ | 298 | 78 | 99 | -21 | 7.13 |
| [0.05, 0.10[ | 280 | 96 | 74 | +22 | 20.71 |
| [0.10, 0.20[ | 440 | 144 | 136 | +8 | 64.22 |
| [0.20, 0.30[ | 315 | 99 | 102 | -3 | 77.88 |
| [0.30, 0.50[ | 325 | 115 | 84 | +31 | 126.02 |
| [0.50, 1.00] | 142 | 50 | 27 | +23 | 87.23 |

Ces intervalles sont descriptifs et n’ont servi à optimiser aucun seuil. Les sorties observées n’établissent pas encore si la cause amont est principalement la représentation acoustique, l’entraînement du critique ou le décalage entre les données servant aux audits et celles évaluées.

## Priorités étayées par cet audit

1. **Auditer puis corriger la probabilité de réussite des décisions retenues** : elle explique 77,6 % de la surestimation du gain. Toute calibration doit être apprise sur des prédictions indépendantes du fragment évalué, avec une validation finale encore distincte.
2. **Distinguer protection et récupération des occasions** : analyser les 522 références abîmées conjointement aux 920 occasions laissées inchangées et aux 122 mauvaises destinations. Un filtre plus conservateur peut réduire les deux côtés à la fois.
3. **Traiter séparément la couverture K4/K5/K6** : le classement des 127 masques ne peut pas inventer un K absent de leurs propositions. Ce constat ne justifie pas à lui seul une tête supplémentaire ; il faut examiner les votes et la fusion sur les cas absents.

Le problème des scores contradictoires est corrigé. Cet audit ne démontre pas que la qualité des probabilités et la couverture des propositions soient suffisantes. La référence reste conservée.

## Cas, preuves et reproduction

- [522 régressions](evidence/v273-consensus-audit/regressions-522.csv), triées par gain annoncé décroissant.
- [582 corrections](evidence/v273-consensus-audit/corrections-582.csv), pour comparer les profils positifs et négatifs.
- [171 anciennes corrections perdues](evidence/v273-consensus-audit/corrections-perdues-171.csv).
- [1 042 occasions correctes manquées](evidence/v273-consensus-audit/opportunites-manquees-1042.csv).
- [Audit complet, calibration par déciles et empreintes](evidence/v273-consensus-audit/audit.json).
- [Vérification des populations de cas et des probabilités](evidence/v273-consensus-audit/verification.json).

```bash
python -m scripts.audit_v273_consensus_failures \
  --consensus pooled_ce --archived-global archived-global \
  --features features --output analysis/evidence/v273-consensus-audit

python -m scripts.verify_v273_consensus_failure_audit \
  --audit analysis/evidence/v273-consensus-audit \
  --consensus pooled_ce --archived-global archived-global
```

Les archives sont vérifiées par SHA-256 contre les preuves du run 37781750476. Les CSV vérifient chaque ID, vrai K, référence, prédiction, disponibilité et gain. Les horodatages viennent des lignes acoustiques archivées, à 44 100 Hz. Dans les CSV, `true_K_reachable` signifie que le vrai K est présent parmi les propositions, y compris la référence lorsqu’elle est déjà juste.

Périmètre inchangé : folds 0/1/2/4 déjà exposés, fold 3/player05 exclus, +160 ms de contexte, un seul entraînement source. Aucun nouveau modèle, ajustement de seuil, test inédit ou affirmation de causalité physique dans cet audit.
