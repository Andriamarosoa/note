# V27.3 — Point de conservation après la boucle S16–S18

**Statut : recherche exploratoire, pas modèle promu.** Cohorte déjà vue de 59 309 événements (dont 7 385 poly K2–K6), folds 0/1/2/4 ; fold3 exclus. Les trois nouvelles têtes ont exclu le morceau évalué de leur apprentissage, mais les décisions parentes et les seuils conservateurs ont été retenus après analyses de cette cohorte. Ne pas confondre rejeu indépendant du calcul avec évaluation sur enregistrements inédits.

## Résultats mesurés, face à freeze_local_combo

| Variante | Corrections sauvegardées | Régressions restantes | Gain net | Correct global | Correct poly | Exact global | Exact poly |
|---|---:|---:|---:|---:|---:|---:|---:|
| Série9 | 2934 | 2226 | +708 | 49162 | 2989 | 82,8913 % | 40,4739 % |
| Série15 OR safe (référence des nouvelles boucles) | 2934 | 2219 | +715 | 49169 | 2989 | 82,9031 % | 40,4739 % |
| Série16 `flow_logistic k23>0.99` | 2934 | 2215 | +719 | 49173 | 2993 | 82,9098 % | 40,5281 % |
| Série17 `flow_logistic k32>0.99` | 2934 | 2214 | +720 | 49174 | 2994 | 82,9115 % | 40,5416 % |
| **Série18 `flow_logistic k21>0.95`** | **2934** | **2210** | **+724** | **49178** | **2998** | **82,9183 %** | **40,5958 %** |

## Preuves et tracés reproductibles

- S16 : https://github.com/Andriamarosoa/note/actions/runs/37855705158, puis rejeu vérifié : https://github.com/Andriamarosoa/note/actions/runs/37856011482. **4 corrections**, 0 régression et 2 changements neutres ; cas natifs corrigés 13660,13925,35268,23156 (vrais K2, initialement classés K3). **108 politiques** et 152 modèles contrôlés. Archive permanente : https://github.com/Andriamarosoa/note/releases/tag/v273-series16-research-37856011482.
- S17 : https://github.com/Andriamarosoa/note/actions/runs/37856164138, puis rejeu : https://github.com/Andriamarosoa/note/actions/runs/37856362983. **1 correction**, 0 régression / 0 neutre : ID 49874, vrai K3 classé K2. 108 politiques et 152 modèles contrôlés. Archive : https://github.com/Andriamarosoa/note/releases/tag/v273-series17-research-37856362983.
- S18 : https://github.com/Andriamarosoa/note/actions/runs/37856499212, puis rejeu : https://github.com/Andriamarosoa/note/actions/runs/37856611495. **4 corrections**, 0 régression / 0 neutre : IDs 12865, 12869, 39410, 73854, vrais K2 classés K1. 108 politiques et 140 modèles contrôlés. Archive : https://github.com/Andriamarosoa/note/releases/tag/v273-series18-research-37856611495.

**Bilan cumulatif depuis S9** : 16 régressions évitées et **zéro correction initiale détruite** sur la cohorte observée. Les résultats négatifs ou compromis sont conservés : des politiques S16–S18 suppriment davantage de régressions mais détruisent également des corrections et ne sont pas retenues.

## Écart à la vraie cible

YourMT3+ : **51 328 corrects global** (86,5434 %) et **4 028 corrects poly** (54,5430 %). Le meilleur conservateur S18 reste à **2 150** bonnes prédictions globales et **1 030** bonnes polyphoniques de YourMT3+. Le meilleur producteur poly série4 obtient 3 162 bonnes poly (42,8165 %), encore **164** de plus que S18. Aucun résultat validé hors cohorte ne justifie une promotion.

## Prochaine boucle prioritaire

Ne pas se satisfaire de seuils qui produisent des gains de 1–4 exemples après de nombreux essais sur les mêmes labels. Reconstituer la **matrice de prédictions et les scores par K du meilleur producteur S4**, puis expérimenter un sélecteur neuronal qui préserve ses cas corrects et distingue sur- et sous-comptages, surtout K3/K4/K5. La sélection devra être évaluée sur des enregistrements non utilisés pour concevoir les garde-fous, à granularité de morceau/composition, avec toutes les corrections et régressions journalisées. Ne pas régler de nouveaux seuils en lisant les étiquettes de cette évaluation finale.
