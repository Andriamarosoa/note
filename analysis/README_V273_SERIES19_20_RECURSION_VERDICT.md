# V27.3 — Tests récurrents A→B→A (séries 19 et 20), bilan factuel

État : **le mécanisme neural fonctionne, mais aucune variante S19/S20 ne satisfait simultanément la protection des 2 934 corrections, le score global et le score poly de la référence S18**. Les essais sont exploratoires : les 59 309 événements, folds 0/1/2/4, avaient déjà servi à l'élaboration de la référence ; pas de généralisation démontrée sur de nouvelles compositions. Aucun nouveau modèle promu.

## Architecture réellement testée

- A : réseau MLP qui produit une distribution K0–K6.
- B : réseau MLP qui produit sept compatibilités de classes à partir de l'audio et de la distribution actuelle de A.
- Après B, A **reçoit le message B comme une nouvelle entrée**, ainsi que les caractéristiques originales et son ancienne distribution : **A → B → A → B → A** avec poids partagés et rétropropagation sur les quatre passages.
- Un test contrefactuel remplace le seul message B sur K4 à audio et premier passage A identiques : la distribution A2 est effectivement modifiée, et K4 peut être réintroduit ultérieurement. Les probabilités B ne sont pas des interdictions fixes.
- S19 apprend depuis zéro ; S20 apprend un résidu ancré sur la prédiction S18 et pénalise les changements qui détruisent une prédiction correcte sur ses **morceaux d'entraînement seulement**.
- Les morceaux évalués sont exclus du fit, de la normalisation et de la supervision ; 19 modèles par architecture, 335 variables originales. L'entrée S18 de S20 est néanmoins un parent sélectionné sur la cohorte exposée.

## Scores

| Politique | Exact global | Exact poly | Corrections vs S18 | Régressions vs S18 |
|---|---:|---:|---:|---:|
| S18 conservatrice | 82,9183 % | **40,5958 %** | 0 | 0 |
| S19 un seul passage A | 82,2270 % | 36,2762 % | 1537 | 1947 |
| S19 A→B→A, deux passages | 82,3096 % | 35,1659 % | 1575 | 1936 |
| S19 A→B→A→B→A, quatre passages | 82,3045 % | 35,1253 % | 1573 | 1937 |
| S20 résiduel, passage1, marge 0,60 | **82,9402 %** | 40,4604 % | 66 | 53 |
| S20 résiduel, passage4, marge 0,60 | 82,9385 % | 40,3927 % | 63 | 51 |
| S20 résiduel, passage4 sans messages B, marge 0,60 | 82,9351 % | 40,5146 % | 36 | 26 |

Le premier passage S20 gagne 13 bonnes réponses nettes, mais le score poly **baisse** et **53 anciennes corrections sont détruites** par rapport au parent. Cela ne satisfait pas la condition de conservation.

## Preuve d'effet B et analyse des régressions causées par l'activation de B

S19 : sur 1 048 événements sélectionnés à probabilité de K4 relativement forte, forcer un message négatif B sur K4 diminue P_A2(K4) dans 839 cas, et change l'argmax A2 dans 68. S20 : même intervention sur 1 048 candidats, diminution K4 dans 760 cas et 47 changements d'argmax. Cela prouve la dépendance computationnelle de A au message de B, pas une règle musicale causale apprise.

La **comparaison appariée S20 passage4, marge0.6 avec messages B activés contre le même réseau entraîné avec B neutralisé** donne :

| Vrai K (utilisé seulement dans l'audit) | Corrigés par B | Détruits par B | Solde | Neutres |
|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | +1 | 12 |
| 1 | 17 | 7 | +10 | 4 |
| 2 | 8 | 9 | −1 | 2 |
| 3 | 2 | 8 | **−6** | 3 |
| 4 | 0 | 2 | **−2** | 2 |
| 5 | 0 | 0 | 0 | 1 |
| 6 | 0 | 0 | 0 | 0 |
| **Total** | **28** | **26** | **+2** | **24** |

**78 décisions diffèrent** entre B actif et B neutralisé ; 28 nouvelles corrections, 26 nouvelles régressions, 24 changements neutres. B aide le plus les vrais K1, mais pénalise les vrais K3 et K4. Attention : au moment de l'inférence, le vrai K n'est pas connu ; interdiction de coder « bloquer B si vrai K3/K4 ». Il faudra apprendre une fiabilité de B à partir des **seules caractéristiques observables et distributions prédites**, avec pénalisation des conflits K3/K4 sans bloquer arbitrairement ces classes.

## Preuves reproductibles, absence de promotion

- S19 entraînement : https://github.com/Andriamarosoa/note/actions/runs/37858017067 ; rejeu des 19 modèles réussi : https://github.com/Andriamarosoa/note/actions/runs/37858212108 ; archive modèles/probabilités/messages : https://github.com/Andriamarosoa/note/releases/tag/v273-series19-research-37858212108. Écarts de rejeu max : A 5,07×10^-7, B 2,38×10^-7.
- S20 entraînement : https://github.com/Andriamarosoa/note/actions/runs/37858383973 ; rejeu réussi : https://github.com/Andriamarosoa/note/actions/runs/37858541337 ; archive S20 : https://github.com/Andriamarosoa/note/releases/tag/v273-series20-research-37858541337. Écarts max : A 4,17×10^-7, B 2,98×10^-7.
- Audit apparié B actif/neutralisé : https://github.com/Andriamarosoa/note/actions/runs/37858615276 ; tous les 78 événements modifiés sont conservés dans `all-B-vs-noB-changed-cases.csv` (artefact d'audit).

## Blocage et suite justifiée

Le défaut n'est plus que les sélections ne repassent qu'une fois : **la boucle est effective**. Le défaut observé réside dans le signal de fiabilité/compatibilité de B, qui corrige 28 erreurs mais en introduit 26 au seuil retenu. L'entraînement actuel de B avec BCE binaire sur les classes K ne capture pas les contradictions harmoniques ou la possibilité qu'une sélection soit elle-même erronée. Une prochaine expérience doit apprendre séparément **l'utilité du message B** sur des morceaux d'entraînement, conditionnellement à la distribution de A et aux caractéristiques sonores. Comparer au contrôle B inactif sur des morceaux exclus de tous choix de conception. Pas de nouveaux ajustements de seuil sur les vrais K des événements évalués. Toujours conserver tous les essais et le parent série18.
