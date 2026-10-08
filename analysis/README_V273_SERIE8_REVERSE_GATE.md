# V27.3 — Série 8 : conserver la polyphonie, corriger les erreurs de confiance

Protocole ajouté après le constat négatif de la série 7, **avant l'exécution de la série 8**.

## Hypothèse et différence contrôlée
Les séries 6 et 7 utilisent le candidat à meilleur score global (G, 83,7799 % / 34,3805 % poly) comme parent et basculent vers le candidat à meilleure polyphonie de série 5 (P, 82,8829 % / 40,4333 % poly). Les 12 portes apprises de la série 7 sont insuffisantes. La série 8 inverse le parent : partir de P sur les 59 309 lignes, et ne basculer vers G que lorsque la probabilité ajustée sur d'autres morceaux indique un avantage de G.

## Réutilisation du protocole imbriqué
Ne modifier ni sources, ni features, ni folds, ni construction du modèle de risque de série 7. Chaque gate est appris seulement sur les autres morceaux du même fold, dont tous les producteurs sont entraînés sans le fold évalué. Label uniquement à l'ajustement de la gate. Aucun résultat de morceau évalué ne contribue à son propre modèle/choix.

Les probabilités d'avantage sont P(seul P correct) − P(seul G correct). Les 2 représentations (votes, votes + 58 caractéristiques), les C de 0.03 / 0.3 sont identiques.

## Nouvelles politiques, déclarées
Pour chaque couple (représentation, C), partir de P ; appliquer G uniquement si avantage < -coût, avec coûts fixes 0, 0.05, 0.10, 0.20, 0.30, 0.50. **24 politiques nouvelles** ; les 12 de série 7 et les deux parents restent conservés, pour un inventaire de 38.

## Critères
Pour chaque politique, exact-K global/poly, K0–K6, scores par fold, corrections/régressions versus G, P, freeze et YourMT3+. Conserver poids, marges, changements, identité de chaque événement et sorties négatives. Pas de promotion automatique. Les données du développement déjà explorées ne sont pas une validation indépendante ; seule une évaluation hors développement peut confirmer un futur gain.
