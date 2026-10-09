# Série30 — audit causal des quatre régressions et validation croisée des nouvelles sélections

Protocole fixé **avant la première exécution**. Le prototype S29 est conservé, ainsi que les 38 modèles A/BA/STOP. La référence S18 reste 49 178/59 309 corrects global et 2 998/7 385 corrects poly.

## Question falsifiable

L'ordonnanceur S29 (λ=2, seuil=0,02) corrige 17 décisions S18 et en casse 4. Les quatre erreurs ne doivent **pas** être converties en exceptions codées en dur. Pour vérifier si A-first/B-first apportent une preuve indépendante, construire des règles déterministes d'accord et un modèle de corroboration **appris sur d'autres morceaux**, sans exploiter l'étiquette vraie de l'événement évalué.

## Sources inchangées

- S29 : run 37865173636, `predictions.npz` et `routes.npz` des 9 politiques ; parent S18.
- S20 : run 37858383973, `probabilities.npz` des 4 passages A-first. Les modèles S20 n'ont pas vu le morceau évalué.
- S25 : run 37862417522, `probabilities.npz` des 4 passages B-first. Les modèles S25 n'ont pas vu le morceau évalué.
- Les distributions A-first/B-first reflètent le même parent S18 historiquement choisi après observation du jeu de développement. Pas de validation externe indépendante.

## Tests fixes

Pour chaque événement dont S29 `series29__lambda2__threshold0.02` propose un K différent du parent, comparer **sans vérité** :
1. `AF2`, `AF4`, `BF2`, `BF4` : chaque tête A-first/B-first à passage 2/4 soutient l'argmax du candidat.
2. Accord `AF4 ∧ BF4`, accord `AF4 ∨ BF4`, accord `AF2 ∧ BF2`, accord `AF2 ∨ BF2`.
3. Avantage de probabilité du K proposé vs K parent, `max_{s∈{AF4,BF4}} (P_s(cand)-P_s(parent))`, `min`, seuils fixés {0.0,0.05,0.10,0.20}. Conserver S18 en absence de confirmation.
4. `B_before_A` : compatibilité B de la tête S25 initiale soutient le K proposé plus que le K parent, pour démontrer ou invalider la fiabilité de la tête la plus faible.
5. Un classificateur de fiabilité apprenant si un changement S29 est réellement correct, à partir de la concordance AF/BF, des probabilités, du K ancien/nouveau, des longueurs de parcours et des variables acoustiques originales; fit **uniquement sur d'autres morceaux**, sans label dans les entrées. Fixer logistic L2 C=0,1 et HGB régularisé min_leaf=20, max_depth=3, 80 itérations. Cutoffs fixés 0,3,0,5,0,7. Comme S29 conservateur change très peu de cas, entraîner sur *toutes les neuf propositions S29* des autres morceaux, en dédoublonnant événements+proposition+parcours pour éviter que le même cas pèse artificiellement neuf fois. Si le fold de formation ne contient pas deux classes, utiliser les morceaux des autres folds, toujours en excluant le morceau évalué; signaler le faible nombre de positifs et de régressions.

## Sorties imposées

Audit de toutes les propositions, même rejetées : nombre de nouvelles corrections, régressions, neutralités face à S18, compte exact de faux positifs bloqués et vraies corrections perdues, K0–K6, fold et morceau, et **les 21 cas documentés**. Audit A/B par sous-population sans transformer le résultat en règle musicale. Comparer aussi les 9 stratégies S29 et les scores S28 et S18. Toutes les stratégies et poids doivent être sauvegardés. Aucun gain sur ce corpus exposé n'est une promotion : revalidation sur compositions inédites indispensable.

Critère strict : zéro perte des bonnes prédictions de S18 **plus** au moins une nouvelle correction et poly≥40,5958 %. Même un candidat qui satisfait ces critères sur ce corpus doit rester une hypothèse exploratoire, non une preuve générale.