# V27.3 — Deuxième série : préserver les réussites et contextualiser la règle

Fixé après la première série, avant de calculer ces routages. Le dernier
cycle donne 512 corrections / 396 régressions (+116), mais seulement 356
anciennes corrections conservées, 210 perdues et 156 nouvelles. Le garde-fou
portait sur le nombre de corrections, pas sur l'identité des anciennes
réussites. La deuxième série teste explicitement cette différence.

Réutiliser exactement les sorties externes et internes de la première
série. Pour chaque morceau test, les choix ci-dessous n'utilisent que les
prédictions doublement exclues et les vérités des autres morceaux du même
fold. Aucun modèle n'est ajusté à nouveau. Les exclusions et limites de
supervision sont identiques à la première série.

Trois cycles supplémentaires, tous conservés :

1. `source_count90` : choisir une des 49 politiques séparément pour chaque
   K initial (2, 3, 4). Énumérer les 49^3 possibilités sur la validation
   interne. Garder les contraintes de nombre de corrections >=90 %, net
   >=original et régressions <=original. Minimiser les régressions, puis
   maximiser net, anciennes réussites conservées, simplicité lexicographique.
2. `source_success90` : même énumération mais ajouter la conservation d'au
   moins 90 % des identités des anciennes corrections sur la validation
   interne. Cette contrainte n'est pas une garantie externe.
3. `transition_success90` : partir du deuxième routage et permettre un
   choix par couple (K initial, meilleure alternative du croisement initial
   avant seuil). Ce couple est calculable sans vérité. Par descente de
   coordonnées, tester toutes les substitutions de politique d'une cellule,
   retenir la meilleure amélioration stricte de l'ordre précédent qui
   respecte toutes les contraintes. Répéter jusqu'à l'absence d'amélioration
   locale. Seules les cellules avec >=60 événements, >=10 corrections
   initiales et >=10 régressions initiales dans la validation interne
   peuvent s'écarter de leur routage source ; le test ne décide pas du
   support. Limite de protection : 200 substitutions, déclarée explicitement
   si atteinte. Ce n'est pas une preuve de minimum global sur les transitions.

Chaque application choisit une politique complète, jamais un veto sur un
constituant d'une combinaison. Toutes les tentatives finales, les chemins
de recherche et les règles par morceau sont conservés. Le routage utilise
les résultats des audits à l'entraînement, pas la vérité du fragment traité.

L'objectif est d'obtenir un compromis meilleur et de vérifier si la
conservation des succès se généralise. Publier les échecs de cette contrainte
sur le test aussi clairement que les réductions de régressions.
