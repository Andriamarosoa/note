# V27.3 — Série 15 : conservation cumulative des corrections K0, protocole post-hoc explicite

Les scores série 13 et série 14 ont été examinés sur la même cohorte **avant** cette combinaison ; celle-ci est donc un **audit de conservation exploratoire**, pas une préinscription ni une preuve de validation indépendante.

Cible : combiner les deux gardes (13 `flow_logistic p0>0.975` et 14 `flow_mean p0>0.9`) qui ont individuellement donné zéro correction perdue sur la cohorte observée. Les deux gardes sont des classificateurs dont la formation exclut le morceau évalué, mais le choix de leurs seuils et de leur combinaison a bénéficié de l'observation des données de développement.

## Sources et assertions

- Série 13 : run 37854384791, politique `series13__flow_logistic__p0_gt0.975`, 3 corrections / 0 régression face à série 9 ; parent `series9_parent`.
- Série 14 : run 37855070118, politique `series14__flow_mean__p0_gt0.9`, 4 corrections / 0 régression face à série 9 ; parent `series9_parent`.
- Conserver les variantes témoin `series13__flow_logistic__p0_gt0.9` et `series14__flow_logistic__p0_gt0.9` qui corrigent davantage mais perdent des corrections.
- Vérifier identité des 59 309 ID, folds et labels, et l'égalité bit-à-bit des parents série 9.
- Construire des politiques OR (revenir à freeze si une des têtes le recommande) et AND (revenir à freeze seulement si les deux le recommandent) pour ces deux paires, sans aucun nouveau paramètre optimisé. Les deux variantes individuelles et le parent sont conservés.

## Mesures exigées

Corrections, régressions, et changements neutres contre série 9 et freeze ; séparation vrais K0–K6 et folds, éventuel recouvrement des événements corrigés, tous les IDs et décisions modifiées. Les combinaisons ne peuvent changer que le sous-ensemble déjà modifié par un garde 13/14, aucune nouvelle classe n'est permise.

Si OR conservatrice apporte un gain >0 sans régression, conserver sa prédiction comme **candidat** supplémentaire. L'archive contient également les cas négatifs et les deux contrôles. Ne pas promouvoir ni généraliser : manque une validation sur morceaux nouveaux et la comparaison à YourMT3+ est toujours défavorable.