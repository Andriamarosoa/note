# V27.3 — Diagnostic croisé des facteurs de risque et de correction

Ce diagnostic est fixé **après** le résultat du run 37815209004, avant de
calculer les croisements. Le réseau contextuel a produit 705 corrections,
743 régressions et un gain annoncé de 676,94 pour −38 observé. Sur ses décisions
changées, il estime la justesse du verdict initial à 15,58 %, contre 29,80 %
observé. Ce constat motive un diagnostic des deux facteurs probabilistes.

Calculer **les deux** croisements symétriques, sans fit, recherche de seuil
ou choix par fragment :

1. Risque du parent local + estimation conditionnelle de correction du nouveau réseau.
2. Risque du nouveau réseau + estimation conditionnelle de correction du parent local.

Le risque `r` est la probabilité que le verdict initial soit correct. La
distribution `q` estime, lorsque ce verdict est faux, la bonne alternative
ou OTHER. La distribution recomposée est `P(initial)=r`,
`P(alternative=k)=(1-r)*q[k]`, `P(OTHER)=(1-r)*q[OTHER]`.
Chaque groupe complet reste évalué par `P(correction)-P(régression)` face à
KEEP=0. Les 255 propositions restent identiques.

Les facteurs de chaque événement viennent de modèles ayant exclu son fold
entier ; leurs propositions, votes et audits doivent être identiques. Le
rejeu avec les deux facteurs provenant du même modèle doit retrouver ses
décisions. Les vérités ne servent qu'aux bilans après la recomposition.

Publier résultats globaux/par K, apports/pertes, profil des 126/92, confiance
et comparaison du risque ancien/nouveau **sur les mêmes 2 493 changements du
nouveau réseau**. Conserver les deux sorties, quel que soit le résultat.

Il s'agit d'un diagnostic a posteriori sur des folds déjà exposés, pas d'une
validation sur données inédites ni d'une sélection de modèle finale. Le
croisement mesure l'effet de substituer des facteurs déjà appris ; il
n'identifie pas à lui seul la cause physique de leurs erreurs ou la raison
complète de leur généralisation insuffisante.
