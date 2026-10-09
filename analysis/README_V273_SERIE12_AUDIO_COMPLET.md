# V27.3 — Série 12 : audio K0–K6 complet et accord sur le veto

## Protocole arrêté avant le run

La série 10 constate 2 226 régressions contre freeze pour 2 934 corrections, dont 907 régressions K0 et 691 transitions 0→1. La série 11 élimine jusqu'à 1 100 régressions mais perd 806 corrections et dégrade la polyphonie, et ne dispose des 58 caractéristiques acoustiques que pour les 7 493 événements initialement éligibles. La série 12 modifie **exclusivement la disponibilité acoustique et l'accord de veto**.

### Données et non-fuite

Reprendre l'unique cohort de 59 309 événements et les 19 morceaux des folds 0/1/2/4. Les deux sources G et P restent les producteurs `open_k0k6_series5__prior_corrected__tilt1.5__freeze__mix1` et `open_k0k6_series5__raw__tilt1.5__freeze__mix1`, produites avec exclusion du fold prédit. La décision série 9 reste le parent; elle ne doit pas alimenter l'entraînement, mais uniquement la comparaison de classes au décodage et l'audit de vérité.

Lire les 4 archives `v273-open-features-0/1/2/4.tar` du run original 37837235723 (GitHub prerelease). Vérifier les SHA256 enregistrés dans chaque rapport, les 58 valeurs finies, les membres, les folds et les valeurs des 7 493 événements acoustiques déjà audités. Les 58 variables audio sont calculées **sans label** depuis l'audio original ; aucune prédiction YourMT3+ n'est utilisée. Les features `votes_time` du protocole série 11 restent les mêmes ; ajouter la totalité des 58 entrées audio et enlever le masque de disponibilité partielle, désormais constant.

Pour chaque morceau tenu à l'écart, entraîner ses deux modèles uniquement sur les **autres morceaux du même fold**, en ne retenant pour le fit que les lignes où au moins G ou P diffère de freeze. Standardisation fit-only et mêmes hyperparamètres qu'en série 11 (logistic C=.1 et HGB iteration 100, depth 4, leaves 15). Aucune sélection de paramètre par le résultat du morceau évalué.

### 24 politiques déclarées

Sur chacune des décisions série 9 différentes de freeze, un veto restaure freeze seulement lorsque l'estimateur estime `p(freeze_K) - p(series9_K) > seuil`. Les modèles individuels logistic et HGB, le consensus `both` (les deux écarts > seuil), et le consensus `mean` (moyenne des écarts > seuil), sont fixés. Coûts 0.00, 0.10, 0.25. Application au périmètre K0 initial uniquement ou à tous K0–K6 : 4 modes × 3 coûts × 2 périmètres = **24 candidats**, plus le parent série 9 et freeze = 26 décision vecteurs. Aucun modèle ne crée une nouvelle classe d'attaque.

Conserver les 19 exclusions, poids des 38 modèles, probabilités originales, règles, 26 prédictions et audit par K/fold. Compter précisément corrections perdues et régressions évitées vs série 9 et freeze, ainsi que le score poly. Aucun seuil n'est choisi sur les labels évalués, toutes les variantes sont rapportées. Ces données ont déjà servi à la recherche : **pas de validation indépendante, pas de promotion automatique**, même si un score augmente. Un benchmark final devra utiliser des enregistrements inédits et des latences comparables.
