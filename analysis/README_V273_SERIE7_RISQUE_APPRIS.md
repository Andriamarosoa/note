# V27.3 — Série 7 : boucle d'arbitrage de risque appris

Protocole figé **avant** l'entraînement ou la lecture des résultats de cette série.

## Objectif et sources
À partir de la série 5, deux décisions déjà entraînées : `open_k0k6_series5__prior_corrected__tilt1.5__freeze__mix1` (parent global G) et `open_k0k6_series5__raw__tilt1.5__freeze__mix1` (parent poly P). Comparaison avec `freeze_local_combo` et le checkpoint fixé YourMT3+ : 51 328 / 59 309 corrects globaux et 4 028 / 7 385 corrects polyphoniques. Cohorte déjà exposée, folds 0/1/2/4, fold 3 et player05 exclus.

## Séparation et supervision
Sur les 7 493 fragments où la prédiction freeze est K2/K3/K4, apprendre seulement lorsque G et P divergent. Les trois labels d'entraînement sont : seul G correct, seul P correct, aucun des deux correct. Le label réel n'entre dans aucune décision sur le morceau évalué.

Pour chaque morceau évalué, ajuster le sélecteur **uniquement** avec d'autres morceaux du **même fold**. Les prédicteurs G/P ont exclu entièrement ce fold de leur entraînement. Ne jamais utiliser des exemples du morceau cible pour fit, normalisation, choix de paramètres ou coût. Conserver pour chaque apprentissage les identifiants, partitions et poids. Les morceaux sans observations d'entraînement suffisantes gardent le parent G.

## Bras et combinaisons annoncés
- Deux représentations : vote seulement (one-hot G, P, freeze + écarts), et vote + 58 caractéristiques acoustiques.
- Régression logistique, coefficients C = 0.03 et 0.3 ; normalisation fit-only ; aucun modèle extérieur ni score YourMT3+ dans les entrées.
- Calcul d'avantage : P(seul P correct) − P(seul G correct). Basculer vers P si avantage dépasse le coût fixé : 0.00, 0.10, 0.20.
- Les deux parents et **12 politiques** (2 × 2 × 3) sont conservés sans sélection rétroactive des variantes.

## Mesures
Exact-K global, Exact-K poly, K0–K6, chaque fold, corrections et régressions appariées face à G, P, freeze et YourMT3+. Conserver les poids de chaque sélecteur, probabilités d'avantage, propositions, bonnes corrections perdues et résultats même négatifs. Aucune promotion automatique. Cette série sur données déjà examinées **n'est pas une validation indépendante**. Le score doit dépasser YourMT3+ à la fois globalement et en polyphonie, puis être confirmé sur des morceaux inédits.
