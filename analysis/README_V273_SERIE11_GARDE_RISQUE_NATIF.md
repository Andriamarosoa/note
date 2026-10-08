# V27.3 — Série 11 : garde-fou appris sur les 2 226 régressions natives

## Protocole fixé avant entraînement

**Objectif** : protéger les 2 934 corrections gagnées sur `freeze_local_combo`, réduire les 2 226 régressions de la série 9 et garder des mesures distinctes global / K0 / K1–K6 / poly. La série 10 a identifié 907 régressions de vrai K0, 691 pour la transition freeze 0 → décision 1. Ces résultats motivent une hypothèse, mais toute amélioration mesurée sur les mêmes 59 309 événements de développement reste exploratoire.

## Préservation des modèles et interdiction de fuite directe

Les deux producteurs de série 5, G (`prior_corrected__tilt1.5__freeze__mix1`) et P (`raw__tilt1.5__freeze__mix1`) sont utilisés comme **entrées et contre-exemples**, jamais les prédictions ou scores YourMT3+. Leurs modèles excluent entièrement le fold de l'événement évalué. Le parent série 9 fournit une **action proposée** à corriger, sans entrer dans l'ajustement d'un modèle. Le sélecteur apprend à prédire le vrai K uniquement sur les sorties G/P/freeze d'autres morceaux du même fold, sans jamais recevoir les sorties série 9 de ces autres morceaux comme variables ou labels. Les vraies étiquettes des morceaux évalués servent **uniquement au rapport**. Les fold3 et player05 restent exclus.

Le fichier original `yourmt3-exactk-summary.zip` sert uniquement à aligner les membres, bornes temporelles, folds, vérité déjà épinglée par preuve, et jamais le champ de prédiction du concurrent. Les 58 caractéristiques audio indépendantes sont disponibles sur les événements déjà éligibles de série 7, avec un masque indiquant leur disponibilité ; les événements hors K2–K4 ne bénéficient donc d'aucune acoustique 58-dimensionnelle explicite. Ne jamais utiliser d'identifiant d'enregistrement, de morceau, ou de fold comme variable du sélecteur.

## Bras préannoncés

Sur chaque morceau évalué, modèles entraînés uniquement sur les autres morceaux du même fold et sur les lignes pour lesquelles au moins un producteur G/P diverge du gel initial.

- Entrées : représentation A = vote de classes freeze/G/P + écarts, durée de groupe d'attaque et décalage de l'attaque précédente ; représentation B = A + les 58 caractéristiques audio éligibles et un masque de disponibilité.
- Modèles : régression logistique multinomiale (C=0.1), HistGradientBoosting (learning_rate=0.05, max_iter=100, max_leaf_nodes=15, min_samples_leaf=40, max_depth=4, l2_regularization=10, random_state=27411).
- Chaque modèle produit une distribution des sept K. Le coût de retour à freeze doit être strictement positif : seuils d'avantage de probabilité de **0.00, 0.10, 0.25**.
- Deux périmètres : garde-fou K0 seulement lorsque `freeze_K==0`, et garde-fou sur toutes les classes ; soit **12 variantes** (2 entrées × 2 modèles × 3 seuils × 2 périmètres = **24 variantes**, correction arithmétique explicite). Les deux parents conservés.
- Le décodeur de chaque variante **conserve** la prédiction série 9, sauf quand `P(freeze_K) > P(series9_K) + coût` (et que le périmètre s'applique) ; alors il conserve freeze. Le sélecteur ne crée pas une nouvelle classe, il peut seulement empêcher une modification.

## Critères et conservation

Aucune optimisation du coût sur le morceau évalué ; tous les 24 candidats sont rapportés. Calculer exact-K K0–K6, global/poly, chaque fold, par vrai K, corrections/régressions versus freeze, série 9 et YourMT3+ ; comptes détaillés des corrections perdues / régressions évitées, notamment K0→K1. Archiver les poids des modèles, partitions, paramètres, probabilités prédites et 59 309 décisions. Rejouer les exclusions et identités. **Pas de promotion automatique**. Le parent série 9 a déjà subi une sélection a posteriori sur ces données, donc les scores dérivés ne constituent pas une validation indépendante. Toute généralisation exige une nouvelle cohorte d'enregistrements.
