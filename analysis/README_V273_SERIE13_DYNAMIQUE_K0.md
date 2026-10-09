# V27.3 — Série 13 : détecter une nouvelle naissance K0→K1 par dynamique harmonique

Protocole fixé **avant entraînement**. Le diagnostic de série 10 (907 régressions K0, dont 691 transitions freeze0→K1) et les séries 11/12 insuffisantes motivent ce nouveau bras. **Ne pas présenter une amélioration sur la cohorte déjà exposée comme validation indépendante.**

## Hypothèse testable
Sur la même fenêtre acoustique d'environ -80 à +160 ms, une **nouvelle attaque sonore** peut se distinguer du son présent par la croissance et la persistance de composantes, pas uniquement par l'énergie instantanée. Les 42 trames × 49 composantes harmoniques disponibles dans les archives d'origine servent de proxys physiques, sans postuler que chaque composante est une note réelle.

## Entrées et supervision
- Garder les 59 309 événements, les seuls folds 0/1/2/4, les 19 morceaux et l'archive audio originale du run 37837235723. Aligner ID, membres, vérité K, freeze, vérification SHA256 identique à la série 12.
- Apprendre uniquement sur les événements dont **freeze_K=0** des morceaux d'apprentissage. Cible binaire `is_no_new_attack=(truth_K==0)`, lue seulement dans les morceaux d'entraînement. Les morceaux évalués sont exclus de tout fit, scaler, calibrage et choix. Aucun label, identifiant de morceau/fold ni prédiction YourMT3+ comme feature.
- Deux représentations : A = votes G/P/freeze + durée et proximité précédente + 58 moyennes acoustiques d'origine ; B = A + profils temporels harmonico-énergétiques (moyennes pré/attaque/fin par 49 composantes, flux positif d'attaque et différence positive d'attaque, fixés par les trois fenêtres temporelles sans filtre adapté à la vérité).
- Régression logistique (C=0.1) et HistGradientBoosting (learning_rate=.05, max_iter=100, max_depth=4, max_leaf_nodes=15, min_samples_leaf=40, l2=10, random_state=27411), identiques à la série 12.

## Décision préannoncée
Partir du parent série 9, et ne modifier que les événements dont freeze_K=0 et série9_K>0. Revenir à freeze_K=0 si la probabilité **apprise hors morceau évalué** d'absence de nouvelle attaque dépasse un des seuils fixes 0.85, 0.90, 0.95, 0.975.

Quatre voies = logistic A, HGB A, logistic B, HGB B ; ajouter les deux voies de consensus sur B : `both` (les deux probabilités dépassent le seuil) et `mean` (leur moyenne dépasse le seuil). Soit **6 × 4 = 24 variantes** + parent série 9 et freeze. Aucune sélection rétroactive.

## Évaluation et conservation
Exact-K global/poly/K0–K6, chaque fold, corrections et régressions par vrai K et transition, particulièrement freeze0→K1, face à freeze et série 9. Conserver les modèles, entraînements, identifiants, probabilités de naissance, tous les vecteurs de prédiction et même les résultats négatifs. Aucune promotion, aucune validation indépendante, ni preuve de causalité ou de latence équivalente.
