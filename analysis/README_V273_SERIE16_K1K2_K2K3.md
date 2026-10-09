# Note/V27.3 — Série 16 : audit et garde-fous ciblés K1→K2 et K2→K3

## Contexte observé (établi AVANT ce run)

Référence de développement conservée : `series15__safe_OR` (run 37855272619), 59 309 événements, 2 934 corrections et 2 219 régressions face à `freeze_local_combo`, exact global 82,9031 % et poly 40,4739 %. Deux directions de régression dominent encore : 319 cas `freeze_K=1→prediction_K=2` et 204 cas `freeze_K=2→prediction_K=3`. L'hypothèse à vérifier est qu'un arbitre probabiliste **spécialisé par transition** protège davantage de décisions correctes que les garde-fous globaux des séries 11/12.

Les séries précédentes ont déjà exposé les mêmes labels : **ni la série 16 ni les seuils choisis sur celle-ci ne constituent une validation indépendante**. Ne pas promouvoir automatiquement.

## Producteurs et exclusions

Les entrées sans labels sont : prédictions `freeze`, producteurs `G` / `P` de la série 5 exclus du fold évalué, temporalité de la fenêtre de départ, 58 résumés acoustiques et 245 composantes de trajectoire harmonique pré/on/late + flux sans labels extraites en série 4. Source : quatre archives `features-fold-{0,1,2,4}.npz` avec contrôle SHA256 et intégrité des ID. **Ne pas utiliser de prédiction YourMT3+, de vérité, d'identité de morceau ou de fold comme variable d'entrée.**

La décision `series15__safe_OR` sert uniquement à déterminer **quelle proposition examiner** et à l'audit, non comme label ou entrée du réseau. Deux ensembles d'apprentissage ciblés indépendants :
- direction 1 : événements `freeze_K=1` ET `series15_K=2` ; cible binaire « freeze vaut vérité ».
- direction 2 : événements `freeze_K=2` ET `series15_K=3` ; même cible.

Pour chaque morceau tenu hors fit, entraîner sur les **autres morceaux du même fold** seulement. Le parent série15 est issu de décisions posthoc sur les données exposées : noter cette limite d'indépendance, même avec exclusion du morceau dans l'arbitre. Vérifier la présence des deux étiquettes binaires, sinon maintenir le parent et consigner le déficit. Conserver tous les identifiants de fit/évaluation, paramètres et poids.

## Bras, décisions et métriques annoncés

Entrées `summary` (32 votes/géométrie + 58 audios) et `flow` (idem + 245 trajectoires). Pour chaque direction, entraîner LogisticRegression C=0.1 ou HistGradientBoosting 100 itérations, profondeur 4, 15 feuilles, min leaf 40, L2=10, random_state=27411. Ajouter les deux modes de vote flow (`both` = min probabilité ; `mean` = moyenne) = **6 scores par direction**.

Six seuils de probabilité `P(vrai_K = freeze_K)` fixes **0.75, 0.85, 0.90, 0.95, 0.975, 0.99**. Si la probabilité dépasse le seuil, revenir à freeze seulement sur le couple de transition d'origine ; ne créer **aucune** nouvelle classe. Pour chaque score et seuil enregistrer **une politique pour chaque direction**, ainsi que leur union lorsque les deux conditions sur les événements disjoints s'appliquent : 6 × 6 × 3 = **108 variantes** + parent série15 et parent freeze.

Calculer scores Exact-K globaux/poly et tous K0–K6, pour chaque fold, corrections/regressions versus série15, `freeze`, et YourMT3+; mesurer explicitement corrections préservées et nouvelles régressions, pas seulement nets, et consigner tous les cas. Ne supprimer aucun candidat négatif. Une politique n'est admissible à une **étape de validation supplémentaire** que si `corrections_lost == 0`, `regressions_avoided > 0` et `exact poly` non diminué. Les valeurs positives sur les données exposées demeurent exploratoires.
