# V27.3 — Série 14 : discrimination conditionnelle K0→K1, avant apprentissage

## Fait observé

L'audit série 10 attribue **691 des 2 226 régressions** du parent série 9 à des événements dont `freeze_K=0` et `serie9_K=1`, alors que le vrai K est 0. La série 13 propose trois corrections supplémentaires, sur 59 309 événements exposés, avec le garde `p(K0)>0.975` et 0 correction perdue. L'analyse des mêmes scores est postérieure aux séries 10–13 : **ne pas considérer la série 14 comme validation indépendante ni affirmer que ses seuils sont préenregistrés avant tout travail de recherche.**

## Expérience avec une nouvelle distribution d'apprentissage

Contrairement à série 13, entraîner les modèles UNIQUEMENT sur les événements pour lesquels `freeze_K=0` et `serie9_K=1`. La cible `correct_regression=(truth_K==0)` est lue uniquement dans les morceaux du même fold **autres que le morceau évalué**. Les producteurs source G/P n'ont pas été entraînés sur le fold évalué. Ne jamais utiliser la vérité, les scores YourMT3+ ou l'identifiant du morceau parmi les entrées.

Caractéristiques : les 58 résumés acoustiques et les votes de classes et intervalles natifs (`summary`), ou ces variables plus la trajectoire de naissance/dissipation 42×49 déjà extraite dans série 13 (`flow`). Les seules actions autorisées sont :
- Garder la prédiction série 9 ;
- Ou revenir à `freeze_K=0` sur la seule transition 0→1, si le modèle estime `P(truth_K=0)>t`.

Algorithmes et hyperparamètres identiques à série 13 : régression logistique C=.1 et HGB 100 itérations, profondeur 4, 15 feuilles et régularisation 10. Aucun réglage sur le morceau évalué.

Les 6 mécanismes de décision pré-déclarés sont : logistic summary, HGB summary, logistic flow, HGB flow, consensus des deux modèles flow (minimum de leurs probabilités K0), moyenne des deux flow. Seuils fixes de probabilité **0.75, 0.85, 0.90, 0.95, 0.975** : 30 variantes et deux parents (série 9 et freeze) sans sélection silencieuse.

Pour chaque variante : scores Exact-K global, poly, K0–K6, corrections/régressions face à série 9 et freeze, détails K0→K1 par fold, tous les poids et 59 309 prédictions. Si un morceau a insuffisamment de données d'apprentissage, conserver le parent et enregistrer la raison. Refuser toute promotion automatique. Exiger une vérification d'intégrité et une validation ultérieure sur de nouveaux morceaux.
