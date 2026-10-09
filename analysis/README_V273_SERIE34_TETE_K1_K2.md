# Série34 — tête spécialisée apprenant si une DEUXIÈME note est réelle (K1 ↔ K2)

**Préenregistrée avant le run de série34.** La série33 a identifié, sur 26 changements S29, 14 corrections K2→K1 lorsque K réel est 1, mais **3 régressions K2→K1 lorsque K réel est 2**. Une autre régression va K1→K2 alors que K réel reste 1. La classe vraie de l'événement à prédire n'est JAMAIS une condition de décision.

## Hypothèse falsifiable

Un expert acoustique spécialisé dans la distinction `une source musicale` (K1) ou `deux sources` (K2) peut aider A/B à réinterpréter une composante parasite comme une note supplémentaire réelle. Entraîner uniquement sur d'**autres morceaux** pour chaque morceau évalué, à partir des événements vrais K1 et K2 de formation (pas seulement 26 événements de test). C'est une **nouvelle sélection apprenante, conditionnelle à une contradiction K1/K2**, pas une liste d'exceptions ni des règles fixes selon l'ID ou le vrai K.

La correction K1/K2 est la plus grande source de gains/régressions S29 ; les autres transitions (K3→K2, K3→K4, etc.) restent inchangées par cette ablation. Cela vérifie le rôle de la tête spécialisée dans la boucle de sélection, mais n'est **pas encore** un ordonnanceur réentraîné conjointement.

## Grille avant mesure

Trois vues acoustiques originalement présentes et chaque pièce exclue du fit :
- `audio58` = 58 résumés du signal, X[:,32:90] ;
- `flow245` = 245 variables de flux/harmoniques, X[:,90:335] ;
- `all335` = 335 variables audio/votes/flux réunies.

Deux experts : `LogisticRegression(C=0.1, class_weight=balanced, solver=liblinear)` ou `HistGradientBoostingClassifier(max_iter=150, max_depth=4, max_leaf_nodes=25, min_samples_leaf=80, l2_regularization=10, learning_rate=.05)`. Variables X standardisées par fold/morceaux du fit seulement. 19 × 3 × 2 modèles si entraînement sur chaque morceau, ou seulement les morceaux portant une transition K1/K2 S29 à filtrer (les autres ne reçoivent aucune nouvelle sélection).

**Quatre seuils fixés** {0,35; 0,50; 0,65; 0,80}. Pour les décisions S29 K2→K1, n'accepter le candidat que si `P(K1|X)>seuil`. Pour les décisions K1→K2, n'accepter que si `P(K2|X)>seuil`. Pour les autres, garder la décision S29. **24 politiques nouvelles**, plus parent S18/S29 et meilleur S31 à comparer. Aucun oracle/vrai K du morceau évalué n'entre dans la sélection. Ce protocole a été motivé par l'audit de la même cohorte : résultat **exploratoire et non indépendant**, même si l'apprentissage par morceau exclut le test.

## Audit obligatoire

Toutes les nouvelles corrections/régressions/neutralités vs S18, par K0–K6, fold, morceau et transition ; auditer les 14 gains K1 et les quatre régressions S29 un à un, avec score P(K1) et statut accepté ou bloqué. Enregistrer chaque modèle et scaler ainsi que les identifiants fit/held. Ne promouvoir aucun modèle sans validation sur composition inédite de bout en bout. Un taux zéro-régression obtenu sur ces quatre erreurs connues ne constituerait pas une garantie.
