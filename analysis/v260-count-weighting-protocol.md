# V26 : effet de la pondération du comptage

V25 ordinal ne dépasse pas V10.4 et son gain sur le témoin catégoriel est concentré sur le fold 2. Le bilan est archivé dans v250-count-analysis.md.

Hypothèse : la pondération inverse-racine des classes contribue aux fausses présences. Cette causalité reste à tester.

## Comparaison fixée avant entraînement

- Deux bras catégoriels, `weighted` (politique V25 exacte) et `uniform` (sept poids unitaires).
- Même encodeur frais V25 catégoriel, mêmes poids initiaux complets, Adam 0,0002, batch 128, seed 16061 et décalages par fold inchangés.
- Pondérations calculées uniquement sur les labels du fit interne ; cette table s’applique au fit et à la NLL de validation interne. Recalcul sur les quatre folds d’apprentissage pour le fit final. Aucune pondération des scores externes.
- Sélection interne : NLL propre au bras, maximum 20 epochs, patience 4, puis réentraînement depuis zéro. On teste la politique complète de pondération, y compris son effet sur l’epoch choisie.
- Cinq folds externes groupés identiques à V25 ; deux bras entraînés séquentiellement dans chaque job. Maximum cinq jobs en parallèle, 300 minutes chacun.
- Classement, candidats et timestamps V24 figés, sources vérifiées par empreintes et reproduction exacte du score V24 avant entraînement.
- Aucun changement de seuil, de décodeur argmax, de perte auxiliaire ou de jeu d’évaluation. Aucune évaluation du jeu verrouillé ni de la validation historique.

Critère principal : F1 agrégé à 50 ms, avec TP/FP/FN. Diagnostics : confusion K par fold et agrégée, fausses présences K=0, omissions K=1, exactitude en polyphonie, réalisations limitées par les candidats, tolérances 5/10/20 ms. Tous les folds sont publiés ; aucune sélection a posteriori. Une seule seed ne permet pas de conclure à une robustesse statistique.

Le témoin pondéré est réentraîné dans le même run. Les archives GuitarSet déjà vérifiées du run V25 34183897369 sont reprises, leurs MD5 officiels revérifiés et partagées aux cinq jobs. Les résultats V25 servent au contexte et ne choisissent aucun hyperparamètre interne V26.

La synthèse imprime aussi les rapports complets de chaque fold pour permettre l’analyse K par fold depuis les journaux. Les modèles, probabilités et rapports sont conservés en artefacts. Les tests TensorFlow doivent réussir avant les entraînements longs.
