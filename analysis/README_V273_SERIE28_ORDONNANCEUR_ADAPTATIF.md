# V27.3 — Série 28 : apprendre un ordonnanceur « quelle tête en premier et combien de repassages ? »

Ce protocole est fixé avant de mesurer S28. S27 évalue les 14 manières de repasser entièrement ou partiellement le réseau A/B. S28 apprend une **porte d'ordonnancement** qui choisit, dès les seules caractéristiques initiales, l'action de sélection à exécuter. Il ne se base pas sur le vrai K au moment de choisir.

## Ensemble d'actions fixe

0. `series18_parent` (ne lancer aucune nouvelle tête).
1. `series27__full_4__margin0.6` (A-first, quatre A, trois B).
2. `series27__full_2__margin0.6` (A-first, deux A, un B).
3. `series27__B_initial_only__margin0.6` (A-first, quatre A, **un B**).
4. `series27__B_if_delta_005__margin0.6` (A-first, B ne repasse que si distribution change assez).
5. `series27__msg_only_parent_candidate__margin0.6` (A-first, messages B limités à K parent + K candidat).
6. `series27__early_stable_005__margin0.6` (A-first, arrêter les têtes sur événements stables).
7. `series25__pass2__margin0.6` (B-first, appris dans cet ordre, deux A, deux B).
8. `series25__pass3__margin0.6` (B-first, trois A, trois B).

Les décisions S27 proviennent des 19 checkpoints S20 A-first recalculés (source run 37863961918). Les décisions S25 proviennent de 19 checkpoints B-first exclus du morceau évalué (source run 37862417522). Les producteurs excluent les événements de leur fold/leur morceau selon leurs protocoles ; le parent S18 est historique et ses seuils ont été choisis après observation de ce corpus (important biais de sélection).

## Apprentissage de l'ordonnanceur

Pour chaque événement d'un morceau à évaluer, fit uniquement sur les autres morceaux **du même fold**. Caractéristiques au choix de route : les 90 premières variables de signal/votes d'origine (pas de prédictions calculées en exécutant toutes les routes), plus le one-hot 7 du K initial `series18_parent` ; total **97** valeurs. Pas de vraie classe, pas de morceau/fold, pas de sortie YourMT3+, pas de score des experts déjà exécutés comme entrée.

Pour chacune des 8 actions candidates hors parent, deux cibles distinctes, uniquement sur fit :
- `fix` = action correcte alors que S18 parent faux ;
- `break` = action fausse alors que S18 parent juste.

Pour chaque action, modèle multi-sortie `Ridge(alpha=100)` sur les 97 caractéristiques normalisées sur fit seulement, sans changement de budget après analyse. Prédire `Pfix`, `Pbreak` (des **scores approximatifs**, pas une probabilité calibrée) en clip [0,1]. Utilité `U=Pfix-λPbreak`, avec λ fixé à 1, 2, 4. Ne quitter S18 que si meilleur U strictement supérieur à l'un des quatre seuils 0, 0.02, 0.05, 0.10. Au plus une action nouvelle par événement ; aucune vérité n'entre dans la route. Exécuter mentalement l'action choisie : les prédictions sont déjà archivées, sans nouvel entraînement réseau à cette étape. L'ordre/profondeur sont définis par l'action.

**12 politiques S28** + S18 et `freeze`. Compter les décisions A-first/B-first, complètes vs partielles, et les corrections et régressions par vrai K0–K6, fold, global/poly. Comparer directement au meilleur parent S18, à S20/S25 et aux variantes S27. Mesurer l'amélioration de la fiabilité des messages par rapport à une route fixe ; conserver tous les scores et poids du scheduler, les seuils négatifs, les décisions et leur provenance.

## Critère
Ne promouvoir aucune politique qui perd une seule correction historique. La vérité du morceau évalué est utilisée **seulement pour calculer les métriques**, jamais pour entraîner sa porte ni décider sa route. La cohorte a été vue dans de nombreuses séries : une réussite dans cette grille resterait exploratoire et nécessiterait un nouveau jeu de compositions totalement inédites.
