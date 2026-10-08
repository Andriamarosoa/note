# V27.3 — Série20 : récursion A/B à résidus ancrés, sans effacer le modèle conservateur

Protocole écrit avant le run S20. **Cohorte déjà consultée** (les séries 9–19 ont exploité les étiquettes) ; aucun résultat ne constitue une validation externe. Ne pas promouvoir automatiquement.

## Pourquoi un second essai

S19 démontre que le message B modifie A en contre-factuel : parmi 1048 événements initialement favorables à K4, l'intervention B négative entraîne 68 changements de classe de A2 et réduit P(K4) pour 839. Mais le modèle neuf perd des prédictions correctes : 82,3096 % global et 35,1659 % poly pour A→B→A, contre 82,9183 % et 40,5958 % pour la référence S18. La récursion doit donc apprendre à protéger les corrections antérieures, non réapprendre toutes les distributions.

## Nouvelle architecture (ablation)

La référence S18 reste la seule proposition initiale, sans être écrasée automatiquement. À chaque événement, calculer un **prior souple** : 0.85 pour la classe S18, 0.025 pour chacune des six autres classes. A reçoit 335 caractéristiques signal originales + prior K0–6 + prédiction précédente + message B précédent. A produit des **logits résiduels** ajoutés aux logits du prior ; B lit le signal, la proposition de A et le prior, et émet sept scores de compatibilité modifiables. Ces scores sont réinjectés à la prochaine exécution de A. Aucun hard masking : B peut réintégrer une classe lors des passages suivants.

Recalculer quatre passages en partageant les mêmes poids A et B, gradients propagés sur les 4 dépliages, CE pondérée aux 4 passages et BCE auxiliaire des compatibilités. Ajouter aux événements d'entraînement **où la référence S18 est correcte** une pénalité CE de conservation de sa classe afin de réduire les régressions. Jamais appliquer cette pénalité avec la vérité du morceau évalué (exclusion complète). Les poids A/B et les contraintes sont apprises sans supervision d'exclusion explicite pendant le passage final.

## Données/fit

59 309 événements, folds 0,1,2,4, 19 morceaux, une exclusion par morceau. Fit du scaler sur autres morceaux du même fold ; 335 features audio/votes d'origine, SOURCE G/P hors-fold ; seul nouveau conditionnement est la décision S18 initiale. Cette décision est **post-hoc sur le corpus étudié** (limite majeure de l'indépendance). Ni prédiction YourMT3+, ni étiquette ni identité du morceau dans les variables.

Hyperparamètres fixés : PyTorch CPU, seed 27320, AdamW LR 0.001, 6 epochs, batch 512, poids CE passage (0.2,0.3,0.5,1.0), poids BCE 0.06, poids CE de conservation 1.0 sur les parents corrects, clipping gradient 2. Utiliser les mêmes dimensions cachées A/B que S19 ; sortie résiduelle A initialisée à zéro pour ne pas déstabiliser le prior au début.

## Décision et audits figés

Conserver la sortie S18 inchangée et évaluer **brut** et **avec marge de correction** : pour passes 1/2/3/4, des variantes si la probabilité de la classe proposée dépasse celle du parent d'une marge fixe 0.10, 0.25, 0.40, 0.60, sinon S18 inchangée ; une ablation passage4 avec messages B neutralisés aux mêmes marges. La marge ne reçoit **jamais** la vérité. Le gel S18 et les variantes sont archivés.

Auditer global, poly K2–K6, K0–K6, folds, corrections/régressions strictement appariées contre freeze et S18, et matrices de transitions entre passages. Provoquer sur les mêmes entrées une exclusion artificielle de K4 et vérifier que A2 change ; observer aussi les réintroductions de K4, non seulement les suppressions. Archiver les 19 modèles, la distribution A à chaque passage, la sortie B à chaque passage et toutes les corrections même négatives. **Ne jamais choisir une marge sur les étiquettes des événements évalués pour prétendre que le résultat est indépendant.**