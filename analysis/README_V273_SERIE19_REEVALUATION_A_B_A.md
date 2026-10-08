# V27.3 — Série19 : sélection neuronale récurrente A→B→A→B→A

## Objectif (protocole fixé avant le run)

L'idée testée est précise : si **B** estime qu'une classe est incompatible (par exemple K4), **A doit recalculer sa distribution K0–K6** à partir du signal acoustique, de sa sortie précédente **et du message de B**. Cette nouvelle distribution retourne à B, puis le cycle est répété. Le message B n'est pas un veto binaire final : il modifie **les variables d'entrée de A**, ce qui peut réhabiliter une classe auparavant contestée. Il est entraîné dans le même réseau par rétropropagation à travers les quatre passages.

### Architecture et contrôles

- A : petit réseau MLP à deux couches cachées, entrée 335 variables acoustiques/votes + 7 probabilités précédentes + 7 messages de B, sortie **logits des sept K**.
- B : MLP à deux couches, entrée 335 variables d'origine + sortie probabiliste actuelle de A, sortie **7 scores de compatibilité souples** (tanh des logits). Chaque exclusion est donc révisable au passage suivant.
- Unroll **quatre passages** avec poids A/B partagés. Pertes supervisées à tous les passages, BCE auxiliaire pour que B apprenne les compatibilités. Aucun masque construit avec la vérité durant l'inférence.
- Baselines/ablation préfixées : A seul **passage1**, A→B→A **passage2**, 3 passages, **4 passages**, et 4 passages où l'entrée issue de B est neutralisée (mais la prédiction précédente de A reste transmise).
- Test contrefactuel : pour les événements K4 sélectionnés initialement par A, injecter un message artificiel **« B exclut K4 »** à la deuxième passe, **à probabilités initiales et acoustique identiques**, et mesurer combien la distribution de A change. Ceci teste réellement la causalité du *message numérique* sur l'algorithme de A ; cela **ne prouve pas que le message apprend une vraie règle musicale**.
- Pas de `freeze` forcé ni de posthoc `YourMT3+` dans la décision. La série18 conserve sa place de référence indépendante du nouveau réseau. Ne modifier aucune politique de production.

### Données, séparations, entraînement

59 309 événements natifs, folds 0/1/2/4, 19 morceaux. Récupérer les signatures sources de 4 archives SHA256, construire les 335 variables existantes (=32 votes et temporalité, 58 audio, 245 flux harmonique). Sources `freeze,G,P` viennent des prédictions hors-fold initiales, mais l'état posthoc série18 n'est **pas une entrée** du réseau récurrent. Pour **chaque morceau évalué**, entraîner sur tous les événements des **autres morceaux du même fold**, donc jamais sur sa vérité. Normaliser les variables à partir des seules lignes de fit. Aucun identifiant de morceau/fold et aucune décision YourMT3+ dans les caractéristiques. Les cibles du morceau évalué servent uniquement au rapport.

Hyperparamètres fixes : PyTorch CPU, seed 27319, AdamW LR 1e-3, batch 512, six epochs, weight_decay 1e-3, clipping grad 2, hidden A (128,64), B (96,64), pondération par classe issue des seuls événements de fit avec plafonds [0.3,4] et moyenne 1. La somme des pertes CE des passages 1/2/3/4 est pondérée 0.20/0.30/0.50/1.00, perte BCE B de poids 0.06 par passage. Limiter les threads CPU à 2. Pas de réglage après observation du résultat test. Produire les 19 modèles et toutes les probabilités par passage, par événement.

### Audit et verdict

Comparer les cinq politiques avec `freeze_local_combo`, le parent conservateur série18 (49 178 global corrects, 2 998 poly corrects ; corrections 2 934, régressions 2 210) et YourMT3+ (51 328 global, 4 028 poly). Rapporter 7 Exact-K, global et poly, gain/perte **à chaque passage**, transition de l'état K précédent, régressions et corrections vs série18 / freeze, par fold et par vrai K. Conserver également la matrice de transition et un test qui prouve que A2 reçoit le message perturbé B.

**Limitations :** la cohorte et ses labels ont servi aux recherches précédentes, notamment au choix du parent série18 ; pas de validation complètement nouvelle, pas de preuve de gain YourMT3+ avant le run, pas d'assurance de latence embarquée. Ne pas promouvoir si les nouvelles prédictions détruisent les corrections ; toujours archiver chaque candidat positif/négatif.