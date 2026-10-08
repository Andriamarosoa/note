# Série24 — inversion réelle de l'ordre des têtes faibles et fortes, à poids fixés

Ce protocole est déclaré avant exécution. Question: si **B** (tête auxiliaire de compatibilité, potentiellement moins performante en Exact-K) parle **avant A** (tête de comptage), l'entrée de A est changée. Comparer à l'ordre `A→B→A`, puis répéter `B→A→B→A` sur les mêmes 59 309 événements, 19 morceaux, folds 0/1/2/4. La référence conservatrice S18 et ses 2 934 corrections restent inchangées.

### Neutraliser les confusions
**Même poids des 19 modèles S20, même signal, même prior S18, même normalisation**, aucune réoptimisation et aucun changement de marge. L'ordre A-first recalcule et reproduit les distributions archivées de S20; B-first commence avec `B(signal,prior,prior)` avant `A(signal,prior,prior,Bmessage)`. Après chaque A, B voit la dernière distribution A et peut réviser ses exclusions. Quatre passages d'A au plus, A/B avec poids partagés. Comparer nombre égal de passages d'A (mais noter que B-first a un appel B de plus). B est entraînée via compatibilité multilabel, pas nécessairement une « pire sélection » au sens de l'accuracy, ce qui doit être **vérifié avec sa prédiction autonome** au lieu de l'affirmer.

### Décisions préfixées
Sorties autonomes B1 (argmax logits BCE, interprété *comme simple diagnostic non calibré*), A-first pass 1/2/3/4 archivées, B-first pass 1/2/3/4 recalculées. Pour chaque sortie B-first pass n, calculer l'argmax K brut et les quatre décodeurs de prudence : revenir au parent S18 si la nouvelle classe n'a pas une marge probabiliste >0.10, 0.25, 0.40 ou 0.60 sur le K du parent. Référence S18, `freeze_local_combo` et YourMT3+ comme comparateurs (YourMT3+ sans entrer dans aucune sélection). La référence B-alone n'est pas une décision promue; elle permet de quantifier la qualité réelle de cette tête faible.

### Audits
1. Preuve par test de changement de variable: la distribution A1 diffère lorsque B parle d'abord, à audio et poids identiques ; la reproduction A-first doit concorder (< 5e-5) avec le run original S20, et **les 19 checkpoints d'origine** doivent correspondre à leurs pièces exclues et aux scalers archivés.
2. Exact-K global, poly K2–K6, chaque vrai K0–K6, chaque fold, **corrections et régressions séparées** contre S18, freeze et A-first à passage égal, plus changements neutres.
3. « La mauvaise tête peut promouvoir la bonne »: parmi les événements où B1 est faux, compter ceux que A_Bfirst1 rend corrects, et les événements auparavant corrects B1 qu'A détériore, avant d'affirmer que commencer par la tête la plus faible est utile.
4. Enregistrer poids source SHA256, chaque prédiction et distribution, tous les changements et les variantes négatives. **Zéro promotion** si les corrections déjà acquises sont perdues; la cohorte est une cohorte de développement déjà exposée, pas un test inédit.

L'expérience ne prétend pas que le simple échange d'ordre est optimal : S20 a été entraîné dans l'ordre A-first. Si B-first n'améliore pas les résultats à poids fixés, la suite nécessaire est **l'entraînement symétrique de l'ordre inverse** avec un protocole de séparation identique, et non une conclusion générale sur l'idée.
