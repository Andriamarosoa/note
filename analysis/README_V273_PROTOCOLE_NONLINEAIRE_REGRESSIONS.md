# V27.3 — Troisième série : frontières non linéaires et complémentarité

Déclaré après les séries 1 et 2. Le routage par K initial réduit les
régressions de 518 à 354 et donne 506 corrections (+152), avec des nets
positifs sur les quatre folds. Protéger 90 % des anciennes réussites dans
la validation interne conserve 485 anciennes corrections en test, mais
donne seulement +53 : le compromis reste important et la protection se
transfère imparfaitement entre morceaux.

Tester une limite précise : les calibrateurs précédents sont linéaires
dans les descripteurs. Apprendre des interactions non linéaires à partir
des mêmes observations et des mêmes exclusions, sans nouvelles propositions.

Deux représentations fixes : (a) K initial, contextes de tous les groupes
complets, logits et disponibilités ; (b) mêmes éléments plus 58 observables
acoustiques. Deux classifieurs d'arbres de gradient par représentation :
risque r et distribution conditionnelle q/OTHER sur les seuls cas initiaux
erronés. Configuration fixe : 80 itérations, learning_rate=0,05,
max_leaf_nodes=7, max_depth=3, min_samples_leaf=40,
l2_regularization=10, early_stopping=False, seed=27402.

Mélanger séparément les facteurs r et q avec les facteurs figés d'origine,
aux poids prédéfinis 0,25 / 0,50 / 0,75. Les probabilités attribuées par
l'arbre à une alternative indisponible sont transférées à OTHER. Les
classes absentes de son apprentissage gardent le soutien du modèle
initial grâce au mélange. Aucune vérité à l'inférence, aucun veto individuel.

Les 6 distributions × 7 coûts donnent 42 politiques supplémentaires.
Évaluer trois sélecteurs imbriqués sur les 49 anciennes +42 nouvelles :
choix global par morceau, routage exact par K initial avec conservation
du nombre à 90 %, puis routage avec conservation des identités à 90 %.
Le test ne sélectionne ni paramètres, ni famille, ni poids, ni coût.
Conserver les 45 politiques, modèles et prédictions internes, y compris
les bilans négatifs. Même limite de supervision par morceau que les
deux séries précédentes ; ce n'est toujours pas une validation inédite.
