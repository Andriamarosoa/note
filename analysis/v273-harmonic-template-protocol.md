# Contrôles des gabarits harmoniques — protocole exploratoire

Point de départ : audit acoustique du commit `0bc2c59`, 488 cas internes et
272 vrais K3. Le cas 7636 a servi de vérification préalable : les pics du
spectre sont proches des trois fréquences annotées ; changer la décroissance
des harmoniques permet au triplet fourni avec annotations de les retrouver.
Ce cas et ces folds sont déjà examinés : aucune validation indépendante n'est
revendiquée.

Objectif : distinguer la rigidité des amplitudes harmoniques, la forme des pics
et le fonctionnement numérique du solveur. Aucun entraînement neuronal et
aucun usage du fold 3. La population, les fenêtres et le spectre positif
post-moins-pré restent figés.

Contrôles décidés avant l'exécution sur la cohorte complète :

1. Garder les huit candidats reconstruits de l'audit, puis ajouter les deux ou
   trois fréquences annotées. Toutes les variantes reçoivent la même liste.
2. Comparer les six combinaisons : forme gaussienne d'écart-type 18 Hz ou
   puissance exacte d'une fenêtre Hann ; décroissance des hauteurs des pics
   `1/sqrt(h)`, `1/h` ou `1/h²`. Les trois pentes sont un test de sensibilité,
   pas des valeurs sélectionnées sur un score de validation. Une pente de
   puissance `1/h` correspondrait à des amplitudes sinusoïdales `1/sqrt(h)` ;
   cela ne prouve pas que le gabarit historique avait cette interprétation.
3. Pour les deux formes, comparer aussi un gabarit à coefficients harmoniques
   libres non négatifs, en gardant exactement trois groupes F0 choisis. Ce
   contrôle a davantage de paramètres et peut favoriser les sous-harmoniques :
   un résidu réduit n'est pas une preuve de meilleure identification.
4. Mesurer la correspondance univoque à 55 cents avec les notes attendues,
   les résidus des meilleurs couples/triplets et le résidu forcé sur toutes les
   fréquences annotées. Présenter tous les groupes et tous les folds internes.
5. Contrôle positif synthétique : somme des gabarits historiques aux fréquences
   annotées avec coefficients connus, puis même recherche exhaustive. La
   capacité à retrouver cette solution teste le solveur dans son propre modèle.
6. Vérifier les résidus du bras historique contre le diagnostic déjà publié,
   et les solveurs contre une résolution NNLS indépendante sur des cas contrôlés.

Les fréquences annotées sont exclusivement diagnostiques. Aucun résultat de
ces bras ne constitue un gain d'Exact-K utilisable en prédiction. Une variante
ultérieure sans annotation doit être mesurée sur les mêmes FIT/VAL, avec
routage de base figé et choix des réglages exclusivement sur FIT.

## Étape sans annotation, décidée après le diagnostic des gabarits

Le diagnostic complet retrouve 0/272 triplets K3 avec la pente historique,
32/272 avec `1/h` et 85/272 avec `1/h²`, en gardant la gaussienne. La fenêtre
Hann seule reste à 0/272. Le contrôle synthétique retrouve 488/488 mélanges.
On teste donc uniquement les pentes dans le pipeline sans annotation : les
neuf combinaisons de pente de saillance et de pente du gabarit parmi 0,5 / 1 / 2.
Grille F0, pool de huit candidats, NMS, gaussienne, fenêtres et normalisation
restent figés. Aucune fréquence annotée n'entre dans cette étape.

Pour chaque fold externe interne (0, 1, 2, 4), choisir une combinaison uniquement
sur FIT : validation tournante de la petite LR sur les trois folds restant dans
FIT, en gardant les populations et le compteur de base d'origine figés. Classer
par gain net Exact-K, puis moins de régressions, puis moins d'actions, puis ordre
de la grille. Inclure l'abstention : si aucun gain FIT strictement positif,
conserver la base. Réajuster ensuite cette LR sur tout FIT et appliquer à VAL au
seuil 0,5. Toutes les actions sur les autres vrais K restent comptabilisées.
Rapporter également les neuf variantes fixes sans en sélectionner une sur VAL.

Ceci est une sélection imbriquée du correcteur sur des folds déjà examinés,
avec réseau et routage figés, pas une nouvelle validation indépendante de
l'ensemble du système. Les résultats ne déclenchent aucune promotion automatique.
