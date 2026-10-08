# V27.3 — Diagnostic du sélecteur cohérent, K3/K4

**8 octobre 2026. Diagnostic après résultats, interventions fixées avant leur
exécution. Aucun nouveau modèle candidat n'est entraîné dans cet audit.**

Le run 37766719862 a produit un gain net de +31 face à `freeze_local_combo`,
avec 405 régressions de vrai K3 et 189 de vrai K4. L'objectif est de documenter
ces 594 événements, de les comparer aux corrections réussies pour les mêmes
transitions et de mesurer la sensibilité du modèle gelé aux entrées.

## Comparaisons descriptives

- Transition définie par la même classe initiale et la même destination.
  Une correction a pour vérité la destination ; une régression a pour vérité
  la classe initiale. Les autres vrais K sont des changements neutres.
- Probabilités annoncées de correction/régression et gain attendu comparés
  aux fréquences observées, globalement, par transition et par fold.
- Tranches de marge fixées à 0/0,05/0,10/0,20/0,40/1. Aucun choix de seuil
  ne sera présenté comme un résultat validé.
- Pour les transitions ayant au moins 20 corrections et 20 régressions,
  comparaison des 58 observables acoustiques : médianes, différences
  standardisées et AUC descriptive. Le signe est contrôlé par fold quand
  les deux groupes y comptent au moins cinq événements. Il s'agit de
  descriptions conditionnées par la décision, sans preuve causale.
- Attention sur les combinaisons, soutien des cinq spécialistes, contexte
  sélectionné et recouvrement avec les régressions de l'ancien sélecteur.
  Les poids d'attention ne sont pas assimilés à une importance causale.

## Rejeu des poids et interventions

Les poids des quatre folds du bras `coherent` sont chargés sans entraînement.
Les producteurs d'entrée sont reconstruits avec les versions numériques
d'origine. Le diagnostic s'arrête si les probabilités ou attentions diffèrent
de plus de 2e-6, si une prédiction diffère, ou si la provenance diffère.

Permutation sans labels, à l'intérieur du même fold de test et de la même
classe initiale. Les donneurs sont un décalage circulaire non nul des IDs
triés, aux seeds 27403/27404/27405. Les groupes sont fixés :

- contexte complet, puis spectral, naissance, amortissement, persistance ;
- descripteurs directs des votes, canaux 0–5 ;
- taux des audits, canaux 7–9 et 18–20 ; les taux globaux constants à source
  fixée ne changent pas avec cette permutation ;
- métadonnées de repli ;
- identité des sous-ensembles, attendue inchangée : contrôle négatif exact.

Les permutations conservent les distributions marginales mais peuvent rompre
les relations entre blocs. Elles mesurent la sensibilité du réseau gelé,
pas l'effet physique d'un signal sur le son ni la performance d'un modèle
réentraîné sans ce signal.

Une intervention distincte examine les deux canaux de volume d'audit (6 et
17). Les références occupent deux folds pour les récepteurs d'entraînement
et trois folds pour le test externe. La transformation fixée est
`log1p(expm1(compte_log) * 2/3)`, sans modification des taux, des poids ou du
décodage. Ce ratio découle du nombre de folds et n'est pas choisi sur les
scores. Il n'égalise pas exactement les effectifs par classe/régime ni la
qualité des producteurs. Son résultat sera une sensibilité descriptive,
pas une nouvelle correction validée.

## Sorties et limites

Les sorties incluent la matrice des 7 493 événements admissibles, les 594
régressions K3/K4 avec IDs/enregistrements/temps/observables, les contrastes
de caractéristiques, les probabilités des replays et les bilans par fold.

Les archives sources sont contrôlées par leurs SHA-256. Les tests vérifient
les donneurs, le périmètre exact des interventions, l'absence de modification
des entrées sources, les égalités de décision et le décompte des neutres.

Les folds 0/1/2/4 restent exposés ; aucun player05/fold3, nouvel audio ou jeu
indépendant n'est utilisé. Aucune promotion, fusion, sélection d'un nouveau
seuil ou affirmation de validation indépendante n'est autorisée par ces
résultats. `freeze_local_combo` reste la référence.
