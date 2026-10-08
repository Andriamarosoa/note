# Série 3 — combinaison apprise des compteurs K0–K6

Protocole fixé après la série 2, avant entraînement de cette série.
Source : run `37837235723`, commit
`34f8bac3686a2f2e4ba1ff288540e10df8a43b67`, terminé avec succès.
Le meilleur global neuf est à 82,8424 % mais seulement 31,0359 % poly.
L'union oracle atteint 93,5558 % global et 77,4137 % poly : la diversité
des bonnes réponses dépasse largement la performance de leur choix.

## Entrées et exclusion du morceau évalué

Pour une ligne du fold F, les 12 distributions neuronales viennent uniquement
des réseaux qui excluent entièrement F. Toutes les lignes de F partagent donc
les mêmes producteurs, sans aucun label de F dans leur entraînement.

Former deux représentations :

- `probabilities` : log-probabilités des 12 producteurs (plancher 1e-7),
  plus le one-hot de la référence initiale, soit 91 valeurs ;
- `context` : les mêmes entrées plus les 58 résumés acoustiques et les 46
  valeurs géométriques, soit 195 valeurs.

Aucune sortie YourMT3+ et aucune identité de morceau ou de personne ne sont
des entrées. Les sorties des correcteurs calibrés précédemment sur des
morceaux de F ne sont pas utilisées pour ajuster ce nouveau modèle.

Pour chacun des 19 morceaux : ajuster le combinateur sur les autres morceaux
du même fold F, puis prédire le morceau exclu. Chaque normalisation et
chaque paramètre du combinateur exclut ce morceau. Il n'y a pas de sélection
interne de configuration : les quatre familles ci-dessous sont fixes.
Ce protocole utilise les labels d'autres morceaux de F ; **ce n'est pas une
évaluation du système final sans aucun label du fold F**. Les données restent
du développement déjà exploré et les choix architecturaux sont adaptatifs.

## Modèles et règles fixes

Deux modèles sur chacune des deux représentations, soit 76 ajustements :

1. Régression logistique multinomiale L2, `C=0.1`, LBFGS, au plus 500
   itérations, tolérance 1e-7 ;
2. `HistGradientBoostingClassifier`, log-loss, 100 itérations,
   learning rate 0.05, au plus 15 feuilles, profondeur 4, feuille minimale
   40, L2=10, arrêt anticipé désactivé, seed 27406.

Perte non pondérée dans les quatre familles. Moyennes et écarts-types
ajustés sur les seules lignes autorisées, valeurs tronquées à ±6.
Si une classe manque dans l'ajustement, conserver explicitement cette
absence ; ne pas introduire d'exemple du morceau évalué pour la compléter.

Chaque distribution est décodée avec deux facteurs fixes sur les classes
K2–K6 : 1 et 1.5, puis renormalisée. Le facteur 1.5 est une règle de décision
favorisant la polyphonie, pas une probabilité annoncée comme calibrée.
Pour chaque facteur, utiliser les poids 0.6, 0.8 et 1 du nouveau modèle,
en mélange avec le one-hot de deux parents figés : `freeze_local_combo`
et le meilleur candidat des 211 politiques initiales. Le second parent
intervient seulement au décodage du morceau évalué, jamais dans les
entrées, la perte ou la sélection du combinateur. Les égalités gardent
le parent. Soit **48 politiques**, dont des alias conservés explicitement.

Les modèles traitent conjointement les distributions complètes : aucun
producteur n'est retiré au motif de sa performance individuelle. Un verdict
peut émerger de leur combinaison même s'il n'est l'argmax d'aucun producteur.

## Conservation et décision

Rapporter chaque politique face à la référence, à l'ancien meilleur candidat
et à YourMT3+, en global, poly, K0–K6 et par fold. Conserver probabilités,
modèles, normalisations, partitions exactes, tous les succès et régressions.
Les 261 politiques antérieures restent présentes : total annoncé **309**.
Le replay doit reproduire toutes les décisions. Une promotion n'est pas
automatique. La validation inédite reste nécessaire, même si les deux
scores de développement dépassent YourMT3+.
