# V27.3 + décroissance dans les entrées natives

## État réel

**Entraînement GuitarSet non lancé : les sources originales nécessaires ont
expiré. Aucun résultat V27.3 + décroissance native n'existe à ce stade.**

Le diagnostic avec correcteur ajouté et le classifieur logistique indépendant
précédents ne constituent pas cette expérience. Leurs résultats ne permettent
pas de conclure sur l'ajout de décroissance dans V27.3.

Cette livraison fournit le calcul de l'indice sur les entrées natives, son
intégration dans les trois réseaux de comptage et les vérifications de parité.
Elle ne fournit pas encore un entraînement apparié complet ni son résultat.
Le workflow `v273-decay-native-preflight.yml` exécute uniquement ces
vérifications et constate l'état des archives. Une étape d'optimisation sur
des tenseurs synthétiques vérifie la branche ; ce n'est pas un entraînement
sur GuitarSet.

## Intégration préparée

V27.3 assemble plusieurs composants. L'extension de
`scripts/v273_decay_native_inputs.py` porte sur les réseaux V26 uniforme,
V26 pondéré et V27.2 conditionnel uniforme, utilisés par V27.3. L'ancre V10.4
et les règles historiques V27.0/V27.1/V27.3 restent celles du protocole.
Les seuils historiques devront être recalibrés à l'intérieur de chaque
partition, selon leurs règles existantes ; aucun correcteur supplémentaire
n'est ajouté à la sortie de V27.3.

Les quatre entrées originales sont conservées : candidats, masque,
statistiques du cluster et spectrogramme **23 × 64 × 3**. L'indice ajouté est
calculé depuis ce spectrogramme, et non depuis le CQT de V28.

Le canal original est `ln(1 + puissance / scalaire)`. L'inversion par
`expm1` permet d'ajuster `ln(puissance / scalaire) = a + b t`. Le scalaire
est constant dans chaque observation ; il ne change donc pas la pente.
La pente représente une décroissance de **puissance**, pas directement
une durée écoulée depuis une cause.

- Ajustement uniquement sur les neuf fenêtres FFT entièrement antérieures
  au cluster ; la dernière se termine 28 échantillons avant son début.
- Comparaison avec les quatorze fenêtres suivantes, dans le budget original
  de 40 ms après le début du cluster.
- Rejet des ajustements plats, croissants, saturés ou trop bruités.
- Moyenne et maximum des excès positifs sur 64 bandes : 128 entrées ajoutées.

La projection apprise de ces entrées est ajoutée au contexte acoustique,
**avant** les deux couches cachées de comptage déjà présentes. Elle est sans
biais et initialisée à zéro. Tous les paramètres historiques sont réutilisés
sans changement. Les deux bras ont la même architecture : indice nul pour
le témoin, indice mesuré pour le traitement. Chaque bras doit être construit
séparément avec la même graine ; les deux objets renvoyés par la factory
partagent leurs couches uniquement pour la vérification de parité.

Les tests vérifient la causalité de l'ajustement, la décroissance analytique,
une nouvelle attaque, le cache float16, les données invalides et la conservation
des quatre entrées. Sous TensorFlow 2.15.1, ils vérifient également les mêmes
probabilités initiales pour chacun des trois composants, le maintien de la
parité avec l'indice nul et l'apprentissage effectif de la nouvelle branche.

La fenêtre d'historique demeure courte, environ 29,7 ms. Les tests synthétiques
ne démontrent ni la fiabilité de la pente sur des enregistrements de guitare,
ni un gain d'Exact-K. Ces deux points nécessitent les données réelles.

## Sources à restaurer

Audit du 16 septembre 2026, identités et empreintes dans
`analysis/v273-decay-native-sources.json` :

| Source | État | Rôle |
|---|---|---|
| `v104-nested-spectral-bundle`, run 33647694565, artefact 9853492767 | Expiré le 9 septembre | Entrées originales, dont `spectral` et `stats` |
| Les 20 artefacts `v104-nested-inner-*` du même run | Expirés le 9 septembre | Calibration interne sans contamination du fold externe |
| Les cinq folds V26, run 34201830555 | Disponibles à l'audit | Budgets d'entraînement historiques |
| Les cinq folds V27.2, run 34287254337 | Disponibles à l'audit | Budgets du modèle conditionnel |
| Les cinq folds V27.3, run 34297767492 | Disponibles à l'audit | Prédictions de référence |

Une sauvegarde du bundle spectral et des partitions internes permettrait de
reprendre la préparation du test. Les empreintes publiées permettent de
contrôler les ZIP originaux. Pour commencer par le fold externe 1, seules ses
quatre archives internes sont nécessaires, en plus du bundle spectral :
`v104-nested-inner-o4-1-0`, `v104-nested-inner-o5-1-2`,
`v104-nested-inner-o6-1-3`, `v104-nested-inner-o7-1-4`.

Les métadonnées conservées pour V28 omettent `stats` et `spectral`. Le spectre
peut être recalculé depuis GuitarSet, mais les statistiques originales ne
peuvent pas être garanties identiques à partir des seuls candidats conservés :
les statistiques utilisaient aussi les candidats ensuite tronqués et les
valeurs avant leur quantification float16. Les anciens poids V8.4/V8.6/V8.7/V8.8
nécessaires au recalcul complet ont eux aussi expiré. Remplacer ces données
silencieusement produirait une autre référence.

## Comparaison à exécuter après restauration

1. Vérifier les sources, l'alignement des lignes et les partitions originales.
2. Rejouer l'ancre V10.4 interne une seule fois et la partager entre les bras.
3. Réentraîner les trois composants de comptage, dans les deux bras, avec les
   mêmes graines, lots, pondérations, Adam 2e-4 et budgets d'époques historiques.
   Conserver les indices d'apprentissage/validation du protocole original.
4. Calibrer les règles V27.1 et V27.3 uniquement sur la validation interne de
   chaque bras ; figer les décisions avant l'évaluation externe.
5. Comparer le témoin réentraîné à l'archive V27.3, puis mesurer le traitement
   contre ce témoin apparié. Rapporter toute dérive du témoin.

La référence officielle reste **4 005 / 9 401 = 42,6019 %** d'Exact-K
polyphonique sur les cinq folds. Sur le seul fold 1, elle est
**796 / 1 776 = 44,8198 %**. Ces deux périmètres ne sont pas interchangeables.
Un premier fold est un diagnostic ; il ne suffit pas pour promouvoir un
remplacement de V27.3.
