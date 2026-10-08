# V27.3 — Protocole du risque contextuel

Fixé avant les résultats de la nouvelle variante. Parent : catalogue à huit
sélections avec audits locaux, run 37807124287 ; 527 corrections, 495
régressions, gain +32 face au freeze.

## Limite observée dans le calcul

La probabilité partagée que le verdict initial soit correct détermine le risque
de régression de tous les groupes qui le changent. Sa branche reçoit le contexte,
le K initial, les votes et la moyenne des neuf descripteurs d'audit des groupes
actifs. L'association entre l'audit et l'identité du groupe disparaît dans
cette moyenne avant l'entrée de cette branche. Permuter les audits entre
groupes actifs en conservant leur moyenne ne peut donc pas changer cette
probabilité, même si les groupes concernés diffèrent. La branche conditionnelle
des corrections dispose déjà de ces associations ; ce constat n'est pas une
preuve que toute la mauvaise généralisation provient de cette seule limite.

## Modification testée

Ajouter à la branche de risque la moyenne des 32 activations non linéaires
déjà calculées pour les groupes complets. Ces activations dépendent conjointement
de l'identité du groupe, de ses membres, de son K proposé, du contexte acoustique
et de son audit. La moyenne intervient alors après ces interactions apprises.

Le risque reste partagé et les probabilités cohérentes : BCE sur la correction
du verdict initial, puis CE catégorielle sur les alternatives lorsque ce verdict
est faux. Tous les 255 groupes restent présents. Aucun veto individuel ou seuil
choisi après lecture des résultats n'est ajouté.

Le nombre de paramètres passe de 13 538 à 14 562 (+1 024). L'apprentissage
commun des groupes est également affecté. Ce n'est donc pas une ablation à
nombre de paramètres constant, ni une preuve causale isolant le seul effet de
la capacité. Ce sont les apports et les pertes mesurés qui décident de ce qui
est établi ; un bilan global négatif ne supprime pas la candidate.

## Comparaison et exclusions

Mêmes votes, propositions, audits, producteurs, 43 caractéristiques de contexte,
seed 27402, 30 époques, batch 192, Adam 0,002 et perte que le parent. Aucun
réglage sur les 126 régressions ou les 92 corrections du profil défavorable.
Leurs labels interviennent uniquement dans les bilans après prédiction.

Quatre entraînements : pour chaque fold 0/1/2/4 évalué, apprentissage et
références d'audit sur les trois autres folds ; producteurs et références
internes gardent les exclusions imbriquées déjà vérifiées. Fold3/player05
restent exclus. Ces quatre folds sont déjà exposés à la recherche : aucune
validation finale sur données inédites n'est revendiquée.

Le parent est rejoué à poids figés et ses décisions doivent être identiques.
Les entrées de la variante doivent être identiques aux siennes. Les sorties,
les paramètres et la provenance sont archivés même en l'absence de gain global.

## Preuves attendues

- Test de permutation : invariance du risque initial à la réaffectation des
  audits, capacité de la nouvelle branche à en tenir compte.
- Identité des propositions, audit des exclusions, cohérence des probabilités,
  test de l'apprentissage et préservation des groupes complets.
- Résultats globaux, par vrai K et fold ; corrections conservées/perdues/ajoutées
  et régressions persistantes/évitées/ajoutées face au parent.
- Sur le profil défavorable du parent : combien des 126 régressions sont évitées
  et combien des 92 corrections restent justes. Les autres profils sont publiés.
- Surestimation, perte probabiliste et écart entraînement/évaluation.
- À poids figés : neutralisation du nouveau vecteur de risque uniquement, puis
  neutralisation des cinq canaux locaux. Ces interventions restent distinctes
  d'un réentraînement et leurs sorties sont conservées.
- Conservation de la candidate et de ses interventions en complément des 41
  sorties déjà archivées, sans sélectionner la meilleure sortie avec la vérité.
