# Audit des 127 combinaisons — sélections par K et défaut du décodeur

**8 octobre 2026.** Exécution terminée :
[GitHub Actions 37750617069](https://github.com/Andriamarosoa/note/actions/runs/37750617069).
Contrôle précédent du décodeur :
[GitHub Actions 37748832939](https://github.com/Andriamarosoa/note/actions/runs/37748832939).

## Portée et méthode

Nous avons audité toutes les combinaisons de têtes *structurellement*
disponibles pour les transitions entre le K initial de \`freeze_local_combo\`
(K2/K3/K4) et chaque hypothèse de sortie K0–K6. Le test utilise les
**59 309 événements natifs**, dont **7 493 révisables**, folds 0,1,2,4,
fold3/player05 exclus.

**Distinction centrale :** ces combinaisons sont des **expériences de
vote direct non apprises** utilisant les mêmes probabilités spécialistes.
Ce ne sont **pas** les sous-ensembles effectivement activés par
l'attention neuronale, ni une analyse causale de ses activations cachées.
Aucun correcteur ni poids du réseau principal n'a été modifié.

Les rapports incluent **4 folds × 2 phases (OOF d'entraînement, audit
descriptif tenu de côté) × 1 018 sous-ensembles = 8 144 audits**
complets, chaque ligne étant détaillée par vrai K0..K6. Les folds
de validation n'ont pas servi à ajuster de poids.

## Sous-ensembles disponibles et bilans diagnostiques tenus à part

| Origine → Cible | Têtes éligibles | Sous-ensembles | Net positif | Net négatif | Sans action |
|---|---:|---:|---:|---:|---:|
| K2→K3 | 7 | **127** | **37** | **58** | 30 |
| K3→K2 | 7 | **127** | **117** | **3** | 4 |
| K3→K4 | 7 | **127** | **59** | **3** | 35 |
| K4→K3 | 7 | **127** | **72** | **13** | 37 |
| K2→K4 | 6 | 63 | 0 | 44 | 14 |
| K2→K5 | 6 | 63 | 1 | 26 | 31 |
| K2→K6 | 6 | 63 | 0 | 4 | 58 |
| K3→K5 | 6 | 63 | 15 | 4 | 33 |
| K3→K6 | 6 | 63 | 0 | 0 | 59 |
| K4→K2 | 6 | 63 | 51 | 6 | 3 |
| K4→K5 | 6 | 63 | 2 | 25 | 34 |
| K4→K6 | 6 | 63 | 0 | 0 | 63 |
| Vers K0 ou K1, origine K2/K3/K4 | 1 | 1 par transition | 0 | 0 | 1 |

Quelques sous-ensembles apportent des actions mais un bilan exactement
nul : les colonnes positif/négatif/sans action ne totalisent donc
pas toujours l'ensemble des 127 ou 63.

Au total : **1 018 définitions** de combinaisons structurelles,
**4 072 audits sur folds tenus de côté** et **4 072 audits OOF sur
données d'entraînement**.

## Sous-ensembles diagnostiques observés sur les folds externes

| Transition | Meilleur bilan descriptif | Pire bilan descriptif |
|---|---|---|
| K2→K3 | H2+H3+H4+H5 : +139/−118 = **+21** | H5 : +113/−169 = **−56** |
| K3→K2 | H2 : +260/−201 = **+59** | H5 : +203/−216 = **−13** |
| K3→K4 | H3 : +59/−46 = **+13** | H5 : +23/−34 = **−11** |
| K4→K3 | H3+H4+H5+C43 : +108/−81 = **+27** | H1 : +218/−255 = **−37** |

Les « meilleurs » ont été identifiés **après avoir regardé les labels
des folds tenus à l'écart** ; ils ne doivent pas servir à programmer
directement des changements ni être interprétés comme performances
validées hors sélection.

## La cause centrale reste dans le décodeur

Pour les destinations **K0 et K1**, le masque actuel n'admet
que **H0**, pourtant son adaptateur n'a **aucun posterior K0/K1**.
Les votes directs sont donc tous des conservations K.
Or, dans la vraie sortie neuronale :
- le décodeur a tout de même proposé **1 822** changements vers K0/K1 ;
- ils ont généré **438 régressions polyphoniques**, dont **361 pour K2→K1** ;
- ces pertes expliquent **438 des 440 bonnes prédictions polyphoniques
  perdues nettes** par le réseau.

**Aucune des 127 combinaisons pour K2→K3 ou K3→K2 ne peut, selon
l'interface déclarée, proposer K1**. Le décodeur est capable d'émettre
cette classe parce que ses logits de sortie ne sont pas contraints
par les probabilités et masques des têtes.

Le problème n'est donc pas « 127 combinaisons choisies au hasard ».
Il comprend simultanément :
1. une sortie **K0/K1 sans évidence de tête** ;
2. un objectif de perte qui favorise des corrections nonpoly au détriment
   de la conservation de certains K2/K3/K4 corrects ;
3. une estimation de risque auxiliaire non raccordée directement au
   décodeur et à l'action KEEP ;
4. des combinaisons polyphoniques dont l'utilité **change fortement
   selon la classe et la direction** (K2→K3 : 58 négatives sur 127,
   contre K3→K2 : 3 négatives sur 127 dans ce vote direct).

## Contenu de l'artefact de preuves

[ZIP sur GitHub](https://github.com/Andriamarosoa/note/actions/runs/37750617069/artifacts/11537304358) :

- \`audit-structure.json\` : contrat de chaque transition, nombre
  de sélections et bilans extrêmes ;
- \`audits-individuels-OOF-et-heldout.csv\` : les **8 144 audits**
  avec détail séparé corrections/régressions/neutres par vrai K ;
- \`audits-agreges-1018-selections.csv\` : une ligne par sous-ensemble,
  regroupant les résultats descriptifs des 4 folds ;
- \`verdict.md\` : synthèse.

## Prochaine preuve à obtenir

Pour identifier **causalement** quelle combinaison a déclenché
une erreur du réseau, il faut instrumenter le décodeur entraîné et
calculer des interventions d'ablation sur chaque entrée active,
puis mesurer le changement de prédiction sur les mêmes événements.
Le vote direct fournit une carte de risque empirique par combinaison,
mais ne remplace pas cette intervention sur les véritables activations
neuronales. L'intervention ne doit pas changer de baseline ni être
promue sans validation fraîche.
