# V27.3 — résultat de la sélection apprise S8

**S8 élargit le catalogue, mais le système étendu perd 22 verdicts justes
face au contrôle.** Une correction apprise peut donc fournir de nouvelles
propositions utilisables ; cet essai ne démontre pas que le sélecteur sait
en tirer un meilleur bilan global.

Run [37789646292](https://github.com/Andriamarosoa/note/actions/runs/37789646292),
terminé avec succès. Source exécutée :
`3f83c2b07308c9114f1db0b5a904c0d80a91b91a`.
Protocole fixé dans [README_V273_PROTOCOLE_SELECTION_APPRISE.md](README_V273_PROTOCOLE_SELECTION_APPRISE.md).
26 tests réussis ; 14 correcteurs hors fold entraînés, puis deux sélecteurs
évalués sur les quatre folds, 30 époques à chaque entraînement. Aucun
réglage après résultats. PR expérimentale, sans promotion.

## Bilan sur la cohorte native

59 309 événements ; 7 493 à K initial 2/3/4 sont modifiables.
Les corrections et régressions ci-dessous sont mesurées face au verdict figé.

| Système | Corrections | Régressions | Net | Exact-K global | Exact-K poly |
|---|---:|---:|---:|---:|---:|
| Verdict figé | — | — | 0 | 81,6976 % | 34,2586 % |
| Contrôle : 7 sélections, 127 groupes | 582 | 522 | +60 | 81,7987 % | 35,0711 % |
| Catalogue : 8 sélections, 255 groupes | 584 | 546 | +38 | 81,7616 % | 34,7732 % |
| S8 appliquée seule | 622 | 562 | +60 | 81,7987 % | 35,0711 % |

Le contrôle reproduit **bit pour bit tous les tableaux de prédictions et
de scores du consensus catégoriel archivé**. Les sept votes originaux, les
127 propositions et leurs historiques restent également identiques dans
le catalogue étendu. Le sélecteur étendu ajoute 544 paramètres ; S8 possède
5 575 paramètres. La comparaison inclut cette modification de capacité et
ne permet pas d'attribuer chaque écart uniquement à la nouvelle information.

Face au contrôle, le système étendu répare 309 décisions et en dégrade 331 :
**−22 net**. Les 1 061 changements comprennent aussi des déplacements entre
deux verdicts faux.

## La couverture s'enrichit effectivement

Les erreurs éligibles avec au moins un groupe donnant le bon K passent de
**1 624 à 1 757**, soit **133 possibilités supplémentaires**. Parmi les
1 313 erreurs polyphoniques précédemment hors couverture, il en reste 1 180.
Cette couverture utilise la vérité uniquement pour l'audit, jamais pour
choisir le verdict d'un fragment.

| Vrai K | Nouvelles erreurs accessibles | Effectivement corrigées |
|---|---:|---:|
| K2 | 18 | 4 |
| K3 | 20 | 7 |
| K4 | 59 | 12 |
| K5 | 33 | 6 |
| K6 | 3 | 0 |
| Total | **133** | **29** |

128 cas deviennent accessibles par S8 seule ; **5 nécessitent une
combinaison**, puisque S8 seule et tous les anciens groupes échouent.
Le système corrige 27 des 128 premiers et 2 des 5 seconds.
Il ne récupère donc que 29/133, soit 21,8 %, de cette nouvelle couverture.

Deux exemples vérifiés :

| Événement | Vérité | Verdict initial | Sélections seules | Combinaison correcte | Verdict final |
|---|---:|---:|---|---|---:|
| 5084 | K2 | K3 | S3 : K3 ; S8 : K4 | S3 + S8 : K2 | K2 |
| 66872 | K3 | K4 | S2 : K2 ; S3 : K2 ; S8 : K4 | S2 + S3 + S8 : K3 | K3 |

Événement 5084 : `00_Jazz1-130-D_comp.jams`, échantillon 53 600.
Événement 66872 : `04_Jazz3-137-Eb_comp.jams`, échantillon 688 163.
Dans le second cas, aucun groupe de taille 1 ou 2 ne donne K3 ; une
combinaison d'au moins trois sélections est nécessaire dans ce catalogue.
Les groupes ayant le même verdict partagent leur score ; le masque choisi
est un représentant, pas une attribution causale de la décision neuronale.

Les 133 cas sont exportés dans
[new-reachable-errors.csv](evidence/v273-learned-selection/new-reachable-errors.csv),
avec enregistrement, position, vérité, verdicts et masques corrects.

## Pourquoi le bilan total baisse

Sur les erreurs déjà accessibles avec l'ancien catalogue, les corrections
passent de 582 à 555 : **−27**. Les 29 nouvelles corrections compensent
juste cette perte, donnant +2 corrections au total. Les régressions passent
de 522 à 546 : **+24**. Le bilan est donc `29 − 27 − 24 = −22`.

En détaillant les événements face au contrôle :

- 419 corrections sont conservées, 163 sont perdues, 165 apparaissent ;
- 144 régressions sont réparées, 378 persistent, 168 apparaissent.

Les 640 bascules de justesse, 309 favorables et 331 défavorables, sont dans
[extended-vs-control.csv](evidence/v273-learned-selection/extended-vs-control.csv).

| Vrai K | Net du contrôle | Net avec S8 |
|---|---:|---:|
| K2 | +101 | +94 |
| K3 | −6 | −24 |
| K4 | −50 | −49 |
| K5 | +15 | +17 |
| K6 | 0 | 0 |

Les vrais K0/K1 gardent un net nul : les actions restent K2–K6. Leur masse
prédite par S8 est transférée au verdict initial pour produire un vote,
sans prétendre rendre ce verdict correct.

| Fold | Net du contrôle | Net avec S8 | Différence |
|---|---:|---:|---:|
| 0 | +16 | +19 | +3 |
| 1 | +23 | +4 | −19 |
| 2 | +1 | +4 | +3 |
| 4 | +20 | +11 | −9 |

La confiance reste trop optimiste : le catalogue étendu annonce **+397,42**
sur ses 1 896 changements, contre **+38 observé**. Il annonce 872,92
corrections contre 584, et 475,50 régressions contre 546. Les 766 autres
changements restent faux. La normalisation des probabilités et l'égalité
des scores pour un même K sont vérifiées ; elles ne prouvent pas leur
calibration empirique.

## Diagnostics de décodage prévus

Mêmes poids du sélecteur étendu et **toutes ses entrées conservées**.
Seules les sorties admissibles changent. Ces diagnostics ne sont ni des
modèles réentraînés ni une validation d'un nouveau filtre.

| Sorties admissibles | Net face au verdict figé |
|---|---:|
| Anciens groupes uniquement | +44 |
| Anciens groupes + singleton S8 | +37 |
| Tous les 255 groupes | +38 |
| Tous les singletons | +39 |
| Singleton S8 soumis au score du sélecteur | +64 |

Ajouter les combinaisons contenant S8 au décodage « anciens groupes + S8
seule » change 8 décisions, avec 2 corrections et 1 régression : +1 net.
S8 reste présente dans les entrées et dans l'agrégation des scores des
autres diagnostics. Le +64 ne justifie donc pas de déployer un filtre
« S8 seulement », ni de veto sur les membres avant leurs combinaisons.

## Vérification et portée

[verification.json](evidence/v273-learned-selection/verification.json)
est produit par `scripts/verify_v273_learned_selection.py`, sans TensorFlow.
Il recalcule les propositions, les probabilités, les décisions, les scores,
les nouveaux cas accessibles et les historiques globaux réellement activés.
Il vérifie également les archives SHA-256, les 14 ensembles d'entraînement
du correcteur, les poids, les standardisations, les 36 chemins de production
hors fold et les exclusions des producteurs des spécialistes.

Commande de reproduction, après téléchargement des artefacts :

```bash
python -m scripts.verify_v273_learned_selection \
  --artifacts catalogue-results \
  --reference selector-repair-results/coherent/predictions.npz \
  --archived-consensus group-consensus-results/pooled_ce \
  --archived-global group127-cached-results/global \
  --producer-cache group127-cached-results/producer-cache \
  --features selector-review-artifacts/features \
  --output analysis/evidence/v273-learned-selection/verification.json
```

Les folds 0/1/2/4 restent des données déjà exposées au développement. Aucun
fold3/player05 ni jeu inédit n'a été utilisé. Les caractéristiques gardent
leur contexte futur de +160 ms. Le correcteur est une proposition apprise
sur les observables existants ; il n'apporte pas de nouvelle mesure sonore.
La fusion des votes demeure une moyenne fixée, et S7 une moyenne dérivée.

L'idée d'enrichir le catalogue a maintenant des exemples concrets, y compris
des réussites seulement combinatoires. **La priorité pour la prochaine
expérience est de mieux exploiter ces propositions dans le sélecteur**,
avant de multiplier les nouvelles sélections. Cet essai ne prouve ni une
capacité de création automatique de sélections, ni une généralisation sur
des données réservées.
