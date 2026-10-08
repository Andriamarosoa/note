# V27.3 — Calibration partielle et lacunes restantes

**Calcul terminé et vérifié.** Pour S8, la calibration réduit l'écart entre
gain annoncé et gain réel de **359,42 à 115,55**, soit une réduction de
**67,85 %**. Le gain net réel passe de **+38 à +57**. La surestimation reste
présente et K3 perd 41 verdicts exacts par rapport au S8 initial : **cette
variante n'est pas promue**.

La calibration a bien rendu les probabilités plus réalistes dans ce test.
Elle révèle aussi que le réseau reconnaît encore mal les fragments sur
lesquels changer est utile, même lorsque le bon verdict est proposé.

## 1. Ce qui a été corrigé et ce que le test démontre

Les réseaux, leurs poids, les votes, les 127/255 groupes et leur classement
sont conservés. Quatre paramètres ajustent les logits de justesse du verdict
initial et des alternatives. Les pentes positives conservent le meilleur K
alternatif ; la décision de le retenir ou de garder le verdict initial peut
changer. La règle reste un gain strictement positif, sans seuil appris.

Le [protocole](README_V273_PROTOCOLE_CALIBRATION.md) a été publié avant le
calcul au commit `8c2352b48b0645b654dd1f831f5f32adb64f29cf`. Pour chacun des
19 morceaux des folds 0/1/2/4, le calibrateur apprend sur les autres morceaux
du même fold, que le réseau figé n'avait pas vu pendant son apprentissage.
Tous les musiciens et versions comp/solo du morceau évalué restent exclus
de ce calibrateur. Chaque événement est évalué une fois, soit 7 493 par bras.

**Le test bénéficie de labels de calibration provenant du fold exclu du
réseau.** Il mesure une adaptation supervisée à d'autres morceaux de ce fold.
Il ne démontre pas qu'un calibrateur appris uniquement sur les trois folds
d'entraînement fonctionnerait aussi bien. Les folds sont déjà exposés au
développement ; fold3/player05 et toute validation sur données inédites
restent exclus. Les 19 évaluations partagent des données de calibration ;
elles ne sont pas 19 expériences indépendantes.

## 2. Résultat principal

| Modèle / condition | Changements | Corrections | Régressions | Net réel | Net annoncé | Excès annoncé |
|---|---:|---:|---:|---:|---:|---:|
| S8 initial | 1 896 | 584 | 546 | +38 | +397,42 | +359,42 |
| S8 calibré | 1 700 | 516 | 459 | **+57** | +172,55 | +115,55 |
| Contrôle 7 initial | 1 800 | 582 | 522 | +60 | +383,20 | +323,20 |
| Contrôle 7 calibré | 1 576 | 486 | 442 | +44 | +162,86 | +118,86 |

Pour S8, il y a 87 régressions de moins, mais aussi 68 corrections de moins :
le bénéfice supplémentaire est **19 verdicts exacts**, pas 87. Pour 100
changements retenus, le système calibré annonce encore +10,15 net pour
seulement +3,35 réel.

Le contrôle perd 16 verdicts exacts malgré de meilleures probabilités. S8
calibré reste également sous le net +60 du contrôle 7 initial. Une amélioration
de calibration ne garantit donc pas une amélioration de la règle de décision.

Pour séparer l'effet sur la confiance de l'effet du nouveau choix des
événements, on recalcule aussi les probabilités sur les **1 896 décisions
initialement retenues par S8**, sans en changer aucune. Le gain annoncé
passe alors de +397,42 à **+134,30**, pour le même +38 réel. La baisse de
surestimation ne provient donc pas uniquement de la baisse du nombre de
changements.

## 3. La confiance s'améliore sans devenir fiable partout

| Mesure S8 | Initial | Calibré |
|---|---:|---:|
| Perte jointe par événement, plus bas = mieux | 1,0775 | 0,9812 |
| Perte de justesse du verdict initial | 0,6317 | 0,6201 |
| Perte conditionnelle si verdict initial faux | 0,6710 | 0,5435 |
| Brier de justesse du verdict initial | 0,2198 | 0,2157 |
| Brier conditionnel si verdict initial faux | 0,4206 | 0,3619 |
| Réussite conditionnelle annoncée sur les changements partant d'un verdict faux | 61,71 % | 48,39 % |
| Réussite conditionnelle observée sur cette population retenue | 43,26 % | 41,58 % |

Les populations retenues diffèrent dans les deux dernières lignes : la
calibration modifie la décision de changer. La perte jointe et les Brier
portent en revanche sur les mêmes événements externes.

Sur les changements retenus partant d'un verdict faux, le réseau calibré
annonce 45,63 % de cas sans alternative correcte (OTHER), contre **52,78 %
observés**. Ce défaut reste important. Sur tous les changements retenus,
il annonce 25,90 % de verdicts initiaux corrects, contre 27,00 % observés.

La perte jointe S8 s'améliore sur 18 des 19 morceaux ; le gain net s'améliore
sur 10, diminue sur 7 et reste identique sur 2. Cela distingue concrètement
qualité probabiliste et résultat de décision.

## 4. Changements retirés et ajoutés

La transformation ne se contente pas de réduire tous les scores : elle
rééquilibre la justesse du verdict initial, les alternatives et OTHER.

| S8 | Corrections | Régressions | Faux vers faux sans alternative correcte | Faux vers faux avec bonne alternative disponible |
|---|---:|---:|---:|---:|
| Changements conservés | 436 | 375 | 497 | 69 |
| Changements annulés | 148 | 171 | 128 | 72 |
| Changements ajoutés | 80 | 84 | 158 | 1 |

Les annulations améliorent le net de +23 : 171 régressions évitées pour
148 corrections perdues. Les ajouts le dégradent de −4 : 80 corrections
contre 84 régressions. Le total est bien +19.

Les 158 nouveaux changements sans alternative correcte montrent que cette
calibration à quatre paramètres ne reconnaît pas suffisamment chaque cas
individuel. Tous les paramètres ont convergé sans atteindre leurs bornes ;
aucun réglage n'a ensuite été choisi en fonction des résultats évalués.

## 5. Les lacunes désormais mieux délimitées

Parmi les 1 184 changements S8 encore ratés après calibration :

| Famille | Avant | Après | Ce qu'elle permet d'examiner |
|---|---:|---:|---|
| Verdict initial correct remplacé | 546 | **459** | Reconnaître quand conserver le verdict |
| Verdict initial faux, aucune alternative correcte | 625 | **655** | Reconnaître l'absence de correction disponible et enrichir les propositions |
| Verdict initial faux, bonne alternative présente mais mauvais choix | 141 | **70** | Classer correctement les propositions |

La baisse de 141 à 70 dans la dernière ligne ne signifie pas que le classement
a été réparé : il est inchangé. Des mauvais choix sont simplement davantage
rejetés par la règle de changement.

Sur les **4 978 erreurs initiales éligibles**, indépendamment de la décision
de changer :

- **3 221** n'ont aucune alternative correcte dans le catalogue ;
- **1 757** ont au moins une alternative correcte ;
- parmi ces 1 757, le verdict déjà classé premier est correct dans **1 456**
  cas et incorrect dans **301** ;
- la version calibrée corrige **516 des 1 456** cas où ce meilleur verdict
  est correct. Les **940 autres sont conservés**.

Ces nombres sont des diagnostics utilisant les labels, pas des règles
exploitables à l'inférence ni une promesse de gain. Ils montrent cependant
deux limites distinctes : la couverture des propositions, et la capacité à
reconnaître les corrections utiles déjà présentes. Une calibration qui
conserve le classement ne peut résoudre les 301 erreurs de classement.

Il reste **479 changements ratés avec un gain annoncé d'au moins 0,1** :
165 régressions, 297 cas sans alternative correcte et 17 mauvais choix malgré
une bonne alternative. Exemples, issus du fichier exhaustif :

| Événement | Enregistrement | Vrai K | Initial → final | Gain avant → après | Échec |
|---|---|---:|---|---|---|
| 66074 | 04_Jazz2-110-Bb_comp.jams | 4 | 4 → 2 | 0,895 → 0,498 | Régression |
| 19710 | 01_Funk2-119-G_comp.jams | 3 | 4 → 2 | 0,886 → 0,495 | Aucun autre verdict correct |
| 43268 | 02_Rock3-148-C_comp.jams | 3 | 3 → 2 | 0,863 → 0,479 | Régression |

Aucune erreur retenue ne dépasse le seuil descriptif 0,5 après calibration.
Cela ne prouve pas qu'elle est résolue : les exemples ci-dessus restent faux
avec un score proche de ce seuil. Aucun seuil n'a été optimisé.

## 6. Les combinaisons restent nécessaires et certaines sont perdues

Le catalogue n'a subi aucun veto individuel. Il reste 238 événements S8 où
un groupe donne le bon verdict alors qu'aucun de ses membres ne réussit seul.
Dans cette famille descriptive, 41 événements étaient corrigés avant et
20 le sont après calibration.

Plus strictement, **17 erreurs initiales sont corrigeables uniquement par
une combinaison**, sans aucun singleton correct dans tout le catalogue.
Les corrections réalisées dans ce sous-ensemble passent de **5 à 3**.

La calibration ne supprime pas ces groupes, mais peut refuser leur verdict.
Ces pertes doivent donc être suivies lors d'une amélioration du sélecteur.
Les groupes proposant le même K partagent leur score agrégé ; le premier
masque affiché n'est toujours pas une preuve de contribution causale.

## 7. Exact-K et hétérogénéité des résultats

| Système | Exact-K global, 59 309 événements | Exact-K poly, 7 385 événements |
|---|---:|---:|
| freeze_local_combo | 81,6976 % | 34,2586 % |
| S8 initial | 81,7616 % | 34,7732 % |
| S8 calibré | **81,7937 %** | **35,0305 %** |
| Contrôle 7 initial | 81,7987 % | 35,0711 % |
| Contrôle 7 calibré | 81,7717 % | 34,8544 % |

| Vrai K | Événements | freeze_local_combo | S8 initial | S8 calibré | Exacts supplémentaires face au S8 initial |
|---|---:|---:|---:|---:|---:|
| K0 | 39 652 | 95,9220 % | 95,9220 % | 95,9220 % | 0 |
| K1 | 12 272 | 64,2846 % | 64,2846 % | 64,2846 % | 0 |
| K2 | 3 445 | 34,3396 % | 37,0682 % | 38,0261 % | +33 |
| K3 | 2 289 | 39,9738 % | 38,9253 % | **37,1341 %** | **−41** |
| K4 | 1 207 | 34,5485 % | 30,4888 % | 33,0572 % | +31 |
| K5 | 355 | 4,2254 % | 9,0141 % | 7,8873 % | −4 |
| K6 | 89 | 0,0000 % | 0,0000 % | 0,0000 % | 0 |

Le gain global masque une dégradation supplémentaire de K3, déjà sous la
référence initiale. La calibration commune de la confiance ne répare pas
cette faiblesse de discrimination entre classes.

| Fold | Net S8 initial face à freeze | Net S8 calibré face à freeze | Différence |
|---|---:|---:|---:|
| 0 | +19 | +29 | +10 |
| 1 | +4 | +17 | +13 |
| 2 | +4 | +3 | −1 |
| 4 | +11 | +8 | −3 |

Les nets par morceau et les paramètres de chaque calibrateur sont publiés
dans les preuves. Aucune conclusion de significativité ni de généralisation
à des données inédites n'est tirée de ces écarts descriptifs.

## 8. Conclusion pour la suite du système

La correction partielle de confiance donne un diagnostic plus précis :
le système peut être moins optimiste tout en restant faible pour décider
quand agir, avec des effets opposés selon K. La priorité reste un apprentissage
du sélecteur dont la généralisation est contrôlée, en suivant les erreurs
OTHER, les pertes de bonnes corrections et les synergies, particulièrement K3.

La couverture insuffisante de 3 221 erreurs motive également l'étude de
nouvelles propositions, mais leur ajout ne dispense pas de reconnaître les
1 456 bonnes premières propositions déjà disponibles. L'audit ne permet pas
d'attribuer ces difficultés à une caractéristique acoustique précise ; les
cas persistants sont maintenant exportés pour cette étude.

Ce test ne justifie ni l'adoption immédiate du calibrateur ni le choix d'un
nouveau seuil sur les mêmes résultats. Il fournit une amélioration mesurée
de confiance, un gain net modeste pour S8 et des régressions de classe à traiter.

## 9. Exécution et preuves

- [Run 37803433204 terminé avec succès](https://github.com/Andriamarosoa/note/actions/runs/37803433204),
  commit `fd0ac61cc0177e6b39fe9c8aaca63db6ac9e61e6`.
- 3 tests numériques réussis ; 38 calibrateurs convergés et réajustés par le
  vérificateur sur leurs seules lignes autorisées ; 14 986 évaluations vérifiées
  pour les deux bras. Les partitions, SHA, probabilités, classements et comptes
  ont été contrôlés. L'archive GitHub a aussi été téléchargée et revérifiée localement.
- [Vérification complète](evidence/v273-recording-calibration/verification.json),
  [comparaison](evidence/v273-recording-calibration/calibration-comparison.csv),
  [manifeste des artefacts](evidence/v273-recording-calibration/artifacts.json).
- Erreurs restantes : [S8](evidence/v273-recording-calibration/corrector8-remaining-failures.csv)
  et [contrôle 7](evidence/v273-recording-calibration/control7-remaining-failures.csv).
- Paramètres et partitions : [S8](evidence/v273-recording-calibration/corrector8-calibrators.json)
  et [contrôle 7](evidence/v273-recording-calibration/control7-calibrators.json).
- [Transformation](../scripts/v273_probability_calibration.py),
  [évaluation](../scripts/evaluate_v273_recording_calibration.py),
  [vérificateur](../scripts/verify_v273_recording_calibration.py),
  [workflow](../.github/workflows/v273-recording-calibration.yml).

Reproduction après extraction de l'archive du run dans `calibration-results/`,
du rejeu figé dans `confidence-results/`, des deux bras originaux dans
`catalogue-results/` et des observables dans `features/` :

```bash
python -m scripts.verify_v273_recording_calibration \
  --results ../calibration-results --replay ../confidence-results \
  --original ../catalogue-results --features ../features \
  --output analysis/evidence/v273-recording-calibration/verification.json \
  --run-id 37803433204 \
  --source-commit fd0ac61cc0177e6b39fe9c8aaca63db6ac9e61e6
```

Les snapshots et prédictions complets sont conservés dans l'artefact
`v273-recording-calibration` (ID `11561872139`, 1 894 769 octets), SHA-256
`cd34922f0fedc3a60605d374f6a7e505cb61ec3a7a11de8dc2df65635df0bc99`.
