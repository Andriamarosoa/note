# V27.3 — Régressions : quatre séries, résultats et limites

**Critère de réussite précisé : battre YourMT3+ en Exact-K global et poly.**
Le meilleur résultat ci-dessous atteint 81,9724 % / 36,4658 %, contre
86,5434 % / 54,5430 % pour YourMT3+ sur les mêmes 59 309 lignes.
Même le choix parfait parmi les 255 groupes et les 211 politiques conservées
plafonne à 85,0394 % global. Il faut donc élargir les propositions du système.
Voir la [comparaison vérifiée et la preuve du plafond](README_V273_OBJECTIF_YOURMT3.md).

## Résultat mesuré

Le croisement de départ réalisait **566 corrections et 518 régressions**,
soit **+48** face à `freeze_local_combo`. Quatre séries successives ont été
exécutées ; leurs protocoles ont été enregistrés avant chaque calcul.
Elles produisent 162 politiques supplémentaires, toutes conservées. L'audit
final conserve ensuite trois gardes déterministes a posteriori, soit
**211 politiques au total** en incluant les 46 précédentes.

Trois compromis utiles se dégagent. Leur présentation ci-dessous est un
choix de développement après comparaison : aucune validation inédite ni
promotion automatique n'est revendiquée.

| Politique | Corrections | Régressions | Net / freeze | Anciennes corrections conservées | Anciennes perdues | Nouvelles corrections |
|---|---:|---:|---:|---:|---:|---:|
| Croisement de départ | 566 | 518 | +48 | 566 | 0 | 0 |
| Mélange non linéaire léger des audits, poids 0,25, coût 1 | **569** | **463** | **+106** | **523** | **43** | **46** |
| Calibration des groupes, coût 1,15 | **544** | **395** | **+149** | 344 | 222 | 200 |
| Routage imbriqué par K initial | **506** | **354** | **+152** | 353 | 213 | 153 |

Le premier candidat améliore les deux comptes et conserve **92,40 %** des
anciennes réussites. Il évite 84 anciennes régressions et en introduit 29,
soit **55 de moins**. C'est le candidat le plus prudent parmi ces trois.

Le routage réduit le nombre de régressions de **31,66 %** (164 de moins),
au prix d'un nombre total de corrections inférieur de 60. Il évite 258
anciennes régressions et en introduit 94. Il conserve seulement 353 anciennes
corrections : le net global ne doit pas masquer ces pertes.

Les trois candidats ont un net positif dans chacun des quatre folds.

| Fold | Mélange léger | Calibration groupes coût 1,15 | Routage par K initial |
|---|---:|---:|---:|
| 0 | +35 | +41 | +48 |
| 1 | +41 | +32 | +29 |
| 2 | +16 | +26 | +35 |
| 4 | +14 | +50 | +40 |

Ces nombres comparent chaque candidat au freeze, pas directement au
croisement de départ dans chaque fold.

| Politique | Exact-K global, 59 309 événements | Exact-K polyphonique, 7 385 événements |
|---|---:|---:|
| Mélange léger | 81,8763 % | 35,6940 % |
| Calibration groupes coût 1,15 | 81,9488 % | 36,2762 % |
| Routage par K initial | 81,9538 % | 36,3169 % |

Les tableaux K0–K6, matrices de confusion et résultats de chaque politique
sont dans les `series*/report.json`. Le périmètre ne correspond pas au
pourcentage historique de la V27.3 officielle.

## Ce que les interventions permettent de conclure

### 1. Une calibration uniforme ne suffit pas

Corriger uniquement les probabilités de correction/OTHER produit 543
corrections /489 régressions (+54). Calibrer uniformément les deux facteurs
produit 504 /457 (+47). Cela ne résout pas l'écart important entre gain
annoncé et gain réel. Une règle uniforme de confiance réduit aussi les
bonnes corrections quand elle devient plus restrictive.

### 2. Les audits des groupes contiennent une information exploitable

À coût identique de 1, la calibration avec transitions seules donne
575 /524 (+51). L'ajout des statistiques de tous les groupes donnant
chaque destination donne 691 /538 (+153). Ce changement utilise les mêmes
propositions et les mêmes exclusions ; il apporte 116 corrections nettes
supplémentaires au compteur, avec 14 régressions supplémentaires.

L'information des audits aide donc à distinguer des actions, mais son
utilisation seule n'assure pas une baisse des régressions. Le coût 1,15
donne ensuite le compromis 544 /395. Les identités de tous les groupes
ayant le même verdict sont agrégées ; aucun premier masque choisi n'est
présenté comme une explication causale.

### 3. Des interactions non linéaires sont utiles, mais un mélange fort peut perdre des succès

Avec les audits de groupes et un mélange de poids 0,25, on obtient
569 /463 et 523 anciennes corrections conservées. À poids 0,75, on obtient
575 /466, mais seulement 347 anciennes corrections conservées. Ces deux
politiques restent en mémoire. Le plus gros changement de modèle n'est
donc pas le plus protecteur des succès déjà acquis.

Ajouter les 58 caractéristiques acoustiques au mélange de poids 0,25
donne 547 /467 (+80), inférieur à cette variante utilisant les audits de
groupes. L'expérience ne justifie pas de désigner une caractéristique
acoustique unique comme la cause résolue.

### 4. Le contexte K et le transfert entre morceaux restent importants

Le choix d'une politique complète séparément pour K initial 2, 3 ou 4
donne 506 /354. En revanche, exiger de conserver 90 % des identités des
anciennes réussites dans la validation interne donne seulement 491 /438
(+53), avec 485 anciennes réussites effectivement conservées en test,
soit 85,69 %. La contrainte d'apprentissage ne devient donc pas
automatiquement une garantie sur un nouveau morceau.

Affiner ce routage par transition donne 490 /438 (+52). La descente de
coordonnées s'arrête sans amélioration locale admissible sur ses données
internes. Cela ne prouve pas un minimum global des régressions.

### 5. Apprendre uniquement sur les changements ne résout pas le problème à lui seul

Un garde apprend correction/régression/neutre sur les changements du
croisement. Il peut uniquement annuler une action : aucune nouvelle
régression n'est possible face au parent. Le choix imbriqué global conserve
532 corrections et laisse 475 régressions (+57). Il sauve 43 régressions,
mais abandonne 34 bonnes corrections. Le routage par K donne 484 /406
(+78), avec 82 bonnes corrections perdues. Ces variantes sont conservées,
mais elles ne remplacent pas les meilleurs compromis précédents.

Ces résultats identifient des limites du mécanisme de décision et mesurent
des interventions utiles. Ils n'identifient pas la cause physique de
chaque ambiguïté sonore. Les associations d'énergie, de persistance et
de source dans `series1/diagnostics.json` sont descriptives, stratifiées
par K initial ; elles ne servent pas de preuve causale ni de règle au test.

## Surestimation restante

| Politique | Net annoncé | Net observé | Excès annoncé |
|---|---:|---:|---:|
| Croisement initial | +354,77 | +48 | +306,77 |
| Mélange léger | +268,29 | +106 | +162,29 |
| Calibration groupes coût 1,15 | +261,31 | +149 | +112,31 |
| Routage par K initial | +275,18 | +152 | +123,18 |

La surestimation diminue sans disparaître. L'audit ne transforme pas les
régressions restantes en catégories nouvelles par simple affirmation.
Il faut distinguer une catégorie trop large, des observations insuffisantes
et une règle qui ne se transfère pas bien entre morceaux.

## Exclusions et portée de la preuve

Les réseaux initiaux et les audits sont figés et excluent le fold de
l'événement. Les calibrateurs utilisent les autres morceaux de ce fold.
Le morceau évalué est exclu de tous les ajustements et du choix de politique ;
une validation supplémentaire par morceau est imbriquée dans les morceaux
autorisés pour choisir le modèle, le coût et le routage.

**Le système final utilise des annotations d'autres morceaux du fold exclu
par le réseau. Ce n'est pas une évaluation sans supervision sur le fold
entier.** Les 19 morceaux ont déjà été examinés au cours du développement.
La sélection de ces trois candidats après les expériences reste exploratoire.
Fold 3 et player05 restent exclus. Une validation réellement inédite est
nécessaire avant promotion.

La vérification locale indépendante reproduit les probabilités linéaires
à 1,78e-15 près, les décisions et les choix imbriqués, contrôle les
standardisations et identités d'entraînement, les exclusions des modèles
non linéaires et les métriques natives. La
[CI 37827415747](https://github.com/Andriamarosoa/note/actions/runs/37827415747)
est **terminée avec succès** : cinq tests, puis réapprentissage des quatre
séries depuis les données figées. Les 162 politiques nouvelles sont
reproduites exactement : **9 608 058 décisions natives et 4 144 077 décisions
internes, zéro divergence**. L'écart probabiliste maximal est 9,04e-10.
Il s'agit d'un rejeu indépendant du calcul, pas de données inédites.

## Audit final des profils et conservation des petits apports

Dans le candidat prudent, les 463 régressions restantes se répartissent
ainsi : 297 dans un profil mixte entre groupes complets, 87 dans un profil
sans groupe favorable à la correction, 78 dans un profil favorable et
une sans support local. Le profil défavorable contient aussi **85 bonnes
corrections** : ce signal n'isole donc pas proprement les échecs.

La définition exacte du profil défavorable est : support local présent,
au moins un groupe avec N>M, et aucun groupe avec M>N parmi tous les
groupes donnant le K proposé. Les égalités sont possibles. La vérité du
fragment n'entre pas dans cette définition.

Après cet audit, trois politiques qui annulent uniquement ce profil sont
calculées et conservées. Ce sont des diagnostics **a posteriori**, pas des
variantes préenregistrées avant les quatre séries ni une validation inédite.

| Parent | Régressions évitées | Corrections perdues | Apport net supplémentaire | Corrections / régressions finales | Net / freeze |
|---|---:|---:|---:|---:|---:|
| Mélange prudent | 87 | 85 | **+2** | 484 /376 | +108 |
| Calibration groupes coût 1,15 | 83 | 69 | **+14** | 475 /312 | +163 |
| Routage par K initial | 84 | 79 | **+5** | 427 /270 | +157 |

Même **+2** devient une politique candidate documentée. Les bonnes
corrections abandonnées restent identifiées et conservées dans son parent.
Le dernier compromis réduit les régressions de 47,88 % face au départ, mais
réduit aussi le nombre total de corrections de 24,56 % : ce n'est pas une
amélioration gratuite. Ces trois règles sont rejouées localement à partir
des sorties fraîchement réapprises par la CI, avec décisions identiques.
Les cas et les vérifications sont dans `posthoc-audit/`.

## Rien n'est supprimé

Les quatre séries font passer la mémoire de **46 à 208 politiques
archivées**, soit 194 vecteurs distincts. Les trois diagnostics finaux
portent le total à **211 politiques et 197 vecteurs distincts**. Les
doublons et les bilans négatifs restent présents. Les anciennes colonnes
sont reproduites à l'identique. Aucun de ces éléments ne devient
automatiquement une tête active du réseau.

L'union des erreurs initiales corrigées par au moins une politique passe
de 1 671 à 1 727 : **56 cas complémentaires supplémentaires** sont conservés.
Cette union est un diagnostic utilisant la vérité, pas le score d'un
sélecteur utilisable en inférence.

`all-candidate-decisions.npz` conserve les 208 premières politiques sur les
59 309 événements, leurs effets individuels et leurs identifiants. Les trois
suivantes sont ajoutées dans `posthoc-audit/appended-candidates.npz` ; le
dernier registre est `posthoc-audit/candidate-registry.json`. `prepared/`
conserve les 58 observables, les contextes complets utilisés, les propositions,
probabilités initiales et coordonnées des événements. Chaque série conserve
les sorties internes, décisions, paramètres et chemins de choix. Les modèles
linéaires sont compressés sans perte ; les objets des arbres sont recompressés
et découpés, avec identité de l'état complet contrôlée par `joblib.hash`.
Le format de stockage et la restauration sont dans `storage-manifest.json`.

Une utilisation comme entrées d'un nouveau sélecteur exigera à nouveau des
prédictions imbriquées adaptées à ses exclusions. L'archivage d'un candidat
ne constitue pas, à lui seul, une preuve autorisant sa promotion.
