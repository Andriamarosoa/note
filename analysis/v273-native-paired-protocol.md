# V27.3 avec/sans indice ln dans les entrées natives

## État et périmètre

La reconstruction des sources a réussi dans le
[run 35086829802](https://github.com/Andriamarosoa/note/actions/runs/35086829802).
Ses huit lots et l'audit final sont terminés. Les poids V8.1/V8.4/V8.6/V8.7/
V8.8 et les tableaux natifs sont sauvegardés dans la release de ce run.

Les entrées reconstruites comportent **74 588 clusters**, dont **9 354
polyphoniques**, sur **240 pistes et 24 groupes de composition**. L'archive
V27.3 comportait 76 768 clusters, dont 9 401 polyphoniques. Les nouveaux
candidats diffèrent : on ne déclare pas restauré le modèle historique.

Le présent workflow reconstruit les experts internes, entraîne les deux bras
de comptage et agrège les cinq folds. Sa publication n'est pas un résultat
d'entraînement ; le résultat ne sera disponible qu'après la fin des jobs.

## Question expérimentale

Sur une même reconstruction de la chaîne V27.3, l'indice de décroissance
acoustique améliore-t-il l'Exact-K polyphonique ?

- **Témoin** : trois réseaux natifs V26 uniforme, V26 pondéré et V27.2 uniforme,
  avec entrée de décroissance nulle.
- **Traitement** : mêmes réseaux et paramètres initiaux, avec l'indice calculé
  depuis le spectrogramme natif 23 × 64 × 3.

L'indice décrit les excès de puissance par rapport à une décroissance estimée
sur les neuf fenêtres antérieures au cluster. Il est injecté avant les couches
de comptage. Il ne s'agit ni d'un classifieur logistique indépendant ni d'un
correcteur ajouté après V27.3. Les règles intrinsèques V27.0/V27.1/V27.3
sont conservées et calibrées selon leurs fonctions historiques.

## Partitions et sources figées

`analysis/v273-native-paired-config.json` contient les empreintes des huit
archives reconstruites, la correspondance des 240 pistes aux folds originaux
et les budgets d'époques des rapports V27.3 archivés. Les ZIP sources et les
rapports/prédictions qui ont fourni cette correspondance sont identifiés par
ID et SHA-256. Les affectations ne sont pas recalculées à partir des nouvelles
tailles de clusters : une composition reste dans son fold historique.

Chaque fold externe conserve les quatre autres pour l'apprentissage. Le
plus petit fold restant fournit la validation interne des réseaux de comptage
et des seuils. Toutes les copies comp/solo et tous les musiciens d'une même
composition restent ensemble. Les pistes de validation historique ne sont
pas ajoutées à l'expérience.

## Ancre commune

Pour chacun des cinq folds externes, quatre paires V10.1/V10.2 sont entraînées
sur trois folds internes et prédisent le quatrième. Une cinquième paire est
entraînée sur les quatre folds restants pour l'ancre externe. Les réglages
historiques sont conservés : **13 époques V10.1, 17 époques V10.2**, décodages
`ordinal_cumulative_050` et `structured_pb_argmax`.

Cela représente **25 paires d'experts**, partagées entre les bras témoin et
traitement. Les codes historiques construisent et entraînent ces experts ;
les ajouts permettent seulement de fournir les compositions figées et de
sauvegarder leurs poids. Le programme historique d'évaluation V10.4 reconstruit
l'ancre de déploiement avec sa calibration interne. L'ancre interne utilisée
par V27.1 est rejouée une fois et partagée entre les deux bras.

L'isolation concerne ces experts et les réseaux de comptage. La chaîne de
proposition V8.x reste la source commune reconstruite selon le protocole
historique ; elle n'est pas réentraînée séparément à l'intérieur des folds.

## Entraînement apparié

| Fold externe | V26 uniforme | V26 pondéré | V27.2 uniforme |
|---|---:|---:|---:|
| 0 | 12 | 8 | 4 |
| 1 | 10 | 12 | 11 |
| 2 | 10 | 3 | 12 |
| 3 | 8 | 8 | 11 |
| 4 | 8 | 8 | 10 |

Ces budgets déjà sélectionnés dans les archives sont figés pour les deux bras,
en phase interne puis lors du réentraînement final. Aucun nouvel arrêt
anticipé ni choix d'architecture n'est effectué à partir des scores externes.

TensorFlow 2.15.1, Adam 2e-4, lots de 128 et graines des programmes de comptage
historiques sont conservés. Un mélange déterministe explicite des lignes par
époque permet de vérifier l'identité de l'ordre entre les bras, tout en lisant
les grands spectrogrammes par lots. Cet ordre est celui de cette expérience ;
on ne prétend pas retrouver la trajectoire numérique de l'ancien entraînement.

Les pondérations V26 sont calculées uniquement sur la partition d'apprentissage.
V27.2 est entraîné uniquement sur les exemples dont le vrai K est au moins 2.
Ses probabilités sont ensuite calculées sans consulter le vrai K à l'inférence.
Les trois modèles repartent de zéro à chaque phase et chaque bras.

Les empreintes des paramètres initiaux et des ordres de lignes, les graines,
budgets et poids de classes doivent être identiques entre bras. La projection
de décroissance du témoin doit rester exactement nulle. Les seuils V27.1 et
V27.3 de chaque bras sont figés et archivés avant la prédiction des comptes
sur son fold externe. L'ancre partagée ne dépend d'aucun résultat du traitement.

## Évaluation et conservation

La métrique principale est l'Exact-K polyphonique, sur les mêmes 9 354 exemples
dans les deux bras. Le rapport fournit les nombres corrigés/dégradés, le score
global, les résultats par K et par fold, et le F1 événementiel à 50 ms. Un
bootstrap par composition donne un intervalle descriptif ; une seule graine
d'entraînement est utilisée.

La comparaison causale de l'indice est **traitement moins témoin apparié**.
Le score archivé V27.3 **4 005 / 9 401 = 42,6019 %** est affiché séparément :
la population de clusters est différente, donc la différence avec ce score
n'est pas attribuable au seul indice ln. Aucun modèle n'est promu automatiquement.

Les poids et sorties terminées ou partielles de chaque job sont conservés
90 jours dans Actions et archivés avec SHA-256 dans la préversion
`v273-native-decay-<run_id>`. Une archive partielle n'est pas un résultat complet.
