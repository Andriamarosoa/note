# V28.0-F — diagnostic du sous-comptage

Diagnostic post-hoc du 12 septembre 2026. **V27.3 reste la référence.**
L'étape réalisée ici est l'analyse des résultats archivés. Le prochain essai
de pondération décrit plus bas n'a pas encore été entraîné.

Source : [run 34462604665](https://github.com/Andriamarosoa/note/actions/runs/34462604665),
commit `50e615bfb6cbce6ac5bbc57843ef3d1378e1bf79`, cinq folds et agrégation
terminés avec succès le 11 septembre. Les calculs détaillés sont dans
[v280-undercount-diagnostic.json](v280-undercount-diagnostic.json) ; le
[script reproductible](../scripts/audit_v280_undercount.py) lit sept ZIP figés.

## Résultat vérifié

| Mesure | V27.3 | V28 harmonique |
|---|---:|---:|
| Exact-K polyphonique | 42,6019 % | 28,1459 % |
| Exact-K global | 81,4897 % | 80,2613 % |
| F1 événements à 50 ms | 79,5625 % | 76,7656 % |
| Sous-comptage parmi les lignes polyphoniques | 40,8254 % | 58,3980 % |

Le déficit est de **1 359 comptages exacts**, soit **−14,4559 points** sur
9 401 lignes polyphoniques. Les cinq folds sont négatifs. L'intervalle
bootstrap apparié par piste à 95 % est `[−16,9959 ; −12,0815]` points.
Ce bootstrap concerne ce développement, après sélection d'architecture ;
il ne constitue pas une validation indépendante.

Le diagnostic vérifie les SHA-256 des sept ZIP, les fichiers référencés dans
les manifestes, l'égalité entre prédictions par fold et prédictions agrégées,
les 76 768 indices uniques et les 240 pistes. Il recalcule les métriques
V28 et V27.3, reproduit les prédictions internes sélectionnées, les partitions,
les ordres de chaque époque et les compteurs de mises à jour. Les poids
évalués correspondent aux états de refit archivés, y compris la reprise
exacte du fold 0. Les trois implémentations scientifiques utilisées sont
également fixées par SHA-256.

Le cache CQT complet et l'audio ne sont pas chargés dans ce diagnostic.
Les F1 événementiels et le bootstrap ci-dessus sont ceux de l'agrégation
figée ; ils ne sont pas recalculés par ce script. Aucune nouvelle inférence,
modification de prédictions ou recherche de seuil n'est réalisée.

## Où se trouvent les pertes

| Vrai K | Lignes | Exact-K V27.3 | Exact-K V28 | Différence de corrects |
|---:|---:|---:|---:|---:|
| 0 | 51 956 | 95,1093 % | 93,8255 % | −667 |
| 1 | 15 411 | 59,2953 % | 66,3228 % | +1 083 |
| 2 | 4 279 | 51,4840 % | 27,5064 % | **−1 026** |
| 3 | 2 952 | 39,1260 % | 39,1938 % | +2 |
| 4 | 1 628 | 33,7838 % | 17,0762 % | −272 |
| 5 | 438 | 19,8630 % | 7,5342 % | −54 |
| 6 | 104 | 9,6154 % | 0,9615 % | −9 |

**K=2 explique 75,50 % du déficit net polyphonique.** Les vrais K=2 sont
prédits K=1 dans 1 351 cas avec V28, contre 483 avec V27.3. La dégradation
ne concerne donc pas seulement les accords très denses. Les K=4 à 6 perdent
335 corrects supplémentaires ; le K=3 en gagne deux.

Sur les lignes polyphoniques, les deux modèles réussissent ensemble 1 448
fois. V28 perd 2 557 cas que V27.3 réussissait et en gagne 1 198 que V27.3
manquait ; 4 198 cas restent faux pour les deux. La perte nette ne doit pas
être confondue avec le nombre total de régressions.

La baisse apparaît chez chacun des cinq interprètes présents dans ces
données. Elle touche les versions `comp` (29,09 % contre 42,31 %, 8 779
lignes polyphoniques) et `solo` (14,79 % contre 46,78 %, seulement 622 lignes
polyphoniques). Ces sous-groupes servent au diagnostic, sans choisir un
décodeur différent après lecture des résultats externes.

## Explications qui ne rendent pas compte de l'ampleur du déficit

**Supervision auxiliaire masquée.** La tête principale conserve toutes les
cibles, avec un poids de 1. Les quatre têtes auxiliaires sont masquées sur
seulement cinq lignes, dont deux polyphoniques. Sur les 9 399 lignes
polyphoniques dont la supervision auxiliaire est disponible, V28 reste à
28,14 %, contre 42,61 % pour V27.3. Le masquage ne peut expliquer le déficit.
Cela ne constitue toutefois pas une validation manuelle de toutes les cibles.

**Fin d'enregistrement.** Les 79 crops tronqués à la fin de l'audio ne
contiennent aucune ligne polyphonique. Ils ne peuvent expliquer les
5 490 sous-comptages polyphoniques.

**Candidats et ranking.** Les candidats disponibles sont moins nombreux que
le vrai K sur 180 lignes ; seules 154 font partie des sous-comptages de V28.
Le ranking figé présente une pénurie sur 174 lignes, dont 149 sous-comptées.
Ces limitations peuvent affecter la récupération des événements, mais
concernent une petite fraction des erreurs. Le ranking top-K est appliqué
après le choix de cardinalité et n'est pas une correction de l'Exact-K.

**Interruption ou mauvais checkpoint.** Les historiques sont complets :
12 époques internes pour chaque fold, puis respectivement 2, 12, 5, 10 et
8 époques de refit. La règle « corrects polyphoniques, puis NLL, puis époque
antérieure » reproduit chaque choix. Le diagnostic vérifie les fichiers et
157 424 mises à jour documentées, pour 97 époques au total. Il ne relève
aucune anomalie dans ces contrôles.

## Hypothèses encore ouvertes

### 1. Objectif d'apprentissage et déséquilibre des exemples

Les K=0 ou 1 représentent **87,7540 %** des lignes, alors que le critère
principal du projet évalue seulement K≥2. Dans les partitions internes
d'ajustement, la part polyphonique varie de 11,75 % à 12,26 %. La perte
principale est une entropie croisée non pondérée.

Le sampler place une amorce polyphonique dans les lots, mais présente chaque
ligne exactement une fois par époque : il modifie l'ordre, pas le poids
total des classes. Le gain de 1 083 corrects sur K=1 accompagné de la chute
sur K=2 est compatible avec un compromis défavorable à la polyphonie.

**Ce n'est pas une cause démontrée.** Les lignes polyphoniques contribuent
déjà 43,81 % de la NLL OOF non pondérée, bien qu'elles ne représentent que
12,25 % des exemples. Cette contribution mesurée sur les prédictions de
développement n'est pas celle des gradients d'entraînement. Il faut un
contrôle expérimental pour savoir si une pondération améliore le résultat.

### 2. Généralisation des probabilités et durée d'entraînement

| Fold | Époque choisie | Perte principale en entraînement, époque 1 → 12 | NLL polyphonique interne, époque 1 → 12 |
|---:|---:|---:|---:|
| 0 | 2 | 0,781 → 0,231 | 1,768 → 1,891 |
| 1 | 12 | 0,774 → 0,231 | 1,881 → 2,272 |
| 2 | 5 | 0,809 → 0,233 | 2,292 → 2,723 |
| 3 | 10 | 0,800 → 0,236 | 2,287 → 2,690 |
| 4 | 8 | 0,805 → 0,230 | 2,126 → 2,806 |

La perte d'entraînement diminue dans les cinq probes, tandis que la NLL
polyphonique de validation finit plus haute qu'à l'époque 1. C'est un
signal compatible avec du surapprentissage ou une mauvaise généralisation
des probabilités. Il ne justifie pas de simplement prolonger à 24 époques.
L'Exact-K interne fluctue et ne suit pas nécessairement la NLL : le fold 1
choisit bien l'époque 12 selon la règle déclarée.

Les pertes d'entraînement sont des moyennes de lots pendant l'optimisation,
pas une évaluation séparée du modèle final sur les données d'apprentissage.
Les archives ne contiennent pas cette dernière mesure.

| Fold | Exact-K du probe sur sa validation interne | Exact-K du refit sur son fold externe |
|---:|---:|---:|
| 0 | 38,23 % | 23,02 % |
| 1 | 31,69 % | 27,82 % |
| 2 | 26,05 % | 24,88 % |
| 3 | 30,60 % | 29,90 % |
| 4 | 29,37 % | 35,41 % |

Ces colonnes concernent des modèles et compositions différents : leurs
écarts ne prouvent pas que le refit endommage un modèle. Le budget est fixé
en époques ; avec quatre folds au lieu de trois, le refit effectue environ
30 à 35 % de mises à jour supplémentaires pour le même nombre d'époques.
C'est une propriété du protocole initial, pas un défaut de reprise identifié.

### 3. Représentation et combinaison des têtes

La sortie est `softmax(log(max(PoissonBinomial(string_birth), 1e-7)) + residual_logits)`.
Un sous-comptage peut provenir des probabilités par corde, du résidu, ou de
la représentation commune. Les NPZ sauvegardés contiennent uniquement la
probabilité finale de cardinalité. Ils ne permettent pas d'attribuer le
déficit à une de ces composantes, ni à l'agrégation harmonique seule.

Parmi les 5 490 sous-comptages, la vraie classe est strictement deuxième
dans seulement 1 908 cas ; 619 erreurs ont une probabilité maximale d'au
moins 0,8. Le problème dépasse donc des égalités numériques ou de très
petites marges. Une température scalaire positive conserve l'argmax et
ne peut, seule, améliorer l'Exact-K de ce décodeur.

La préférence harmonique de V28.0-E comparait deux variantes V28 sur un seul
split ; elle ne démontrait pas une supériorité sur V27.3. Les variantes E
avaient aussi des nombres de paramètres différents.

## Prochain essai recommandé : pondérer seulement la perte principale

L'hypothèse à tester en premier est le poids relatif de la polyphonie dans
l'objectif. Un essai apparié interne est plus informatif qu'une relance
identique des cinq folds. Cette proposition est formulée après observation
de F ; tout résultat futur sur ces compositions reste du développement.

1. Utiliser d'abord la partition interne du fold externe 0 : ajustement sur
   les folds 2, 3, 4 ; validation sur le fold 1. Le fold 0 ne sert pas à
   sélectionner cet essai. Réutiliser le cache préparé dont le SHA est figé.
2. Entraîner deux bras depuis zéro, avec le même seed `28035`, les mêmes
   initialisations, données, ordres de lots, architecture harmonique, Adam
   `2e-4`, batch 32, 12 époques et sélection de checkpoint que F.
3. Bras témoin : poids principal 1. Bras expérimental :
   `w(K<2)=N/(2*N_nonpoly)` et `w(K>=2)=N/(2*N_poly)`, calculés uniquement
   sur les lignes d'ajustement. Chaque groupe représente alors la moitié
   du poids total ; le poids moyen reste 1. Pour ce premier split :
   `N=46921`, `N_poly=5514`, `N_nonpoly=41407`, soit environ 4,2547 pour
   une ligne polyphonique et 0,5666 pour une autre ligne.
4. Conserver toutes les pertes et tous les masques auxiliaires existants.
   Ne modifier simultanément ni le sampler, ni le budget, ni la tête de
   cardinalité. Le facteur de pondération n'est pas recherché sur les
   prédictions externes de F.
5. En validation interne, conserver les probabilités finales, Poisson-binomial,
   par corde et les logits résiduels. Rapporter Exact-K polyphonique et global,
   confusion par K, sous/sur-comptage, NLL et Brier pour les deux bras.
   Comparer leur meilleure époque selon la règle fixée, en publiant aussi
   les courbes complètes et le dernier état.
6. Examiner le résultat apparié avant de décider d'une confirmation sur une
   seconde partition interne, puis éventuellement d'un nouveau protocole
   externe. Un gain face au témoin V28 ne constitue pas un gain face à V27.3.
   Les critères de gain matériel de F face à V27.3 ne sont pas redéfinis.

La pondération est une hypothèse testable, pas une correction validée. Le
diagnostic n'établit pas encore une cause unique du sous-comptage.

## Reproduction

Télécharger depuis le run source les sept artefacts suivants, en conservant
ces noms : `v280-f-outer-comparison.zip`, `v280-f-preparation-audit.zip` et
`v280-f-fold-0.zip` à `v280-f-fold-4.zip`. IDs et SHA-256 exacts sont fixés
dans `ARTIFACTS` du script et reproduits dans le JSON.

Depuis la racine du dépôt, avec NumPy disponible :

```bash
python -B scripts/audit_v280_undercount.py \
  --input-dir /chemin/vers/les/sept/zip \
  --output analysis/v280-undercount-diagnostic.json
```

L'analyse fonctionne sans TensorFlow, GPU, audio, ni téléchargement du cache
complet de features. Elle rejette des archives ou implémentations scientifiques
différentes de celles prévues. Les seules écritures du script sont le JSON
de diagnostic et son répertoire de sortie.
