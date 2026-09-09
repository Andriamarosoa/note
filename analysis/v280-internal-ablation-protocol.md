# V28.0-E — entraînement comparatif sur partitions internes

Protocole fixé avant le premier entraînement V28.0-E.

L'expérience entraîne deux variantes de V28-CHEC depuis zéro sur les données
GuitarSet réelles : agrégation harmonique à plusieurs profondeurs et contrôle
sans agrégation harmonique. Le score interne sert à comparer ces deux variantes.
V27.3 reste la référence ; ce jalon n'entraîne aucune promotion et ne lance
aucune évaluation outer.

## Sources immuables

| Source | Run | Commit | Artefact | Digest ZIP GitHub |
|---|---:|---|---|---|
| Métadonnées V28 issues de V10.4 | `34346175521` | `97c158e8c8b110b85eeac80a336a06768a95b936` | `v280-cluster-metadata` | `sha256:dccf400794cd8d50a9aab8b13a1292a1d6e97696315a96ba21beb5ec8e423d8c` |
| GuitarSet vérifié | `34287254337` | `d11f73079a9eb12b2ebce3c8d1df8209b0bd6bdf` | `v272-verified-guitarset` | `sha256:e90bd513877f8592c7f749410167aa581feb7a4b5d3250f12c980a3b8dc74587` |
| CQT causal V28 | `34323015277` | `3d2fe62b4527236ab91e86dad3cc9c5d81e67b08` | `v280-causal-cqt-outer-clean-cache` | `sha256:2cb10b3042a0ff77ca6e7f59aacbef7a43ff04690aa6f7476a376f3c39eee2ff` |

Le payload NPZ des métadonnées a le SHA-256
`666a5f55e62a9d8f78f45a432e771a998341ea2662980c400aa807edbe97f8ba`.
Le workflow exige des runs sources réussis, leurs commits et digests exacts,
les MD5 GuitarSet et l'intégrité de chaque piste CQT. Le manifeste compact
préserve également la provenance ultime V10.4.

## Partitions de compositions

La fonction `_balanced_group_folds` de V10.4, reprise par V24, assigne les
compositions selon les identités et le nombre de lignes, sans consulter les
valeurs cibles. Les 240 pistes outer-clean et 76 768 lignes sont identiques à
la source figée.

| Rôle V28.0-E | Folds canoniques | Lignes | Pistes |
|---|---|---:|---:|
| Entraînement interne | 2, 3, 4 | 46 921 | 150 |
| Validation interne | 1 | 14 001 | 40 |
| Réservé pour une étape ultérieure | 0 | 15 846 | 50 |

Les versions `comp` et `solo`, ainsi que tous les interprètes d'une même
composition, restent dans la même partition. Les listes de groupes, effectifs
et empreintes des indices globaux sont enregistrés avant de construire les
cibles MIDI. Aucun groupe n'est choisi d'après un résultat d'entraînement.

Les métadonnées compactes de toutes les lignes sont désérialisées pour
reproduire les identités et le split. Seuls les JAMS et crops des folds 1 à 4
sont utilisés ensuite. V28.0-E n'entraîne ni ne prédit les lignes du fold 0.
Les 60 pistes de validation historique sont indexées uniquement par noms pour
vérifier leur exclusion ; leurs annotations ne sont pas lues. Locked12 n'est
pas indexé.

Les trois groupes de V28.0-D ont déjà servi au diagnostic de supervision :
`BN2-166-Ab` appartient au fold 0, `BN3-154-E` au fold 1 et `BN1-147-Gb` au
fold 3. Les poids de D ne sont pas réutilisés. Ce contexte et les expériences
antérieures interdisent de présenter le présent score interne comme une
validation indépendante. La sélection des variantes sur ce split ne devra pas
être présentée ultérieurement comme invisible à chacun des cinq folds.

## Cibles et représentation

Les entrées restent des cartes CQT causales `24 × 238 × 3`, configuration
`5fae41ea91e90b5cb2a7611edb41c6dba633eaf8853341a481ea3a2c9ba1e805`.
Le crop se termine à `min(cluster_start + 1764, longueur_audio)` échantillons.
À la fin d'un enregistrement, la décision est vidée sur la dernière trame
causale disponible : aucun échantillon futur ni silence artificiel n'est
ajouté, et la ligne conserve sa cible de comptage. Le nombre de lignes ainsi
traitées, leurs indices dans la piste et la durée post-onset manquante sont
enregistrés. Les crops qui disposent des 40 ms complets restent identiques.
Les cartes sont
matérialisées en float16 et converties en float32 par lot. Le cache préparé
est commun aux deux variantes et vérifié par SHA-256 à chaque chargement.

Les cinq sorties et poids de pertes sont ceux du graphe V28.0-B : Exact-K
catégoriel `1,0`, naissance par corde `0,25`, corde/frette `0,15`, hauteur MIDI
`0,10` et cardinalité Poisson-binomiale `0,10`.

Toutes les lignes conservent leur cible Exact-K et leur poids principal `1`.
Les hauteurs MIDI continues sont quantifiées au demi-ton le plus proche. Les
masques suivants sont figés avant le premier résultat :

- si le nombre de cordes occupées diffère d'Exact-K, masquer les quatre pertes
  auxiliaires pour cette ligne ;
- si l'appariement MIDI est incomplet, une hauteur tombe à la frontière de
  50 cents ou la position sort de la grille de 20 frettes, masquer uniquement
  les pertes corde/frette et hauteur (en plus de la règle précédente) ;
- les annotations actives non finies interrompent la préparation ;
- conserver et publier les nombres de lignes masquées, les raisons, les
  diagnostics de quantification et d'appariement par piste.

Cette règle évite qu'une limitation des cibles auxiliaires retire une ligne
de l'évaluation Exact-K. Il n'y a ni exclusion de lignes selon le score ni
modification des cibles de comptage.

## Budget et équité de la comparaison

| Paramètre | Valeur commune |
|---|---|
| Initialisation | depuis zéro, seed `28035` |
| Optimiseur | Adam, learning rate `0,0002` |
| Époques | 12 complètes, sans early stopping |
| Taille des lots | 32, dernier lot partiel conservé |
| Passages par ligne d'entraînement | exactement 12 |
| Mises à jour par variante | 17 604 |
| Augmentation et pondération de classes | aucune |
| Préentraînement / teacher / poids D | aucun |

Chaque époque parcourt toutes les lignes une fois, en mélangeant les files
par groupe de composition. Une ligne polyphonique amorce chaque lot tant que
le stock le permet, sans duplication ni suppression de lignes. Les empreintes
des ordres d'entraînement sont comparées entre variantes.

L'initialisation Glorot des couches communes dépend de leur nom, afin de
garantir les mêmes valeurs initiales malgré les couches supplémentaires du
modèle harmonique. Les biais et la dernière couche du résidu catégoriel restent
nuls. Les deux architectures gardent leurs dimensions déjà implémentées :
110 402 paramètres pour le modèle harmonique, 67 298 pour le contrôle.
Il s'agit d'une suppression de module, pas d'une comparaison à nombre égal
de paramètres ; la différence de capacité doit accompagner l'interprétation.

Les deux entraînements tournent successivement, un seul job lourd à la fois.
Chaque variante dispose d'un plafond de six heures. Les poids du meilleur
checkpoint et du dernier état, les pertes, les prédictions de validation et
les journaux sont conservés, y compris les sorties partielles en cas d'échec.

## Sélection interne et rapport

À chaque époque, évaluer toutes les 14 001 lignes de validation, sans
rééchantillonnage. Sélectionner le checkpoint ayant le plus de lignes
polyphoniques exactes ; en cas d'égalité, préférer la NLL polyphonique la plus
faible, puis l'époque la plus ancienne. Les deux variantes terminent leurs
12 époques même si leur meilleur checkpoint est antérieur.

Le comparateur relit les prédictions, vérifie les identités, digests, budgets,
initialisations et ordres de lots, puis recalcule les métriques. La préférence
interne va au modèle harmonique uniquement s'il corrige strictement plus de
lignes polyphoniques que le contrôle. En cas d'égalité ou de régression, elle
va au contrôle. Cette règle sélectionne une piste de développement, sans
promouvoir V28 ni déclencher automatiquement les folds externes.

Publier pour chaque variante : Exact-K polyphonique et global, NLL globale et
polyphonique, Brier, confusion `0..6`, effectifs/corrects/sous-comptes/sur-comptes
par vrai K, historique des 12 époques, époque retenue, durée et nombre de
paramètres. Publier le delta apparié entre variantes, même s'il est négatif.
Un seul seed et un seul split interne ne constituent pas une preuve de
robustesse. Les métriques événementielles et le ranking ne sont pas évalués
dans cette comparaison de compteurs.

Le cache préparé est conservé 7 jours, les deux entraînements et l'audit
30 jours, la comparaison finale 90 jours dans les artefacts GitHub Actions.

## Correction avant le premier entraînement

Le run `34350199265` a passé les 37 tests mais s'est arrêté pendant la
préparation de `00_BN1-147-Gb_solo.jams` : une décision demandée à 864 202
échantillons dépassait la longueur audio de 863 725. Aucun entraînement ni
score de validation n'avait commencé. Le traitement EOF ci-dessus est fixé
avant relance, avec un test qui vérifie l'absence de lecture de futur et
l'identité des crops ordinaires. Les partitions, lignes, budgets, pertes et
règles de sélection sont inchangés.
