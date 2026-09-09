# V28.0-D — protocole figé du mini-overfit réel interne

Statut au gel : **implémenté et testé localement, exécution GuitarSet réelle pas encore lancée**.

Ce jalon vérifie uniquement que le CQT causal V28, les clusters V10 figés et
les annotations corde/hauteur GuitarSet alimentent correctement le graphe
V28-CHEC. Il répète un même petit lot réel jusqu'au surapprentissage. Il ne
produit aucun score de généralisation et ne peut promouvoir aucun modèle.

## Entrées immuables

| Source | Run | Commit | Artefact | Digest |
|---|---:|---|---|---|
| Clusters V10.4 | `33647694565` | `345dea1e92f6281583c58590a0b1286ce4df459b` | `v104-nested-spectral-bundle` | `sha256:d54768cae40a9cf9b2c97a1be58fb6f52a681af3d79b79b148b5c92a427a9d99` |
| GuitarSet vérifié | `34287254337` | `d11f73079a9eb12b2ebce3c8d1df8209b0bd6bdf` | `v272-verified-guitarset` | `sha256:e90bd513877f8592c7f749410167aa581feb7a4b5d3250f12c980a3b8dc74587` |
| CQT causal V28 | `34323015277` | `3d2fe62b4527236ab91e86dad3cc9c5d81e67b08` | `v280-causal-cqt-outer-clean-cache` | `sha256:2cb10b3042a0ff77ca6e7f59aacbef7a43ff04690aa6f7476a376f3c39eee2ff` |

Le workflow valide d'abord les trois runs, commits, conclusions et digests.
Il contrôle ensuite les MD5 GuitarSet figés. L'artefact V10.4 expire le
9 septembre 2026 ; il est donc téléchargé en premier et remplacé dans ce
jalon par une copie V28 compacte et vérifiable.

## Copie compacte des clusters

Les huit shards V10.4 sont ouverts sans pickle. Seuls les tableaux utiles à
V28 sont désérialisés :

- `sequence`, `mask` et `top_samples`, nécessaires pour reconstruire les
  positions absolues des candidats ;
- `exact`, `slot_targets` et `members`, nécessaires à la supervision ;
- `track_members`, nécessaire au contrat d'appartenance outer-clean.

Les anciennes cartes `spectral`, ainsi que `stats` et `target`, ne sont ni
chargées ni copiées. Le fichier `v280-cluster-metadata.npz` contient les 240
pistes outer-clean, jamais les 60 pistes de validation historique ni le
locked12. Son manifeste enregistre le SHA-256 du fichier, le digest des
identités, les huit chemins source et la provenance V10.4. Une seconde lecture
complète doit réussir avant l'entraînement. L'artefact compact est conservé
90 jours.

## Sélection interne figée

La sélection est déterministe et ne consulte d'abord que les identités :

1. regrouper `comp` et `solo` par composition avec `group_stem` ;
2. trier les noms de compositions ;
3. prendre exactement les trois premiers groupes et laisser tous les autres
   intacts ;
4. ouvrir les payloads JAMS de ces seuls groupes ;
5. apparier chaque note à la candidate causale la plus proche dans un rayon
   maximal de 20 ms ;
6. exiger que l'occupation des six cordes reproduise exactement les cibles
   `slot_targets` figées ;
7. parmi les lignes où le nombre de cordes occupées égale `exact`, prendre
   28 lignes par tour de rôle entre les classes Exact-K disponibles.

Les valeurs cibles servent donc à équilibrer les 28 lignes après le choix des
groupes, mais jamais à choisir les groupes. Ce mini-lot n'est ni un train/test
split, ni un fold externe.

Pour chaque ligne retenue, le crop CQT a la configuration figée
`5fae41ea91e90b5cb2a7611edb41c6dba633eaf8853341a481ea3a2c9ba1e805`.
Sa décision s'arrête à `cluster_start + 1764` échantillons. Les cibles sont :

- Exact-K catégoriel `0..6` ;
- naissance par corde ;
- position corde/frette sur 6 × 20 cases ;
- hauteur MIDI binaire entre 40 et 83 ;
- Exact-K Poisson-binomial auxiliaire.

La corde 0 est le mi grave et la corde 5 le mi aigu. Deux cordes jouant la
même hauteur restent deux cibles corde/frette, mais une seule cible de hauteur.

## Entraînement et gates

Le graphe harmonique figé `v280_chec_direct` comporte exactement 110 402
paramètres. Avec le seed `28034`, le même lot de 28 lignes est présenté pendant
160 pas Adam. Aucun checkpoint ou teacher n'est chargé.

Le jalon ne réussit que si :

1. toutes les pertes initiales, intermédiaires et finales sont finies ;
2. la perte totale baisse d'au moins 20 % ;
3. l'accuracy Exact-K sur le lot mémorisé atteint au moins 70 % ;
4. cette accuracy est strictement supérieure à sa valeur initiale ;
5. tous les contrôles d'appartenance, de causalité, de forme et de provenance
   restent vrais.

L'accuracy ci-dessus est seulement un test de mémorisation du lot
d'entraînement. Le rapport interdit explicitement de la présenter comme une
mesure holdout ou de généralisation. La validation historique, le locked12 et
les folds externes restent fermés. En cas de réussite, l'étape suivante sera
une ablation harmonique sur partitions internes, sous un protocole séparé.
