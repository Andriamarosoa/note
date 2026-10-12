# S205–S207 — Pont expérimental S45→S46 et audit des 19 branches

Date : 12 octobre 2026. Branche : `codex/v273-scope-rescue`.

**Verdict : S80 N'EST PAS promu.** Référence officielle inchangée :
`freeze_local_combo`. Aucun score Exact-K indépendant n'est mesuré ici.
Les 57 événements GOAT sont des exemples de débogage déjà exposés ;
leurs labels/MIDI n'ont jamais été ouverts dans S205, S206 ni S207.

## Preuve exécutable du pont expérimental

- [S205 réussi (38236726481)](https://github.com/Andriamarosoa/note/actions/runs/38236726481) — quatre folds représentatifs, 16 familles, leurs 304 checkpoints Ridge historiques et 193 sorties S46 par fold.
- [S206 réussi (38237264206)](https://github.com/Andriamarosoa/note/actions/runs/38237264206) — **19 branches originales conservées**, réparties entre folds 0/1/2/4 comme 5/4/5/5, 16 familles S45 et 193 sorties S46 pour chacune, sur les mêmes 57 événements. [Artefact S206](https://github.com/Andriamarosoa/note/actions/runs/38237264206/artifacts/11740897871).
- [S207 réussi (38237535336)](https://github.com/Andriamarosoa/note/actions/runs/38237535336) — audit **indépendant et en lecture seule** des 19 sorties S206, sans vérité terrain, sans sélection de politique, sans entraînement. [Artefact S207](https://github.com/Andriamarosoa/note/actions/runs/38237535336/artifacts/11741517275).

Sources : [S205](../scripts/s205_experimental_s45_s46_bridge.py),
[S206](../scripts/s206_all_19_s45_s46_branches.py),
[S207](../scripts/s207_audit_original_19_branch_stability.py).

Les 5 familles S1/S2/S3/S4/repair_coherent proviennent du
[runtime historique reconstruit](README_V273_RECOVERED_FIVE_FAMILIES.md).
Les 11 autres sont les sorties sauvegardées de S173, avec des parents
S5/S18 *adaptés*, dont S35 avec S20 en remplacement du vrai S29.
**Ce pont ne reconstitue pas un S45 historique fidèle à 16 familles.**
Les risques calculés utilisent cependant les 304 checkpoints Ridge
originaux S45 sur les véritables parents S18 par fold ; c'est un test
des interfaces et des divergences, non un score de candidat S80.

## Résultat S207 : stabilité et divergences sans annotations

| Fold | Nombre de branches | Différences entre décisions S46 au sein du fold | Différences de propositions S45 (vs première branche, cumul) | Différences de disponibilité des risques Ridge |
|---|---:|---:|---:|---:|
| 0 | 5 | 0 | 2 | 0 |
| 1 | 4 | 0 | 8 | 6 |
| 2 | 5 | 0 | 6 | 0 |
| 4 | 5 | 0 | 8 | 0 |

- **0 différence de décision S46 à l'intérieur d'un même fold** malgré les changements de proposition / risque S45.
- **904 / 11 001 couples (événement, politique) instables** entre les quatre folds; **10 097 / 11 001 stables**. L'unité est le couple événement–politique, **pas** le nombre de morceaux, d'événements erronés, de corrections ou de régressions.
- Nombre de cellules différentes de leur propre parent S18, sur les 192 politiques alternatives, par fold : **8, 58, 146, 192** (folds 0, 1, 2, 4). Cela ne signifie **pas** gain net ni Exact-K correct.

### Conséquences vérifiées

1. Le choix entre les **19 branches à l'intérieur d'un même fold** n'a pas modifié les décisions S46 sur GOAT (bien que des éléments intermédiaires aient varié).
2. Le choix entre les **quatre folds** peut modifier les décisions S46. Une règle de déploiement **doit** donc être définie et figée *avant* d'ouvrir les labels d'un nouveau corpus ; il serait incorrect d'en sélectionner une à partir des annotations GOAT déjà exposées.
3. Les parents S18 reconstruits et ceux de S173 divergent sur **3, 4, 5 et 4 événements** respectivement pour les folds 0, 1, 2 et 4. Les onze anciennes sources doivent être recalculées avec les mêmes véritables parents avant de prétendre à une équivalence historique.
4. L'exécution S46 n'est **pas** l'exécution de l'aval S69/S72/S79/S80. Le véritable S29 et la gestion des quatre folds doivent aussi être traités ; les performances globales et polyphoniques doivent ensuite être évaluées **en paire sur de nouveaux morceaux indépendants** contre `freeze_local_combo`.

Aucun changement de référence officielle, aucune déclaration de promotion,
aucun résultat d'Exact-K indépendant.


## S208 — Attribution des 904 divergences au parent S18

[S208 — Run indépendant réussi (38237895315)](https://github.com/Andriamarosoa/note/actions/runs/38237895315) ;
[preuve archivée](https://github.com/Andriamarosoa/note/actions/runs/38237895315/artifacts/11741313372).
Le script [S208](../scripts/s208_s46_parent_divergence_attribution.py)
relit les seuls vecteurs S206, confirme le résultat S207 (904 couples divergents)
et classe chaque couple sans consulter les annotations.

| Diagnostic entre les quatre folds | Mesure |
|---|---:|
| Événements dont les 4 parents S18 divergent | **6 / 57** |
| Événements dont les 4 parents S18 sont égaux | **51 / 57** |
| Couples événement–politique divergents entre folds | **904 / 11 001** |
| Couples divergents sans aucun changement S46 par rapport aux 4 parents | **858 / 904** |
| Couples divergents avec au moins un changement S46 par rapport à son parent | **46 / 904** |
| Couples divergents lorsque les 4 parents sont déjà unanimes | **0** |

**Conclusion circonscrite :** parmi les 904 cellules instables, **858
reproduisent directement les divergences entre les parents S18**,
sans qu'aucun des quatre S46 ne modifie son parent. Les 46 autres
incluent une intervention S46 dans au moins un fold ; ceci n'établit
pas une causalité exclusive de S46. Aucune divergence n'apparaît
lorsque les parents sont unanimes, sur les 57 événements GOAT.

Il faut donc **prioriser la stabilité et le routage des parents S5→S18
entre folds** avant d'attribuer la fragilité à la logique S46. Les
résultats GOAT sont du diagnostic exposé, **pas** une validation
indépendante, encore moins une preuve que S80 est supérieur à la
référence gelée. Aucune politique de fusion ou promotion n'est choisie.
