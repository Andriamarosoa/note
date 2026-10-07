# Verdict V27.3 — référence hidden1 historique

Date : 7 octobre 2026.

## Question confirmatoire

Le groupe historique `candidate_hidden1 [42,52,61,64]`, appliqué au-dessus de
`freeze_local_combo`, est-il assez stable pour devenir la référence robuste ?

Deux runs ont testé exactement ce groupe, sans recherche d'un autre groupe :

- [nested fixe 37572714248](https://github.com/Andriamarosoa/note/actions/runs/37572714248),
  commit `a828c5fea0169c4d83fe0f49b2d12e7988066109` ;
- [confirmation pairwise 37577749778](https://github.com/Andriamarosoa/note/actions/runs/37577749778),
  commit `f4dd100cfd53a00547fb7146d374ef5f09b1da12`.

Les deux runs sont `completed / success`. Le second a entraîné une seule fois
chacune des six paires de folds, puis a évalué les deux folds autorisés restants.
Il a été lancé avant observation du résultat du premier run.

## Périmètre vérifié

- entraînement, sélection et validation : folds `0,1,2,4` seulement ;
- fold 3 exclu ;
- players `00–04` seulement, player 05 exclu ;
- même groupe fixe `[42,52,61,64]` et même règle préenregistrée ;
- aucune recherche d'un autre neurone, groupe, seuil ou feature ;
- aucune promotion automatique.

## Résultats

Le groupe fixe donne les mêmes résultats descriptifs externes dans les deux
runs :

| fold externe | global | low | poly |
|---:|---:|---:|---:|
| 0 | +2 | +1 | +1 |
| 1 | +8 | +5 | +3 |
| 2 | +14 | +14 | 0 |
| 4 | +17 | +7 | +10 |
| **total** | **+41** | **+27** | **+14** |

L'ancienne règle externe est donc satisfaite. La règle nested pré-déclarée ne
l'est pas :

| fold externe | premier run | pairwise | somme inner premier run | somme inner pairwise |
|---:|:---:|:---:|---:|---:|
| 0 | rejeté | rejeté | +2 | +12 |
| 1 | rejeté | rejeté | +10 | -4 |
| 2 | rejeté | rejeté | -10 | -10 |
| 4 | rejeté | rejeté | +27 | -2 |

Dans les deux réalisations, **0 rotation sur 4** est acceptée. La politique
nested s'abstient donc partout : net global sélectionné **0**.

## Audit de reproductibilité numérique

Les checkpoints uniform des paires `(0,1)`, `(0,4)`, `(1,4)` et `(2,4)` sont
bit à bit identiques entre les deux runs. Ceux des paires `(0,2)` et `(1,2)`
diffèrent, alors que les listes d'ordre des huit époques sont identiques. Les
deux copies de chaque paire à l'intérieur du premier run sont, elles, identiques.

Cette observation établit une variation numérique entre runners pour deux
entraînements ; elle n'en établit pas à elle seule la cause matérielle ou
logicielle. Elle empêche d'annoncer une reproduction bit à bit de tous les
scores inner. Elle ne rend pas le verdict ambigu : les deux jeux de checkpoints
rejettent les quatre rotations selon la même règle gelée.

Les empreintes et valeurs comparées sont archivées dans
[`reconciliation.json`](evidence/v273-reference-hidden1-confirmation/reconciliation.json).

## Référence conservée et Exact-K

Le groupe `[42,52,61,64]` est **non confirmé**. La référence reste
`freeze_local_combo` :

| vrai K | exacts / cas | exact |
|---:|---:|---:|
| 0 (silence) | 38 035 / 39 652 | 95,9220 % |
| 1 | 7 889 / 12 272 | 64,2846 % |
| 2 | 1 183 / 3 445 | 34,3396 % |
| 3 | 915 / 2 289 | 39,9738 % |
| 4 | 417 / 1 207 | 34,5485 % |
| 5 | 15 / 355 | 4,2254 % |
| 6 | 0 / 89 | 0,0000 % |

Exact global : **48 454 / 59 309 = 81,6976 %**. Exact polyphonique K2–K6 :
**2 530 / 7 385 = 34,2586 %**.

## Preuves publiées

- [release du run nested fixe](https://github.com/Andriamarosoa/note/releases/tag/v273-reference-hidden1-nested-37572714248) ;
- [release du run pairwise](https://github.com/Andriamarosoa/note/releases/tag/v273-reference-hidden1-pairwise-37577749778) ;
- `report.json` pairwise vérifié :
  `897820e58a90ca95b7c24757f7df2b6011f126601eccd5da426847bd92dc4147` ;
- archive pairwise vérifiée :
  `5adfd69ff483de382a0afeb13de8d756be30125b0dc3675a3329cdb946d9bfe0`.

## Décision

Ne pas promouvoir `candidate_hidden1 [42,52,61,64]`. Conserver
`freeze_local_combo`. La re-sélection nested conditionnelle n'est pas exécutée,
car son prérequis — confirmation du groupe fixe — a échoué. Aucun nouvel
entraînement ni retry n'est nécessaire pour cette décision.
