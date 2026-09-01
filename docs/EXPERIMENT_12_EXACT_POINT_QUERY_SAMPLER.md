# Expérience 12 — audit du sampler ponctuel causal exact

État figé le 1er septembre 2026. Le sampler candidat a été simulé et audité,
mais il n'a pas été intégré au code d'entraînement. Aucun modèle, aucune loss,
aucun décodeur et aucun comportement live n'ont été modifiés. Aucun
entraînement n'a été lancé.

## Verdict

Le sampler est **mathématiquement cohérent**, mais **pas prêt pour un
entraînement**.

Il respecte bien la cible demandée : une requête porte sur une seule position
`t` et produit deux comptes anonymes indépendants :

```text
onset_count[t]
offset_count[t]
```

Une position contenant trois onsets reste une seule position tirable avec la
cible `onset_count[t] = 3`. Elle ne reçoit pas trois fois plus de chances
d'être sélectionnée. Une position contenant simultanément onset et offset
conserve les deux cibles positives.

Le cycle candidat tire successivement :

1. une position contenant au moins un onset ;
2. une position contenant un offset mais aucun onset ;
3. une position sans frontière.

Cette partition est exacte et sa correction analytique de prior fonctionne.
Deux défauts des données la rendent cependant inutilisable telle quelle :

- le fond uniforme montre presque aucun négatif immédiatement voisin d'une
  vraie frontière ;
- les comptes simultanés rares `2` et surtout `3` sont presque absents des
  batches.

Le premier défaut est directement compatible avec une sur-prédiction de pics
trop larges : le modèle voit le centre positif, mais presque jamais les
échantillons adjacents qui devraient redevenir négatifs.

## Partition exacte du flux

Les positions communes onset+offset appartiennent à `onset-bearing`, sans
perdre leur cible offset.

| Split | Onset-bearing | Offset-only | Fond | Onset+offset au même `t` | Total |
|---|---:|---:|---:|---:|---:|
| Train | 44 059 | 43 786 | 320 054 505 | 47 | 320 142 350 |
| Validation | 9 496 | 9 444 | 82 928 740 | 7 | 82 947 680 |

La densité réelle est donc extrêmement faible :

| Split | Type | Positions positives | Instances | Densité du flux |
|---|---|---:|---:|---:|
| Train | Onset | 44 059 | 44 220 | 0,013762 % |
| Train | Offset | 43 833 | 43 894 | 0,013692 % |
| Validation | Onset | 9 496 | 9 541 | 0,011448 % |
| Validation | Offset | 9 451 | 9 459 | 0,011394 % |

Les 326 offsets train et 82 offsets validation placés à la fin exclusive du
WAV restent hors de la cible acoustique. Ils réconcilient exactement les
annotations brutes et restent gérés par la finalisation EOF de l'expérience
11.

Tous les histogrammes, nombres d'instances, maxima et offsets EOF ont été
comparés à l'audit indépendant verrouillé de l'expérience 10. Tous les
contrôles sont exacts.

## Ce que le cycle `1/3, 1/3, 1/3` change

Les tirages fixes sont :

| Split | Requêtes | Onset-bearing | Offset-only | Fond |
|---|---:|---:|---:|---:|
| Train, seed 1337 | 1 600 | 534 | 533 | 533 |
| Validation, seed 1338 | 400 | 134 | 133 | 133 |

Sans correction, environ un tiers des requêtes porte un onset et un tiers un
offset. Ce n'est pas la fréquence rencontrée en live :

| Split | Type | Flux réel | Batch candidat non pondéré | Surreprésentation | Ancien sampler V7 |
|---|---|---:|---:|---:|---:|
| Train | Onset | 0,013762 % | 33,375 % | 2 425,10× | 2,62× |
| Train | Offset | 0,013692 % | 33,375 % | 2 437,60× | 2,53× |
| Validation | Onset | 0,011448 % | 33,500 % | 2 926,23× | 2,80× |
| Validation | Offset | 0,011394 % | 33,250 % | 2 918,22× | 2,61× |

Le sampler candidat résout donc la rareté apparente des positifs dans le
batch, mais crée un prior artificiel beaucoup plus grand que V7. Ses sorties
non pondérées ne peuvent pas être qualifiées de calibrées pour le live.

## Correction de prior

Pour mesurer le flux réel depuis ces batches, l'audit applique seulement comme
diagnostic :

```text
poids(strate) = fréquence dans le flux / fréquence dans le batch
```

| Split | Poids onset | Poids offset-only | Poids fond | Rapport max/min | ESS |
|---|---:|---:|---:|---:|---:|
| Train | 0,000412354 | 0,000410568 | 3,001052 | 7 309,52× | 533,29 / 1 600, soit 33,33 % |
| Validation | 0,000341737 | 0,000342421 | 3,006832 | 8 798,68× | 133,06 / 400, soit 33,27 % |

La distribution jointe pondérée reproduit analytiquement le flux avec une
erreur maximale de `0` en train et `1,36e-20` en validation. Le calcul est donc
correct.

Mais ces poids rendent de nouveau environ `99,97 %` de la masse au fond. Ils
sont valides pour estimer une métrique représentative du flux ; cet audit ne
les approuve pas comme loss d'entraînement.

## Défaut 1 — négatifs trop loin des frontières

Exemple simple : si l'onset réel est à `t=100`, les positions `99` et `101`
doivent apprendre qu'elles ne sont pas l'onset. Une loss appliquée uniquement
au point tiré ne le leur apprend que si ces positions voisines sont réellement
échantillonnées.

Dans le fond train uniforme :

| Distance maximale d'une frontière | Positions du fond | Attendu parmi 533 tirages | Observé | Attendu sur 20 époques fraîches |
|---|---:|---:|---:|---:|
| ±1 échantillon | 175 349 | 0,292 | 0 | 5,840 |
| ±15 échantillons | 2 581 162 | 4,299 | 3 | 85,970 |
| ±63 échantillons | 10 054 344 | 16,744 | 16 | 334,878 |

Le critère préenregistré demandait au moins 20 négatifs à ±1 sur 20 époques.
Le sampler n'en fournit en moyenne que `5,84`. Sur une époque de 1 600
requêtes, il est normal de n'en voir aucun.

En validation, les 133 tirages de fond donnent respectivement `0,061`, `0,895`
et `3,527` exemples attendus ; le tirage fixe en observe `1`, `1` et `4`.

## Défaut 2 — cardinalités rares absentes

Les pools train contiennent :

| Type | Compte 1 | Compte 2 | Compte 3 |
|---|---:|---:|---:|
| Onset | 43 901 | 155 | 3 |
| Offset | 43 773 | 59 | 1 |

Avec le cycle actuel :

| Classe | Tirages attendus par époque | Attendus sur 20 époques fraîches | Probabilité d'être vue au moins une fois |
|---|---:|---:|---:|
| Onset count 2 | 1,879 | 37,572 | presque 100 % |
| Onset count 3 | 0,036 | 0,727 | 51,68 % |
| Offset count 2 | 0,718 | 14,364 | 99,9999 % |
| Offset count 3 | 0,012 | 0,243 | 21,61 % |

Le batch train réellement tiré ne contient aucun onset de compte 2 ou 3, et
aucun offset de compte 3. Il contient un seul offset de compte 2.

Les probabilités sur 20 époques supposent de nouveaux tirages indépendants à
chaque époque. Si exactement le même batch était mis en cache et répété, une
classe absente resterait absente ; le JSON rapporte aussi ce cas séparément.

## Causalité : ce qui est acquis et ce qui ne l'est pas

Le contrat candidat définit le contexte :

```text
audio[max(0, t-4092) : t+1]
```

La requête est donc le dernier échantillon du contexte, sans padding droit.
Les données contiennent 371 positions onset train et 92 validation qui exigent
l'état zéro de début de flux. Elles sont conservées. Les offsets internes dans
les 4 092 derniers échantillons sont également conservés : 87 train et 10
validation.

Ce contrat ne constitue pas encore une implémentation de crop audio. Aucun
modèle ponctuel compatible n'existe dans cette expérience ; l'égalité entre
prédiction sur crop et prédiction en flux continu reste donc explicitement
**non vérifiée** et bloque tout entraînement long.

## Décision et prochaine contrainte

La décision exacte est :

```text
structurally_admissible_with_prior_correction_but_not_training_ready
```

« Structurellement admissible » concerne uniquement la partition des cibles,
le tirage et la correction analytique de prior. Cela ne valide ni une loss, ni
un modèle, ni un crop audio, ni la calibration live.

La prochaine expérience doit corriger **une seule contrainte** : remplacer une
partie du fond uniforme par des négatifs proches des frontières, sans modifier
encore les pools positifs, la loss, le modèle ou le live. Après son audit
avant/après, la couverture séparée des comptes `2/3` pourra être traitée dans
une expérience distincte.

Conformément au protocole, le travail s'arrête ici avant cette nouvelle
modification et avant tout entraînement.

## Intégrité et artefacts

- protocole préenregistré au commit `c31ff88`, SHA-256
  `A57062D99AC907E35E081BF8A73CE12CBE9A7E9477D8CB915D11D88D1D25804F` ;
- amendement négatifs proches au commit `661371d`, SHA-256
  `DB3D5937B0A4F5DF7C004E7D7742ACC1FD5A5268E4AC876C1B4E4CCA0D701930` ;
- corrections de revue au commit `0255913`, SHA-256
  `A54B4CE82DA79A67AAAB1D46F572F0B2CB237B3EA1C485DD4A891A1D25FCDDA6` ;
- implémentation et résultat au commit `aa91b1c` ;
- script d'audit, SHA-256
  `4F8DDA1D84E41910E1DFEA17DAF4150BD278FDDEFD02415D20A6DA812D12B193` ;
- tests dédiés, SHA-256
  `787ED1B1BF84581567D5477706ADBEA7750ACF5CB3491C3B7E38B2DC376F5734` ;
- résultat JSON, SHA-256
  `1056B0349E826FE9D2E8845AA8CE88D5E4F5E9C719E209CC73C41F9A19D1EF56` ;
- `8/8` tests ciblés et `203/203` tests complets réussis ;
- double vérification indépendante sans divergence ;
- aucun contenu annotation ou audio Player05 ouvert ou utilisé.
