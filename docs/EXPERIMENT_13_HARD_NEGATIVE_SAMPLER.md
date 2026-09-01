# Expérience 13 — fond stratifié par distance aux frontières

État figé le 1er septembre 2026. Cette expérience modifie uniquement la
distribution candidate des requêtes de fond de l'expérience 12. Elle ne
modifie ni les cibles positives, ni leur nombre de tirages, ni la loss, ni le
modèle, ni le décodeur, ni le live. Aucun entraînement n'a été lancé.

## Verdict

La contrainte des négatifs immédiatement voisins est **réparée pour des
époques fraîchement rééchantillonnées**, mais le sampler complet reste **non
prêt pour l'entraînement**.

La grille préenregistrée sélectionne `h=1`, le plus petit candidat passant les
gardes train. Il fournit exactement un négatif à distance `1` par époque, soit
`20` tirages sur 20 époques. C'est exactement le minimum demandé, sans marge.

```text
background_constraint_repaired_but_sampler_not_training_ready
```

## Unique changement

Les `533` requêtes de fond train de l'expérience 12 sont divisées en quatre
strates disjointes selon la distance à l'onset ou à l'offset interne le plus
proche dans la même piste :

| Strate | Train | Validation |
|---|---:|---:|
| Distance exactement 1 | 175 349 | 37 822 |
| Distance 2 à 15 | 2 405 813 | 520 337 |
| Distance 16 à 63 | 7 473 182 | 1 640 990 |
| Distance au moins 64 | 310 000 161 | 80 729 591 |
| **Total fond** | **320 054 505** | **82 928 740** |

Ces comptes reproduisent exactement les cumuls `±1`, `±15` et `±63` de
l'expérience 12. Les quatre bandes sont exhaustives, sans chevauchement et ne
contiennent aucune vraie frontière.

Les pools positifs restent exactement :

| Split | Onset-bearing | Offset-only |
|---|---:|---:|
| Train | 44 059 | 43 786 |
| Validation | 9 496 | 9 444 |

## Grille préenregistrée

Pour chaque valeur `h`, les quotas train sont :

```text
distance 1       = h
distance 2..15   = 4h
distance 16..63  = 16h
distance 64+     = 533 - 21h
```

La sélection utilise uniquement les gardes analytiques train. La validation
n'intervient qu'après verrouillage du candidat.

| `h` | Fond train `1 / 2..15 / 16..63 / 64+` | Exposition distance 1 sur 20 époques | Fond 64+ | ESS train | Décision des gardes |
|---:|---|---:|---:|---:|---|
| **1** | **1 / 4 / 16 / 512** | **20** | **96,06 %** | **33,2481 %** | **passe, sélectionné** |
| 2 | 2 / 8 / 32 / 491 | 40 | 92,12 % | 32,3182 % | passe |
| 4 | 4 / 16 / 64 / 449 | 80 | 84,24 % | 29,7561 % | passe |
| 8 | 8 / 32 / 128 / 365 | 160 | 68,48 % | 24,2724 % | rejeté |

`h=8` échoue à la fois au minimum ESS de `25 %` et à la conservation d'au
moins `80 %` de fond lointain. Conformément à la règle préenregistrée, le fait
que `h=4` offre davantage de marge ne permet pas de remplacer après coup le
premier candidat passant `h=1`.

Les quotas fixes du candidat sélectionné sont :

| Split | Requêtes | Onset-bearing | Offset-only | d=1 | d=2..15 | d=16..63 | d=64+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 1 600 | 534 | 533 | 1 | 4 | 16 | 512 |
| Validation | 400 | 134 | 133 | 1 | 1 | 4 | 127 |

Tous les tirages fixes ont été vérifiés à leur distance réelle. Leur exécution
est déterministe avec des générateurs indépendants pour les deux pools
positifs et chaque bande de fond.

Le batch train représente `236/240` pistes : quatre pistes reçoivent zéro
requête, la médiane basse vaut 6 requêtes et le maximum 25. Le batch validation
représente `60/60` pistes, avec une médiane basse de 6 et un maximum de 16.
Les `1 600` et `400` requêtes se réconcilient exactement. Cette couverture est
rapportée sans imposer après coup un minimum par piste qui n'était pas
préenregistré.

## Comparaison à l'expérience 12

| Mesure | Expérience 12 | Expérience 13 `h=1` |
|---|---:|---:|
| Négatifs d=1 attendus par époque train | 0,2920 | **1 exact** |
| Négatifs d=1 sur 20 époques train | 5,8403 | **20 exacts** |
| ESS train | 33,3308 % | **33,2481 %** |
| ESS validation | 33,2652 % | **32,9123 %** |
| Tirages onset-bearing train | 534 | **534** |
| Tirages offset-only train | 533 | **533** |

La couverture d=1 est multipliée par `3,4245` en train. La perte ESS reste
faible : `−0,0827` point en train et `−0,3529` point en validation.

Avec de nouveaux tirages indépendants à chaque époque, les 20 tirages train
correspondent à `19,9989` positions uniques attendues parmi 175 349. Si le
même batch fixe était répété 20 fois, il fournirait toujours 20 passages dans
la loss mais une seule position unique. Le passage du garde ne doit donc pas
être interprété comme une garantie de diversité avec un batch mis en cache.

## Prior et diagnostic de pondération

Le changement de fond ne modifie pas les quotas positifs. Le prior non pondéré
reste donc artificiel :

| Split | Tête | Batch candidat | Flux réel | Surreprésentation |
|---|---|---:|---:|---:|
| Train | Onset | 33,3750 % | 0,013762 % | 2 425,10× |
| Train | Offset, espérance analytique | 33,3481 % | 0,013692 % | 2 435,64× |
| Validation | Onset | 33,5000 % | 0,011448 % | 2 926,23× |
| Validation | Offset, espérance analytique | 33,2747 % | 0,011394 % | 2 920,39× |

La correction analytique `p_live / p_batch` reproduit les six priors de strate
et le prior joint :

| Split | Erreur max prior six strates | Erreur max prior joint | Erreur moyenne du poids | Rapport poids max/min |
|---|---:|---:|---:|---:|
| Train | 8,67e-19 | 1,11e-16 | 0 | 7 370,28× |
| Validation | 1,36e-20 | 1,36e-20 | 0 | 8 970,01× |

Ces poids restent un diagnostic de distribution. Leur rapport extrême et la
masse dominante rendue au fond interdisent de les déclarer loss
d'entraînement sans une expérience séparée.

## Contrainte non réparée : comptes rares

L'exposition des comptes `2/3` est mathématiquement identique à l'expérience
12, ce que le nouvel audit vérifie champ par champ :

| Classe train | Positions | Tirages attendus sur 20 époques |
|---|---:|---:|
| Onset count 2 | 155 | 37,5723 |
| Onset count 3 | 3 | 0,7272 |
| Offset count 2 | 59 | 14,3640 |
| Offset count 3 | 1 | 0,2435 |

Le minimum reste `0,2435`, très inférieur au garde `20`. Cette expérience ne
devait volontairement pas réparer simultanément les négatifs proches et les
cardinalités rares.

## Décision et arrêt

Tous les contrôles structurels passent : population identique à l'expérience
12, partition du fond exacte, quotas exacts, positions valides, correction des
six priors exacte, ESS validation suffisante, exposition rare inchangée et
absence du joueur verrouillé.

L'entraînement reste toutefois interdit pour cinq raisons :

1. les comptes rares restent sous-exposés ;
2. `h=1` passe le garde adjacent exactement à sa limite, sans marge ;
3. le prior non pondéré reste très éloigné du live ;
4. aucune loss ni aucun modèle ponctuel n'est sélectionné ;
5. le crop audio ponctuel et son égalité avec le flux continu ne sont pas
   implémentés ni vérifiés.

La prochaine expérience éventuelle doit traiter une seule de ces contraintes
et nécessite une nouvelle approbation. Aucun smoke ni entraînement long ne
doit commencer à partir de ce seul résultat.

## Vérification

- sortie déterministe après normalisation du seul horodatage ;
- test statistique déterministe de 10 000 tirages uniforme sur six positions
  réparties entre deux pistes de longueurs différentes ;
- `12/12` tests Exp13 réussis ;
- `215/215` tests complets réussis ;
- `11` tests optionnels ignorés parce que TensorFlow ou NumPy n'est pas
  installé dans cet environnement ;
- revue indépendante finale : **PASS, aucun P0 à P3 restant** ;
- aucun audio décodé ;
- aucun contenu du joueur verrouillé ouvert ;
- aucun modèle, loss, décodeur ou comportement live modifié.

## Artefacts

- protocole :
  `model/causal-boundaries-hard-negative-sampler-protocol.json` ;
- clarification train/validation :
  `model/causal-boundaries-hard-negative-sampler-protocol-amendment-01.json` ;
- correction de rapport enregistrée avant l'audit final :
  `model/causal-boundaries-hard-negative-sampler-protocol-amendment-02.json` ;
- corrections de revue indépendante enregistrées avant l'audit final :
  `model/causal-boundaries-hard-negative-sampler-protocol-amendment-03.json` ;
- implémentation d'audit : `scripts/audit_hard_negative_sampler.py` ;
- tests : `test/test_hard_negative_sampler.py` ;
- résultat : `model/causal-boundaries-hard-negative-sampler-audit.json`.

Empreintes SHA-256 finales :

| Artefact | SHA-256 |
|---|---|
| Protocole | `D3FF0F3F9A35C6B0AE8079B9781E961DE9FAE380A42A601922217D2530999CBB` |
| Amendement 01 | `46C739E90BD03469D3DB160DEF3A6215605C1C395E3FBF60865612C8EFD3F6EF` |
| Amendement 02 | `1E130543D20C2B91AB620A26E9228FABE90FF3D5C52C5BBE338A1BC4DB38C055` |
| Amendement 03 | `69F12014A41766611F0ADE37080C68213E68C2ADD752A923FB84A2EE9DB21680` |
| Script | `294A3B17BF887B3493B6BC1BCA5867EADA3145B425CF3D24B178C14CF4B3DC04` |
| Tests | `2F74911FEF245F1CD63162A5ACCD7800CEB7C494A4E5D62D7DD17CE75E113D43` |
| Résultat | `A2AD4C04946CB4C06ECDDE894118E9D7BF2B9D2B1818B0AD736513ADD6EB46A5` |
