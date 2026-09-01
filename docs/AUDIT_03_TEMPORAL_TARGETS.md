# Audit avant expérience 03 — cibles temporelles

## Périmètre

Audit en lecture seule de GuitarSet limité aux joueurs `00` à `04`, du split
figé `seed=1337` et du modèle `causal-boundaries-elementwise-v3.keras`.
Le joueur `05` n'est pas inclus.

L'objectif est de décider si l'échec au seuil `0.5` vient du seuil, du
décodeur, ou de la représentation des cibles d'apprentissage.

## Densité complète

Chaque tête contient `nombre d'échantillons audio × 6 slots` décisions.

| Split et tête | Positifs | Total | Pourcentage positif | Négatifs par positif |
|---|---:|---:|---:|---:|
| Train onset | 44 220 | 1 920 854 100 | 0.0023021 % | 43 437.58 |
| Train offset supervisé | 43 894 | 1 920 854 100 | 0.0022851 % | 43 760.20 |
| Validation onset | 9 541 | 497 686 080 | 0.0019171 % | 52 161.88 |
| Validation offset supervisé | 9 459 | 497 686 080 | 0.0019006 % | 52 614.08 |

Les 326 offsets train et 82 offsets validation manquants sont exactement à la
fin exclusive du WAV : aucun échantillon audio ne permet de les superviser à
cette position.

La cible actuelle mesure un seul échantillon : `1 / 44100`, soit
`0.0226757 ms`. La tolérance de 50 ms appartient seulement à l'évaluateur ;
elle n'aide pas l'apprentissage.

## Cinquante batches fixes de validation

La grille brute contient :

```text
50 × 8 × 8192 × 6 = 19 660 800 éléments par tête
```

Après masquage du warmup causal, 9 840 000 éléments restent scorés par tête.

| Tête | Positifs scorés | Pourcentage | Négatifs par positif |
|---|---:|---:|---:|
| Onset | 527 | 0.0053557 % | 18 670.73 |
| Offset | 489 | 0.0049695 % | 20 121.70 |

Avec un poids positif de 64, l'optimum d'une prédiction BCE presque constante
est donc :

```text
onset  = 64×527 / (64×527 + 9 839 473) = 0.0034161
offset = 64×489 / (64×489 + 9 839 511) = 0.0031706
```

## Scores réellement appris

| Tête et label | p50 | p90 | p99 | Maximum | Nombre ≥ 0.5 |
|---|---:|---:|---:|---:|---:|
| Onset positif | 0.004102 | 0.005399 | 0.005572 | 0.005623 | 0 / 527 |
| Onset négatif | 0.003269 | 0.005278 | 0.005596 | 0.005747 | 0 / 9 839 473 |
| Offset positif | 0.003630 | 0.004101 | 0.004204 | 0.004261 | 0 / 489 |
| Offset négatif | 0.003322 | 0.003953 | 0.004215 | 0.004374 | 0 / 9 839 511 |

Le modèle a convergé vers la sortie presque constante prédite par le ratio de
classes. Le maximum négatif dépasse le maximum positif pour les deux têtes.
Baisser le seuil ne crée donc pas une séparation fiable et un peak picker ne
peut pas retrouver des pics qui ne sont pas appris.

Les scores offset supérieurs à `0.5` observés dans la grille brute appartiennent
tous au warmup masqué et ne constituent pas des prédictions validées.

## Compatibilité d'une fenêtre causale de 512

Sur les 53 761 notes de développement :

- durée minimale : 1 280 échantillons ;
- distance minimale entre deux onsets consécutifs du même slot : 2 204 ;
- aucun intervalle onset-onset du même slot n'est inférieur ou égal à 512 ;
- aucune collision onset ou offset n'est perdue par la binarisation actuelle.

Une cible onset `[t, t+512)` ne fusionne donc aucun onset du même slot dans ces
données. Elle reste strictement causale : aucun échantillon antérieur à `t`
n'est étiqueté positif. Une cible symétrique autour de `t` est exclue, car elle
demanderait au modèle de prédire avant l'arrivée de l'événement.

## Décodeur actuel

Le runtime n'utilise pas un peak picker. Il émet au premier franchissement
montant du seuil et mémorise l'état haut/bas entre les blocs. Il n'emploie ni
maximum local, ni hystérésis, ni seuil adaptatif.

Le lookahead algorithmique est nul. Le bloc de 512 ajoute au plus 11.6 ms de
tampon, plus l'inférence.

## Décision expérimentale

L'expérience 03 change uniquement la cible onset :

```text
avant : positif uniquement à t
après : positifs sur [t, t+512), tronqués aux limites du crop
```

La cible offset reste une impulsion d'un échantillon. Les poids, le modèle, les
six slots, le split, le sampler, le seuil et le décodeur restent inchangés.

Cette expérience n'affirme pas que les six slots sont l'architecture finale.
Deux contraintes distinctes restent volontairement hors périmètre :

- l'association publique dépend encore de la bonne prédiction du slot interne ;
- un retrigger sans offset préalable n'est pas encore implémenté.

Elles devront être traitées séparément après avoir vérifié que la frontière
onset elle-même peut être apprise.
