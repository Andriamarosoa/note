# Baseline de travail V7-e8

État figé le 31 août 2026 pour servir de référence aux prochaines expériences.
Cette baseline est expérimentale : elle n'est pas un modèle de production.

## Identité

- checkpoint local :
  `model/causal-boundaries-weight28-window512-v7.epochs/epoch-08.keras` ;
- architecture causale à 6 slots et 26 580 paramètres ;
- sorties publiques limitées à `onset(event_id, position)` et
  `offset(event_id, position)` ;
- cibles onset et offset : fenêtres causales de 512 échantillons ;
- poids positif commun : `28` ; poids négatif : `1` ;
- GuitarSet, joueurs `00` à `04`, graine `1337`, split groupé `240/60` ;
- seuil du rapport baseline initial : `0.5` ;
- calibration live officielle onset/offset : `0.55` ; blocs causaux :
  512 échantillons.

V5 reste le témoin historique de surdétection avec un poids positif de `64`.
V6 reste le témoin négatif de collapse avec un poids positif de `1`.

## Provenance de l'entraînement

Le run V7 a terminé dix époques, puis son processus a été interrompu sans
traceback pendant l'époque 11. Les dix checkpoints terminés sont intègres.
L'époque humaine 8 est le meilleur checkpoint terminé :

| Loss de validation | Époque 8 |
|---|---:|
| `val_loss` | 0,8084229827 |
| `val_onset_loss` | 0,3552986383 |
| `val_offset_loss` | 0,4531241655 |

L'époque 8 est donc verrouillée comme baseline de travail récupérée. Elle ne
doit pas être décrite comme le résultat d'un run ayant atteint normalement son
arrêt anticipé.

## Audit des 50 batches fixes

Les labels sont strictement identiques à V5 et V6 : 261 423 éléments onset
positifs et 249 851 éléments offset positifs.

| Tête | Précision | Rappel | F1 | FPR négatif |
|---|---:|---:|---:|---:|
| Onset | 11,7821 % | 76,9148 % | 20,4341 % | 15,7176 % |
| Offset | 6,3634 % | 68,6773 % | 11,6476 % | 26,3283 % |

V7-e8 ne présente ni le collapse de V6, ni l'amplitude de surdétection de V5.

## Évaluation live historique à 0,50 sur les 60 pistes

| Mesure | V5 | V7-e8 |
|---|---:|---:|
| Événements de référence | 9 541 | 9 541 |
| Événements complets prédits | 264 943 | 143 089 |
| Vrais positifs onset | 9 217 | 9 030 |
| Précision onset | 3,4764 % | 6,3032 % |
| Rappel onset | 96,6041 % | 94,6442 % |
| F1 onset | 6,7113 % | 11,8193 % |
| Vrais positifs offset | 8 197 | 8 023 |
| Précision offset | 3,0939 % | 5,6070 % |
| Rappel offset | 85,9134 % | 84,0897 % |
| F1 offset | 5,9727 % | 10,5130 % |
| Intervalles correctement associés | 1 579 | 3 006 |
| Précision intervalle | 0,5960 % | 2,1008 % |
| Rappel intervalle | 16,5496 % | 31,5061 % |
| F1 intervalle | 1,1505 % | 3,9389 % |

V7-e8 réduit les faux intervalles de 46,81 % et augmente les intervalles
corrects de 90,37 %. Il produit néanmoins encore 143 089 événements pour
9 541 références, soit 14,997 fois le nombre de références. Parmi les
événements prédits, 97,899 % restent faux.

## Calibration live post-baseline

Un sweep pré-enregistré a comparé un seuil commun onset/offset de `0.50` à
`0.90` par pas de `0.05`, sans modifier le modèle, les données, le décodeur,
les blocs ou les tolérances. Le candidat `0.50` reproduit exactement le rapport
baseline, jusque dans les comptes et les résultats des 60 pistes.
Le protocole figé est
`model/causal-boundaries-weight28-window512-v7-epoch08.threshold-sweep-protocol.json`
et le rapport complet est
`model/causal-boundaries-weight28-window512-v7-epoch08.threshold-sweep-common-0.50-0.90.json`.

| Mesure live | Seuil 0,50 | Seuil 0,55 | Évolution |
|---|---:|---:|---:|
| Événements complets prédits | 143 089 | 110 987 | -32 102 |
| Vrais positifs onset | 9 030 | 8 856 | -174 |
| Vrais positifs offset | 8 023 | 7 980 | -43 |
| Intervalles correctement associés | 3 006 | 3 488 | +482 |
| Faux intervalles | 140 083 | 107 499 | -32 584 (-23,26 %) |
| Précision intervalle | 2,1008 % | 3,1427 % | +1,0419 point |
| Rappel intervalle | 31,5061 % | 36,5580 % | +5,0519 points |
| F1 intervalle | 3,9389 % | 5,7879 % | +1,8489 point |
| Macro-F1 intervalle comp/solo | 5,8704 % | 8,6209 % | +2,7505 points |

`0.55` est l'unique candidat qui respecte les minima pré-enregistrés dans les
deux régimes. `0.60` est rejeté malgré son meilleur F1 global, car ses vrais
positifs solo tombent sous les minima onset (`2 140 < 2 210`) et offset
(`1 950 < 2 044`). Le F1 intervalle progresse sur les 60 pistes et sur les
12 couples famille-arrangement. Le gain net d'intervalles vrais vient surtout
du comp (`+476`) ; le solo gagne `+6`.

La calibration ne résout pas la surdétection : 96,857 % des 110 987
intervalles prédits à `0.55` restent faux. Elle devient néanmoins la
configuration live officielle de V7-e8. C'est une calibration sur validation,
pas une estimation finale non biaisée ; le joueur `05` n'a pas été lu.

## Comp et solo

| Régime | Références | Prédictions à 0,50 | Ratio à 0,50 | Prédictions à 0,55 | Ratio à 0,55 |
|---|---:|---:|---:|---:|---:|
| Comp | 6 919 | 112 561 | 16,268× | 90 277 | 13,048× |
| Solo | 2 622 | 30 528 | 11,643× | 20 710 | 7,899× |

Le gain initial face à V5 venait principalement du solo. La calibration à
`0.55` réduit ensuite les prédictions dans les deux régimes, sans supprimer le
défaut dominant de V7-e8 : la production de frontières à de faux instants,
surtout dans les accompagnements polyphoniques.

## Runtime

L'évaluation traite 1 880,90 secondes d'audio avec un facteur temps réel moyen
de `0,671`. La médiane par bloc vaut `6,981 ms` et le p95 `11,800 ms`. Le débit
moyen est plus rapide que l'audio, mais le p95 dépasse légèrement le budget de
`11,61 ms` d'un bloc de 512 échantillons.

Le sweep à neuf seuils a pris `9 879,30 s` (`164,66 min`, facteur temps réel
`5,115`) parce que l'outil exécute neuf décodages Python et leurs matchings. Ce
temps mesure l'outil d'audit multi-seuil, pas le chemin live à seuil unique.

## Règle pour les expériences suivantes

Chaque expérience prend le checkpoint V7-e8 avec sa calibration live `0.55`
comme référence de comparaison et ne modifie qu'un seul facteur. Le rapport à
`0.50` reste le contrôle historique de calibration. Avant tout entraînement
long, l'audit doit comparer les données transformées avant/après, leur densité,
leurs chevauchements, la masse de loss, le sampler et l'optimum d'une sortie
constante. Après l'entraînement, le même audit fixe précède toute évaluation
live.

V7-e8 devient la référence principale de comparaison. V5 reste le contrôle
historique versionné ; V6 reste un contrôle local non inclus dans ce commit.
