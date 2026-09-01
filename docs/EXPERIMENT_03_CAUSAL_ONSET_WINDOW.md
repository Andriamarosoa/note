# Expérience 03 — fenêtre causale onset de 512 échantillons

## Question

Une cible onset causalement élargie permet-elle au modèle d'apprendre un signal
de frontière que l'impulsion exacte d'un échantillon ne permettait pas
d'apprendre ?

Cette expérience change uniquement la cible onset :

```text
avant : [onset, onset + 1)
après : [onset, onset + 512)
```

La cible offset reste une impulsion exacte d'un échantillon. Le modèle, les
poids, les six slots, le split, le sampler, le seuil `0.5` et le décodeur live
restent inchangés.

## Contrôle

Les expériences v3 et v4 utilisent exactement :

- GuitarSet, joueurs `00` à `04`, sans lecture du joueur `05` ;
- 240 pistes d'entraînement et 60 pistes de validation, graine `1337` ;
- 44 220 notes d'entraînement et 9 541 notes de validation ;
- fenêtres de 8 192 échantillons, batches de 8, 200 pas par époque et 50
  batches fixes de validation ;
- le même modèle causal à deux sorties, six slots et 26 580 paramètres ;
- les mêmes poids élémentaires : `64` pour un positif et `1` pour un négatif ;
- le même décodeur par front montant et les seuils onset/offset `0.5`.

Le run v4 s'arrête après 19 époques et restaure l'époque humaine 16. Les poids
finaux sont identiques à ceux du checkpoint 16. À ce checkpoint,
`val_loss=0,5343511105`, `val_onset_loss=0,5227648020` et
`val_offset_loss=0,0115862619`.

## Scores sur les 50 batches fixes

L'élargissement fait passer les éléments onset positifs scorés de `527` à
`261 423`, soit un facteur `496,06` après découpage aux limites des crops.

| Mesure onset au seuil 0.5 | v3 : impulsion 1 | v4 : fenêtre 512 |
|---|---:|---:|
| Vrais positifs | 0 | 235 766 |
| Faux négatifs | 527 | 25 657 |
| Faux positifs | 0 | 2 828 094 |
| Vrais négatifs | 9 839 473 | 6 750 483 |
| Rappel élémentaire | 0 % | 90,1856 % |
| Précision élémentaire | non définie | 7,6951 % |
| F1 élémentaire | 0 % | 14,1802 % |
| Taux de faux positifs sur les négatifs | 0 % | 29,5252 % |

Pour l'onset positif v4, la médiane du score vaut `0,886410` et le maximum
`0,998076`. Pour l'onset négatif, la médiane vaut encore `0,314380` et le
maximum `0,998031`. Le signal onset est donc appris, mais reste trop large et
insuffisamment séparé des négatifs.

L'offset reste effondré : aucun des `489` éléments positifs scorés ne dépasse
`0.5`, et leur maximum vaut seulement `0,033892`.

## Résultat live sur les 60 pistes

| Mesure au seuil 0.5 | v3 | v4 |
|---|---:|---:|
| Références onset/offset | 9 541 | 9 541 |
| Onsets prédits | 0 | 360 |
| Vrais positifs onset | 0 | 88 |
| Faux positifs onset | 0 | 272 |
| Précision onset | 0 % | 24,4444 % |
| Rappel onset | 0 % | 0,9223 % |
| F1 onset | 0 % | 1,7776 % |
| Offsets prédits | 0 | 0 |
| Événements complets | 0 | 0 |
| Événements ouverts sans offset | 0 | 360 |
| F1 intervalle associé | 0 % | 0 % |

Les 360 onsets correspondent exactement à six onsets simultanés sur chacune
des 60 pistes. Comme aucun offset ne ferme les slots, le décodeur conserve ces
six événements ouverts et bloque tous les onsets suivants. Le rappel
élémentaire de `90,19 %` ne représente donc pas le rappel événementiel live,
qui reste à `0,92 %`.

Les 88 correspondances onset sont toutes précoces : aucune n'est causale au
sens de l'évaluateur. La médiane vaut `-429,5` échantillons (`-9,739 ms`) et le
p90 `-284,4` échantillons (`-6,449 ms`).

Le facteur temps réel mesuré est `1,106`. Il reste légèrement plus lent que le
temps réel et dépend de la charge CPU ; il n'est pas attribué au changement de
cible.

## Décision

La fenêtre onset de 512 débloque l'apprentissage d'un signal, contrairement à
l'impulsion d'un échantillon. Elle n'est toutefois pas validée comme cible
finale : son taux de faux positifs est élevé et son résultat live est bloqué
par l'absence totale d'offset.

L'expérience suivante la conserve uniquement comme base contrôlée et change
seulement l'offset de `[offset, offset + 1)` vers
`[offset, offset + 512)`. Cela permettra de vérifier séparément si la même
contrainte de parcimonie explique l'échec offset.

Les six slots internes, leur association et le retrigger sans offset restent
des défauts de conception distincts qui ne sont pas corrigés par cette
expérience.

## Limites

- Une seule graine ne donne aucune estimation de variance.
- La `val_loss` v3/v4 n'est pas directement comparable puisque le nombre de
  cibles positives change.
- Le rapport live est dominé par les six événements initiaux jamais fermés.
- Les 82 offsets à la fin exclusive du WAV ne peuvent toujours pas être
  supervisés à leur position exacte.
