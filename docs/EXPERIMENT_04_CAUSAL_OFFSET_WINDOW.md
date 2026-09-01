# Expérience 04 — fenêtre causale offset de 512 échantillons

## Question

L'impulsion offset d'un seul échantillon empêchait-elle l'apprentissage de
cette frontière, comme pour l'onset ?

Cette expérience change uniquement la cible offset :

```text
avant : [offset, offset + 1)
après : [offset, offset + 512)
```

L'onset reste `[onset, onset + 512)`. Le modèle, les poids, les six slots, le
split, le sampler, les seuils `0.5` et le décodeur live restent inchangés.

## Contrôle des cibles

Sur les mêmes 50 batches fixes, l'onset reste exactement à `261 423` éléments
positifs scorés. L'offset passe de `489` à `249 851` éléments positifs parce
que chaque frontière occupe maintenant jusqu'à 512 positions consécutives,
avec découpage aux limites des crops.

Ces nombres comptent des éléments de tenseur d'apprentissage. Ils ne comptent
ni les offsets uniques du jeu de données, ni les offsets détectés par le
modèle.

## Entraînement

Le run exécute les 20 époques et sélectionne l'époque humaine 20 :

| Loss de validation au checkpoint final | Valeur |
|---|---:|
| `val_loss` | 1,2427628040 |
| `val_onset_loss` | 0,5411149859 |
| `val_offset_loss` | 0,7016478777 |

Les 24 tenseurs du modèle final sont identiques au checkpoint 20, avec un
écart absolu maximal de `0`. Le modèle conserve 26 580 paramètres.

Les losses v4 et v5 ne sont pas directement comparables : le nombre et la
distribution des cibles offset positives ont changé.

## Scores sur les 50 batches fixes

| Mesure au seuil 0.5 | Offset v4 : largeur 1 | Offset v5 : largeur 512 |
|---|---:|---:|
| Positifs scorés | 489 | 249 851 |
| Positifs dépassant le seuil | 0 | 228 895 |
| Rappel élémentaire | 0 % | 91,6126 % |
| Faux positifs | 0 | 5 817 820 |
| Taux de faux positifs négatifs | 0 % | 60,6645 % |
| Précision élémentaire | 0 % | 3,7854 % |
| F1 élémentaire | 0 % | 7,2705 % |

L'offset est donc débloqué, mais son score est encore plus large que celui de
l'onset : la médiane vaut `0,733775` sur les positifs et `0,569044` sur les
négatifs. Un négatif médian dépasse déjà le seuil `0.5`.

L'onset n'a pas été modifié. Son rappel élémentaire passe néanmoins de
`90,1856 %` à `86,3600 %`, sa précision de `7,6951 %` à `7,5869 %` et son taux
de faux positifs de `29,5252 %` à `28,7093 %`. Ces variations viennent du
nouvel entraînement conjoint et ne prouvent pas une amélioration onset.

## Résultat live sur les 60 pistes

| Mesure au seuil 0.5 | v4 | v5 |
|---|---:|---:|
| Événements de référence | 9 541 | 9 541 |
| Onsets prédits | 360 | 265 130 |
| Vrais positifs onset | 88 | 9 217 |
| Précision onset | 24,4444 % | 3,4764 % |
| Rappel onset | 0,9223 % | 96,6041 % |
| F1 onset | 1,7776 % | 6,7113 % |
| Offsets prédits | 0 | 264 943 |
| Vrais positifs offset | 0 | 8 197 |
| Précision offset | 0 % | 3,0939 % |
| Rappel offset | 0 % | 85,9134 % |
| F1 offset | 0 % | 5,9727 % |
| Événements complets prédits | 0 | 264 943 |
| Intervalles correctement associés | 0 | 1 579 |
| Précision intervalle | 0 % | 0,5960 % |
| Rappel intervalle | 0 % | 16,5496 % |
| F1 intervalle associé | 0 % | 1,1505 % |

V5 prédit `27,77` fois plus d'événements complets qu'il n'existe de
références. Parmi les 264 943 intervalles produits, 263 364 sont faux selon la
tolérance de 50 ms et l'association onset-offset.

Le rappel élevé ne signifie donc pas que la détection est bonne : presque
toutes les vraies frontières se trouvent au milieu d'un très grand nombre de
fausses frontières.

Parmi les 9 217 onsets appariés, 6 753 sont précoces et 2 464 sont causaux.
Parmi les 8 197 offsets appariés, 3 480 sont précoces et 4 717 sont causaux ;
la latence offset signée médiane vaut `+2,698 ms`.

## Runtime

L'évaluation traite 1 880,90 secondes d'audio en 950,12 secondes, soit un
facteur temps réel `0,505`. La médiane par bloc vaut `5,556 ms`, le p95
`6,759 ms` et le maximum `165,394 ms`. Le débit moyen est compatible avec le
live, mais le maximum dépasse ponctuellement le budget d'un bloc de 512
échantillons (`11,61 ms`).

## Décision

L'expérience confirme que l'impulsion offset d'un échantillon empêchait
l'apprentissage. La fenêtre binaire de 512 n'est cependant pas une solution
finale : elle transforme l'absence de détection en sur-détection massive.

Elle ne doit pas être présentée comme un modèle live utilisable. Avant tout
nouvel entraînement, la prochaine analyse doit mesurer la forme temporelle des
plateaux et des pics afin de choisir une seule correction contrôlée pour
réduire les déclenchements, sans masquer le problème par le seul rappel.

Les six slots internes, l'association dépendante du slot et le retrigger sans
offset préalable restent des défauts de conception distincts.

## Limites

- Une seule graine ne donne aucune estimation de variance.
- Le seuil `0.5` est conservé pour la comparaison ; il n'est pas calibré.
- Les métriques comp sont meilleures que solo, mais les deux sous-ensembles
  sur-déclenchent fortement.
- Les mesures de runtime dépendent de la charge CPU locale.
