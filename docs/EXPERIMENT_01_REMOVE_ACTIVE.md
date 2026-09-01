# Expérience 01 — suppression de la cible `active`

## Question

La cible auxiliaire `active` dominait-elle l'apprentissage au point d'empêcher
la détection des onsets et offsets demandés ?

Cette expérience retire uniquement la tête, la cible, la loss et le poids
`active`. Les six slots internes sont encore conservés.

## Contrôle

Les deux entraînements utilisent exactement :

- GuitarSet, joueurs `00` à `04`, sans lecture du joueur `05` ;
- les mêmes 240 pistes d'entraînement et 60 pistes de validation, dans le même
  ordre de split, avec la graine `1337` ;
- 8 exemples par batch, fenêtres de 8 192 échantillons, champ réceptif 4 093 ;
- 200 pas par époque et 50 batchs fixes de validation ;
- les poids positifs `onset=64` et `offset=64` ;
- le même tronc causal, 24 filtres et dix dilatations ;
- le même seuil événementiel `0.5` et les mêmes tolérances de 50 ms.

Le changement retire 150 paramètres : 26 730 deviennent 26 580.

## Résultat d'entraînement

Les `val_loss` totales ne sont pas directement comparables : dans le baseline,
`active_loss` représente 94,445 % de la loss finale. Les composantes onset et
offset restent comparables.

| Loss au checkpoint réellement restauré | Baseline | Sans `active` | Écart relatif |
|---|---:|---:|---:|
| `val_onset_loss` | 0.01171461 | 0.01132428 | -3.332 % |
| `val_offset_loss` | 0.01106136 | 0.01063222 | -3.880 % |
| Somme onset + offset | 0.02277597 | 0.02195650 | -3.598 % |

Le baseline sélectionne l'époque humaine 20. Le modèle sans `active` sélectionne
l'époque 2 et s'arrête après cinq époques. Il effectue donc 1 000 mises à jour,
contre 4 000 pour le baseline.

## Résultat événementiel sur les 60 pistes

| Mesure au seuil 0.5 | Baseline | Sans `active` |
|---|---:|---:|
| Références | 9 541 | 9 541 |
| Onsets prédits | 0 | 0 |
| Offsets prédits | 0 | 0 |
| Événements complets | 0 | 0 |
| F1 onset | 0.0 | 0.0 |
| F1 offset | 0.0 | 0.0 |
| F1 intervalle associé | 0.0 | 0.0 |
| Erreur absolue de cardinalité onset | 9 541 | 9 541 |

Les taux de faux événements valent 0/h uniquement parce qu'aucun événement
n'est émis. Ce n'est pas un résultat favorable : le rappel vaut également zéro.
Le constat est identique pour les 30 pistes `comp` et les 30 pistes `solo`.

## Décision

La suppression d'`active` est conservée parce que cette cible n'appartient pas
à la sortie demandée et n'est pas utilisée au runtime. Elle améliore les losses
de frontières, mais elle ne résout pas la détection live au seuil `0.5`.

La prochaine contrainte doit donc viser la loss réellement appliquée aux six
slots : le poids positif actuel est appliqué à toute la ligne temporelle après
réduction des slots, et non uniquement à l'élément positif.

## Limites

- Une seule graine ne donne aucune estimation de variance.
- Retirer une tête modifie aussi l'initialisation aléatoire de la tête offset.
- La politique d'early stopping change parce que la loss auxiliaire disparaît.
- Les 82 offsets situés à la fin exclusive du WAV restent inclus dans cette
  comparaison ; leur censure sera une expérience séparée.
- Les mesures de vitesse varient avec la charge système et ne permettent pas
  d'attribuer un gain causal à cette modification.
