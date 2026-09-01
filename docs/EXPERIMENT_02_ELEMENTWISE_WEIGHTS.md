# Expérience 02 — pondération élémentaire des frontières

## Question

La pondération positive par ligne empêchait-elle le modèle d'apprendre les
onsets et offsets ?

Cette expérience conserve le modèle sans cible `active` et change uniquement
la sémantique des poids de loss :

- avant : poids de forme `(batch, time)` appliqué aux six slots après leur
  réduction ;
- après : poids de forme `(batch, time, slots)` appliqué avant réduction, avec
  `64` sur le seul élément positif, `1` sur chaque négatif et `0` pendant le
  warmup.

Pour une ligne contenant un positif et cinq négatifs tous prédits à `0.5`, la
loss mesurée passe de `44.361404` avec l'ancien poids de ligne à `7.971189`
avec la pondération élémentaire. L'ancien calcul amplifiait donc aussi les cinq
négatifs.

## Contrôle

Les deux entraînements utilisent exactement :

- GuitarSet, joueurs `00` à `04`, sans lecture du joueur `05` ;
- les mêmes 240 pistes d'entraînement et 60 pistes de validation, dans le même
  ordre, avec la graine `1337` ;
- 44 220 notes d'entraînement et 9 541 notes de validation ;
- fenêtres de 8 192 échantillons, batches de 8, 200 pas par époque et 50
  batches fixes de validation ;
- le même tronc causal, les mêmes deux sorties `onset` et `offset`, six slots
  et 26 580 paramètres ;
- le même seuil événementiel `0.5`, des blocs causaux de 512 échantillons et
  des tolérances de 50 ms.

Les deux runs s'arrêtent après cinq époques et restaurent l'époque humaine 2.
Les poids du modèle final sont identiques bit à bit à ceux du checkpoint 2.

## Résultat d'entraînement

| Loss au checkpoint restauré | Poids par ligne v2 | Poids élémentaires v3 |
|---|---:|---:|
| `val_onset_loss` | 0.01132428 | 0.01129021 |
| `val_offset_loss` | 0.01063222 | 0.01060150 |
| `val_loss` | 0.02195650 | 0.02189171 |

La `val_loss` observée est inférieure de `0.295 %`, mais sa définition a changé
avec la réduction et la pondération. Cette différence ne constitue donc pas à
elle seule une amélioration comparable.

## Résultat événementiel sur les 60 pistes

| Mesure au seuil 0.5 | Poids par ligne v2 | Poids élémentaires v3 |
|---|---:|---:|
| Références | 9 541 | 9 541 |
| Onsets prédits | 0 | 0 |
| Offsets prédits | 0 | 0 |
| Événements complets | 0 | 0 |
| F1 onset | 0.0 | 0.0 |
| F1 offset | 0.0 | 0.0 |
| F1 intervalle associé | 0.0 | 0.0 |
| Erreur absolue de cardinalité onset | 9 541 | 9 541 |

Les objets `counts` et `metrics` globaux des deux rapports sont identiques.
Le constat est aussi identique séparément pour les 30 pistes `comp` (6 919
références) et les 30 pistes `solo` (2 622 références).

## Décision

La pondération élémentaire est conservée : elle corrige le sens mathématique
du poids positif et son format est testé, y compris après sauvegarde et
rechargement Keras.

Elle ne résout toutefois pas la détection. La contrainte suivante doit porter
sur la représentation temporelle extrêmement clairsemée des cibles et sur la
conversion causale des scores en pics, sans réintroduire corde, case, hauteur
ou cible `active` dans la sortie publique.

## Limites

- Une seule graine ne donne aucune estimation de variance.
- Les losses v2 et v3 n'ont pas exactement la même définition.
- Le temps d'inférence v3 (`RTF 1.464`) est plus lent que cette exécution v2
  (`RTF 1.229`), mais la charge système n'est pas contrôlée : cet écart n'est
  pas attribué à la pondération, absente du runtime.
- Les 82 offsets situés à la fin exclusive du WAV restent inclus ; leur censure
  demeure une expérience séparée.
