# V27 : résultats de la fusion conditionnelle aux classes basses

Source : [run 34256957861](https://github.com/Andriamarosoa/note/actions/runs/34256957861), commit `1d0fe4db42a932bec91236c437d27798b557cfaa`. Le job, les contrôles de provenance, la reproduction V10.4, les tests et l'artefact final ont réussi.

## Résultat agrégé

| Bras | Exact K | Exact K polyphonique | F1 à 50 ms | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| V10.4 gelé | 81,9339 % | 36,5068 % | 80,3842 % | 34 017 | 6 399 | 10 203 |
| V26 uniforme sur le ranking V10.4 | 82,7076 % | 28,0608 % | 78,0565 % | 30 508 | 3 441 | 13 712 |
| `null_veto` | 82,6373 % | 36,5068 % | 79,7910 % | 32 828 | 5 237 | 11 392 |
| `low_k_fusion` | 82,4953 % | 37,5705 % | 80,1896 % | 33 409 | 5 696 | 10 811 |

Par rapport à V10.4 :

- `null_veto` gagne 0,7034 point d'exact-K et 540 fenêtres exactes, sans changer l'exact-K polyphonique, mais perd 0,5933 point de F1 ;
- `low_k_fusion` gagne 0,5614 point d'exact-K et 431 fenêtres exactes, gagne 1,0637 point d'exact-K polyphonique, mais perd 0,1946 point de F1 ;
- le léger écart du bras V26 uniforme par rapport au rapport V26 vient du ranking V10.4 imposé à tous les bras V27, au lieu du ranking V24 utilisé dans V26.

## Exact-K par fold

| Fold | V10.4 | `null_veto` | `low_k_fusion` | Poly V10.4 | Poly `low_k_fusion` |
|---|---:|---:|---:|---:|---:|
| 0 | 81,0741 % | 81,5474 % | 81,1498 % | 32,5438 % | 34,6755 % |
| 1 | 80,5157 % | 80,9656 % | 80,9156 % | 39,3581 % | 40,3153 % |
| 2 | 83,2417 % | 84,0662 % | 83,8568 % | 31,0731 % | 32,4292 % |
| 3 | 83,2522 % | 83,8842 % | 83,9095 % | 40,5612 % | 41,1735 % |
| 4 | 81,4681 % | 82,5809 % | 82,5114 % | 38,9666 % | 39,1281 % |

Le gain d'exact-K n'est pas concentré sur un seul fold. `low_k_fusion` ne perd aucun fold en exact-K polyphonique, conformément au garde-fou structurel.

## Décision

Ne promouvoir aucun bras V27 : le protocole exigeait aussi de ne pas baisser le F1 événementiel par rapport à V10.4. La fusion basse cardinalité est néanmoins validée comme direction utile, car elle améliore l'exact-K global et polyphonique sur les cinq folds avec une perte F1 limitée à 0,1946 point.

La suite la plus ciblée est un veto conditionné par la confiance de V26 uniforme, calibré uniquement dans les partitions internes de chaque fold. Il doit conserver les corrections K=0/K=1 les plus fiables tout en évitant de retirer les événements V10.4 qui deviennent les 608 FN supplémentaires de `low_k_fusion`. Les mêmes folds externes ne doivent pas servir à choisir puis à présenter ce seuil comme une validation indépendante.
