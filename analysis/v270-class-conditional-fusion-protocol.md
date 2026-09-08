# V27 : fusion de comptage conditionnelle aux classes basses

Le run V26 [34201830555](https://github.com/Andriamarosoa/note/actions/runs/34201830555) a terminé les cinq folds. Le retrait de la pondération augmente l'exactitude K globale, mais déplace fortement le compromis vers le sous-comptage.

| Modèle | Exact K global | Exact K polyphonique | F1 événementiel à 50 ms | FP | FN |
|---|---:|---:|---:|---:|---:|
| V10.4 outer-clean | 81,9339 % | 36,5068 % | 80,38 % | 6 399 | 10 203 |
| V26 pondéré | 80,7133 % | 37,1982 % | 78,8486 % | 9 189 | 9 460 |
| V26 uniforme | 82,7076 % | 28,0608 % | 78,0591 % | 3 440 | 13 711 |

Le modèle uniforme améliore surtout K=0 et K=1, mais ne doit pas remplacer V10.4 sur toute la distribution. Les deux têtes V26 sont en désaccord sur 8 969 des 76 768 fenêtres ; l'oracle qui choisit la bonne tête parmi elles atteindrait 85,7375 % d'exact-K. Cette borne est descriptive et n'est pas un résultat déployable.

## Traitement V27 fixé

V10.4 conserve les candidats, leurs timestamps et leur classement. V26 uniforme sert uniquement de spécialiste des faibles cardinalités. Deux règles déterministes sont publiées ensemble :

- `null_veto` : remplacer K=1 de V10.4 par K=0 seulement lorsque V26 uniforme prédit K=0 ;
- `low_k_fusion` : utiliser V26 uniforme lorsque V10.4 prédit K=0 ou K=1, sinon conserver V10.4.

Il n'y a aucun paramètre entraînable, seuil ajusté ou consultation des annotations au runtime. Pour une cible réellement polyphonique K>=2, une règle ne peut modifier qu'une prédiction V10.4 déjà erronée, puisque toute prédiction V10.4 correcte vaut au moins 2. L'exact-K polyphonique ne peut donc pas diminuer par construction.

## Protocole et limites

L'audit réutilise les prédictions outer-clean V10.4 du run 33664141756, les probabilités V26 du run 34201830555 et le cache spectral imbriqué du run 33647694565. Il reproduit d'abord TP/FP/FN V10.4, puis évalue les deux règles avec le même ranking événementiel V10.4. La validation historique et locked12 ne sont ni indexées ni évaluées.

Les cinq folds externes ont déjà été examinés pour formuler ces règles : V27 est donc un audit de développement, pas une validation indépendante. Tous les folds et les deux règles seront publiés. Aucun échec GitHub Actions ne dépendra d'un gain métrique.

Le résultat est favorable seulement si l'exact-K agrégé progresse, si la progression n'est pas limitée à un fold, et si le F1 événementiel à 50 ms ne baisse pas par rapport à V10.4. Aucun bras ne sera promu sur le seul gain K=0.
