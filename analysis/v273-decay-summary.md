# Diagnostic interne de décroissance acoustique

**Résultat : conserver V27.3. La variante testée ne montre aucun gain.**

Ces scores agrégés proviennent de l'[exécution GitHub terminée](https://github.com/Andriamarosoa/note/actions/runs/35065190353).
Le diagnostic porte sur 14 001 exemples, dont 1 776 polyphoniques, répartis
en quatre compositions de développement. Le correcteur apprend sur trois
compositions et prédit la quatrième, à tour de rôle.

| Variante | Exact-K polyphonique | Exact-K global |
|---|---:|---:|
| V27.3 gelée | 44,8198 % | 80,3585 % |
| V27.3 + correcteur témoin | 35,3041 % | 79,5657 % |
| V27.3 + correcteur avec décroissance | 34,0090 % | 79,1158 % |

L'ajout de décroissance perd **1,2950 point polyphonique face au contrôle**,
avec une baisse sur les quatre compositions. Le correcteur témoin dégrade
également V27.3 : tout l'écart à la référence ne vient donc pas de la décroissance.

Les 24 tests et les huit ajustements ont réussi. Les scores ont été recalculés
à partir des résultats sauvegardés.

La conclusion porte sur cette formulation et ce correcteur léger, sur des
données déjà étudiées. Ce diagnostic n'est pas une validation indépendante.
Le score V27.3 de 44,8198 % dans ce sous-ensemble est distinct de sa référence
agrégée de 42,6019 %. Aucun modèle n'est promu et aucun nouvel entraînement
n'est déclenché par cette synthèse.
