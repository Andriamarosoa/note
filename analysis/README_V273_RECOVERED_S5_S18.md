# S5 reconstruit : vrais parents et chaîne originale jusqu'à S18

12 octobre 2026 — `freeze_local_combo` reste la référence officielle. S80 n'est pas promu.

Le constructeur S5 précédemment vérifié a été branché sur les **57 événements acoustiques GOAT** conservés par S97 et S116. Les quatre checkpoints S5 originaux produisent leurs probabilités, puis les deux votes historiques : correction du prior pour le parent global, probabilités brutes pour le parent poly, multiplication des classes K2–K6 par 1,5 avant argmax. Les deux votes concordent exactement avec les archives sur les 59 309 événements historiques : **118 618 décisions vérifiées, zéro différence**.

La chaîne choisie S8 → S9 → S13/S14 → S15 safe_OR → S16 → S17 → S18 consomme maintenant ces votes S5 réels. Les 133 checkpoints aval sélectionnés sont inchangés. Le rejeu des six politiques entraînées sélectionnées (S8, S13, S14, S16, S17, S18) retrouve exactement leurs 59 309 décisions archivées chacune. Les raccordements aux parents S9, S15, S16 et S17 sont également vérifiés. Aucun vrai K n'est ouvert ; aucun modèle n'est réentraîné.

## Inférence sur GOAT

Les quatre branches S5 restent distinctes. Chaque branche utilise uniquement les checkpoints de risque du même fold d'origine, moyennés sur leurs modèles exclus par morceau. Les seuils et périmètres des politiques restent historiques. En particulier, S8 intervient seulement pour une référence K2/K3/K4 et un désaccord global/poly.

| Branche originale | K0 | K1 | K2 | K3 | K4 | K5 | K6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fold 0, sortie S18 | 45 | 8 | 4 | 0 | 0 | 0 | 0 |
| Fold 1, sortie S18 | 45 | 8 | 4 | 0 | 0 | 0 | 0 |
| Fold 2, sortie S18 | 46 | 8 | 2 | 1 | 0 | 0 | 0 |
| Fold 4, sortie S18 | 44 | 11 | 1 | 1 | 0 | 0 | 0 |

Il s'agit de nombres de **prédictions**, pas de notes correctement comptées. Sur ces événements, les veto aval ne changent aucune des décisions de leur parent poly S5. Aucune règle de fusion finale des quatre branches n'est choisie ici. L'agrégation des modèles exclus par morceau est une règle explicite de déploiement sur nouveau morceau, pas une politique historique déjà évaluée sur ces WAV.

## Reproduction et préservation

- [Runtime](../scripts/v273_recovered_s5_s18.py), deux étapes `--stage s5` et `--stage s18`.
- [Workflow dédié](../.github/workflows/v273-recovered-s5-s18.yml), un seul run déclenché par les fichiers concernés.
- Entrées acoustiques SHA-vérifiées conservées dans [`inputs/`](evidence/v273-recovered-s5-s18/inputs), issues de S97, run `38182078525`, artefact `11718121757`, et S116, run `38184381730`, artefact `11719226507`.
- [Rapport S5](evidence/v273-recovered-s5-s18/s5_report.json) et [rapport S18](evidence/v273-recovered-s5-s18/s18_report.json), avec empreinte de chaque checkpoint exécuté.
- Prédictions sans annotations : [S5](evidence/v273-recovered-s5-s18/external_s5.npz) et [S5→S18](evidence/v273-recovered-s5-s18/external_s5_to_s18.npz).

S5 exige scikit-learn 1.8.0 ; les poids historiques S8–S18 exigent 1.4.2. Le workflow utilise deux environnements isolés, vérifie les empreintes des sources originales, rejoue toute la cohorte historique puis l'inférence GOAT. Les archives de poids proviennent de trois releases permanentes épinglées dans le script. **Le rejeu local et la [CI #38224327985](https://github.com/Andriamarosoa/note/actions/runs/38224327985) ont réussi.** Les empreintes des sorties S5 et S18 sont identiques entre les deux exécutions. [Preuve CI conservée](evidence/v273-recovered-s5-s18/github_ci.json), artefact `11735608748`.

## Limites restantes

GOAT est un test technique déjà exposé, pas une validation indépendante de précision. Cette récupération ne constitue pas encore le pipeline S80 complet : les cinq familles S45 manquantes, les seize propositions cohérentes, les risques S45 et le déploiement des étages S46–S80 restent à terminer. Un vrai gain global et polyphonique sur une population indépendante reste nécessaire pour la promotion.
