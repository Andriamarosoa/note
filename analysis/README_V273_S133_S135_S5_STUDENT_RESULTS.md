# S133–S135 — S5 d'origine non reproduit ; nouvelles alternatives déployables

**Branche** : `codex/v273-scope-rescue` — travail de recherche uniquement.
**Référence officielle** : `freeze_local_combo` inchangée.
**S80 original** : **non promu**.
**S5 original** : les quatre checkpoints et leurs sorties historiques sont
conservés, mais le vrai producteur des 416 colonnes `owned527`
demeure introuvable dans les sources auditées ([S132](https://github.com/Andriamarosoa/note/actions/runs/38189691837)).

## La nouvelle chaîne

1. Source authentique GuitarSet — 58 résumés + 46 valeurs géométriques +
   7 one-hot K gelés (111D), sans les vrais labels en entrée.
2. Professeur : probabilités brutes et décisions S5 **originales**
   archivées par fold, SHA vérifié. Les labels `k` historiques ne sont
   jamais utilisés pour entraîner les nouveaux modèles.
3. Apprentissage sur les folds **1/2/4** ; audit des décisions et
   probabilités S5 sur le fold **0** (15 952 événements).
4. Pour une prédiction nouvelle, apprentissage d'une version séparée
   sur les quatre folds disponibles, puis inférence sur **57 événements
   acoustiques GOAT**, sans lecture de leurs annotations.
5. Ces étudiants **n'utilisent pas les poids originaux S5** et ne
   reconstituent **pas** le réseau S80 de bout en bout.

## Résultats S133 — [run 38190510117](https://github.com/Andriamarosoa/note/actions/runs/38190510117)

| Étudiant (entrées) | Accord K avec S5 fold0 | Corrections / régressions vs S129 | Rappel macro K2–K4 du professeur |
|---|---:|---:|---:|
| S129 (111D, HGB) | 14 557 / 15 952 (91,26 %) | base | 58,54 % |
| **S133 (111D, LightGBM)** | **14 619 / 15 952 (91,64 %)** | **246 / 184 = +62** | **60,85 %** |
| S133 (111D + 245D flux temporel) | 14 586 / 15 952 (91,44 %) | 261 / 232 = +29 | 58,76 % |

Les deux variantes S133 produisent sur les 57 événements externes GOAT :
`[46,9,2,0,0,0,0]` de K0 à K6. Cette distribution **ne mesure pas**
l'Exact-K réel.

## Résultats S134 — [run 38190754219](https://github.com/Andriamarosoa/note/actions/runs/38190754219)

La variante `S134` entraîne **sept régressions LightGBM sur les
probabilités S5 originales** (cibles *soft*), toujours sur le seul 111D.
Les sorties positives sont normalisées sur la simplexe K0–K6.

| Sortie | Erreur absolue moyenne des 7 probabilités S5 | Accord K original |
|---|---:|---:|
| S129 | 0,028867 | 14 557 |
| S133 111D | 0,030387 | 14 619 |
| **S134 régression soft** | **0,022063** | **14 550** |

S134 reproduit mieux les probabilités de S5 mais perd **69** accords de
classe par rapport à S133. Le bilan S134 est de **269 corrections pour
338 régressions** face à S133 (gain net −69).
GOAT K prédits : `[44,9,4,0,0,0,0]`.

## Résultats S135 — [run 38190945241](https://github.com/Andriamarosoa/note/actions/runs/38190945241)

Combinaison **sans entraînement supplémentaire** :

- Garder le K prédit par **S133 111D**.
- Garder les sept probabilités proposées par **S134** tant que leur
  `argmax` est identique.
- Sinon, relever uniquement la probabilité de la classe K de S133 à
  l'ancien maximum + `1e-6`, puis renormaliser.
- Ne pas consulter les vrais K, les labels GOAT, ni la vérité archivée
  pour la décision.

Résultat fold0 : **14 619 / 15 952 décisions identiques** à S5,
**MAE = 0,02195535**, 708 événements ajustés par la règle.
GOAT : **2 événements** ajustés, histogramme
`[46,9,2,0,0,0,0]`.

La règle S135 a été proposée après lecture des résultats de S133 et
S134 sur fold0 : **l'ensemble a servi à la recherche et ne constitue pas
une validation indépendante**. S135 n'est pas le checkpoint S5 original
et il reste **1 333 désaccords K** sur l'historique.

## Verdict de déploiement

- Le code d'inférence et les poids des **nouveaux** étudiants (S133/S134)
  existent, ainsi qu'un correcteur S135 déterministe et sans étiquette.
  Ils forment une piste de `S5-student-v1`, pas une restauration bit-exacte.
- Les probabilités/choix nouveaux sur GOAT sont calculés **sans annotation**.
  Aucune exactitude de K réel n'y est mesurée.
- L'intégration S45/S69/S80 n'est **pas** démontrée avec cet étudiant.
  Il faudrait un système distinct, versionné, pré-enregistré et validé
  sur nouveaux morceaux annotés après gel des décisions, comparé avec
  `freeze_local_combo`.
- La promotion de **S80 historique** reste bloquée par l'absence des
  caractéristiques originales `owned527` et des producteurs nécessaires.
- Les archives de modèles étudiants devaient être préservées au-delà
  des 30 jours de rétention des Actions ; un workflow S136 est
  prévu pour créer une prerelease GitHub immuable avec les trois jeux
  de preuves et leurs SHA.

**Tous les chiffres de ce rapport sont de la fidélité aux sorties
du professeur S5 historique, pas des pourcentages d'accuracy Exact-K.**
