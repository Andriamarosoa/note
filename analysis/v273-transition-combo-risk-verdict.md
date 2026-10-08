# V27.3 — verdict expérimental du routeur neuronal par transitions et sous-ensembles

**8 octobre 2026 — résultats reproduits dans GitHub Actions.**

- [Run 37755994598 (succès)](https://github.com/Andriamarosoa/note/actions/runs/37755994598).
- [Artefact JSON/CSV/NPZ, audité par train-fold](https://github.com/Andriamarosoa/note/actions/runs/37755994598/artifacts/11540256425).
- Script: `scripts/learn_v273_transition_combo_risk.py`.
- Évaluation: `scripts/evaluate_v273_transition_combo_risk.py`.
- Tests: `test/test_v273_transition_combo_risk.py` : **3/3 réussis**.
- Même cohorte native **59 309 événements**, **7 493 révisables**,
  folds 0,1,2,4 ; fold3/player05 non utilisés.

## Résultat global

| Méthode | Exact-K global | Exact-K poly K2–K6 | Fixes | Régressions | Net |
|---|---:|---:|---:|---:|---:|
| freeze_local_combo | **81,6976 %** | **34,2586 %** | — | — | — |
| Risque par K (précédent) | 81,6942 % | 34,2316 % | 217 | 219 | **−2** |
| Nouvelle sélection par K initial × candidat × combinaisons | 81,6672 % | 34,0149 % | 426 | 444 | **−18** |

Le nouveau routeur augmente les nombres de corrections ET de
régressions. Ce n'est pas une amélioration de l'Exact-K natif.
La sélection explicite des combinaisons et l'entraînement de risque
fonctionnent mais leur gain n'est pas démontré.

## Audit réel par vrai K

| Vrai K | Corrections | Régressions | Net |
|---|---:|---:|---:|
| K0 | 0 | 0 | 0 |
| K1 | 0 | 0 | 0 |
| K2 | 89 | 65 | **+24** |
| K3 | 116 | 249 | **−133** |
| K4 | 132 | 130 | **+2** |
| K5 | 75 | 0 | **+75** |
| K6 | 14 | 0 | **+14** |

**C'est maintenant K3 qui concentre la défaillance.** Les 249
régressions K3 proviennent des transitions :

- K3→K4 : 137 pertes contre 130 corrections.
- K3→K2 : 64 pertes contre 62 corrections.
- K3→K5 : 42 pertes contre 41 corrections.
- K3→K6 : 6 pertes contre 1 correction.

À l'inverse, les classes K5/K6 gagnent 89 prédictions correctes
au total, mais cela ne compense pas K3.

## Transitions nettement déficitaires

| Origine → Sortie | Corrections | Régressions | Net |
|---|---:|---:|---:|
| K4→K6 | 13 | 35 | **−22** |
| K4→K5 | 32 | 48 | **−16** |
| K2→K4 | 2 | 10 | −8 |
| K3→K4 | 130 | 137 | −7 |
| K3→K2 | 62 | 64 | −2 |
| K2→K3 | 68 | 48 | **+20** |
| K4→K2 | 27 | 5 | **+22** |

Le net par transition n'est pas le même que le bilan par vrai K :
chaque transition peut corriger des cas provenant d'une autre
vérité terrain ; une transition avec bilan légèrement négatif peut
comporter de nombreuses pertes K3.

## Effet comparé au précédent routeur apprenant le risque par classe K

Le gain du routeur précédent (par rapport à freeze_local_combo)
était : K2 +52, K3 −59, K4 −22, K5 +27, K6 0.

La nouvelle version a changé ces bilans de :
K2 −28, K3 −74, K4 +24, K5 +48, K6 +14
= **−16 nouvelles prédictions correctes nettes perdues**
par rapport au précédent routeur.

La capacité d'identifier plus finement les combinaisons n'a pas
empêché les régressions vers K4/K5/K6 sur des K3/K4 déjà corrects.

## Audits et limites méthodologiques

- Entraînement différenciable de la sélection des sous-ensembles
  selon le K original, le K cible et les caractéristiques audio.
- 64 combinaisons maximales contenant H0 lorsque Cxy est applicable,
  sinon 32 pour K2–K6. Ce ne sont pas les 127 sous-ensembles
  théoriques sans la contrainte de H0.
- Des audits correction/régression par sous-ensemble sont calculés
  à partir des autres folds internes et du jeu externe TRAIN ;
  le fold externe testé ne fournit aucun vrai K au sélecteur.
- Le vrai K n'est pas une feature à l'inférence, et les
  statistiques des classes présentées ici sont diagnostiques.
- Les sorties OOF des autres folds internes peuvent provenir
  d'experts entraînés sur des lignes du fold courant : il faut
  distinguer exclusion directe des étiquettes d'audit et
  indépendance complète des représentations.
- **Les folds externes ont déjà été inspectés par les expériences
  précédentes. Cela ne constitue PAS une validation indépendante.**
- Le joueur/fold3 manque à la cohorte actuellement utilisée.
  Les métriques sur des données inédites ne peuvent pas être
  établies sans une nouvelle cohorte avec références Exact-K.
- Les anciennes variantes/fixes historiques ne sont pas encore
  toutes raccordées sous forme de têtes exécutables.

**Verdict** : rejet du nouveau candidat comme référence. Les
données montrent qu'un routeur plus flexible peut amplifier certaines
régressions K3 même avec un risque conditionnel explicite. La suite
scientifique exige un audit de calibration **K initial × K cible ×
groupe de signal**, puis des données nouvelles pour évaluer un
changement d'objectif ou de représentation. Ne pas optimiser
une nouvelle règle fixe sur ces folds examinés.

`freeze_local_combo` reste intact.
