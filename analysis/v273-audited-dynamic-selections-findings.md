# Audit individuel avant pondération — 63 sélections dynamiques

**Date : 2026-10-08.** Implémentation et tests :
[GitHub Actions run 37744137558](https://github.com/Andriamarosoa/note/actions/runs/37744137558)
(**succès**). Il s'agit d'un protocole exploratoire, pas d'une
modification du modèle `freeze_local_combo`.

## Règle désormais implémentée

**Chaque sélection / combinaison est auditée individuellement AVANT
l'inférence du fold de test.**

Pour les six têtes disponibles (H0–H5), il existe 63 sous-ensembles non
vides. Dans **chacun des quatre folds externes** 0,1,2,4, les 63
combinaisons sont évaluées sur les données d'apprentissage avec
**probabilités d'experts hors-fold**. L'artefact contient donc
**252 audits de sélection**, chacun documentant les 7 classes
**réelles** K0..K6, les corrections, les régressions, les changements
neutres, et les transitions de classe originale K2/3/4 vers classe
proposée K2..6.

Il y a donc deux tables complémentaires :

- **Diagnostic réel par vrai K** : corrige, régresse, neutre, solde.
  Sert à comprendre les points forts et faibles d'une hypothèse.
- **Fiabilité conditionnelle par prédiction initiale et proposition** :
  `base K3 -> proposition K2` possède son propre taux de gain/perte.
  Cette table peut être consultée **à l'inférence** sans connaître le vrai K.

Les poids des 63 combinaisons sont désormais affectés par ces audits :
`poids final proportionnel à (poids du routeur neuronal / sqrt(nombre
de têtes)) * exp(6 * utilité_audit)`.
Cette utilité mélange : bénéfice global de la sélection (35 %),
bénéfice conditionnel K initial → K proposé (45 %) et bénéfice
moyen des têtes composantes (20 %).

Le score de bénéfice est lissé :
`(corrections - régressions) / (nombre de modifications + régularisation)`,
avec régularisation 30 pour une transition particulière et 65 au
niveau global. Une expérience minuscule n'obtient donc pas un poids
exagéré. **Le vrai K du fold externe ne participe jamais au calcul
du poids.**

## Exemple : audit individuel de H2 sur le training OOF du fold externe 2

| Vrai K | Corrections | Régressions | Solde |
|---|---:|---:|---:|
| K0 | 0 | 0 | 0 |
| K1 | 0 | 0 | 0 |
| K2 | 376 | 214 | **+162** |
| K3 | 247 | 354 | **−107** |
| K4 | 120 | 191 | **−71** |
| K5 | 6 | 0 | +6 |
| K6 | 0 | 0 | 0 |
| **Total** | **749** | **759** | **−10** |

Ce n'est pas la performance de la tête sur le **fold 2 lui-même**.
L'audit a été calculé **sur les trois autres folds**, puis utilisé
pour prédire le fold 2 sans utiliser ses réponses.

Il prouve le point soulevé : **H2 est utile pour corriger K2
et dangereuse lorsqu'il faut préserver K3/K4**. Son poids doit donc
dépendre de la transition proposée et du contexte, et non du seul
gain global.

Les mêmes rapports existent pour **les 63 combinaisons** sur chacun
des quatre folds, incluant les H0–H5 isolées.

## Premier test complet de la pondération auditée

Nombre d'événements natifs : **59 309**, dont **7 493** initialement
prédits K2/K3/K4 peuvent être révisés. Baseline officielle
`freeze_local_combo` : **81,6976 % Exact-K global,
34,2586 % Exact-K poly**.

| Routeur | Exact-K global | Exact-K poly K2..K6 | Corrections | Régressions | Bilan |
|---|---:|---:|---:|---:|---:|
| Référence gelée | 81,6976 % | 34,2586 % | — | — | — |
| Pondération auditée, souple | 81,7734 % | 34,8680 % | 140 | 95 | +45 |
| **Pondération auditée, prudente** | **81,7667 %** | **34,8138 %** | **78** | **37** | **+41** |
| Audit puis choix d'un sous-ensemble | 81,7414 % | 34,6107 % | 110 | 84 | +26 |
| Routeur précédent, sans abstention (même re-run) | 81,7869 % | 34,9763 % | 349 | 296 | +53 |

Pour la variante auditée prudente, selon le **vrai K évalué** :
K0 0, K1 0, **K2 +78**, **K3 −25**, **K4 −12**,
K5 0, K6 0. Le système diminue les régressions par abstention
mais n'empêche toujours pas toutes les dégradations K3/K4.

**Important :** H0–H5 proposent uniquement K2..K6 sur une
population où la référence prédisait K2, K3 ou K4. On ne peut
donc pas obtenir une correction directe vers K0 ou K1 dans
cette configuration. Pour auditer des bénéfices K1 comme
dans l'exemple de l'utilisateur, il faudra des têtes
globales K0–K6 et une politique de risque poly adaptée.

## Fichiers audités et preuves

- `scripts/evaluate_v273_audited_selections.py`
- `test/test_v273_audited_selections.py`
- `.github/workflows/v273-audited-dynamic-selections.yml`
- Artefact du run 37744137558 :
  `v273-audited-dynamic-selections` comprenant
  `audits-all-selections.json` (63 audits individuels pour chacun des 4
  folds), `all-audited-selections.npz` (poids des 63 sélections et
  toutes les prédictions), `report.json`, `report.md`.

Aucune fuite volontaire de vérité terrain du fold tenu de côté,
aucune promotion du modèle, fold 3 et player05 exclus.
Les folds 0/1/2/4 ont été inspectés auparavant, donc
**pas de validation réellement indépendante**. La fenêtre comprend
160 ms après l'événement, elle n'a donc pas la même contrainte
causale que le modèle principal.

## Pour franchir le blocage suivant

Une table de fiabilité par transition seule ne suffit pas à
identifier avec certitude le vrai K. Les K3/K4 masqués peuvent
ressembler à des K2. Avant d'augmenter à H6–H10, il faudra
former un arbitre de risque avec des prévisions hors-fold,
conditionné par les probabilités calibrées, les audits par K
et les caractéristiques signal, et lui apprendre à **s'abstenir**
lorsque le coût d'une régression probable excède le bénéfice.
