# Exact-K V27.3 — réseau neuronal de correction à risque couplé

**Expérience du 8 octobre 2026 — recherche, PAS remplacement de la référence.**

## Question

Une tête d'audit peut prédire le risque, mais si le décodeur ne la consulte
jamais, le modèle peut continuer à détruire un K2 correct en prédisant K1.
Nous testons un réseau qui **apprend la probabilité de réussite de chaque
correction K2–K6 ainsi que l'intérêt de KEEP**, avec les audits OOF
comme caractéristiques et non comme poids de règles prédéfinis.

Le calcul des logits d'action passe désormais **exclusivement par les
têtes de fiabilité apprises**. La tête de risque individuelle H0–H13
est reliée, via la sélection conditionnelle par K, à cette décision.

K0 et K1 ne possédant actuellement pas de tête avec un posterior
nonpoly légitime, nous **interdisons une nouvelle proposition K0 ou
K1** plutôt que de laisser le réseau produire cette classe sans
évidence. Cette mesure protège de la régression dominante K2→K1,
mais empêche aussi de corriger les faux K2/K3/K4 dont le vrai K est
K0/K1 : il faut un vrai adaptateur entraîné K0/K1 avant de
réactiver ce type de correction.

## Code et tests

- \`scripts/learn_v273_risk_coupled_selector.py\` : classe
  \`RiskCoupledClassSelector\`, qui calcule l'avantage relatif
  \`log P(K cible correcte) - log P(KEEP préférable)\`.
- \`scripts/evaluate_v273_risk_coupled_selector.py\` : entraînement
  nested OOF et métriques natives complètes.
- \`test/test_v273_risk_coupled_selector.py\` : gradients de la sortie
  vers la tête de risque, masque K0/K1, pertes de calibration
  et entraînement réel sur un petit lot.
- \`.github/workflows/v273-risk-coupled-neural.yml\` :
  workflows CI d'entraînement et évaluation.

La prédiction n'utilise ni le vrai K de l'événement évalué, ni un
veto manuel dérivé des labels de folds test. La loss comporte
une classification des actions, une calibration probabiliste de
chaque correction et de KEEP, un audit auxiliaire du risque de
chaque tête, et une pénalité différentiable de destruction d'une
prédiction polyphonique déjà correcte. Ces termes ont des
hyperparamètres fixes **pour l'entraînement**, sans poids de
sélection statiques à l'inférence.

## Résultats numériques du premier passage (4 folds terminés)

Le premier passage a entraîné tous les folds et exporté
\`report.json\`/\`.npz\`. Un défaut **d'affichage dans report.md**
a ensuite interrompu le job ; le calcul chiffré est toutefois
complet. Le correctif d'export a été appliqué et le run complet
est [37754337532](https://github.com/Andriamarosoa/note/actions/runs/37754337532).

| Modèle | Exact-K global | Exact-K poly | Corrections | Régressions | Net |
|---|---:|---:|---:|---:|---:|
| \`freeze_local_combo\` | **81,6976 %** | **34,2586 %** | — | — | — |
| Neuronal conditionnel par K précédent | 82,3012 % | 28,3006 % | 1 402 | 1 044 | +358 |
| **Risque réellement couplé au choix** | **81,6942 %** | **34,2316 %** | **217** | **219** | **−2** |

Par vrai K, réseau à risque couplé vs \`freeze_local_combo\` :

| Vrai K | Corrections | Régressions | Net |
|---|---:|---:|---:|
| K0 | 0 | 0 | 0 |
| K1 | 0 | 0 | 0 |
| K2 | 95 | 43 | **+52** |
| K3 | 58 | 117 | **−59** |
| K4 | 37 | 59 | **−22** |
| K5 | 27 | 0 | **+27** |
| K6 | 0 | 0 | 0 |
| **Total** | **217** | **219** | **−2** |

Bilan des folds : 0 **+5**, 1 **−1**, 2 **−8**, 4 **+2**.

## Les corrections qui régressent encore

| Ancien → nouveau K | Actions | Corrections | Régressions | Net |
|---|---:|---:|---:|---:|
| K3→K4 | 135 | 33 | 48 | **−15** |
| K4→K3 | 103 | 26 | 39 | **−13** |
| K3→K2 | 175 | 58 | 63 | **−5** |
| K2→K4 | 26 | 4 | 8 | −4 |
| K2→K3 | 85 | 32 | 33 | −1 |
| K4→K2 | 78 | 37 | 7 | **+30** |

## Verdict et recherche suivante

**L'architecture supprime l'erreur de conception dominante du
décodeur : les sorties sans spécialiste K0/K1, et le risque auxiliaire
non raccordé à la décision.** Elle réduit les régressions de
**1 044 à 219** mais n'améliore pas encore le score
polyphonique de référence : 34,2316 % au lieu de 34,2586 %.

Les régressions restantes sont principalement **K3↔K4** et
les cas K3→K2. L'objectif d'entraînement ne distingue pas encore
assez finement les situations dans lesquelles une telle transition
permet de sauver un accord et celles où elle détruit un accord déjà
juste.

Le prochain travail devra mesurer des risques
\`P(correction juste | K initial, K proposé, audio, auditions)\` avec
soutien par transition (et non seulement par K de destination),
sur des données supplémentaires réellement inédites pour
éviter de surajuster les folds déjà inspectés.

**Aucune promotion à \`freeze_local_combo\`.**
