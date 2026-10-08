# Audit forensique — pourquoi le routeur neuronal conditionnel à K régresse

**8 octobre 2026.** Le but de ce document est de trouver les **causes concrètes de régression**, et NON de présenter une nouvelle idée de correcteur.

**Audit GitHub Actions réussi :**
https://github.com/Andriamarosoa/note/actions/runs/37748832939

Les deux artefacts du réseau à 14 têtes ont été comparés événement par
événement sur les **59 309 mêmes cas natifs** (folds 0/1/2/4, sans player05/fold3).
Aucun modèle n'a été réentraîné par cet audit ; la référence
\`freeze_local_combo\` n'a pas été modifiée.

## 1. Reproduction

| Système | Exact-K global | Exact-K poly K2–K6 | Corrections | Régressions | Net global |
|---|---:|---:|---:|---:|---:|
| Référence gelée | 81,6976 % | 34,2586 % | — | — | — |
| 14 têtes, routeur global initial | 82,4327 % | 28,3819 % | 1 616 | 1 180 | +436 |
| 14 têtes, routeur conditionnel à K | 82,3012 % | 28,3006 % | 1 402 | 1 044 | +358 |

La modification « poids par tête ET par K » a diminué le nombre de
régressions de 136, mais a aussi perdu 214 corrections : **−78 bonnes
prédictions globales par rapport à l'ancien routeur**. Parmi les deux
versions : 837 prédictions auparavant correctes sont devenues fausses,
contre 759 améliorations dans l'autre sens ; **2 407 décisions ont changé**.
Le masque conditionnel seul n'a donc pas corrigé l'architecture.

## 2. Défaut dominant identifié et mesuré

Le routeur conditionnel modifie **1 822** prédictions K2/K3/K4 vers
**K0 (386)** ou **K1 (1 436)**.

| Destination | Modifications | Corrections | Régressions | Neutres |
|---|---:|---:|---:|---:|
| K0 ou K1 | 1 822 | **798** | **438** | **586** |
| Autres destinations K2–K6 | 1 987 | **604** | **606** | **777** |
| Total | 3 809 | 1 402 | 1 044 | 1 363 |

Les **438 régressions** vers K0/K1, parmi des K2–K4 correctement prédits
par la référence, expliquent **438 des 440 prédictions polyphoniques
correctes perdues nettes** (99,55 % de la baisse poly en nombre de
bonnes prédictions). Les autres transitions K2–K6 ont un gain net
**−2 seulement**, malgré près de 2 000 modifications.

**Contrefactuel purement diagnostique** : si l'on remet uniquement les
sorties K0/K1 du routeur à leur prédiction gelée, le score poly passe de
**28,3006 % à 34,2316 %** (référence 34,2586 %). Le global revient à
**81,6942 %**, presque la référence 81,6976 %.
Ceci ne constitue ni une évaluation indépendante ni une politique à
promouvoir sans nouveaux tests : il localise les pertes.

### 2.1 Les transitions dangereuses, chiffres réels

| Ancien -> Nouveau K | Actions | Corrigées | Régressées | Neutres | Net |
|---|---:|---:|---:|---:|---:|
| **K2 -> K1** | **1 198** | 517 | **361** | 320 | +156 |
| K2 -> K0 | 319 | 159 | 47 | 113 | +112 |
| K3 -> K1 | 204 | 74 | 29 | 101 | +45 |
| K3 -> K0 | 65 | 33 | 1 | 31 | +32 |
| K4 -> K1 | 34 | 14 | 0 | 20 | +14 |
| K4 -> K0 | 2 | 1 | 0 | 1 | +1 |
| K2 -> K3 | 500 | 140 | 166 | 194 | **−26** |
| K3 -> K4 | 267 | 74 | 93 | 100 | **−19** |
| K4 -> K3 | 361 | 103 | 118 | 140 | **−15** |
| K3 -> K2 | 576 | 202 | 167 | 207 | +35 |
| K4 -> K2 | 141 | 46 | 20 | 75 | +26 |

Ainsi, le principal comportement destructeur est **K2 -> K1**
(361 régressions). Mais les transitions poly-polys n'apportent
pratiquement aucune valeur nette à cette version.

## 3. Pourquoi le code permet cette anomalie

### Cause A — Le masque « tête ouverte pour K » ne contraint pas la sortie finale

Fichier : \`scripts/evaluate_v273_neural_history_mix.py\`.

Le masque de support laisse H0 (baseline) seul contribuer aux hypothèses
K0 et K1 : \`support[:,0,:]=True\`, mais les spécialistes H1–H5 ne
supportent que K2–K6. Dans cet adaptateur, H0 attribue pourtant une
probabilité infinitésimale aux K0/K1 (normalisation des seules classes
K2..K6). Le réseau final produit malgré tout **386 K0 et 1 436 K1**
à partir d'un décodeur libre. **Les poids par K sont parfaitement
normalisés et valides, mais ne contraignent pas les logits de décision.**

Contrôle numérique sur les **7 493** événements admissibles :
- Pour K0, l'intégralité du poids des têtes vaut **H0=1**, H1–H13=0.
- Pour K1, l'intégralité du poids des têtes vaut **H0=1**, H1–H13=0.
- Aucun spécialiste ne produit réellement d'évidence K0/K1 dans le
  contrat actuel.

Le décodeur peut donc prédire K0/K1 avec seulement une représentation
apprise des états cachés/du contexte, sans le support K annoncé.

### Cause B — La perte d'action privilégie le global, pas le coût d'une perte poly

Fichier : \`scripts/learn_v273_neural_head_selector.py\`.

\`form_targets\` définit \`target = KEEP\` si baseline juste, sinon
\`target = vrai K\` (donc K0/K1 compris).
\`train_loss\` utilise une entropie croisée d'action **non pondérée**
par le vrai K ni par le préjudice de modifier une bonne prédiction
polyphonique. Le réseau est ainsi incité à produire des corrections
K0/K1, nombreuses, sans coût spécifique pour les bonnes prédictions
K2/K3/K4 détruites.

Ce constat explique la direction de l'optimisation, mais pas à lui
seul la part causale de la perte : l'ablation ci-dessus mesure celle
des actions K0/K1.

### Cause C — Le risque appris n'est pas relié au décideur

\`ClassConditionalAuditSelector.call\` calcule \`decision\`, **puis**
\`risk=self.risk(state)\`; l'argmax \`decision\` est utilisé directement
par \`fit_and_predict\`. Les probabilités de risque ne modifient
jamais le choix final d'action. La perte auxiliaire de risque affecte
indirectement le tronc partagé via les gradients, mais ne constitue
ni un arbitre ni un veto appris au niveau de la décision.

Donc « la tête apprend la régression » ne signifie pas que le
modèle apprend réellement à s'abstenir face à une régression probable.

### Cause D — Les corrections et fixes intégrés ne correspondent pas
aux correcteurs audités historiquement

Le système comporte 4 adaptateurs de transition et 4 adaptateurs KEEP,
issus de formules artificielles. Pour \`C23/C32/C34/C43\`, la sortie
est construite comme :
\`P(KEEP)=1-.95q\`, \`P(K_dest)=.95q\`, où \`q\` est un score moyen
des autres experts. L'adaptateur préfère KEEP dès que
\`q <= 1/1.9 ≈ 0,5263\`. De très nombreuses propositions
restent donc de type abstention au niveau du signal d'audit.

Les anciens correcteurs/fixes sont bien **répertoriés** mais pas
raccordés comme sorties individuelles. Il est incorrect de dire que
leurs avantages et défauts passés sont déjà appris dans les
14 têtes effectives.

### Cause E — La modification conditionnelle réorganise les décisions
sans résoudre leur sûreté

Le changement de routage modifie **2 407 prédictions** par rapport à la
première version : **759 gains, 837 pertes**, soit **−78** net.
Cela démontre que la représentation conditionnelle fournit de nouveaux
degrés de liberté, mais sans signal de coût/abstention propre pour
préserver ce que la référence sait déjà prédire.

## 4. Autres résultats par K réel

| Vrai K | Corrections | Régressions | Net |
|---|---:|---:|---:|
| K0 | 193 | 0 | +193 |
| K1 | 605 | 0 | +605 |
| K2 | 248 | 579 | **−331** |
| K3 | 243 | 304 | **−61** |
| K4 | 76 | 161 | **−85** |
| K5 | 37 | 0 | +37 |
| K6 | 0 | 0 | 0 |

## 5. Verdict d'ingénierie

**Ce n'est pas d'abord un problème de seuil** ni une insuffisance de
nombre de têtes. Il y a une incohérence entre (1) les têtes structurées
par K, (2) le décodeur libre, (3) le risque auxiliaire qui ne commande
aucune action, et (4) la perte qui ne compte pas la destruction des
accords comme un événement de coût distinct.

L'**ablation diagnostique** montre que l'écart poly dominant vient des
sorties K0/K1 sans soutien spécialisé ; les transitions restantes ne
gagnent rien net. La priorité n'est donc pas d'ajouter d'autres fixes :
elle est de **faire respecter le contrat de proposition de chaque tête
par le décodeur**, de **relier le risque de régression au choix KEEP**
et d'**apprendre la valeur réelle des corrections avec un coût explicite
de destruction des notes existantes**. Ces trois interventions doivent
être testées séparément, puis ensemble, sous un protocole frais.

Ce diagnostic est établi **sans modifier \`freeze_local_combo\`**.

## Reproduction et fichiers

- [Run de forensic GitHub Actions](https://github.com/Andriamarosoa/note/actions/runs/37748832939)
- \`scripts/audit_v273_classconditional_regression.py\`
- \`.github/workflows/v273-classconditional-forensic-audit.yml\`
- Artefact : \`v273-classconditional-rootcause-audit\` contenant
  \`audit.json\`, \`audit.md\`, \`transitions.csv\`, et
  \`decision-details.csv\` (toutes les 7 493 décisions).
