# V27.3 — Point de boucles S21/S22 : fiabilité B et accord A→B→A

**Référence non modifiée** : série 18, 49 178/59 309 corrects global (82,9183 %), 2 998/7 385 corrects poly K2–K6 (40,5958 %), **2 934 corrections et 2 210 régressions** face à `freeze_local_combo`. Les tests proviennent tous de la cohorte GuitarSet **déjà exposée** : pas de validation externe ni de modèle promu.

## Série 21 — Arbiteur apprenant à faire confiance à B

Protocole préannoncé : [README_V273_SERIE21_FIABILITE_B.md](README_V273_SERIE21_FIABILITE_B.md). Entraînement : https://github.com/Andriamarosoa/note/actions/runs/37859454327 ; rejeu indépendant : https://github.com/Andriamarosoa/note/actions/runs/37859537595.

96 variantes ont été testées : 4 propositions de A issues des passages initiaux/final avec ou sans B, 2 classificateurs de fiabilité, 2 jeux de caractéristiques avec/sans scores explicites B, et 6 seuils fixés. Chaque classificateur est formé sur d'autres morceaux du même fold ; **76 modèles rechargés**, **96 politiques reproduites exactement**, écart max de probabilité **2,98×10⁻⁸** au rejeu. Cette vérification ne fait pas de la cohorte une nouvelle évaluation indépendante.

| Variante | Global | Poly | Nouveaux corrects vs S18 | Anciens corrects détruits |
|---|---:|---:|---:|---:|
| S18 gardée | 82,9183 % | **40,5958 %** | 0 | 0 |
| S21 meilleur global `with_B logistic ABA2 seuil 0.6` | **82,9402 %** | 39,6344 % | 181 | 168 |

**Aucun des 96 candidats n'a un gain strict sans perdre de correction**.

Archive permanente des 76 modèles et audits : https://github.com/Andriamarosoa/note/releases/tag/v273-series21-research-37859537595.

## Série 22 — confirmation mutuelle entre passages

Protocole : [README_V273_SERIE22_CONSENSUS_ABA.md](README_V273_SERIE22_CONSENSUS_ABA.md). Run mesuré : https://github.com/Andriamarosoa/note/actions/runs/37859786345.

108 variantes à action déterministe, sans nouvel apprentissage : 6 structures d'accord entre passages de A, 3 vérifications de fiabilité dont des contraintes apportées par B, 6 seuils. La vérité n'entre dans aucun des décodages et aucun ancien cas K0–K6 n'est oublié de l'audit.

| Variante | Global | Poly | Nouveaux corrects vs S18 | Anciens corrects détruits |
|---|---:|---:|---:|---:|
| S22 meilleur global `ABA2_ABA4_Bsupports, min logistic/HGB, seuil 0.3` | **83,0161 %** | 39,3365 % | 255 | 197 |

Ce modèle gagne +58 au global, mais **détruit 197 anciennes bonnes prédictions** et régresse en poly. **0 des 108 variantes** respecte le garde-fou strict de conservation des corrections.

## Interprétation et blocage

Le mécanisme A→B→A est implanté depuis S19 et confirmé par un test de changement de message ; l'obstacle observé est la **fiabilité de B et des propositions réintroduites par A**, notamment sur la polyphonie. La récursion seule n'apporte aucune garantie. Le gain brut de centaines de corrections peut cacher presque autant de régressions ; aucune amélioration exploratoire globale seule n'autorise une promotion.

La suite utile n'est pas un nouveau réglage a posteriori de ces seuils. Priorité : **récupérer les décisions exactes par événement du meilleur ancien producteur poly série4 (42,8165 % poly)**, comparer ses corrections/régressions *avec* S18, puis utiliser les messages récurrents comme variables de sélection pour arbitrer deux producteurs diversifiés. Requiert un protocole d'exclusion des morceaux inédits et une validation réellement nouvelle avant toute revendication.
