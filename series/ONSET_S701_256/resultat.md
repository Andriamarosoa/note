# Onset S613 + S701 — poids entraînés à 256 : résultat FIT80 SIGNÉ

**Run GitHub Actions** : [#39716211840](https://github.com/Andriamarosoa/note/actions/runs/39716211840) (science `signed_fit80` échec intentionnel de non-régression ; `smoke` réussi). **Artefact** : [#12399607233](https://github.com/Andriamarosoa/note/actions/runs/39716211840/artifacts/12399607233), `report.json` complet, 80 pistes, 306 536 décisions PCM256, aucun S708 chargé.

## Comptes authentiques

| Modèle | TP | FP | FN | Précision | Rappel | F1 |
|--|--:|--:|--:|--:|--:|--:|
| S613 natif, poids 256 | 8099 | 4600 | 3168 | 63,777 % | 71,882 % | **67,5874 %** |
| S613 + S701 shadow, poids 256 | 8113 | 4603 | 3154 | 63,802 % | 72,007 % | **67,6563 %** |
| Référence historique S708/H0 | 8528 | 3489 | 2739 | 70,966 % | 75,690 % | **73,2520 %** |

S701 améliore S613 de **+14 TP, +3 FP, −14 FN**, soit seulement **+0,0688 point F1**. **15 des 80 pistes** seulement ont un changement dans leurs comptes par rapport à S613. Le gain marginal de S701 ne répare pas le déficit principal : les fausses attaques S701 sont **1114** de plus que la référence, et le rappel est inférieur d'environ **3,68 points**.

## Répartition par composition (chiffres extraits du report complet)

| Composition | S613 TP/FP/FN | S613 F1 | S701 TP/FP/FN | S701 F1 |
|--|--|--:|--|--:|
| BN1 | 1212 / 726 / 489 | 66,61 % | 1217 / 726 / 484 | 66,79 % |
| Funk2 | 1749 / 674 / 729 | 71,37 % | 1753 / 675 / 725 | 71,46 % |
| Jazz1 | 1196 / 484 / 489 | 71,08 % | 1196 / 485 / 489 | 71,06 % |
| Rock3 | 2420 / 1333 / 899 | 68,44 % | 2421 / 1334 / 898 | 68,45 % |
| SS1 | 1522 / 1383 / 562 | 61,01 % | 1526 / 1383 / 558 | 61,13 % |

**Diagnostic :** FP particulièrement élevés sur SS1 et Rock3; le faible effet de S701 est confirmé partout, avec légère régression Jazz1. Ces observations ne prouvent pas encore les causes acoustiques fondamentales. Un seuil unique ajusté a posteriori sur FIT80 serait une fuite de sélection, pas une amélioration prouvée.

## Verdict et suite

**NON VALIDÉ — régression de performance, pas erreur technique.** Le `signed_fit80` lève correctement `AssertionError: Onset trained256 native below preserved historical H0 threshold`. Le statut scientifique doit rester rouge ; retirer cette assertion serait une falsification de validation.

Réutiliser cet artefact immuable pour l'audit `python -m onset256.audit --report report.json --out audit.json`. Pour progresser scientifiquement, entraîner des spécialistes distincts sur compositions TRAIN et sélectionner uniquement sur inner-CAL avec sorties OOF, puis évaluer un OUTER indépendant. Évaluer explicitement les FP par registre, harmonique, texture et budget de notes, sans régression Onset/Offset, sans tenir une note plus longtemps, et sans ajouter de lookahead.

Les cinq autres caractéristiques et le MIDI officiel ne sont pas modifiés. FIT80 est déjà exploré et ne certifie pas la généralisation externe.
