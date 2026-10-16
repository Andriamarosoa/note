# S637 — Vérification réelle GuitarSet des fenêtres ×1, ×2 et ÷2

**Verdict : rejet des fusions naïves et conservation de S614.**

## Protocole reproductible

- Branche `codex/s637-guitarset-three-view-groundtruth`.
- CI vérifiée :
  https://github.com/Andriamarosoa/note/actions/runs/38916329738
- Artefact immuable GitHub Actions :
  `S637_REAL_GUITARSET_ANNOTATED_TP_FP_FN_F1/report.json`.
- 12 enregistrements GuitarSet **réels**, durée totale 284,294 s,
  groupes de compositions DEV BN3/Funk3/Rock1, annotations originales.
  Les compositions TEST sont exclues. DEV avait déjà été observé ;
  ce n'est pas une validation de généralisation indépendante.
- Modèle et tête K0..6 natifs gelés :
  SHA256 `7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f`.
- 14 variantes de décision fixées, dont référence inchangée. Chaque
  source est évaluée à **tolérance ±80 ms** selon le même `event_score`.
- Trois vues causales de signal à chaque hop 256 :
  vue originale, fréquence doublée et notes remappées −12, fréquence
  divisée par deux et notes remappées +12. Pas de nouvelle donnée future.
- Le score original **recoupe exactement** le contrôle historique
  `998 TP / 585 FP / 582 FN, F1=63,10465 %`.

## Résultats globaux

| Stratégie | TP | FP | FN | F1 | Δ F1 points |
|---|---:|---:|---:|---:|---:|
| S614 original | 998 | 585 | 582 | 63,105 % | 0 |
| Bass ×2, seuil 0,60 | 1068 | 1267 | 512 | 54,559 % | −8,545 |
| Bass ×2, seuil 0,80 | 1044 | 834 | 536 | 60,382 % | −2,723 |
| Bass ×2, seuil 0,90 | 1018 | 655 | 562 | 62,588 % | −0,516 |
| High ÷2, seuil 0,60 | 998 | 903 | 582 | 57,340 % | −5,765 |
| High ÷2, seuil 0,80 | 998 | 813 | 582 | 58,862 % | −4,243 |
| High ÷2, seuil 0,90 | 998 | 717 | 582 | 60,577 % | −2,528 |
| Bass+High, seuil 0,80 | 1044 | 1062 | 536 | 56,647 % | −6,458 |
| Bass+High, octave veto, K-support 0,35 | 1024 | 828 | 556 | 59,674 % | −3,431 |

## Sous-groupes réels

### GuitarSet 6 pistes `_comp`

| Cas | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Référence | 689 | 488 | 485 | 58,613 % |
| Bass 0,90 | 709 | 543 | 465 | 58,450 % |

Bass 0,90 : +20 bonnes attaques, mais +55 faux positifs.

### GuitarSet 6 pistes `_solo`

| Cas | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Référence | 309 | 97 | 97 | 76,108 % |
| Bass 0,90 | 309 | 112 | 97 | 74,728 % |

Bass 0,90 : aucune bonne attaque supplémentaire, +15 faux positifs.

## Importance de la population

GuitarSet DEV contient :
- 1196 vrais événements MIDI E2–D#4 ;
- 383 vrais événements E4–D#5 ;
- **un seul** vrai événement E5+.

Il est donc **impossible d'utiliser GuitarSet DEV seul pour juger le High
Pitch Rescue E5+**. Les centaines de propositions aiguës observées ici
sont presque toutes des faux positifs, mais il faut inclure URMP ou un
autre corpus réellement riche en aigus pour mesurer son rappel.

La fusion simple augmente les attaques et les confusions de registre.
L'octave veto et le veto K à seuil constant atténuent les faux positifs,
mais **ne parviennent pas à inverser le verdict**.

## Conclusions et pistes concrètes

1. **Ne pas promouvoir** un seul des 14 correcteurs évalués.
2. Préserver les têtes NOTE originales et le K natif.
3. Entraîner sur compositions TRAIN (pas DEV) une tête de sélection
   par candidat/fenêtre utilisant : nouveauté de fondamentale, ratio
   harmoniques, activité d'octave, soutien K, état de la note et
   score de la branche native. Elle doit apprendre l'abstention.
4. Faire un audit apparié des vrais bass récupérés et des faux bass
   ajoutés selon solo/polyphonie/style (pas de règles statiques
   généralisées à partir du DEV).
5. Garder le High-Pitch Rescue pour des données E5+ représentatives
   avec vérité terrain séparée.
6. Avant toute version LIVE, mesurer l'overhead de calcul et la file
   microphone ; l'exécution S637 était **offline**, pas PortAudio.

**Attention :** ce rapport représente les données réelles de GuitarSet,
mais pas une nouvelle validation sur TEST scellé.
