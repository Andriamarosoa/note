---
serie: S837
statut: termine
type: experience
reference: S708
branche: codex/note-s837-six-axis-80pct-independent-loops256
run_id: 39998150139
commit_sha: 6216833aaeeec2880c804c4c0d04d4b9b3d34b1a
promotion: false
---

# S837 — Résultats signés : six boucles, gains et régressions

## Exécution
- **GitHub Actions [run #39998150139](https://github.com/Andriamarosoa/note/actions/runs/39998150139) : SUCCESS**.
- 30 tests unitaires PASS, sélection CAL-other-compositions sans utiliser OUTER pour choisir sa politique.
- Artifact `S837_REAL5_SIX_AXIS_CV_POLICY_GRID_AND_ERROR_AUDIT`, ID 12519934315 : rapport détaillé et erreurs par composition.
- 5 vrais GuitarSet `_comp` signés SHA, un BN1/Funk2/Jazz1/Rock3/SS1, **20 062 fenêtres de décision PCM256**.
- Première tentative [#39997037972](https://github.com/Andriamarosoa/note/actions/runs/39997037972) arrêtée pour convention tp/TP ; premiers deux scores K identiques à la reprise, autopsie préservée dans [audit_loop1.md](audit_loop1.md).
- Aucun autre axe source ni modèle S708 original modifié, aucune promotion. Le décodeur officiel MIDI est intact.

## Métriques
Les six métriques sont des résultats SHADOW **sur cinq pistes**, pas un MIDI fusionné ni les valeurs FIT80 (80 WAV).

| Caractéristique | Témoin identique | Nouvelle variante OUTER | Gain en points |
|---|---:|---:|---:|
| Exact-K accuracy | 56,5995 % | 56,2706 % | **−0,3290** |
| Active Exact-K accuracy | 46,7725 % | 45,4503 % | **−1,3222** |
| Onset F1 | 68,4475 % | 64,8356 % | **−3,6120** |
| Offset F1 événement | 18,1818 % | **49,3175 %** | **+31,1357** |
| Frame Pitch F1 | 84,9695 % | 84,9036 % | **−0,0658** |
| Full Note F1 | 67,7172 % | 68,3574 % | **+0,6402** |

**Aucune promotion**, puisque seules des améliorations isolées ont été obtenues ; un seul axe Frame Pitch dépasse 80 % sur ce jeu d'essai, et lui-même recule un peu.

### Onset F1 — autopsie
TP **884→838**, FP **407→455**, FN **408→454**, F1 −3,61 points. Funk2 : **45 TP perdus**, alors que la politique `on0.85_s0.20_r0.25` n'en récupère aucun (FP −16). Rock3 : **10 TP récupérés mais 17 perdus**, FP nets +2 ; SS1 : **6 TP récupérés mais +62 FP nets**, malgré le seuil Onset plus bas. 44 nouvelles FP SS1 sont associées temporellement à une note grave distante de 12 ou 19 demi-tons, **sans preuve** que l'harmonique soit la cause physique. La CAL-other-composition n'identifie pas les décisions à risque.

### Offset F1 — gain important, faux événements restants
La sélection de politique `off0.50_s0.65_r0.25` sur les cinq OUTER a fait passer **TP 131→560**, **FP 36→437**, **FN 1 143→714**. Elle récupère respectivement 49/116/81/114/85 bonnes fins de note (GT) sur BN1/Funk2/Jazz1/Rock3/SS1, et perd 4/9/2/1/0 anciennes. Elle ajoute 51/119/88/93/86 nouveaux événements faux, et en supprime 6/6/4/9/11. Le caractère systématique du gain rappelle que la règle originelle `release=.80, low_sustain=.35` était beaucoup trop restrictive pour émettre des releases à 50ms, tandis que la règle moins restrictive produit des nouvelles fausses fins de notes. **49,32 % reste loin de 80 %.**

### Full Note F1 — petit gain et régressions locales
TP **838→876**, FP **345→395**, FN **454→416**. Gains BN1 +4,94 points, Funk2 +2,13 et Jazz1 +2,67, mais Rock3 **−1,22** et SS1 **−1,03** points. L'ajout de TP ne compense pas toujours les nouvelles FP du décodeur Full Note.

### Frame Pitch F1 — régression interdite
TP **57 365→58 502**, FP **7 981→9 627**, FN **12 314→11 177**. Les 1 137 TP supplémentaires s'accompagnent de **1 646 FP supplémentaires** ; SS1 baisse d'environ un point malgré BN1/Rock3 en progression. Même supérieur à 80 %, cette variante n'est pas acceptable.

### Exact-K et Active Exact-K — diagnostique par vrai K0–K6
- Exact-K : **−66 blocs correctement comptés nets** (Jazz1 −22, Rock3 −44 ; autres inchangés).
- Active-K : **−254 blocs corrects nets** (BN1 −112, Jazz1 +31, Rock3 +21, SS1 −194, Funk2 = 0).
- Pour BN1 Active-K : 723 décisions de compte changées, 166 récupérées mais **278 perdues**, toutes vers un plus petit K ; le seuil d'activation plus haut supprime à la fois fausses notes et vraies notes. SS1 : 990 décisions changées, 248 récupérées mais **442 perdues**. Jazz1 et Rock3 illustrent les petits gains qui restent éligibles comme nouvelles **sélections de fragments** si un veto appris peut bloquer les cas opposés.
- Le rapport fournit pour chaque composition les changements détaillés par K0–K6, nombre de corrections, erreurs perdues et exemples par échantillon. La cause acoustique harmonique précise reste à confirmer par de vraies analyses FFT, pas par un simple nom de règle.

## Comparaison à S708
Le témoin historique S708/H0 sur 80 WAV possède Onset F1 73,252019 % et Offset accuracy 69,886229 %. Ces métriques **ne se confondent pas** avec les cinq pistes ci-dessus ou avec Offset F1 ; la source S708 avait été entraînée au pas 1024 et est ici rejouée causalement tous les 256. Le MIDI officiel et les caractéristiques S835 sont conservés, aucune fusion fantaisiste.

## Audits
| Rôle | État | Preuve / limite |
|---|---|---|
| Données | PASS | 5 fichiers originaux SHA, 5 compositions ; 75 autres non évalués |
| Acoustique | PASS | 20 062 nouveaux blocs causaux PCM256, aucun hold |
| Architecture | PASS | six décodeurs indépendants, état et MIDI originaux protégés |
| Apprentissage | PASS | calibration excluant la composition OUTER ; modèle source 1024 préentraîné |
| Métriques | FAIL | Exact-K, Active-K, Onset et Frame Pitch régressent ; quatre têtes loin de 80 et une cinquième sous 80 |
| LIVE | UNKNOWN | aucune mesure p95/p99 Windows/iRig ni MIDI fusionné validé |

## Verdict
**S837 terminé en recherche, PROMOTION REFUSÉE**. La progression Offset est documentée comme nouvelle *candidature isolée* mais son FP reste très haut, et une amélioration d'une tête n'autorise pas de modifier les cinq autres. Le protocole des seuils fixes ne peut pas être présenté comme une solution générale vers 80 %.

## Prochaine action
S838 expérimente un sélecteur **par fragment 256** appris sur TRAIN pour acceptation/rejet des corrections Exact-K et Active Exact-K, calibration CAL disjointe, puis évaluation OUTER. Onset/Offset demandent des modèles de gestion des FP/FN privés avec décision actuelle seulement. Exécuter ensuite les 80 WAV intégralement et auditer les p99 sur Windows/iRig avant le moindre assemblage MIDI.
