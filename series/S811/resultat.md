---
serie: S811
statut: termine
type: experience
reference: S708
branche: codex/note-s811-blind-nvidia-s810-error-attribution
run_id: 39391755628
commit_sha: c0268b86424280b696f079213543e32709c43002
promotion: false
---

# S811 — résultats de la consultation contradictoire indépendante

## Exécution

[S811 GitHub NVIDIA indépendant #39391755628](https://github.com/Andriamarosoa/note/actions/runs/39391755628) : workflow **SUCCESS**, Gemma et Nemotron ont tous deux rendu des textes et des erreurs vérifiables ; 1/3 atteint le seuil textuel automatique (Nemotron). Kimi a renvoyé **HTTP410**. Le run correctif [#39392258041](https://github.com/Andriamarosoa/note/actions/runs/39392258041) n'a réinterrogé que la troisième place via `z-ai/glm-5.3`, mais ce fournisseur a expiré sans réponse. Les deux premiers modèles n'ont pas été relancés. Prompt et input SHA256 rigoureusement identiques ; aucune opinion ne circulait entre modèles.

**2 avis NVIDIA réellement reçus ; pas de troisième avis et aucun consensus scientifique automatique.** Les affirmations des modèles ont été réfutées dans la suite par des calculs sur les vraies 80 pistes FIT80 et les originaux SHA signés de S810. Programme original, preuves numériques, 4 artefacts fournisseurs, tests et journaux dans `S811_NVIDIA_contre_audit_FIT80.zip`, SHA256 `0832f0719d5d3b6ae30db2a25018163f44b3408b82ba0f53c7bcc08a42cb0c1e`, archive créée dans la conversation. Ce code d'ablation n'a pas été rejoué sur GitHub Actions ; ses résultats sont **locaux et reproductibles**, pas 6 audits CI.
## Métriques

Les métriques SHADOW de S810 à la cadence 1024 (FIT80) sont :

| Axe | S810 diagnostic individuel | Nature |
| --- | ---: | --- |
| Onset F1 | 11,3986 % | 10002 vrais, 154245 faux, 1247 manqués |
| Offset F1 | 12,7457 % | 10479 vrais, 142827 faux, 647 manqués |
| Début Full Note F1 | 14,8887 % | 8518 vrais, 94655 faux, 2731 manqués |
| Exact-K accuracy | 76,8042 % | 58832 / 76600 trames |
| Active Exact-K accuracy | 76,5656 % | 47708 / 62310 trames actives |
| Frame Pitch F1 | 78,7230 % | 132708 TP / 64190 FP / 7546 FN |

Aucun score officiel S704 du MIDI S810 final n'existe à cette étape. Référence S708 CAL32 figée : Onset F1 74,2639 % ; Exact-K 62,0455 % ; Active Exact-K 56,2762 % ; Offset accuracy 77,3292 % ; Full Note F1 57,1319 % ; Frame Pitch F1 76,4626 %. Les deux horloges et méthodologies ne sont pas interchangeables.

### Résultats de falsification — mêmes 80 pistes FIT80

| Test S811 | Onset F1 SHADOW | Offset F1 SHADOW | Détail |
| --- | ---: | ---: | --- |
| S810 original, seuil .50, balanced | 11,40 % | 12,75 % | Parité contrôlée TP/FP/FN avec S810 d'origine |
| Même entraînement sans `class_weight=balanced` | **16,01 %** | **15,16 %** | Seule variable changée, 10 fits, 5 compositions exclues |
| Checkpoints S810 figés, seuil .70 | 16,62 % | 16,09 % | Aucun nouvel entraînement |
| Checkpoints S810 figés, seuil .90 | 22,81 % | 24,12 % | Aucun nouvel entraînement |
| Checkpoints S810 figés, seuil .95 | 26,14 % | 27,36 % | Aucun nouvel entraînement |
| Checkpoints S810 figés, seuil .99 | **34,51 %** | **32,72 %** | Analyse exploratoire FIT, non promotable |

Le seuil .99 réduit les faux positifs Onset **154 245 → 21 055** ; pour Offset **142 827 → 25 055**. Les métriques demeurent loin de 80 % : ce problème n'est pas seulement une erreur de seuil. **La suppression de la pondération équilibrée est le seul changement dont l'effet sur FP est ici isolé par un nouveau réentraînement.** Le seuil est une exploration sur les mêmes FIT80, pas une validation indépendante.

**Full Note, 4 classes**, matrice de confusion [classe vraie][classe prédite] (0 silence, 1 Onset, 2 sustain, 3 release) :
```text
3481217   81890   14198   27769
    992    8518     841     898
   4383   11871   98064   14667
    375     894    1170    5653
```
Sur **94 655 faux débuts** de Full Note, **81 890 (86,5 %) sont en réalité du silence**, contrairement à l'hypothèse d'une seule confusion de maintien. La proportion naturelle « silence » de ces étiquettes est **96,048 %**, supérieure à l'accuracy quatre états du modèle S810 **95,739 %**. Un classifieur prédire toujours silence est toutefois musicalement inutilisable : ce calcul démontre seulement que l'accuracy globale est un mauvais indicateur ici.

**Critique des deux avis externes :** la famille de causes liée à la fréquence des positifs est cohérente avec les mesures : environ **0,30 % de vrais Onset/Offset** sur toutes les trames/hauteurs, contre environ **33–35 % de positifs dans le train plafonné S810**, puis pondération équilibrée. L'ablation confirme une contribution réelle de `class_weight` sans démontrer une cause unique. Les modèles suggèrent aussi une fenêtre de label trop stricte ; or **11 264/11 264 débuts éligibles et 11 129/11 129 fins éligibles** reçoivent un label dans S810 : **0 rejet de l'événement éligible dû à cette condition**. Nemotron lit par erreur les indices de colonnes `FEATURE_COLUMNS` comme des « pitches », alors que ce sont les 23 variables de caractérisation acoustique par hauteur. Les comparaisons directes SHADOW 1024 ↔ S704 MIDI256 sont rejetées.

## Comparaison à S708

S708 MIDI non modifié. Les scores officialisés sont inchangés ; aucun verdict de « baisse » sur les autres axes du moteur courant n'est autorisé à partir des F1 SHADOW. Toute modification de production attendrait six mesures finales sur un même flux MIDI et une preuve de latence non régressive.

## Audits

| Audit | Statut | Preuve |
| --- | --- | --- |
| Données | PASS | Vérification locale SHA 80 traces et 15 checkpoints, zéro split réservé ouvert |
| Acoustique | UNKNOWN | Étude fondée sur les features S800 signées, pas d'extraction WAV brut indépendante |
| Architecture | PASS | Seuls axes SHADOW contrôlés ; flux MIDI S708 inchangé, pas de fusion |
| Apprentissage | PASS | 10 réentraînements contrôlés sur 5 folds, `balanced` vs `None` |
| Métriques | PASS | Même 80 FIT, parité des TP/FP/FN, confusion Full Note complète, distinction S704 |
| LIVE | UNKNOWN | Zéro modification LIVE, pas de mesure de latence iRig ou CPU |

## Verdict

**S811 TERMINÉ comme audit contradictoire, sans promotion.** Seulement **deux réponses NVIDIA reçues, dont une conforme aux contrôles textuels** ; les modèles sont des avis, non la source de la preuve. L'ablation sur les vraies traces prouve que la pondération `balanced` aggrave les FP Onset/Offset, mais il reste des dizaines de milliers de faux positifs même sans elle. La condition de fenêtre temporelle **ne jette aucun événement éligible**, donc les hypothèses centrées sur une fenêtre « trop stricte » sont réfutées pour S810. La majorité des faux Onset Full Note sont du silence réel (81 890/94 655). S708 officielle et LIVE préservés, aucune nouvelle performance MIDI256 démontrée. Ces audits locaux ne constituent pas une confirmation 9/9, ni six audits scientifiques NVIDIA complets.
## Prochaine action

Traiter la surestimation de la probabilité d'attaque par calibration **hors composition**, sans fixer un seuil depuis le test réservé ni supposer que `class_weight=None` suffit. Ajouter des fenêtres de décision et un compteur strictement causaux si justifiés, sans tampon ni hold ; diagnostiquer les FP silencieux et harmoniques sur audio original. Retester les six métriques depuis le **même** flux MIDI256 à la fin de chaque expérimentation et mesurer CPU/iRig avant toute promotion. Ne pas déployer S811 ni modifier la référence S708.
