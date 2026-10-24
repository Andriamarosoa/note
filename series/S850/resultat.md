---
serie: S850
statut: termine
type: experience
reference: S708
branche: codex/note-s850-causal-multinote-rank256
run_id: 40097584744
commit_sha: 992e7fa5bcb91383482ee882cec18991280d4187
promotion: false
---

# S850 — Résultat final : deux vrais entraînements appris en PCM256, BCE vs RANK

## Exécution
[GitHub Actions S850 #40097584744](https://github.com/Andriamarosoa/note/actions/runs/40097584744) **SUCCESS**. Les deux archives originales micro GuitarSet (SHA256 237cdc...) et JAMS (SHA256 8daa...) ont été vérifiées, et chaque note source est une **vraie capture enregistrée**, non un sinusoïde synthétique ou un pitch-shift DSP. Une entrée de la tête contient **256 nouveaux échantillons**, et le réseau apprend une GRU sur 32 décisions causales passées, sans future audio ni attente supplémentaire.

Deux réseaux entraînés **indépendamment depuis une initialisation fraîche** :
- BCE : CNN brut PCM256 + GRU + tête note source + tête transposée ±12 + activité/silence, apprentissage supervisé par vérité terrain réelle.
- RANK : même architecture et mêmes données, ajout d'une contrainte d'ordonnancement des vraies notes au-dessus des mauvaises notes concurrentes (hard-negative ranking coefficient 0,75).

Les deux ont été entraînés trois époques, checkpoint le plus faible en erreur **CAL**. Le vainqueur a été choisi **uniquement sur CAL, pas OUTER**.

## Vraies données et partitions signées
| Partition par guitariste | Blocs PCM256 | E2 actifs | E3 actifs | E2+E3 actifs |
|---|---:|---:|---:|---:|
| TRAIN 00,01,02,03 (64 pistes) | **98 304** | 9 053 | 11 001 | 3 818 |
| CAL 04 (16 pistes) | **24 576** | 2 613 | 2 643 | 1 478 |
| OUTER 05 (16 pistes) | **24 576** | 1 670 | 3 006 | 988 |

32 pas consécutifs par séquence au TRAIN ; une décision à chaque 256 samples au Live. Aucune piste ni guitariste partagé entre partitions. L'apprentissage utilise la vraie vérité terrain des hauteurs MIDI pendant TRAIN seulement.

## Métriques de notes activées, OUTER 05 réellement exclu des décisions de modèle
| Cible musicale décalée, même OUTER | BCE TP/FP/FN | **F1 BCE** | RANK TP/FP/FN | F1 RANK |
|---|---:|---:|---:|---:|
| Source E2 vers cible E3 +12 | 1463 / 7782 / 207 | **26,8071 %** | 1255 / 16716 / 415 | 12,7794 % |
| Source E3 vers cible E4 +12 | 1993 / 4977 / 1013 | **39,9559 %** | 2950 / 20925 / 56 | 21,9486 % |
| Source E3 vers cible E2 −12 | 2385 / 6006 / 621 | **41,8531 %** | 2890 / 20340 / 116 | 22,0308 % |

**La perte RANK testée n'améliore pas le compromis TP/FP/FN ; elle tend à prédire beaucoup plus de positifs.** Elle augmente seulement la probabilité que la note classée numéro un fasse partie de l'ensemble GT sur quelques catégories. Ce n'est pas une amélioration musicale globale.

| Mesure de hiérarchie +12 | BCE | RANK |
|---|---:|---:|
| CAL choix moyen sur les 5 cibles (F1) | **31,8878 %** | 19,5417 % |
| OUTER top1 contenue dans une vraie note active +12 | 8,4935 % | **12,9408 %** |
| OUTER nombre de blocs avec vraie note cible | 20 192 | 20 192 |

La métrique top1 n'est **pas** une F1 exacte Onset ; la cible GT peut être polyphonique. Le score classe simple est très faible et les FP restent massifs. **BCE sélectionné uniquement par CAL**, RANK non promu.

## Référence standard conservée
Sur la guitare enregistrée S848, le pitch shift standard +12 offline garde **90,384 %** de corrélation des enveloppes de dynamique avec l'original ; l'ancien modèle S847 ne donne que **11,643 %** contre standard et choisit MIDI73 sur 603/603 blocs. S849 a prouvé que ce blocage persiste avec resets de l'ancienne GRU. Le score S850 n'est pas comparable en points au F1 S847 (nouveau split et nouveaux fragments). La [suite S851](https://github.com/Andriamarosoa/note/tree/codex/note-s851-identical-guitar-standard-ab256) doit comparer **les vrais poids BCE/RANK** au même benchmark externe et au même standard avant de conclure sur la qualité du transposeur.

## Comparaison à S708
S708/H0 historique (1024 échantillons d'entraînement) n'entre dans aucun de ces deux modèles natifs à 256. F1 de présence par bloc ≠ F1 Onset événement ±80ms. **Aucun nouveau F1 Onset de 80 WAV, aucune amélioration Onset de référence prouvée.**

## Audits
| Domaine | État | Détail |
|---|---|---|
| Données | PASS | les deux archives originales SHA, 64/16/16 pistes par joueurs 00–03/04/05 |
| Acoustique | PASS logiciel | CNN raw256, GRU seulement passée, transposition de code apprise, aucun futur ou hold |
| Architecture | PASS logiciel | deux entraînements distincts BCE vs RANK, sortie multi-label et gate de silence |
| Apprentissage | PASS | sélection checkpoint et variante sur CAL uniquement, 3 époques |
| Métriques | FAIL pour promotion | vrais FP milliers, top1 de vraie note encore extrêmement faible |
| LIVE | UNKNOWN | 5,805ms cadence input, p99 machine Windows/iRig non mesuré |

## Verdict
**S850 TERMINÉ, NON PROMU.** BCE remporte cette ablation sur CAL et OUTER pour les cibles testées, mais reste trop imprécis pour une transposition crédible. En particulier la détection d'une note correcte est bien différente de la capacité à transposer une **performance audio complète**. La référence de travail Onset PCM256 **69,2584331 % F1** (7607 TP, 3093 FP, 3660 FN), les six têtes et MIDI officiel sont inchangés.

## Prochaine action
Auditer la même guitare S848 vs standard sur BCE et RANK et deux politiques d'état sans sélectionner sur benchmark externe, avant une éventuelle nouvelle architecture. Ne pas masquer la performance par post-traitement dynamique ou multiplicité de notes non prouvées.
