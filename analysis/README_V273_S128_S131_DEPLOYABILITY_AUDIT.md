# S128–S131 — Promotion S80 : audit des producteurs S4/S5 et nouveaux étudiants

**Date locale : 11 octobre 2026 ; exécutions GitHub datées du 12 octobre en UTC.**

**Verdict : quatre nouveaux workflows réussis, mais S80 non promu.** La référence
officielle demeure `freeze_local_combo`. Aucun score Exact-K de S80 complet
sur GOAT n'a été mesuré. Les nouveaux étudiants S5 sont des **nouveaux modèles**
et non le modèle S5 historique.

## Résultats vérifiés

| Étape | Run | Résultat | Portée |
|---|---|---|---|
| S128 | [38188250730](https://github.com/Andriamarosoa/note/actions/runs/38188250730) | 1 882 désaccords historiques entre S5 gelé et reconstruction log527, dont 1 665 sous-comptages ; biais ordinal β=0,32 : 352 reconciliations et 152 pertes, **1 682 désaccords restants** | Fidélité aux sorties archivées, **pas l'accuracy** |
| S129 | [38188446402](https://github.com/Andriamarosoa/note/actions/runs/38188446402) | Un étudiant S5 111D a appris les prédictions S5 des folds 1/2/4 ; **14 557/15 952 = 91,255 %** de fidélité aux décisions gelées sur fold 0 ; 57 prédictions GOAT sans labels | Nouveau modèle, pas S5 original |
| S130 | [38188595287](https://github.com/Andriamarosoa/note/actions/runs/38188595287) | Pondérations de classes K rares : amélioration du rappel moyen K2–K4 avec régressions globales | Comparaison teacher-only sur le fold historique |
| S131 | [38188773741](https://github.com/Andriamarosoa/note/actions/runs/38188773741) | Ajout de flux 245D puis morphologie 490D : résultats inférieurs au modèle 111D | Aucun gain retenu |

### Résultats S129 par classe de *K prédit par le professeur S5*

| Classe | Décisions S5 original | Identiques S5-student111 | Fidélité |
|---|---:|---:|---:|
| K0 | 11 079 | 10 928 | 98,64 % |
| K1 | 2 937 | 2 485 | 84,61 % |
| K2 | 954 | 634 | 66,46 % |
| K3 | 632 | 343 | 54,27 % |
| K4 | 286 | 157 | 54,90 % |
| K5 | 61 | 10 | 16,39 % |
| K6 | 3 | 0 | 0 % |

### S130 : pondération K rare

| Paramètre | Rappel enseignant moyen K2/K3/K4 | Corrections vs S129 | Régressions vs S129 | Bilan net |
|---|---:|---:|---:|---:|
| S129 non pondéré | 58,54 % | — | — | — |
| α=0,25 | 61,08 % | 202 | 210 | −8 |
| α=0,50 | 63,12 % | 252 | 277 | −25 |
| α=0,75 | 65,32 % | 294 | 376 | −82 |

**Pas de promotion :** ces corrections sont des accords supplémentaires
avec les *prédictions du professeur historique*, pas des corrections de
vérité terrain. La pondération ne démontre aucun meilleur Exact-K réel.

### S131 : entrée temporelle/morphologie

| Étudiant | Fidélité au professeur fold0 | Gains vs S129 | Pertes vs S129 | Bilan |
|---|---:|---:|---:|---:|
| 111D initial | 14 557/15 952 | — | — | — |
| 111+245 flux | 14 520/15 952 | 178 | 215 | **−37** |
| 111+245+490 morphologie | 14 507/15 952 | 193 | 243 | **−50** |

Le modèle S131 n'a pas surmonté le déficit polyphonique. Ces paramètres
réduisent même la fidélité globale aux prédictions S5 sauvegardées.

## Qu'est-ce qui a *réellement* été débloqué ?

Les 57 événements GOAT sont passés au travers d'une **nouvelle voie déployable
S5-student111**, à partir des authentiques 58D acoustiques, 46D géométriques,
K de la référence gelée et sans entrée MIDI/annotation. Sa distribution
K0..K6 sur les 57 événements est **[46, 9, 2, 0, 0, 0, 0]**.
Le nouveau checkpoint final S129 (appris sur les 4 folds historiques)
est archivé avec SHA256 et des prédictions externes.

Cela démontre qu'un **modèle alternatif** peut recevoir un nouveau WAV.
Cela **ne démontre pas** que l'ancien S5 de 527D est reconstitué, que les
16 familles S45 originales sont déployables, ni que S80 a été évalué.
Il serait méthodologiquement incorrect d'injecter ce nouvel étudiant dans
S80 puis d'attribuer au montage les anciennes métriques historiques S80.

## Blocage et prochaine transition de promotion

1. Soit retrouver/reconstituer **exactement** les 416 colonnes originales
   S4-owned/S5 et prouver l'identité per-event de ses sorties historiques ;
   soit figer un **nouveau modèle `S80-deploy-v1`**, avec un nouveau protocole
   et de nouvelles métriques.
2. Faire exécuter les 16 familles S43/S44/S45 sur GOAT sans consulter la vérité,
   puis faire circuler les vrais vecteurs dans S46→S69→S72→S80.
3. Fixer à l'avance les règles de nouveaux morceaux (304 modèles de risque,
   folds S69 et quatre classificateurs K1 de S80).
4. Stocker les prédictions avant l'ouverture des annotations, puis mesurer
   une comparaison **appariée indépendante** contre `freeze_local_combo`.
   Les anciens 19 morceaux GuitarSet ne constituent pas une validation inédite.

**Statut officiel : BLOCKED, non promu.** Aucun run S128–S131 ne peut
être pris pour une validation indépendante S80.
