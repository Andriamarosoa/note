# S641 — Transposer +12 puis remapper −12 : pourquoi ce n'est pas une preuve et comment utiliser la fondamentale native

## Verdict reproductible (GuitarSet réel)

- Branch: `codex/s641-roundtrip-fundamental-consistency`
- CI succès : https://github.com/Andriamarosoa/note/actions/runs/38996134511
- Artefact original : `S641_REAL_GUITARSET_FUNDAMENTAL_CONSISTENCY/report.json`
- Entraînement sur **36 pistes GuitarSet TRAIN, 9 compositions** distinctes de DEV et TEST.
- Contrôle S640 à **18 pistes**, identiques à la sélection historique.
- Validation sur **12 pistes GuitarSet DEV annotées** (groupes BN3/Funk3/Rock1).
- Aucun TEST ouvert, aucune nouvelle donnée future, même S614 natif
  et prédicteur K, MIDI LIVE non modifié.
- Note : DEV est *historiquement exposé*. Score positif = résultat
  exploratoire vérifié, **pas généralisation indépendante, pas promotion**.

## Limite physique du raisonnement +12 puis −12

Si f est une fondamentale, le déplacement d'octave donne 2f puis f.
La réattribution MIDI −12 n'ajoute aucune information acoustique :
un faux candidat à 2f pourrait lui aussi être remappé à f.

La vue x2 est le signal transformé, et la sélection recherche dans
le **spectre acoustique ORIGINAL** :
- énergie de la fondamentale f à 3 résolutions ;
- contraste avec ses deuxième et quatrième harmoniques ;
- contraste avec les bins aux demi-tons voisins et l'octave supérieure ;
- nouveauté physique de la fondamentale, comparée à celle de l'octave ;
- posterior K original et scores des attaques concurrentes.

La bande f n'est pas une preuve absolue : les instruments, les
résonances et la prise micro peuvent produire des composantes mixtes.

## Training-only group CV

Les seuils sont choisis par GroupKFold 5 plis, chacun excluant
une ou plusieurs **compositions entières** des exemples TRAIN.
Une proposition correcte vaut +1, une fausse −0,5 ; la politique
peut choisir l'abstention complète (seuil 1,01).
Pas d'optimisation du seuil via DEV.

| Famille | Exemples TRAIN | Vrais candidats utiles | Seuil GroupKFold TRAIN-only | TP/FP des propositions OOF |
|---|---:|---:|---:|---|
| S640 41 caractéristiques, 18 pistes | 388 | 19 | 1,01 (abstention) | 0/0 |
| S640 41 caractéristiques, 36 pistes | 761 | 50 | **0,90** | **17/20** |
| S641 65 caractéristiques spectrales, 36 pistes | 761 | 50 | 0,97 | 11/11 |

Les 24 nouvelles caractéristiques spectrales ne surpassent pas ici le
meilleur classifieur 41D, mais elles renforcent la prudence : moins de
faux positifs contre un gain de rappel plus faible.

## Résultat DEV réel sur les 12 morceaux GuitarSet

| Modèle | TP | FP | FN | F1 (%) | Rappel attaques avec overlap (TP/1147) | K3+ TP/910 |
|---|---:|---:|---:|---:|---:|---:|
| S614 original | 998 | 585 | 582 | **63,1046** | 697 | 550 |
| Bass ×2 brut | 1044 | 834 | 536 | 60,3817 | 740 | 586 |
| S640 TRAIN18, seuil CV abstention | 998 | 585 | 582 | 63,1046 | 697 | 550 |
| **S641 TRAIN36 41D, seuil CV 0,90** | **1009** | **594** | **571** | **63,3993** | **707** | **557** |
| S641 TRAIN36 65D physique, seuil CV 0,97 | 1003 | 586 | 577 | 63,3007 | 702 | 552 |
| S641 TRAIN36 41D, seuil exploratoire 0,85 | 1010 | 602 | 570 | 63,2832 | 708 | 558 |
| S641 TRAIN36 65D, seuil exploratoire 0,85 | 1010 | 604 | 570 | 63,2436 | 707 | 557 |
| S641 TRAIN36 41D, seuil exploratoire 0,95 | 1005 | 587 | 575 | 63,3670 | 704 | 554 |

**Meilleure variante OOF TRAIN : 41D sur 36 pistes**.
- +11 TP vrais, +9 FP, F1 +**0,294661 point** versus S614.
- Exactement 11 annotations nouvelles récupérées et **0 ancienne
  annotation perdue** dans cette configuration.
- 23 occasions de rescue acceptées ; elles ne sont pas nécessairement
  toutes des événements NoteOn inédits.
- +10 NoteOn corrects réellement superposés, dont **+7 K3+** et +3 K2.
- Les gains par piste (TP, FP supplémentaires) incluent :
  `Rock1-130-A_comp`: (+5, +0), `Funk3-112-C#_comp`: (+3, +2),
  `BN3-119-G_comp`: (+1, +0),
  `BN3-154-E_comp`: (+1, +5),
  `Rock1-90-C#_solo`: (+1, +0).
- Sur les autres enregistrements le TP reste inchangé ; le FP
  augmente sur 2 autres pistes (Funk3-98-A-comp, Rock1-90-C#-comp).

## Conclusion et limites

La transposition ×2 aide réellement à révéler des graves masquées,
mais redescendre de −12 MIDI n'est pas une validation physique.
La nouveauté utile vient d'une sélection apprenant à *s'abstenir*,
et de davantage de vrais accords disponibles sur TRAIN.

Ce résultat dépasse S614 sur le DEV **déjà souvent évalué** ;
un modèle final ne peut pas être promu sur ce seul résultat.
Une évaluation scellée sur compositions TEST GuitarSet et instrument
indépendant (URMP/IDMT), en figeant modèle, seuil et émissions avant
d'ouvrir les labels, demeure indispensable. Le runtime 3 vues
est hors production et pas encore benchmarké en microphone live.

Les courbes et sources détaillées sont dans l'artefact du run.
