# S68 — Repassage conditionne au K et apprentissage (non promu)

Audit du 9 octobre 2026. Sources developpement : folds 0/1/2/4, soit 59 309 evenements. Cible : 111 regressions S58 sans veto sur fold 1.

IMPORTANT : les 12 producteurs neuronaux PR16 ne recoivent pas le K courant ; leurs sorties q(x) ne changent pas en repassage. Les 36 decodeurs conditionnes par K (alpha*q+(1-alpha)*one_hot(K)) ont ete re-calcules apres chaque changement de K. Leurs 36 sorties initiales ont ete reproduites exactement sur toute la cohorte. Une nouvelle tete MLP 123->32->16->1 recoit effectivement le K courant et est reevaluee apres chaque action.

Sur les 111 : ordre original 36 cas corriges, ordre inverse 15 ; 50 cas ont des sorties differentes. 103/111 changent au moins une proposition apres changement de K. 103 cas retrouvent le vrai K dans les 36 sorties mixtes, mais seulement 60 dans les 12 argmax acoustiques bruts : l'ancrage au K initial cree une partie du signal apparent.

Sur fold1 entier, base S58 = 217 corrections / 118 regressions / +99 net. Repassage sans apprentissage: +(-66) net pour les 36 dans l'ordre original, -62 inverse; 24 melanges prudents avec 2 passes +47.

Controles appris avec fit folds 0,2,4 seulement et choix des seuils uniquement en OOF interne de ces trois folds :
- LogisticRegression penality 1.5: 409 corrections / 224 regressions / +185 net sur fold1 ; 2/111 restaures.
- LogisticRegression penality 2: 274 / 140 / +134 ; 0/111.
- MLP recurrente avec cout 1: 589 / 419 / +170 ; 17/111.
- MLP avec cout 1.5: 283 / 154 / +129 ; 0/111.

Un cas (global index 20232, vrai K0) alterne K1 et K2 en fin de tour dans le MLP permissif. Une memoire anti-cycles est requise.

LIMITES: Les producteurs amont ne sont pas imbriques pour un meta-apprentissage avec holdout fold1 (certains q d'entrainement peuvent provenir de reseaux ayant vu fold1); tous les folds ont historiquement servi au developpement. Aucune independance globale ni promotion automatique. Le producteur principal est INCHANGE. Sept tests de relecture numerique ont reussi.

Preuves et poids numeriques en NPZ : dossier local S68_REPASSAGE_APPRENTISSAGE.zip et rapport S68_REPASSAGE_APPRENTISSAGE.md ; dependencies: artefact PR16 v273-open-summary et audios historiques s60_head_design.npz.