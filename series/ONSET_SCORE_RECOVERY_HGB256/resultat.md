# Résultat — nouvelle récupération HGB

État : entraînement terminé ; nouveau replay des 80 WAV en cours.

Le calcul sur les événements archivés donne **TP 8050 / FP 4231 / FN 3217**,
**F1 68,370987 %**. Le gain potentiel contre le run 39915038301 est de
**0,308293 point**, avec **+38 TP et −33 FP**. Il n'est pas encore présenté
comme un résultat audio vérifié.

Les 120 modèles internes et 5 modèles finaux ont été entraînés. Tous les plis
ont choisi la politique `cautious` à partir des seules compositions internes.

| OUTER | Modèle retenu | Seuil |
|---|---|---:|
| BN1 | HGB 15 feuilles | 0,45 |
| Funk2 | HGB 15 feuilles | 0,375 |
| Jazz1 | HGB 31 feuilles | 0,40 |
| Rock3 | HGB 31 feuilles | 0,40 |
| SS1 | HGB 15 feuilles | 0,375 |

La sélection sur les événements reproduit exactement les anciennes sorties
de récupération pour les 80 pistes. Deux tests supplémentaires vérifient
les frontières de réattaque, les seuils float32 et l'appariement un-à-un.

La séparation de la nouvelle tête ne supprime pas les recouvrements historiques
des enseignants. Ce résultat concerne FIT80, sans généralisation externe prouvée.
