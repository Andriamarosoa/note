# Point de reprise — Exact-K

## Dernière étape terminée

Audit des gabarits et test des pentes harmoniques, le 5 octobre 2026.
Base de cette étape : commit `0bc2c59226a3d87a44e085febd1e4d36b885d4fe`.
Résultats : [README](README-harmonic-template-audit.md),
[protocole](v273-harmonic-template-protocol.md),
[preuves](evidence/v273-harmonic-templates/summary.json).
Le commit contenant ce fichier est le checkpoint de reprise de cette étape.

- 488 diagnostics internes : les gabarits à décroissance `1/h²` retrouvent
  85/272 triplets K3 avec gaussienne, contre 0 avec `1/sqrt(h)`, lorsque les
  fréquences annotées sont fournies. Le contrôle synthétique retrouve 488/488.
- Sans fréquences annotées : neuf variantes sur 1 666 lignes uniques, 845 cas
  VAL d'action. Nets fixes : −15, +1, −17, −17, −5, −27, −14, −10, −25.
- Sélection exclusivement sur FIT : aucune variante ne passe ; abstention
  dans les quatre folds, net 0. Aucun modèle promu.
- Les huit candidats contiennent les trois notes attendues dans seulement
  3, 10 ou 8 cas sur 272 K3 selon la pente de saillance. Le meilleur triplet
  sans annotation n'est complet que dans 5 cas au maximum.
- 20 tests locaux réussis. Rejeu des 36 modèles : écart de probabilité 0.

## Prochaine étape autorisée

Auditer les rangs de saillance et les éliminations par NMS des fondamentales
attendues, sur les mêmes cas internes. Séparer : mauvais classement avant NMS,
concurrence de fréquences proches, élimination après NMS, et insuffisance de
capacité du pool. Les annotations ne servent qu'à ce diagnostic.
Ne pas relancer la grille des neuf pentes déjà rejetées sans information nouvelle.

Vérifier d'abord le HEAD et les derniers runs de la branche
`codex/v273-failure-clustering`. Le contrôle CI léger initial `37362400113`
était encore en attente au début de cette étape ; vérifier son état actuel et
celui du commit portant ce checkpoint. Les calculs scientifiques ci-dessus ont
été exécutés localement et ne dépendent pas de cette attente CI.

## Contraintes persistantes

Folds 0, 1, 2, 4 uniquement ; fold 3 exclu. Base et populations figées. Pas de
nouvel entraînement neuronal complet sans défaut précis et intérêt établi.
Pas de promotion à partir d'un gain supposé ou d'une sélection sur VAL. Les
résultats du chemin normal ne démontrent pas ceux du chemin compressé.
Réutiliser les artefacts vérifiés, éviter les runs concurrents et préserver les
changements d'autres travaux. L'automatisation reste utile pour cette prochaine
question ; aucun blocage définitif d'accès ou de données n'a été constaté.
