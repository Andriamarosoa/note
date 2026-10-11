# S88–S89 — Les sélecteurs S46 et S69 deviennent exécutables sans labels

**Date : 11 octobre 2026.** [CI S89 #38178147414](https://github.com/Andriamarosoa/note/actions/runs/38178147414) : job S46/S69 label-free **SUCCESS**. Attention : *test de reproductibilité sur 59 309 fragments historiques*, **PAS** une mesure Exact-K S80 sur nouveau corpus.

## S88 : S46 193 politiques en inférence pure

- Nouveau script : `scripts/s88_unlabeled_s46_runtime.py`. Contrat strict NPZ : `row_id, parent_K, family_ids, proposal_K, risks, independent_votes`. Refuse les champs de vérité terrain et toute référence H9/YourMT3 comme famille.
- Applique sans réapprentissage les règles originales `scripts/audit_v273_s46_classwise_compromise.py` et renvoie **193 politiques** : parent S18 + 192 combinaisons groupées.
- Historiquement, **193/193 colonnes de 59 309 fragments exactes**, en partant d'un NPZ sans les labels d'origine. Les `NaN` de risque des propositions qui ne changent pas de K sont préservés ; les risques des changements sont contrôlés finis.
- Artefact GitHub : `v273-s88-unlabeled-original-s46-runtime`.

## S89 : S69 deux veto historiques en inférence pure

- Nouveau script : `scripts/s89_label_blind_s69_runtime.py`. Contrat S46 sans labels `row_id,fold,variant_ids,predictions` ; votes PR16 `row_id,predictions` ; règles figées provenant du rapport S69, sélectionnées antérieurement sur d'autres folds.
- Applique, **sans aucune vérité terrain**, les seuils de veto enregistrés pour chaque fold 0/1/2/4 aux deux politiques S69 `balanced` et `polymax`.
- Historique : **59 309/59 309 égalité stricte**, pour la référence S18, les deux sources S46 et les deux décisions `balanced_protected_K`, `polymax_protected_K` ; contrôlé en CI contre l'artefact original S69.
- Artefact GitHub : `v273-s89-frozen-s69-label-free-rules`.

## Contraintes scientifiques résiduelles, non escamotées

Cette réussite **ne signifie pas** que les 16 sorties S45 existent sur les 57 groupes externes GOAT. Les 304 modèles Ridge S45 sont spécialisés par **19 pièces tenues à l'écart**. Il faut une vraie règle de production pour le nouveau morceau, puis reconstruire les 16 politiques sources S43/S44 qui ne se confondent pas avec PR16. Le script S89 emploie **une affectation de fold** pour choisir ses seuils ; aucun fold n'est défini pour un morceau extérieur, et arbitrer après avoir vu les labels créerait un biais. Les parents S70/S71/S72 ne disposent toujours pas d'un pipeline de production autonome, et l'étage S80 lui-même a quatre classifiers K1 avec des seuils 0,50/0,70/0,65/0,65.

**Verdict :** plusieurs modules maintenant *exactement rejouables à partir d'entrées sans labels*. Mais les modèles encore manquants empêchent une **prédiction S80 bout-en-bout** sur nouvel audio. Le score externe et la promotion officielle restent **NON MESURÉS / NON AUTORISÉS** ; `freeze_local_combo` reste inchangé. Une version déployable devra être spécifiée et figée **avant** de lire les annotations, et si son comportement diffère, elle doit être versionnée comme un **nouveau modèle**, pas présentée comme la S80 historique.
