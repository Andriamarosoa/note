# S86–S87 — Rejeu des modèles S80 et des 193 politiques S46 sans réentraînement

**Date : 11 octobre 2026. Run GitHub [38177477884](https://github.com/Andriamarosoa/note/actions/runs/38177477884) : quatre jobs réussis.**

## Résultats réellement vérifiés

1. **S86 — Sortie finale S80 gelée.** Script [`scripts/verify_v273_s80_frozen_inference.py`](../scripts/verify_v273_s80_frozen_inference.py) ; les quatre fichiers `K1_gate_fold{0,1,2,4}.joblib` sont chargés et appliqués **sans réentraîner**, avec les 58 résumés acoustiques, 46 formes, 12×7 probabilités PR16, 36 votes PR16 et 193 votes S46. Toutes les identités natives sont alignées. **59 309/59 309 prédictions S80 reproduites à l'identique** ; décisions S79, routes K0/K2 et scores K1 enregistrés égalent les originaux, **erreur maximale K1 : 0**. Résultats historiques inchangés : **49 281 corrects globaux / 3 240 corrects poly**. Seuils par fold : **0,50 / 0,70 / 0,65 / 0,65** ; 67 interventions K1 au total. Artefact du run : `v273-s86-verified-frozen-s80-tail`.
2. **S87 — Sélecteur S46 original.** Le code source officiel `scripts/audit_v273_s46_classwise_compromise.py` a été restauré depuis la branche PR16. Source fixée [release S45](https://github.com/Andriamarosoa/note/releases/tag/v273-series45-research-37891074478) (16 familles, 304 modèles de risque pièce-exclue). **193/193 politiques × 59 309 fragments** recalculées et **identiques à leurs 193 colonnes d'archive**. Aucun apprentissage de nouveaux paramètres pendant ce test. Artefact : `v273-s87-s46-original-frozen-selector-replay`.
3. **S83–S85 (jobs du même run)** : nouveaux WAV GOAT réellement passés dans le réseau V8.4 gelé, groupes V8.6/V9.0, score de la référence `freeze_local_combo`, puis PR16 (12 réseaux et 36 politiques) ; événements individuels sans consulter les annotations. Ces jobs réussis **ne produisent pas les 16 vecteurs S45 sur GOAT**.

## Contrat de déploiement : causes précises de non-promotion

S46 requiert **quatre entrées calculées sur le même événement nouveau** : `original_parent_K`, `proposal_K` (N×16), `risk_fix_break` (N×16×2), `independent_votes` (N×16). Ces données sont présentes uniquement pour les **anciens** fragments dans `S45/predictions.npz` et `S45/risk_scores.npz`. **Les 304 prédicteurs Ridge sont formés par 19 morceaux exclus au sein de leurs folds ; aucune affectation de modèle à un nouveau morceau GOAT n'a été gelée ni prouvée.** De plus, chaque famille source S43/S44 doit elle-même produire une proposition sur un nouveau WAV ; ce n'est pas un simple vote des 36 politiques PR16. Remplacer les 16 colonnes par PR16 ou assigner au hasard un `piece` **changerait le modèle**.

Même après déploiement S45/S46, les K parents **S69→S70→S71→S72** nécessitent leur apprentissage et leur application avec exclusion de la musique évaluée. Les quatre classificateurs K1 S80 sauvegardés ne disposent pas non plus d'une politique fixée pour les **nouveaux** morceaux hors folds {0,1,2,4} : choisir un des quatre, une moyenne, ou un cinquième entraînement implique une nouvelle **version de déploiement**, à figer avant le test.

## Verdict scientifique et opérationnel

- **Faits positifs :** dernier étage S80 et règles S46 vérifiés individuellement jusqu'à la dernière décision pour **tous les 59 309 fragments historiques** ; réseau original et référence exécutés sur nouveau WAV.
- **Faits non démontrés :** aucune prédiction **S80 complète** sur les nouveaux WAV ; **aucun score Exact-K S80 externe** ; aucune supériorité indépendante sur `freeze_local_combo`.
- **Promotion officielle : BLOQUÉE**, sans modifier `freeze_local_combo`.
- **Solution de développement honnête :** définir explicitement et geler une variante déployable (par exemple `S80-deploy-v1`), qui entraîne un modèle de risque S45 unique ou un ensemble selon **des règles fixées sans les labels GOAT**, implémente les 16 générateurs de famille et le parent S69–S72, puis vérifie l'équivalence de l'extraction et de la chaîne sur les morceaux historiques. Tant que cette nouvelle version diffère de la S80 expérimentale, ses métriques historiques **ne sont pas transférables** automatiquement. La mesurer sur un corpus neuf avec une comparaison appariée et des intervalles par morceau.

Voir aussi [pré-enregistrement S81](README_V273_S81_INDEPENDENT_VALIDATION.md) et [rapport S83–S85](README_V273_S83_S85_EXTERNAL_NATIVE_PIPELINE.md).
