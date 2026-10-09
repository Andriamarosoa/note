# Note / série 17 — arbitres K3→K2 et K3→K4

## Protocole fixé avant apprentissage (cohorte déjà explorée)

Point de départ : série16 `series16__flow_logistic__k23__pbase_gt0.99` confirmée par run indépendant 37856011482. Baseline appariée `freeze_local_combo` : 2934 corrections conservées / 2215 régressions restantes, 49 173 prédictions correctes globales / 2993 correctes polyphoniques.

Les transitions qui ont motivé les essais viennent de l'audit antérieur de la série10 : 188 vraies notes K3 sous-comptées en K2 et 138 vraies K3 surcomptées en K4. On teste donc **uniquement** lorsque `freeze_K=3, serie16_K=2` ou `freeze_K=3, serie16_K=4`. Retour possible seulement à la classe freeze, jamais nouvelle classe.

Deux têtes binaires indépendantes dont la cible est `truth_K==freeze_K`, chacune entraînée sur **les autres morceaux du même fold** et uniquement sur les événements de sa transition. Ne pas fournir la vérité du morceau tenu de côté, les prédictions YourMT3+ ou l'identité du morceau/fold aux entrées. Utiliser les producteurs G/P hors fold et les features audio vérifiées : votes temporels + résumés 58 (summary), et summary + 245 flux harmoniques (flow). Les labels ont été regardés dans les décisions de séries antérieures ; l'évaluation S17 est **exploratoire**, non indépendante.

Chaque tête : logistic C=.1 ou HGB profondeur4/100 itérations, conservés comme séries 13–16. Scorers = summary_logistic, summary_hgb, flow_logistic, flow_hgb, flow_both (minimum des deux derniers), flow_mean (moyenne). Seuils fixés 0.75, 0.85, 0.90, 0.95, 0.975, 0.99 ; chaque scorer×seuil s'applique seulement à K3→K2, seulement à K3→K4, ou aux deux (6×6×3 = **108** vecteurs, plus parent et freeze).

Auditer exhaustivement corrections, nouvelles régressions, neutralités, global/poly, K0–K6, folds 0/1/2/4 et toutes les décisions d'origine. Enregistrer tous les poids/probabilités/hachages. Promotion interdite tant que pas de nouvelle cohorte. Pour conserver un candidat expérimental sans correction sacrifiée sur cette cohorte : exige >0 régressions corrigées, **0** correction perdue, et aucune baisse polyphonique.
