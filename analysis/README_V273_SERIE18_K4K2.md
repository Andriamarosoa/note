# Note/V27.3 — série18 : protéger K4→K3 et K2→K1

Protocole déclaré avant lancement, sur la même cohorte déjà examinée : ne constitue **pas** une validation indépendante.

Point de départ série17 `series17__flow_logistic__k32__pbase_gt0.99`, rejouée dans run 37856362983. Mesures natives : 59 309 événements / 49 174 globalement corrects / 2 994 poly corrects / 2 934 corrections vs freeze / 2 214 régressions vs freeze. Aucun de ces scores ne dépasse encore la référence YourMT3+.

Les deux directions de régression natives de l'audit série10 sont `freeze_K=4→decision_K=3` (130 regressions) et `freeze_K=2→decision_K=1` (115 regressions). Elles sont différentes des tâches des séries16/17. Ne protéger que ces deux couples et laisser toutes les autres décisions de série17 **bit-à-bit inchangées**.

Pour chaque morceau évalué, ajuster deux modèles binaires de correction indépendants avec seulement d'autres morceaux du **même fold**, étiquetés `truth_K==freeze_K`, sans vérité du morceau testé. Les producteurs G/P excluent le fold évalué. Les entrées sont uniquement leurs votes, la temporalité géométrique et les caractéristiques acoustiques d'origine : résumés 58, ou ceux-ci + 245 variables de flux harmonique. Aucune prédiction YourMT3+, aucune identité du morceau ni fold comme variable. La politique de départ est néanmoins post-hoc et les événements déjà exposés ; conserver cette limitation dans chaque rapport.

Six têtes (summary_logistic, summary_hgb, flow_logistic, flow_hgb, flow_both et flow_mean), six seuils immuables **0.75/0.85/0.90/0.95/0.975/0.99**, trois périmètres (K4→3, K2→1, union). Total **108 candidats** + parents serie17 et freeze, tous archivés y compris les régressifs ou sans changement. Retour à freeze uniquement lorsqu'un arbitre hors morceau estime `p(freeze correct)>seuil`. Maintenir le parent si peu de données d'apprentissage.

Comparaison complète par true K0–K6, folds 0/1/2/4, Exact global/poly et corrections/régressions strictement appariées. Une correction est admissible à validation ultérieure uniquement si elle évite au moins 1 régression et ne perd **aucune** des 2 934 corrections déjà acquises. Aucune promotion automatique.