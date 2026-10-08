# V27.3 — Série 21 : apprendre la fiabilité de B avant de transférer une sélection vers A

Protocole fixé avant lancement de la série 21. Aucune promotion ; les événements du corpus de développement ont déjà été observés pendant les séries 9–20.

## Question testée
Un message de B peut améliorer A sur K1 mais régresser sur K3/K4 (audit série20 : +28/−26 changements appariés face au réseau sans B, dont vrais K3 +2/−8 et K4 +0/−2). Un arbiteur apprendra **sur les morceaux d'entraînement** la probabilité qu'une *proposition issue d'un passage de A* soit correcte. Il verra les scores de compatibilité de B à toutes les classes, la trajectoire de A, le signal, l'ancien K et le K proposé, pour décider de **laisser le parent S18** ou d'**accepter la nouvelle sélection**.

## Sources et sécurité de séparation
- 59 309 événements natifs, folds 0,1,2,4, 19 morceaux. Source S20 : run 37858383973, logits probabilistes A par passage, B par passage (et control B neutralisé). Parent S18 : même décision que la série 20, 49 178 corrects global, 2 998 corrects poly ; 2 934 corrections / 2 210 régressions vs freeze.
- Les 335 variables acoustiques et votes proviennent des quatre archives SHA256 d'origine et des producteurs G/P excluant chaque fold. Les messages B issus de S20 sont hors-morceau (les réseaux S20 ont été entraînés sans le morceau évalué).
- Pour **chaque morceau évalué**, entraîner l'arbitre uniquement sur les **autres morceaux du même fold**. L'étiquette de fit `candidate_K==true_K` n'est utilisée que pour les lignes d'entraînement où la proposition diffère du parent ; si le candidat est faux et le parent également, l'apprentissage cible une abstention. Aucun K vrai, identité de morceau, ou score YourMT3+ comme entrée ou règle.
- **Limite majeure** : le parent S18 et le choix de cette expérience ont été élaborés sur le même corpus ; scores exploratoires, pas de nouvelle validation indépendante.

## Candidates et ablations
- Actions concurrentes fixées : sortie brute A au passage1, passage2, passage4, et passage4 avec message B neutralisé. **Ne pas entraîner de nouvelles prédictions sur les labels des lignes évaluées.**
- 2 vues de l'arbitre : `with_B` = 32 votes + 58 audios + 35 probabilités des cinq sorties A + 28 scores B + one-hot 7 ancien K / 7 K proposé + one-hot 4 source + comparaisons de marge et entropie ; `no_B` = même vue, les 28 B remplacés par zéro (ablation d'information B, avec le reste inchangé).
- Deux estimateurs : régression logistique C=0.1 et HistGradientBoosting (max_iter=100, max_leaf_nodes=15, min_samples_leaf=40, max_depth=4, l2=10, lr=0.05), standardisation et imputation fit-only. Score = P(proposition correcte) calculé hors-morceau.
- Six seuils d'acceptation fixes **0.20, 0.30, 0.40, 0.50, 0.60, 0.70**. Si la proposition diffère et que score > seuil, sélectionner le K proposé ; sinon conserver S18. Ne pas utiliser les labels test pour décider. Cela donne 2 vues × 2 modèles × 4 sources × 6 seuils = **96 politiques** + 2 parents.

## Métriques imposées
Exact-K K0–K6 et poly, fold, correction et régression vs S18 et freeze, effet de B vs ablation, corrects sur les seules décisions changées, toutes probabilités, partitions exactes et poids. Distinguer un gain net d'une **conservation sans aucune régression** : seules les politiques avec 0 ancienne correction détruite et une correction positive sont candidates à une validation complémentaire. **Aucune promotion automatique et aucune sélection de seuil présentée comme indépendante.**
