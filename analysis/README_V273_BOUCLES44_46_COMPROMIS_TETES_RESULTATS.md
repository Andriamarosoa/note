# S44–S46 — compromis historiques par tête : audit, apprentissage et polyphonies

**Date : 9 octobre 2026. Tous les résultats ci-dessous ont été réellement produits et leurs sorties originales indépendamment rejouées.** Branche `codex/v273-open-k0-k6`. Reference S18: 49 178/59 309 = **82,9183 % global**, 2 998/7 385 = **40,5958 % poly**. Aucune entrée YourMT3+/H9 dans les sélecteurs, S18 inchangée, H8 sans faux pitch-shift.

## S44 — premières boucles par familles, masques label-blind

- Exécution réussie [37890613101](https://github.com/Andriamarosoa/note/actions/runs/37890613101) ; vérification indépendante et [archive 37890938098](https://github.com/Andriamarosoa/note/releases/tag/v273-series44-research-37890938098).
- **16 familles**, 20 propositions de têtes distinctes brutes/poly, **320** compromis programmés et contrôlés pour chaque événement. **198** compromis non dominés à l'intérieur de leur famille (corrections conservées/régressions restantes), *sans prétendre à une amélioration globale du poly*.
- Plus grand **gain global net** de la meilleure règle S40 à transitions descendantes : 1 051 corrections / 660 régressions = **+391 net** ; mais **368 corrections poly pour 660 régressions poly**, soit −292 poly. S11 +306 net (1022/716), S38 +297 (972/675), S12 +279 (972/693), S19 +271 (958/687), S24 +243 (1649/1406), S25 +242 (1684/1442), S1 +124 (1325/1201).
- **Aucune** des 320 règles ne combine correction positive, zéro régression et maintien poly S18. Les accords de plusieurs familles sont corrélés; ce n'est pas une validation causale de la note.

## S45 — nouvelle tête de fiabilité apprise par famille

- Exécution réussie [37891039201](https://github.com/Andriamarosoa/note/actions/runs/37891039201), preuve indépendante et [archive 37891074478](https://github.com/Andriamarosoa/note/releases/tag/v273-series45-research-37891074478).
- **304** modèles de risque `Ridge` : **16 familles × 19 morceaux tenus à l'écart** ; 139 signaux audio/votes/flux + K ancien/proposé one-hot + taille/direction de transition + votes des autres familles, deux cibles apprises `fix` / `break` sur les **autres morceaux du même fold**. Enregistrement des modèles, scalers, identifiants fit/hold et propositions. **204** politiques : 192 individuelles + 12 arbitres entre familles. Rejeu numérique 304/304 et 204/204.
- Meilleur global `S24_Bfirst__lambda1__cut0.02` : **83,4393 % global**, mais seulement **35,3555 % poly**, 1 145 corrections / 836 régressions = **+309 net**, sans maintien du poly.
- S40 appris : 772 corrections / 591 régressions, mais corrections poly 321 / régressions poly 508. S38 appris : 696 / 505, poly 188 / 498. Le fit individuel réduit des pertes mais manque une preuve de source musicale indépendante.
- **Aucune** des 204 stratégies n'atteint le critère strict zéro régression avec gain poly.

## S46 — risques asymétriques pour K2–K6, avec sous-groupes spécialisés

- Exécution réussie [37891336823](https://github.com/Andriamarosoa/note/actions/runs/37891336823), **192** stratégies utilisant les poids S45 **gelés** avec pénalité poly λ={2,4,8}, seuil poly ={0,0.05,0.10,0.20}, corroboration par 0 ou 1 autre famille, structure libre ou voisin K±1 sans chute K≥2→K≤1. Quatre sous-catalogues : toutes les 16, temporal/récurrent/morphologie, acoustiques historiques, têtes d'ordre B-first.
- Vérification indépendante **192/192** et [archive 37891464117](https://github.com/Andriamarosoa/note/releases/tag/v273-series46-research-37891464117).
- **96 politiques avec gain net poly positif**, mais encore beaucoup de régressions individuelles. La **meilleure poly seule** obtient **41,8009 % poly**, contre 40,5958 % S18 (+89 bonnes prédictions poly), avec 296 corrections poly /207 régressions poly et **82,6839 % global** (contre 82,9183 % S18) : **non acceptable comme remplacement global**.
- **Deux seulement** améliorent simultanément le nombre correct poly **sans diminuer** le nombre correct global :
  1. **`historic_acoustic polyLambda2 cut0 votes1 neighbor_and_no_drop_to_K1`** : **41,4760 % poly**, contre 40,5958 % S18, **+65 vrais accords poly net** ; global **82,9183 % identique à S18**, avec **671 corrections et 671 régressions globales**. Remplace donc 671 anciennes bonnes réponses : ce n'est **pas** du zéro régression malgré le global stable.
  2. **`ordering polyLambda2 cut0 votes1 neighbor_and_no_drop_to_K1`** : **40,9208 % poly**, **+24 accords poly net**, **82,9233 % global**, avec **483 corrections et 480 régressions** (gain net global **+3**).
- Les deux compromis ne sont pas prouvés indépendants : ils reclassent des décisions sur les mêmes compositions déjà utilisées depuis les recherches historiques.

## Plan d'intégration prudente de **nouvelles têtes** (pas de promotion automatique)

1. Préserver les **16 têtes candidates S44/S45**, chacune avec ses poids/règles et chaque correction/régression par K, fold et morceau. Ne jamais remplacer le routeur S35 ou S18 silencieusement.
2. **T-comp-acoustic** : utiliser le signal de compétition S1/S2/S3/S4 et la confirmation inter-familles comme expert à spécialiser sur K2/K3/K4, en gardant la stratégie S46 historique comme point de comparaison expérimental.
3. **T-comp-ordering** : spécialiser S24/S25 avec S19 récurrent comme décision d'ordre conditionnelle et ses propres risques, surveillant notamment les transitions K2→K3/K4.
4. **T-comp-temporal** : préserver le CNN S40 et la morphologie S38 — leurs corrections exclusives sont précieuses malgré des régressions fortes. Vérifier l'existence de **sources indépendantes** (fondamentales, attaques et énergie) au niveau du candidat avant de décider d'un nouveau K.
5. Ne pas entraîner les nouvelles têtes sur les **mêmes morceaux déjà exploités comme validation** : les champions S43 ont été choisis ex post sur cette cohorte, et certaines anciennes sources n'ont pas de preuve complète de prédiction hors fold. Exiger un jeu indépendant acquis et verrouillé **avant** de choisir la politique à promouvoir. Les 192 choix S46 sont des résultats de recherche, pas un estimateur non biaisé.

## Fichiers reproductibles

- `scripts/audit_v273_s44_compromise_loops.py`, `scripts/verify_v273_s44_compromises.py` ;
- `scripts/train_v273_s45_specialized_risk_heads.py`, `scripts/verify_v273_s45_specialized_risk_heads.py` ;
- `scripts/audit_v273_s46_classwise_compromise.py`, `scripts/verify_v273_s46_poly_compromises.py`.

Chaque archive permanente contient les prédictions par événement, les poids ou règles, les audits corrections/régressions et les contrôles indépendants. **Aucun résultat négatif supprimé.**