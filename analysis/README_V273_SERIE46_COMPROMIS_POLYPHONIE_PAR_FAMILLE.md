# S46 — compromis polyphoniques à risques de régression asymétriques

**Déclaration avant exécution.** La S44 a audité 320 filtres par famille, la S45 a appris 304 modèles de risque (16 familles ×19 morceaux) et produit 204 stratégies. Les meilleurs scores globaux dégradent toujours K2–K6. Il faut donc introduire une **fonction de risque différente pour les accords polyphoniques**, sans utiliser le vrai K au moment de la décision.

Conserver absolument les producteurs originaux S43, tous les estimateurs S45 gelés et S18 intacts. Ne pas réentraîner ni déduire de nouvelles étiquettes. L'entrée S46 est le tableau `risk_scores.npz` (Pfix, Pbreak et proposition de K par famille S45), déjà produit avec exclusion du morceau évalué par les métamodèles. Ces valeurs sont des scores de risque **non calibrés**. H9 ne doit être ni action ni entrée.

Choisir la meilleure proposition parmi quatre catalogues fixés :
- `all16` : tous les mécanismes historiques correcteurs ;
- `unique_poly` : S40 temporel, S19 récurrent, S38 morphologie ;
- `historic_acoustic` : S1/S2/S3/S4 et réparation cohérente ;
- `ordering` : S24/S25 B-first et S19 A–B–A.

**Décision observable uniquement** : si K initial S18 <=1, utilité estimée `Pfix−1×Pbreak`, seuil 0,02. Si K initial >=2, utilité estimée `Pfix−λpoly×Pbreak` avec λpoly ∈ {2,4,8}, seuil ∈ {0,0.05,0.10,0.20}. Exiger 0 ou ≥1 confirmation d'une autre famille (une seule voix par famille, déjà gelée), et limiter ou non les transformations poly à ΔK=1 tout en empêchant, dans la variante stricte, les transferts K2–K6 vers K0/K1. Ce sont **4×3×4×2×2 = 192 politiques**, toutes définies avant observation des sorties S46. STOP/KEEP S18 si aucune proposition admissible ; un seul candidat retenu au score max.

Auditer chaque politique par corrections, régressions, nets global/poly, par vraie K0–K6, source K, folds et famille sélectionnée. Garder les cas rejetés et les modèles S45 intacts, et comparer obligatoirement S18 82.9183% global/40.5958% poly, S36b 2/0, S44, S45. Ne pas inventer un zéro régression ni promouvoir une sélection sur les compositions déjà utilisées. En particulier l'origine acoustique historique n'a pas toute sa provenance OOF prouvée ; l'absence de fuite ne peut être affirmée pour ces modèles.