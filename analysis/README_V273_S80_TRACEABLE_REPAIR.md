# S76–S80 — Audit correctif des sélections S73

Date: 11 octobre 2026. Statut: développement sur folds historiques 0,1,2,4, **non promu**. 59 309 fragments; 7 385 vrais K≥2.

| Variante | Correct global | Correct poly | Delta global S72 | Delta poly S72 |
|---|---:|---:|---:|---:|
| S72 | 49 238 (83,02 %) | 3 202 (43,36 %) | — | — |
| S76 LightGBM multiclasses, seuil inner-OOF | 49 236 | 3 200 | −2 | −2 |
| S77 routeur K privilégiant le global | 49 465 | 2 869 | +227 | −333 |
| S78 routeur conjoint poly/global | 49 198 | 3 246 | −40 | +44 |
| **S79 filtre acoustique K1→poly** | **49 244** | **3 232** | **+6** | **+30** |
| **S80 S79 + routes K0/K2** | **49 281 (83,09 %)** | **3 240 (43,87 %)** | **+43** | **+38** |

S79: utiliser la politique n°22 des 36 têtes PR16 comme candidate uniquement quand S72 donne K1 et que la candidate donne K2–K6. Filtre StandardScaler + LogisticRegression sur acoustique (58 résumés+46 géométries), probabilités PR16 12×7, votes PR16 7 et S46 7, avec 3 espaces de variables et trois régularisations. Chaque fold cible est exclu du fit et de la sélection de seuil, qui exploite uniquement le inner-OOF des trois autres folds. 67 actions; 30 corrections, 24 régressions, 13 erreurs→erreurs.

S80 ajoute, sans chevauchement, 46 décisions depuis K0 vers une classe ≥2 privilégiée par la moyenne des 12 têtes PR16, et 61 depuis K2 par vote majoritaire S46 lorsqu'une marge de confiance du mélange PR16/S46 atteint 0,40. Le choix de ces deux routes est rétrospectif après les audits des folds historiques; il ne prouve pas une généralisation.

**S80 par vrai K :** K0 37 931 (−3); K1 8 110 (+8); K2 1 764 (+34); K3 1 047 (+4); K4 389 (0); K5 39 (0); K6 1 (0).
**S80 bilan complet :** 174 changements, 86 corrections, 43 régressions, 45 erreurs→erreurs. Gains globaux par fold 0/1/2/4 +4/+22/+7/+10, gains poly +12/+9/+9/+8.

Rejeu: scripts/audit_v273_s80_traceable_repair.py et .github/workflows/v273-s80-traceable-repair.yml. Les artefacts S72 (run 38155427761), PR16 (run 37837235723), S46 (run 37891336823) et features PR16 (run 37837235723) sont vérifiés et alignés. Archivage de chaque prédiction native, métriques K0–K6, changements, scores, classes et 4 modèles de filtrage K1.

**Limites:** PR16 amont pas intégralement nested; producteur n°22 et règles de S80 sélectionnés avec connaissance de résultats des folds historiques; aucune nouvelle composition véritablement indépendante. Ne pas promouvoir ni changer freeze_local_combo. S80 est un candidat expérimental reproductible, pas une preuve de dépassement des 50 % polyphoniques.