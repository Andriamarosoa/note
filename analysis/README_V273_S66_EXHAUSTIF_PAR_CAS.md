# S66 — audit exhaustif de toutes les combinaisons par fragment (fold 1)

Correction de S65 : ne plus sélectionner une seule combinaison globale, mais **étudier toutes les possibilités de chaque `global_index`**, sans échantillonnage.

## Portée et résultats

- **13 868 événements** du fold 1.
- **2^7=128** sous-ensembles de 7 têtes de veto S58, évalués pour **chaque événement**, soit **1 775 104** décisions.
- **36 politiques gelées PR16** par événement, soit **499 248** décisions.
- **9 politiques de référence supplémentaires**, soit au total **173 sorties archivées par événement et 2 399 164 décisions comparées**.
- **2 573 erreurs S58** : vraie classe présente parmi les sorties disponibles dans **1 487 cas**, absente dans **1 086 cas**.
- **118 régressions** : seulement **7** activent au moins un veto existant (**111** ne peuvent être corrigées par les 128 sous-ensembles actuels), alors que la bonne classe est proposée par au moins une politique PR16 dans **110 cas**.
- **217 corrections** : **8** potentiellement détruites par un des veto actuels.
- **28 signatures** distinctes (K initial, K S58, masque des têtes) dont **11 contradictoires**, couvrant **109 régressions** et **202 corrections** sur les mêmes signatures. Ce n'est pas un problème résoluble par une simple table de veto.
- Répartition des **1 086** erreurs sans proposition juste : K0 120, K1 526, K2 220, K3 101, K4 63, K5 41, K6 15.

## Contrôle de l'ordre

Les sept têtes sélectionnées sont des **refus purs** et l'opérateur est OR : l'ordre est mathématiquement sans effet sur le résultat. Les **13 700** séquences ordonnées sans répétition sur 0–7 têtes se ramènent aux mêmes 128 sous-ensembles. Les **36 politiques ne sont pas des têtes indépendantes** et leurs repassages neuronaux successifs n'ont pas été simulés avec ces seules sorties gelées.

## Exemple par cas

- Index 982 : vérité K4, initial K4, S58 K3, aucun veto ; les 36 politiques votent K4.
- Index 993 : vérité K4, initial K4, S58 K3, aucun veto ; 22 politiques K4 et 14 K3.
- Index 986 : vérité K1, initial et S58 K0 ; zéro politique pour K1, impossible à récupérer par sélection seule.

## Données reproductibles

Analyse locale S66 complète enregistrée pour chaque fragment, avec les matrices de 128+36+9 décisions, les sept K envisageables, votes, ambiguïtés par signature et résultats par transition. Scripts `s66_exhaustive_per_case.py`, `s66_case_detail.py`, rapport `README_S66_AUDIT_EXHAUSTIF.md`, archive `S66_EXHAUSTIF_PAR_CAS.zip`.

Ce sont des **données de développement déjà exposées** ; présence du vrai K dans un candidat = plafond oracle, **pas une prédiction réalisable**. Aucun réseau promu ou modifié. Ne pas choisir la meilleure combinaison pour un fragment en lisant sa vérité terrain : la décision doit être apprise sur d'autres folds et éprouvée sur données inédites.