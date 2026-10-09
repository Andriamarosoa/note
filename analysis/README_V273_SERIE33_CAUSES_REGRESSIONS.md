# Série33 — rechercher des régularités acoustiques dans les quatre régressions S29

Protocole fixé avant exécution. Il s'agit d'un **audit descriptif**, pas de nouvelle politique produisant une bonne note par connaissance de sa vérité. Après S30–S32, confirmer par plusieurs estimations ne suffit pas à empêcher les 4 régressions de S29 tout en conservant 17 corrections.

Données : les 59 309 mêmes événements S29, vrais K **uniquement pour l'audit**, 335 caractéristiques (32 votes, 58 audio, 245 flux/harmoniques), routes réellement exécutées par l'ordonnanceur ; scores d'incertitude S31. Les 26 décisions changées sont 17 corrections, 4 régressions, 5 neutres.

Étapes d'audit :
1. Compter les 4 régressions par vrai K0–K6, ancien K, nouveau K, transition K2↔K3/K4 et fold, morceau, sans sélection post-hoc ;
2. Enregistrer le parcours A/B réellement suivi à chaque étape avant la décision ;
3. Calculer les niveaux relatifs et normes des trois groupes de caractéristiques pour les groupes corrigé/régressé/neutre, avec standardisation descriptive sur la cohorte (pas utilisable comme entrée de futurs modèles sans fit-only) ;
4. Pour chacune des quatre régressions, trouver dans les 17 corrections la voisine la plus proche selon audio58, flux245, et toutes 335 caractéristiques, en laissant les identifiants et distances explicites ;
5. Comparer les scores de la meilleure porte S31 `all335_logistic` pour les corrections/régressions/neutres, sans choisir de seuil à partir de la vérité.
6. Émettre un verdict prudent : des ensembles de quatre événements et de multiples caractéristiques **ne peuvent pas établir une cause musicale statistiquement générale**. Exiger de nouveaux événements acquis indépendamment et des ablations de caractéristiques avant de modifier les poids des têtes.

Enregistrer tous les cas et les métriques détaillées, y compris les changements neutres ; ne modifier ni S18 ni l'ordonnanceur retenu.