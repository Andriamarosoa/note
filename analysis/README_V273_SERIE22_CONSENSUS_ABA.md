# V27.3 — Série 22 : confirmation mutuelle entre passages et arbitres de fiabilité

Étude exploratoire sur la cohorte GuitarSet déjà exposée (59 309 événements) ; la série 21 et tous ses scores ont été consultés avant ce protocole. Aucune validation indépendante, aucun modèle promu.

## Hypothèse testée
S19/S20 prouvent qu'un message B modifie effectivement A au passage suivant, mais il provoque aussi des régressions polyphoniques. Série 21 : 96 portes apprises de fiabilité ne préservent pas tous les anciens cas corrects. Cette série teste une **confirmation mutuelle à la fois entre passages récurrents et entre modèles de fiabilité** avant d'implémenter le changement de classe.

## Décodage sans nouveau fit
Utiliser les fichiers exacts de série20 (probabilités des cinq sorties A et messages B) et de série21 (16 probabilités de correction apprises avec exclusion du morceau évalué). Référence protégée = série18 `series18__flow_logistic__k21__pbase_gt0.95`. Ne jamais utiliser la vraie classe K ni le score YourMT3+ dans une sélection.

Règles d'accord entre propositions :
- `A1_ABA2` : passage 1 et passage 2 proposent le même K différent du parent ;
- `ABA2_ABA4` : passages 2 et 4 d'accord ;
- `ABA4_noB4` : quatrième passage avec et sans B d'accord ;
- `A1_ABA2_ABA4` : les trois passages récurrents d'accord ;
- `ABA2_ABA4_Bsupports` : passages 2 et 4 d'accord, et la compatibilité B finale pour K proposé dépasse celle de K parent ;
- `all4` : tous les quatre candidats de A (premier, deuxième, quatrième et contrôle sans B) d'accord.

Trois manières préannoncées de décider si l'action est fiable : `logistic_AND_hgb` (minimum des estimations avec B, aux deux passages), `logistic_AND_noB_hgb` (minimum des estimations B logistic et sans-B hgb), `average_models` (moyenne des estimations logistic et HGB aux passages accordés). Les probabilités proviennent **uniquement des réseaux déjà entraînés hors morceau**.

Seuils fixes 0.30, 0.40, 0.50, 0.60, 0.70, 0.80. Six règles × trois modes × six seuils = **108 sorties**, plus série18 et freeze = 110. Ne produire que des classes réellement proposées par les passages de A. Tous les scores et masques sont une fonction déterministe de l'audio et des messages, sans vérité.

Mesurer exact global/poly et K0–K6, folds, corrections/régressions/neutres contre S18 et freeze, jeux de cas modifiés, précision des nouvelles sélections et cohérence par rapport aux 2 934 anciennes corrections. Ne pas confondre la sélection sur ce corpus avec une validation externe. Comparer aussi les 96 variantes S21, sans en faire un nouveau jeu d'entraînement.