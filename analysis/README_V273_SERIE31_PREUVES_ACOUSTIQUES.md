# Série31 — contre-épreuve acoustique des sélections du scheduler S29

Protocole fixé avant tout calcul, après que la série30 a démontré que la simple concordance A-first/B-first ne discrimine **pas** les quatre régressions. **Erratum S29 : 26 décisions changées = 17 corrections, 4 régressions et 5 changements neutres** ; les anciennes synthèses qui citent « 21 changements » avaient compté seulement les corrections/régressions. On conserve tous les 26 événements.

## Objectif
Vérifier si une **source de preuve distincte du vote des têtes** détecte les mauvaises promotions. Les 335 caractéristiques source se composent de 32 votes, 58 résumés audio et 245 variables harmoniques/flux archivées ; le signal est identique à S19–S30, mais le modèle de corroboration S30 ne consultait que les 90 premières variables et les probabilités A/B.

## Trois vues acoustiques et deux estimateurs
- `flow245` : les 245 variables de flux/harmonique **seules**, complétées des 14 one-hot parent/candidat et de deux marges globales A/B pour indiquer quelle décision est évaluée.
- `audio_plus_flow303` : les 58 variables audio et 245 flux, même information de classe candidate.
- `all335` : les 335 variables, même information de classe candidate.

Estimateurs appris séparément : `LogisticRegression(C=0.1, class_weight='balanced', solver='liblinear')` ou `HistGradientBoosting(max_iter=80,max_depth=3,min_samples_leaf=20,l2=10)`, fit et standardisation uniquement sur autres morceaux. Labels fit : « le K proposé est-il juste ? » provenant **uniquement des morceaux d'entraînement**, parmi toutes les propositions distinctes des neuf politiques S29, dédoublonnées (événement, K proposé). Pour le morceau évalué, aucun label ne peut entrer dans le modèle, la mise à l'échelle, l'extraction de caractéristiques ou la sélection.

Quatre seuils de confiance figés `0.25,0.40,0.55,0.70`. Nombre total : 3 vues × 2 modèles × 4 seuils = **24 politiques**, plus S18 et référence S29. Sans corroboration suffisante : ne pas modifier S18. Tous les 26 cas S29 consignés, même les cinq neutres.

## Validation, coûts et limites
Global Exact-K, poly, chaque vrai K0–K6, fold, correction/régression et neutralité vs S18, nombres des 17 corrections maintenues et des quatre régressions évitées. Archivage complet des scores, modèles et identités des morceaux exclus. Compare S30 et S29. **Un résultat sur ce corpus déjà exposé ne constitue pas une preuve indépendante**, car S18 et les producteurs antérieurs ont été sélectionnés sur ce même développement ; ne rien promouvoir sans validation sur musique inédite. La vue « flux » est un assemblage de variables acoustiques **préexistantes**, pas une mesure démontrée d'une loi physique de Navier–Stokes.
