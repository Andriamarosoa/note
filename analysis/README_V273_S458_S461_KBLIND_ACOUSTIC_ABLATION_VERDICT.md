# S458–S461 — L'ablation du K révèle des faux positifs acoustiques

**14 octobre 2026.** Résultat mesuré, non promu. Le projet demeure `son → E3 / [E3,G3,B3]`, avec original S80 Exact-K K0–K6 intact. Aucune modification Live/Offset.

## Pourquoi l'ablation ?

[S457](https://github.com/Andriamarosoa/note/actions/runs/38535320952) prouvait que le modèle expérimental S454, alimenté par le K original S80, ne proposait **aucune correction parmi 120 validations internes** (4 modèles × 5 classes K prédites × 6 seuils). Hypothèse : le réseau copie peut-être la référence K plutôt que d'apprendre une autre preuve acoustique.

S458 teste **seulement** la suppression du K de la matrice de caractéristiques. Le réseau reçoit **1090 caractéristiques acoustiques** : résumé58, flux245, morphologie490, physique294, géométrie3. Le K original n'est utilisé **qu'après** le réseau pour définir les règles d'abstention. Apprentissage d'arbres ExtraTrees en quatre folds **par compositions GuitarSet disjointes**, avec quatre validations internes pour la sélection de seuils par K prédit. Le réseau ne voit pas les labels IDMT, YourMT+ ou un K oracle à l'inférence.

## Résultats effectivement vérifiés

| Expérience | Population | S80 original | K candidat | Corrections | Régressions |
|---|---:|---:|---:|---:|---:|
| [S458](https://github.com/Andriamarosoa/note/actions/runs/38536654031), développement GuitarSet global | 59 309 | 53 529 | 53 529 | 0 | 0 |
| S458, vrais événements poly GuitarSet | 7 385 | 6 054 | 6 054 | 0 | 0 |
| [S459](https://github.com/Andriamarosoa/note/actions/runs/38536698802), sortie IDMT native gelée sans vérité | 506 | tous K conservés | **0 K modifiés** | — | — |
| [S460](https://github.com/Andriamarosoa/note/actions/runs/38536736584), IDMT vrais K évaluables | 45 | 25 | 25 | 0 | 0 |
| S460, vrais accords poly IDMT | 21 | 7 | 7 | 0 | 0 |

Sur GuitarSet, **aucun des quatre folds S458 n'a ouvert une tête de correction**. Sur IDMT, l'accord exigé entre au moins trois modèles distincts n'a entraîné **aucun K modifié**. Tous les originaux S80 restent invariants.

[S461](https://github.com/Andriamarosoa/note/actions/runs/38537170051) a audité **160 seuils entraînement uniquement** (4 folds × 5 K prédits × 8 seuils de 0,15 à 0,85). Sur ces 160 choix :

- **160/160** ont un gain net insuffisant pour le critère de sécurité.
- **93/160** comptent moins de deux corrections par régression.
- **123/160** ont moins de cinq corrections.
- **25/160** régressent sur les vrais K2 de validation, **18/160** sur les vrais K3, **12/160** sur les vrais K4.

Un seuil trop strict n'est donc **pas** l'unique problème. Sans le K source, le réseau produit des alternatives acoustiques à faibles seuils, mais **elles apportent trop de fausses corrections**. L'hypothèse « seul le K injecté causait le blocage » est infirmée dans ce protocole.

Les sommes de validations internes agrègent des partitions provenant de plusieurs folds et ne constituent pas un échantillon externe indépendant. Les 45 événements IDMT appartiennent à un groupe annoté historiquement et ne doivent pas servir à calibrer le prochain seuil.

## Conséquence architecturale

Un nouveau classificateur multinomial global K0–K6 n'a pas montré de bénéfice sûr, **avec ou sans K original en entrée**. La prochaine piste à tester sur GuitarSet seulement est une architecture **résiduelle à deux questions** :

1. *Le comptage K actuel est-il contestable ?* Détecteur d'anomalie d'événement utilisant des preuves indépendantes de fondamentales, attaques, harmoniques, persistance et morphologie.
2. *Si oui, combien de notes corriger, et dans quel sens ?* Experts de transition K2↔K3, K3↔K4 et erreurs K0, avec abstention par défaut ; poids/choix des seuils déterminés **uniquement sur compositions tenues à l'écart**.

L'objectif n'est pas de forcer les corrections : les caractéristiques des régressions doivent être apprises, les scénarios impossibles doivent rester abstinents, et tout candidat doit subir une **validation prospective sur de nouveaux WAV non annotés jusque-là**, puis une comparaison vraie correction/régression par K, séparée des 59 309 scores historiques.

**Verdict : S454 et S458 sont rejetés comme remplacements de S80 ; garder les modèles et leurs audits comme contrôles négatifs. Aucune promotion.**
