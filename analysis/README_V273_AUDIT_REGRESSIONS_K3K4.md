# V27.3 — Audit des régressions K3/K4 du sélecteur cohérent

**Audit terminé le 8 octobre 2026.** Les 594 régressions K3/K4 sont indexées,
les quatre modèles gelés sont rejoués et 28 interventions diagnostiques ont
été vérifiées. Aucun nouvel entraînement, seuil déployé ou changement de référence.

Le constat le plus solide est une **surestimation du gain de correction**,
accompagnée d'une sensibilité beaucoup plus forte au contexte acoustique
direct qu'aux votes et aux taux d'audit variables. L'audit ne démontre pas
une cause acoustique unique des erreurs ni l'inutilité des spécialistes.

[Protocole des interventions](README_V273_PROTOCOLE_AUDIT_K3K4.md) ·
[Résultat du modèle audité](README_V273_RESULTATS_CORRECTION_SELECTEUR.md) ·
[Run de diagnostic réussi](https://github.com/Andriamarosoa/note/actions/runs/37769970667).

## 1. Où vont les régressions ?

| Vérité et classe initiale | Nouvelle prédiction | Régressions |
|---|---|---:|
| K3 | K2 | 222 |
| K3 | K4 | 177 |
| K3 | K5 | 6 |
| **Total K3** | | **405** |
| K4 | K2 | 33 |
| K4 | K3 | 125 |
| K4 | K5 | 31 |
| **Total K4** | | **189** |

Parmi les 405 cas K3, 200 étaient déjà dégradés par l'ancien sélecteur complet
et 205 sont nouvellement dégradés. Pour K4, les nombres sont 88 et 101.
Les erreurs concernent respectivement 84 et 54 enregistrements.

La comparaison suivante fixe la **même classe initiale et la même décision**.
Pour K3→K2, une correction est un vrai K2 et une régression est un vrai K3.
Cela évite de comparer des groupes définis par des décisions différentes.

| Décision | Changements | Corrections | Régressions | Neutres Exact-K | Net | Gain annoncé par les probabilités |
|---|---:|---:|---:|---:|---:|---:|
| K2→K3 | 558 | 172 | 198 | 188 | −26 | +128,51 |
| K2→K4 | 9 | 0 | 5 | 4 | −5 | +0,87 |
| K2→K5 | 1 | 0 | 0 | 1 | 0 | +0,39 |
| K3→K2 | 825 | 271 | 222 | 332 | +49 | +196,25 |
| K3→K4 | 520 | 163 | 177 | 180 | −14 | +109,00 |
| K3→K5 | 24 | 4 | 6 | 14 | −2 | +5,01 |
| K4→K2 | 183 | 57 | 33 | 93 | +24 | +48,61 |
| K4→K3 | 405 | 136 | 125 | 144 | +11 | +78,71 |
| K4→K5 | 83 | 24 | 31 | 28 | −7 | +14,04 |
| K4→K6 | 3 | 1 | 0 | 2 | +1 | +0,80 |
| **Total** | **2 611** | **828** | **797** | **986** | **+31** | **+582,19** |

Le net d'une transition n'est pas le net d'une classe de vérité. Par exemple,
K3→K2 est globalement positif, mais ses 222 régressions comptent dans les
pertes de vrai K3 ; les 271 corrections bénéficient aux vrais K2.

## 2. Une décision cohérente ne garantit pas des probabilités fiables

Le décodage applique correctement `P(destination) − P(classe initiale)`.
Sur les décisions effectivement modifiées, le réseau annonce environ
**1 200,73 corrections et 618,54 régressions**, soit **+582,19**. Les nombres
observés sont **828 et 797**, soit **+31**. Il surestime donc la justesse des
destinations et sous-estime le risque de dégrader une bonne prédiction.

| Fold | Gain annoncé | Net observé |
|---|---:|---:|
| 0 | +192,69 | −16 |
| 1 | +149,08 | +28 |
| 2 | +108,97 | +11 |
| 4 | +131,45 | +8 |

L'écart apparaît sur les quatre folds. C'est un constat descriptif sur les
données exposées, sans test de significativité supposant les événements
indépendants. La formule du décodeur est maintenant cohérente ; la fiabilité
empirique des probabilités qui l'alimentent reste un blocage.

Les tranches de marge suivantes ont été fixées avant le rejeu diagnostique :

| Marge annoncée en faveur de la correction | Événements | Corrections | Régressions | Net observé |
|---|---:|---:|---:|---:|
| [0 ; 0,05[ | 404 | 107 | 141 | −34 |
| [0,05 ; 0,10[ | 364 | 112 | 116 | −4 |
| [0,10 ; 0,20[ | 603 | 181 | 203 | −22 |
| [0,20 ; 0,40[ | 822 | 273 | 225 | +48 |
| [0,40 ; 1] | 418 | 155 | 112 | +43 |

Les faibles marges sont défavorables ici, mais **242 des 594 régressions
K3/K4 ont encore une marge d'au moins 0,20**, dont 76 d'au moins 0,40.
Une simple règle de confiance ne sépare donc pas toutes les erreurs. Aucun
seuil n'est retenu à partir de ce tableau ; il faudrait l'apprendre et
l'évaluer dans des partitions distinctes.

## 3. Quelles entrées influencent les décisions du modèle gelé ?

Chaque permutation échange un bloc entre événements du même fold et de
la même classe initiale, sans utiliser les labels. Trois donneurs
déterministes sont utilisés. Les distributions marginales sont conservées,
mais les relations entre blocs peuvent être rompues : il s'agit de
**sensibilité à poids fixes**, pas d'une ablation réentraînée ni d'une cause
physique du son.

Le point de comparaison est le sélecteur cohérent à +31 face à la référence.
Les variations du tableau sont mesurées **face à ce sélecteur**, sur tous
les événements natifs ; les changements de décision portent sur les 7 493
événements admissibles.

| Bloc échangé | Décisions différentes, selon le donneur | Variation du nombre d'exacts face au modèle intact |
|---|---:|---:|
| Contexte acoustique complet | 3 205–3 253 | −336 à −314 |
| Contexte spectral direct | 2 281–2 346 | −163 à −151 |
| Naissance | 2 167–2 218 | −171 à −128 |
| Amortissement | 2 097–2 125 | −149 à −107 |
| Persistance | 2 163–2 234 | −205 à −140 |
| Descripteurs directs des votes | 177–186 | +12 à +21 |
| Taux d'audit variables | 34–44 | +3 à +6 |
| Métadonnées de repli | 235–259 | +2 à +8 |
| Identité des combinaisons, contrôle négatif | 0 | 0 |

Le contexte direct est donc fortement utilisé, y compris le spectre
réintroduit. Les votes changent seulement environ 2,4 % des décisions
admissibles sous cette intervention ; les taux d'audit, environ 0,5 %.
Le réseau étudié se comporte surtout comme un classifieur utilisant le
contexte direct. Cela motive un contrôle réentraîné « contexte seul » pour
mesurer l'apport propre des combinaisons ; cela ne suffit pas à conclure
que les votes ou les audits sont inutiles.

La permutation des taux laisse inchangés les taux globaux constants à
classe initiale/fold fixés ; son faible effet renseigne principalement sur
la part contextuelle variable. Il ne mesure pas toute l'influence des
audits globaux. Les gains des perturbations ne sont pas présentés comme des
améliorations déployables : leurs donneurs proviennent du lot de diagnostic
et les relations entre entrées sont artificiellement modifiées.

### Volume des références d'audit

Les audits d'entraînement utilisent deux folds de référence, ceux du test
trois. Le contrôle prédéfini multiplie leurs comptes par 2/3 avant `log1p`,
en conservant taux, poids et décodeur.

Il change 207 décisions et apporte **+6 exacts** face au sélecteur intact,
mais ne restaure que **8 des 594 régressions K3/K4** et perd 17 corrections
originales. Les nets par vrai K changent de +18/+3/−12/−3/0 pour K2–K6.
Le test aggrave donc K4. Dans les deux agrégats de compte sélectionnés,
aucun des 2 611 changements originaux ne dépasse le maximum d'entraînement
de sa classe initiale/destination/fold. Ce contrôle limité ne soutient pas
le volume des références comme explication suffisante des régressions.

## 4. Corrections et régressions ont-elles une signature distincte ?

Les 58 observables sont comparées séparément dans les mêmes transitions.
Les meilleures AUC ci-dessous sont **descriptives, orientées après lecture
et maximisées parmi 58 caractéristiques**. Elles ne mesurent pas la
performance indépendante d'un filtre appris.

| Transition | Corrections / régressions comparées | Meilleure AUC univariée descriptive |
|---|---:|---:|
| K3→K2 | 271 / 222 | 0,573 |
| K3→K4 | 163 / 177 | 0,574 |
| K4→K3 | 136 / 125 | 0,638 |
| K4→K2 | 57 / 33 | 0,682 |
| K4→K5 | 24 / 31 | 0,755 |

Les deux grandes transitions K3 ont des distributions fortement
chevauchantes pour chaque observable prise seule. Des tendances existent :
K3→K2 présente moins de masse de naissance intermédiaire et une variabilité
d'amortissement plus forte dans les régressions, avec le même signe dans
les quatre folds mais de faibles différences standardisées (environ 0,22).
Pour K4→K3, la masse perdue est plus faible dans les régressions dans les
quatre folds (différence standardisée −0,32). Le petit groupe K4→K5 a un
contraste de naissance plus fort, mais seuls deux folds ont assez de cas
dans les deux catégories pour comparer ce signe.

L'attention ne fournit pas non plus une séparation nette : pour K3→K2,
l'entropie normalisée moyenne vaut 0,9325 dans les corrections et 0,9358
dans les régressions ; pour K3→K4, 0,9492 et 0,9498. Les poids restent
répartis sur de nombreuses combinaisons. C'est une description de
l'attention, pas une preuve que cette diffusion cause les erreurs.

Le désaccord entre experts et arbitre n'est pas un veto sûr : parmi les
177 régressions K3→K4, **94 n'ont aucun des cinq spécialistes qui privilégie
K4 à K3**, mais c'est également le cas de **68 des 163 corrections réussies**.
Pour K3→K2, une majorité de spécialistes soutient la décision dans 143/222
régressions et 183/271 corrections. Supprimer aveuglément les décisions
sans accord des spécialistes ferait donc perdre des corrections.

## 5. Les 594 événements sont traçables

[Index des 594 régressions](evidence/v273-k34-coherent-audit/594-regressions-index.csv) :
ID natif, fold, enregistrement, repère temporel, vérité, classe initiale,
nouvelle et ancienne décisions, probabilités, marge, soutien des experts
et attention. Le temps est `start_sample / 44100`, le repère de la fenêtre
native. L'archive complète ajoute les 58 observables et les descripteurs
sélectionnés pour chaque cas. Cet audit est numérique ; il ne prétend pas
à une écoute manuelle de chaque extrait.

Exemples pris à la marge médiane de chaque groupe de régressions :

| ID | Enregistrement | Repère (s) | Vérité | Décision | P(initial) | P(destination) | Experts favorables / 5 |
|---|---|---:|---:|---|---:|---:|---:|
| 53603 | `03_Jazz3-150-C_comp.jams` | 24,605 | 3 | 3→2 | 0,3033 | 0,4841 | 2 |
| 29661 | `01_SS2-107-Ab_comp.jams` | 0,272 | 3 | 3→4 | 0,3661 | 0,5225 | 0 |
| 17837 | `01_BN3-154-E_comp.jams` | 6,836 | 4 | 4→3 | 0,3244 | 0,4610 | 5 |

Ces exemples illustrent plusieurs configurations de désaccord ; ils
n'établissent pas à eux seuls la cause acoustique de l'erreur.

## 6. Vérifications et décision

Run `37769970667`, commit `4329829cf9953363b95f6e13dab68c42eaa2367c` :
**13 tests passent**, puis les quatre modèles retrouvent exactement toutes
les classes prédites. L'écart maximal de probabilité après chargement des
poids et inférence par lots est 7,16e-7, sous la tolérance annoncée de 2e-6.
La provenance reconstruite correspond aux manifestes archivés.

Le [vérificateur séparé](../scripts/verify_v273_k34_audit.py) a recalculé les
bilans natifs et par fold des 28 interventions depuis leurs probabilités,
contrôlé les IDs des 594 cas et les soutiens des spécialistes.
[Preuve machine](evidence/v273-k34-coherent-audit/verification.json) ·
[Tous les contrastes de caractéristiques](evidence/v273-k34-coherent-audit/feature-contrasts.csv).

[Artifact complet 11546734202](https://github.com/Andriamarosoa/note/actions/runs/37769970667/artifacts/11546734202),
SHA-256 `5979dc1b465f467a8b463035fbfb1c1c41d91fb9362549e73bd58c69f7eb3043`.
Il contient les 7 493 lignes de décision, les 594 cas détaillés, les
probabilités des interventions et le rapport exhaustif.

**`freeze_local_combo` reste la référence ; le candidat cohérent reste à
+31 événements exacts, sans promotion.** Les scores des perturbations ne
remplacent pas ce score mesuré après entraînement.

La priorité suivante est une comparaison réentraînée, à protocole fixé,
du sélecteur complet avec un contrôle utilisant seulement le contexte,
pour mesurer l'apport propre des combinaisons et audits. Le choix
correction/KEEP doit ensuite être évalué avec une calibration apprise sur
des prédictions internes réellement hors entraînement, en surveillant
séparément K3/K4. L'objectif est de rendre fiable le gain attendu utilisé
pour décider, avant d'ajouter de nouveaux spécialistes.

Ces conclusions ne ferment pas la recherche sur les signaux acoustiques :
la faible séparation univariée ne prouve pas qu'une combinaison de signaux
ou une autre représentation ne pourrait pas réussir. Toute nouvelle
variante reste à confirmer sur des données inédites. Les folds 0/1/2/4
sont exposés ; aucun player05/fold3 ni nouvel audio n'a été évalué ici.
