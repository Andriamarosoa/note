# V27.3 — Résultats de l’apprentissage des 127 groupes

**Expérience terminée et vérifiée, 8 octobre 2026. Aucun remplacement de `freeze_local_combo`.**

Le réseau commun évalue les regroupements complets sans éliminer leurs membres individuellement. Les complémentarités demandées existent dans les sorties réelles : 16 fragments ont un groupe juste alors qu’aucun des sept candidats seuls ne donne le bon K. Le réseau en récupère 6 à 7 selon le bras.

Le meilleur des trois bras complets atteint **+57 corrections nettes**, mais son décodage limité aux candidats seuls atteint **+68** avec les mêmes poids. Les historiques globaux aident dans cette expérience ; ajouter les voisins similaires ne fait pas mieux au total. Le choix parmi les groupes et la calibration du gain restent insuffisants.

[PR #8](https://github.com/Andriamarosoa/note/pull/8) · [Run contrôlé terminé](https://github.com/Andriamarosoa/note/actions/runs/37777844182) · [Protocole fixé avant entraînement](README_V273_PROTOCOLE_GROUP127.md)

## Scores mesurés

Global : **59 309 événements natifs**. Poly : **7 385 vrais K≥2**. Seuls les **7 493 fragments prédits initialement K2/K3/K4** peuvent changer. « Net » signifie corrections moins régressions face à `freeze_local_combo`.

| Système | Exact-K global | Exact-K poly | Corrections | Régressions | Net |
|---|---:|---:|---:|---:|---:|
| freeze_local_combo | 81.6976 % | 34.2586 % | — | — | 0 |
| Coherent archivé (PR #7) | 81.7498 % | 34.6784 % | 828 | 797 | +31 |
| 127 groupes, sans historique | 81.7380 % | 34.5836 % | 686 | 662 | +24 |
| 127 groupes, historique global | 81.7937 % | 35.0305 % | 675 | 618 | +57 |
| 127 groupes, global + voisins similaires | 81.7886 % | 34.9898 % | 640 | 586 | +54 |

Les entrées acoustiques, votes, 127 propositions, audits avant masquage, architecture (12 994 paramètres), seed et budget d’entraînement sont identiques entre les trois nouveaux bras. Seules les entrées d’historique autorisées changent : aucun historique, global seul, global + local.

Le global gagne **+33 net** face au bras sans historique (110 corrections / 77 régressions). Le local gagne **+30** face au bras sans historique, mais perd **3** face au global (132 / 135). Le voisinage réduit certains dégâts en K4 mais en ajoute en K3. Ce résultat ne démontre pas l’inutilité générale de la similarité ; il ne valide pas son apport supplémentaire avec ces caractéristiques et ces réglages.

Face au modèle `coherent` archivé, les trois nets sont −7 / +26 / +23. Cette comparaison change plusieurs éléments de conception ; elle ne permet pas d’attribuer le gain aux seuls audits.

## Détail K0–K6 et folds

| Vrai K | Effectif | Référence | Sans historique | Global | Global + local |
|---|---:|---:|---:|---:|---:|
| K0 | 39652 | 95.9220 % | 95.9220 % (+0) | 95.9220 % (+0) | 95.9220 % (+0) |
| K1 | 12272 | 64.2846 % | 64.2846 % (+0) | 64.2846 % (+0) | 64.2846 % (+0) |
| K2 | 3445 | 34.3396 % | 37.8520 % (+121) | 37.8229 % (+120) | 37.7358 % (+117) |
| K3 | 2289 | 39.9738 % | 38.7505 % (-28) | 39.1874 % (-18) | 38.6195 % (-31) |
| K4 | 1207 | 34.5485 % | 27.4234 % (-86) | 29.2461 % (-64) | 30.6545 % (-47) |
| K5 | 355 | 4.2254 % | 9.0141 % (+17) | 9.5775 % (+19) | 8.4507 % (+15) |
| K6 | 89 | 0.0000 % | 0.0000 % (+0) | 0.0000 % (+0) | 0.0000 % (+0) |

Les nombres entre parenthèses sont les nets face à la référence. L’Exact-K des vrais K0/K1 reste identique ; certaines prédictions déjà fausses peuvent changer entre K2–K6. Les prédictions hors du périmètre initial K2/K3/K4 restent exactement inchangées.

| Fold | Sans historique | Global | Global + local |
|---|---:|---:|---:|
| 0 | +12 | +18 | +17 |
| 1 | +25 | +26 | +25 |
| 2 | -24 | -20 | -13 |
| 4 | +11 | +33 | +25 |

Le fold 2 reste négatif dans les trois bras. Les pertes K3/K4 persistent face à la référence ; le gain total est principalement porté par K2.

## La complémentarité est réelle, sa sélection reste imparfaite

Deux mesures doivent être distinguées :

- **147 fragments** ont au moins un groupe correct dont tous les membres inclus sont faux seuls. Parmi eux, 87 étaient faux à la référence et 60 déjà justes : ces derniers sont des protections, pas des corrections supplémentaires.
- **16 fragments** satisfont un critère plus fort : aucun des sept candidats seuls n’est juste, mais un groupe l’est. Ils comprennent 1 vrai K2, 6 K3 et 9 K4. Le réseau en corrige respectivement **6 / 6 / 7** pour direct / global / local.

Ces 16 cas disposent tous déjà d’au moins une paire correcte. Les données observées ne prouvent donc pas qu’une association de sept membres soit nécessaire ; les tests synthétiques vérifient séparément que le code accepte aussi cette situation.

Exemple réel : fragment **8909**, `00_Rock1-90-C#_comp.jams`, début à l’échantillon **522477**, fold 4, vrai **K4**, référence **K3**. S4 choisit K5, S5 choisit K3, mais leur moyenne de distributions choisit K4. Les trois bras sélectionnent cette paire, masque 24.

| Vote | K2 | K3 | K4 | K5 | K6 | Verdict |
|---|---:|---:|---:|---:|---:|---:|
| S4, harmonique | 0,2122 | 0,2052 | 0,2280 | 0,2359 | 0,1187 | K5, faux |
| S5, fondamentales | 0,2216 | 0,2587 | 0,2488 | 0,1546 | 0,1163 | K3, faux |
| S4 + S5 | 0,2169 | 0,2320 | 0,2384 | 0,1952 | 0,1175 | K4, juste |

Il s’agit de la moyenne des distributions, pas de la moyenne des nombres K. [Les 16 cas complets](evidence/v273-group127/synergy-all-seven-wrong.csv) fournissent les sept verdicts individuels, tous les masques corrects et les choix des trois bras. Un masque est la somme des bits `2^(j-1)` des candidats Sj ; 0 désigne KEEP.

## Pourquoi les 127 groupes ne gagnent pas encore face aux singletons

Le réseau a été entraîné sur tous les groupes. On redécode ensuite ses mêmes scores en n’autorisant que les sept singletons et KEEP. C’est un diagnostic du choix des actions, **pas un modèle réentraîné sur les singletons**.

| Bras | Net avec singletons seuls | Net avec tous les groupes | Effet des groupes sur ce décodage |
|---|---:|---:|---:|
| direct | +50 | +24 | 67 corrections / 93 régressions = -26 |
| global | +68 | +57 | 66 corrections / 77 régressions = -11 |
| local | +64 | +54 | 80 corrections / 90 régressions = -10 |

Le réseau récupère certaines complémentarités, mais les nouveaux choix lui font perdre davantage de justesses ailleurs. Le résultat +57 ne démontre donc pas une supériorité du routage des 127 groupes.

Les 127 masques ne créent en moyenne que **1,837 verdict distinct** par fragment. Le septième candidat est une moyenne dérivée des cinq spécialistes ; il n’apporte pas une nouvelle source indépendante. Les bilans des 127 groupes sont conservés pour chaque bras dans [group-outcomes.csv](evidence/v273-group127/group-outcomes.csv). Les comptes de groupes se recouvrent : les additionner ne donne pas un nombre d’événements indépendants.

## Calibration et potentiel du catalogue

| Bras | Somme des gains prédits des groupes choisis | Net observé |
|---|---:|---:|
| direct | 496.830 | +24 |
| global | 488.942 | +57 |
| local | 473.273 | +54 |

La surestimation après choix du meilleur groupe demeure forte. Des probabilités normalisées et un risque commun de référence ne suffisent pas à garantir une bonne calibration sur les groupes retenus. Aucun seuil n’a été ajusté après lecture de ces résultats.

Un oracle qui connaîtrait les labels pourrait corriger **1 624 erreurs** avec les propositions disponibles et KEEP : plafond descriptif **84,4358 % global / 56,2492 % poly**. Ce n’est ni une prévision atteignable garantie, ni le score du réseau. Les 1 624 opportunités incluent celles déjà accessibles à un candidat seul ; seules 16 exigent ici un regroupement.

## Ce qui a effectivement été entraîné et contrôlé

Un critique neuronal partagé apprend la fiabilité de chaque groupe complet en fonction du fragment, de ses membres et, selon le bras, de ses historiques correction/régression/neutre. Aucun membre n’est écarté à cause de son bilan individuel. **La fusion des votes reste une moyenne fixe** : ce prototype apprend la fiabilité des regroupements, pas une nouvelle règle de fusion.

La supervision utilise une probabilité commune de justesse de la référence et la justesse conditionnelle du verdict de chaque groupe. Les pertes sont moyennées par événement, sans traiter les 127 groupes corrélés comme 127 observations indépendantes. Les audits portent sur le même groupe, la même classe initiale et la même destination ; le local ajoute les 64 voisins autorisés et un lissage fixé à 12.

Les **11 tests** passent dans chacun des trois jobs, dont les synergies de deux et sept membres, l’absence de veto, les exclusions de labels, les probabilités, les gradients et un entraînement court. Le vérificateur séparé a recalculé les scores natifs, décisions, issues de tous les groupes, synergies, provenance des 14 ensembles de producteurs et 24 chemins imbriqués par bras. Aucun chevauchement d’IDs interdits n’a été trouvé dans ces contrôles.

Le premier run [37776065908](https://github.com/Andriamarosoa/note/actions/runs/37776065908) avait de petites différences numériques entre sorties de spécialistes (maximum 1,0133e−6). Il est conservé dans [initial-run.json](evidence/v273-group127/initial-run.json). Sans modifier le modèle ni les hyperparamètres, un job commun a ensuite figé toutes les prédictions de producteurs avant de relancer les trois bras. Les votes, propositions et audits du run contrôlé sont **identiques bit à bit entre bras**. Les nets restent +24/+57/+54.

Le run contrôlé utilise le commit `71b162b10e1100f6efd2ec6e258406023077a219`. [verification.json](evidence/v273-group127/verification.json) contient les IDs d’archives, SHA-256, scores complets, comparaisons et preuves de vérification. Les archives CI incluent les poids et sorties par groupe ; leur rétention est de 90 jours. Les tableaux et preuves JSON/CSV de ce dossier restent dans le dépôt.

Pour rejouer la vérification, extraire les archives dans `artifacts/direct`, `artifacts/global`, `artifacts/local`, `artifacts/producer-cache`, garder les ZIP correspondants et copier `artifact_manifest` du JSON de preuve dans `artifacts/artifacts.json` :

```bash
python -m scripts.verify_v273_group127 \
  --artifacts artifacts \
  --reference coherent/predictions.npz \
  --features features \
  --output verification/verification.json
```

## Limite et décision

Les folds 0/1/2/4 ont déjà servi au développement. La séparation des producteurs et des audits empêche les chevauchements contrôlés, mais **ne transforme pas ces résultats en validation sur des données inédites**. Fold 3 et player05 sont exclus. Le contexte acoustique conserve une anticipation de +160 ms. Un seul seed a été évalué ; aucune significativité du gain n’est revendiquée.

La priorité révélée par ce test est de fiabiliser le choix du groupe et ses gains annoncés, en examinant les erreurs créées face au décodage par singletons. L’apport local doit être retesté dans un protocole fixé à l’avance avant de conclure sur la similarité. Une validation indépendante reste nécessaire avant promotion ; ni le bras global à +57 ni le décodage singleton à +68 ne remplacent la référence dans cette PR.
