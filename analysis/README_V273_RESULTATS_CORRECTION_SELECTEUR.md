# V27.3 — Résultat de la correction et du réentraînement du sélecteur

**Expérience terminée le 8 octobre 2026. Gain exploratoire de +31 événements
exacts face à `freeze_local_combo`. Référence conservée ; aucune promotion.**

Les défauts identifiés justifiaient une revue du système. Leur correction,
suivie d'un réentraînement avec une décision probabiliste cohérente, produit
ici un petit gain global et poly. Elle ne résout pas le problème K3 et ne
démontre pas encore une amélioration sur de nouvelles données.

[Protocole fixé avant résultats](README_V273_PROTOCOLE_CORRECTION_SELECTEUR.md) ·
[Revue de conception](README_V273_REVUE_CONCEPTION_SELECTEUR.md) ·
[Run terminé avec succès](https://github.com/Andriamarosoa/note/actions/runs/37766719862) ·
[PR expérimentale #7](https://github.com/Andriamarosoa/note/pull/7).

## Comparaison native

Les trois bras ont été entraînés sur les mêmes folds 0/1/2/4 et évalués sur
les mêmes 59 309 événements, dont 7 385 vrais K≥2. Les changements restent
limités aux 7 493 événements initialement prédits K2/K3/K4. Les scores poly
portent sur tous les vrais K≥2, y compris ceux hors de ce périmètre.

| Système | Exacts globaux | Exact-K global | Exacts poly | Exact-K poly | Corrections | Régressions | Net face à la référence |
|---|---:|---:|---:|---:|---:|---:|---:|
| `freeze_local_combo` | 48 454 | 81,6976 % | 2 530 | 34,2586 % | — | — | — |
| `legacy` : reproduction du sélecteur complet précédent | 48 423 | 81,6453 % | 2 499 | 33,8389 % | 479 | 510 | −31 |
| `contracts` : entrées et producteurs corrigés, ancien arbitre réentraîné | 48 432 | 81,6605 % | 2 508 | 33,9607 % | 577 | 599 | −22 |
| `coherent` : mêmes entrées corrigées, nouvelle décision réentraînée | 48 485 | 81,7498 % | 2 561 | 34,6784 % | 828 | 797 | +31 |

Le contrôle reproduit l'archive `both` du run 37758789956 : **zéro prédiction
différente**. Il s'agit du sélecteur complet avec identité et régimes, et non
du bras nommé `baseline` de l'ancienne expérience.

Le gain de `coherent` face à la référence est de **+0,0523 point global** et
**+0,4198 point poly**. Il apporte +62 exacts face à `legacy` et +53 face à
`contracts`. Ces écarts concernent cette exécution à seed fixé et ces folds
déjà exposés. Ils ne constituent pas une preuve de gain reproductible.

## Répartition du gain et des pertes

Les nets suivants comptent les corrections moins les régressions face à
`freeze_local_combo`, par vrai K.

| Vrai K | Effectif | Référence : exacts | `legacy` : net | `contracts` : net | `coherent` : corrections | `coherent` : régressions | `coherent` : net |
|---|---:|---:|---:|---:|---:|---:|---:|
| K0 | 39 652 | 38 035 | 0 | 0 | 0 | 0 | 0 |
| K1 | 12 272 | 7 889 | 0 | 0 | 0 | 0 | 0 |
| K2 | 3 445 | 1 183 | +17 | +36 | 328 | 203 | +125 |
| K3 | 2 289 | 915 | −130 | −152 | 308 | 405 | −97 |
| K4 | 1 207 | 417 | −8 | +36 | 163 | 189 | −26 |
| K5 | 355 | 15 | +75 | +44 | 28 | 0 | +28 |
| K6 | 89 | 0 | +15 | +14 | 1 | 0 | +1 |

Le gain total de +31 repose surtout sur K2. K3 descend de **39,9738 % à
35,7361 %**, et K4 de **34,5485 % à 32,3944 %**. Les résultats K5/K6 restent
supérieurs à la référence, mais inférieurs à ceux du sélecteur précédent.
La nouvelle décision redistribue donc les erreurs ; elle n'améliore pas
toutes les classes.

L'égalité des exacts K0/K1 ne signifie pas que leurs prédictions sont toutes
inchangées. La variante cohérente change 69 événements de vrai K0 et 324 de
vrai K1 entre destinations K2–K6 : ils restent faux pour l'Exact-K. Au total,
elle change 2 611 prédictions : 828 corrections, 797 régressions et 986
changements neutres pour cette métrique.

| Fold externe évalué | `legacy` : net | `contracts` : net | `coherent` : corrections | `coherent` : régressions | `coherent` : net |
|---|---:|---:|---:|---:|---:|
| 0 | −12 | −10 | 237 | 253 | −16 |
| 1 | −7 | −2 | 238 | 210 | +28 |
| 2 | −11 | −22 | 144 | 133 | +11 |
| 4 | −1 | +12 | 209 | 201 | +8 |

Trois folds sont positifs et un reste négatif. Aucun intervalle de confiance
ni test de significativité n'est revendiqué ; les événements d'un même
enregistrement ne doivent pas être traités comme des observations indépendantes.

## Ce qui a été corrigé et ce que la comparaison permet de conclure

- Le contexte direct passe de 26 caractéristiques tronquées à 43, dont les
  14 spectrales absentes de l'ancienne entrée directe.
- La présence Cxy décrit sa participation effective au sous-ensemble.
- H0 encode l'action initiale disponible ; sa constante de 0,84 et les
  anciennes constantes de confiance des replis ne sont plus utilisées
  comme des probabilités de justesse dans les entrées réparées.
- Les producteurs des références d'audit, ainsi que leur scaler/KMeans,
  excluent le fold receveur et le test externe. La normalisation du contexte
  brut reste un prétraitement sans labels ajusté sur le train externe.
- `coherent` apprend une distribution unique sur les sept vrais K avec une
  perte catégorielle non pondérée. La justesse de KEEP et le risque de
  régression sont la même quantité, `P(Y=K_initial)`. Le choix d'une
  destination compare son gain attendu à zéro ; une égalité conserve KEEP.

Les trois bras ont reçu 30 epochs, le seed 27402, Adam à 0,002, des lots de
192 et une largeur de 32. Ils comportent respectivement 6 596, 8 228 et
8 261 paramètres par modèle. La capacité n'est donc pas strictement égale.
Aucun nouvel expert acoustique n'a été ajouté. Les spécialistes utilisent
toujours les mêmes familles d'observables harmoniques.

Le bras `contracts` reste perdant malgré les entrées réparées : **−22**.
Le changement conjoint de paramétrisation et d'objectif dans `coherent`
donne **+53 exacts face à `contracts`**. Cela soutient l'intérêt d'examiner
l'arbitrage, sans isoler l'effet de chaque correction ni celui de la perte
seule. Une distribution normalisée et une perte propre n'établissent pas
automatiquement la calibration empirique.

## Vérifications et traçabilité

Le run 37766719862, tentative 1, est terminé avec succès sur le commit
`3774441791695c23bb81b4039892d65e7faa77ae`. Dans chacun des trois jobs, les
10 tests ont passé avant l'entraînement : séparation des producteurs,
modification des labels des folds interdits, contrats d'entrée, contre-exemple
de décision, gradients TensorFlow et entraînement court.

Un [vérificateur séparé](../scripts/verify_v273_selector_repair_artifacts.py),
sans import de l'évaluateur ni de TensorFlow, a ensuite relu les archives et
recalculé les métriques, matrices de confusion et bilans appariés. Son
[résultat machine](evidence/v273-selector-repair/verification.json) confirme :

- les empreintes SHA-256 des trois archives, les IDs natifs, labels, folds,
  classes initiales et événements admissibles identiques à la référence ;
- zéro changement hors périmètre et zéro divergence du contrôle `legacy` ;
- les 14 ensembles distincts de fit des spécialistes par bras réparé et
  les 24 chemins imbriqués des producteurs de références d'audit, contrôlés
  contre les IDs réels ; aucune intersection avec les folds interdits ;
- des manifestes identiques entre `contracts` et `coherent` ;
- les 7 493 distributions cohérentes normalisées, une dispersion exactement
  nulle des risques de régression entre destinations, zéro divergence entre
  le décodage par gain attendu et les prédictions exportées ;
- les masques des 32/64 sous-ensembles et leur attention normalisée.

Ces contrôles de provenance ne transforment pas les folds de développement
en validation indépendante. Les poids des quatre folds, les prédictions et
les rapports complets restent disponibles dans les artifacts du run :

| Bras | Artifact | SHA-256 de l'archive ZIP |
|---|---|---|
| `legacy` | [11545671181](https://github.com/Andriamarosoa/note/actions/runs/37766719862/artifacts/11545671181) | `e447a18990c534cf4b5cd405b1bba430fe5c046de75552faa0327110f71a3022` |
| `contracts` | [11544644114](https://github.com/Andriamarosoa/note/actions/runs/37766719862/artifacts/11544644114) | `1581c57d5b35db3fdbf0cf9ad21c188da15458aa0d48894cf455d27187c0c309` |
| `coherent` | [11544713559](https://github.com/Andriamarosoa/note/actions/runs/37766719862/artifacts/11544713559) | `58717a0298c5dd899622326660f10a0b597236ecc2bb70210caebc81d13c9b91` |

## Décision et priorité suivante

**`freeze_local_combo` reste la référence.** `contracts` est rejeté comme
remplacement selon le protocole, car son bilan est négatif. `coherent`
franchit le critère exploratoire de gain global et poly, mais reste un
candidat expérimental : gain faible, pertes K3/K4 et fold 0 négatif.

La revue du système était donc une priorité fondée. La suite utile est
maintenant un diagnostic ciblé des décisions restantes, en particulier les
405 régressions K3 et 189 régressions K4 de cette nouvelle variante, et de
leur lien avec les corrections obtenues. Une éventuelle modification devra
avoir un nouveau protocole fixé avant son évaluation ; aucun seuil ni
hyperparamètre n'a été retouché après ce run.

Une confirmation sur des données inédites reste nécessaire avant promotion.
Les folds actuels ont déjà servi à la recherche ; player05/fold3 n'a pas
été utilisé. Les résumés acoustiques consultent jusqu'à +160 ms et les
1 918 erreurs poly hors périmètre restent hors de portée de cette correction.
Ces résultats ne justifient pas encore l'ajout de nouvelles têtes acoustiques.
