# V27.3 — Résultats du consensus entre groupes de même verdict

**Terminé et vérifié le 8 octobre 2026. Référence conservée ; aucune promotion.**

Le partage d’un score par verdict supprime les contradictions mesurées entre groupes. La variante catégorielle atteint **+60 net**, soit seulement **3 justesses de plus** que les 127 groupes globaux précédents (+57). Elle reste sous leur décodage singleton (+68). La variante conservant la BCE atteint +45 : partager les scores ne suffit pas à améliorer le résultat.

[PR #9](https://github.com/Andriamarosoa/note/pull/9) · [Run terminé](https://github.com/Andriamarosoa/note/actions/runs/37781750476) · [Protocole antérieur à l’entraînement](README_V273_PROTOCOLE_CONSENSUS.md)

## Comparaison sur la même cohorte

59 309 événements natifs, dont 7 385 vrais K≥2 pour le score poly. Les mêmes 7 493 fragments prédits initialement K2/K3/K4 peuvent changer. Net = corrections moins régressions face à `freeze_local_combo`.

| Variante | Global | Poly | Corrections | Régressions | Net |
|---|---:|---:|---:|---:|---:|
| freeze_local_combo | 81.6976 % | 34.2586 % | — | — | +0 |
| Group127 global archivé | 81.7937 % | 35.0305 % | 675 | 618 | +57 |
| Ancien décodage singleton, mêmes poids | 81.8122 % | 35.1794 % | 624 | 556 | +68 |
| pooled_bce | 81.7734 % | 34.8680 % | 558 | 513 | +45 |
| pooled_ce | 81.7987 % | 35.0711 % | 582 | 522 | +60 |

La variante `pooled_bce` utilise la moyenne des logits des groupes proposant le même K, puis une sigmoïde, avec exactement l’ancienne perte BCE par événement. `pooled_ce` normalise aussi les destinations disponibles avec une issue « aucune proposition juste » (OTHER) et une vraisemblance catégorielle par événement. Elle change donc à la fois cette normalisation et la perte conditionnelle.

Les 127 groupes passent toujours dans le réseau partagé. Leurs membres ne subissent aucun veto individuel. La fusion qui définit leurs propositions reste une moyenne fixe de votes ; le réseau apprend leur fiabilité. Le masque retenu est un représentant déterministe du K gagnant, **pas l’attribution de ce résultat à ce seul groupe**.

La variante catégorielle produit **96 régressions de moins**, mais aussi **93 corrections de moins**, que le bras global archivé : le solde n’est que +3. La comparaison appariée entre leurs prédictions compte 249 corrections et 246 régressions. Face à l’ancien décodage singleton : 231 / 239, soit −8. Ce faible écart ne suffit pas à revendiquer une meilleure généralisation.

## Ce que le correctif répare effectivement

Avant le correctif, 942 fragments du bras global avaient au moins une destination évaluée avec des gains de signes opposés selon le groupe. Après le correctif, **l’écart de gain entre groupes de même verdict est exactement zéro**, dans les deux variantes.

Dans `pooled_bce`, les probabilités des différentes destinations restent indépendantes : leur somme, avec la référence, dépasse 1 sur **265 fragments** (maximum 1,9241). Dans `pooled_ce`, les probabilités de la référence, des alternatives disponibles et de OTHER somment à 1 à la précision numérique près ; **aucun dépassement supérieur à 1e−6**. OTHER évite qu’une seule alternative disponible soit considérée automatiquement juste si la référence est fausse.

Les votes, propositions et audits bruts sont identiques bit à bit à l’archive globale précédente. La probabilité apprise de justesse de la référence est également **identique bit à bit dans les deux nouveaux bras et l’ancien** : les différences viennent ici de l’estimation conditionnelle des alternatives et des décisions qui en résultent. Architecture inchangée : 12 994 paramètres, mêmes 30 epochs, seed 27402, Adam 0,002 et batch 192.

## Résultats K0–K6 et par fold

| Vrai K | Effectif | Référence | Group127 global | pooled_bce | pooled_ce |
|---|---:|---:|---:|---:|---:|
| K0 | 39652 | 95.9220 % | 95.9220 % (+0) | 95.9220 % (+0) | 95.9220 % (+0) |
| K1 | 12272 | 64.2846 % | 64.2846 % (+0) | 64.2846 % (+0) | 64.2846 % (+0) |
| K2 | 3445 | 34.3396 % | 37.8229 % (+120) | 37.4746 % (+108) | 37.2714 % (+101) |
| K3 | 2289 | 39.9738 % | 39.1874 % (-18) | 38.6632 % (-30) | 39.7117 % (-6) |
| K4 | 1207 | 34.5485 % | 29.2461 % (-64) | 30.1574 % (-53) | 30.4060 % (-50) |
| K5 | 355 | 4.2254 % | 9.5775 % (+19) | 9.8592 % (+20) | 8.4507 % (+15) |
| K6 | 89 | 0.0000 % | 0.0000 % (+0) | 0.0000 % (+0) | 0.0000 % (+0) |

Entre parenthèses : net face à la référence. La variante catégorielle réduit les pertes K3 de −18 à −6 et K4 de −64 à −50, mais cède 19 justesses K2 et 4 K5 face à l’ancien bras global. Les scores K0/K1 sont inchangés ; les prédictions hors du périmètre initial K2/K3/K4 sont identiques à la référence.

| Fold | Group127 global | pooled_bce | pooled_ce |
|---|---:|---:|---:|
| 0 | +18 | +11 | +16 |
| 1 | +26 | +24 | +23 |
| 2 | -20 | -25 | +1 |
| 4 | +33 | +35 | +20 |

Les quatre folds catégoriels sont positifs dans ce run. Le fold 2 ne gagne qu’un événement : ce résultat reste fragile et ne constitue pas une validation indépendante.

## Groupes, singletons et cas de complémentarité

Le décodage singleton de chaque réseau réutilise ses mêmes poids **et les informations des 127 groupes** ; seule la liste des actions exécutables est réduite. Il ne s’agit pas de réentraîner le réseau avec uniquement les candidats individuels.

| Variante | Net avec tous les groupes | Net du décodage singleton | Effet des actions supplémentaires | Cas récupérés où les 7 singletons sont faux |
|---|---:|---:|---|---:|
| pooled_bce | +45 | +43 | 5 corrections / 3 régressions = +2 | 5 / 16 |
| pooled_ce | +60 | +61 | 6 corrections / 7 régressions = -1 | 6 / 16 |

La récupération des vraies complémentarités reste possible. Elle n’augmente pas face au bras global précédent : 6/16 auparavant, 5/16 en BCE et 6/16 en catégoriel. Les différences entre les deux décodages tombent de 264 décisions modifiées dans l’ancien global à 16 en BCE et 19 en catégoriel. Le score commun retire ainsi les divergences de préférence entre groupes et singletons qui proposent exactement le même K.

Les [16 cas](evidence/v273-group-consensus/synergy-all-seven-wrong.csv) et les [bilans des 127 groupes](evidence/v273-group-consensus/group-outcomes.csv) sont exportés. Les groupes de même verdict ont le même gain ; les compteurs de masques sélectionnés dépendent désormais du représentant choisi et ne mesurent pas une contribution causale des membres.

## Suivi des erreurs auditées et du gain annoncé

L’audit préalable avait recensé 77 régressions et 66 corrections ajoutées par les groupes face aux singletons de l’ancien modèle. Sur ces mêmes événements :

| Variante | Anciennes régressions récupérées / 77 | Anciennes corrections conservées / 66 |
|---|---:|---:|
| pooled_bce | 50 | 26 |
| pooled_ce | 43 | 25 |

Le [suivi des 143 cas](evidence/v273-group-consensus/followup-original-143-cases.csv) évite de ne montrer que les régressions réparées : d’anciennes corrections sont aussi perdues. Ce sous-ensemble n’est pas un nouveau jeu de test.

| Variante | Gain total annoncé sur les choix retenus | Net réel |
|---|---:|---:|
| Ancien global | 488.942 | +57 |
| pooled_bce | 353.882 | +45 |
| pooled_ce | 383.195 | +60 |

La surestimation reste forte. Une distribution cohérente entre verdicts ne garantit donc pas des gains bien calibrés. Aucun seuil, seed ou hyperparamètre n’a été modifié après ces résultats. Le diagnostic préalable de moyenne des probabilités sans réentraînement (+69 pour l’ancien global) est conservé dans [l’audit initial](evidence/v273-group-consensus/duplicate-audit.json) ; il ne faut pas le confondre avec l’apprentissage de moyenne des logits évalué ici.

## Vérification et décision

Les **17 tests passent dans chacun des deux jobs** : contrats des producteurs, séparation des labels, synergies, score commun, invariance à l’ordre des groupes, cas OTHER, vraisemblance, gradients et entraînement court. Les deux entraînements complets ont terminé avec succès au commit `2caa02556eec3896f896509ebb0b9c401d83e6b1`.

Le vérificateur indépendant a recomputé les scores, décisions, synergies, issues des groupes, comparaisons, probabilités issues des logits et provenance des 14 producteurs / 24 chemins imbriqués par bras. Les empreintes des entrées correspondent exactement au bras global précédent. Les SHA-256 des archives téléchargées ont été validés. [verification.json](evidence/v273-group-consensus/verification.json) contient les preuves et IDs des archives CI, conservées 90 jours.

Pour reproduire, garder les ZIP `pooled_bce.zip` et `pooled_ce.zip`, extraire chacun dans le dossier du même nom et copier `artifact_manifest` de la preuve dans `artifacts/artifacts.json` :

```bash
python -m scripts.verify_v273_group_consensus \
  --artifacts artifacts \
  --reference coherent/predictions.npz \
  --archived-global archived-global \
  --features features \
  --output verification/verification.json
```

Le défaut de cohérence des scores est corrigé ; le gain de performance est limité. La variante catégorielle reste sous l’ancien décodage singleton et perd encore des justesses K3/K4 face à `freeze_local_combo`. Elle n’est pas promue. La priorité restante est de comprendre les gains positifs mal estimés et les corrections perdues, avec un protocole fixé avant tout nouvel essai.

Folds 0/1/2/4 déjà exposés, fold 3/player05 exclus, un seul seed, contexte acoustique jusqu’à +160 ms conservé. Aucune donnée inédite n’a été évaluée et aucune significativité du faible gain n’est revendiquée.
