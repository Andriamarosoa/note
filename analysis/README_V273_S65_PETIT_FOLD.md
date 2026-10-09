# S65 — audit du plus petit fold (fold 1)

**Statut : diagnostic reproductible local, non promu.** 59 309 fragments de référence, folds 0/1/2/4. Plus petit par population totale : fold 1, **13 868 événements**, **1 762 vrais K≥2** et **4 morceaux**. Les têtes ont été développées historiquement sur les mêmes folds ; ceci **n'est pas une validation indépendante**.

## Résultat principal

Référence S58 sur fold 1 : 2 672 erreurs initiales, 217 corrections, 118 régressions, 145 changements mauvaise→mauvaise, **+99 net**, **2 573 erreurs persistantes** ; 668/1 762 polyphoniques correctement classés.

Les 7 têtes de veto existantes S60 (Ridge et context-transition), S61 (voisins et transition), S62 (morphologie non linéaire), S63 (contre-votes source et transition) forment **128 sous-ensembles**. La sélection du sous-ensemble par gain net sur les folds **0, 2 et 4**, sans choisir sur fold 1, retient **S62 + S63**. **Aucun des 128 sous-ensembles ne dépasse +99 net sur fold 1 : 32 à +99, 96 inférieurs (minimum +97)**. S62 et S63 ne bloquent **aucune** décision sur ce fold (abstention décidée par leurs propres modèles), d'où la stagnation. S60-Ridge économise 2 régressions mais perd 3 corrections ; S63 transition économise 5 régressions mais perd 7 corrections. L'oracle local ne justifie pas leur promotion.

Les 118 régressions de S58 sur ce fold concernent principalement K3→2 (33) et K2→3 (25), alors que ces mêmes transitions contiennent 61 et 49 corrections légitimes. K4 réel perd 8 bonnes réponses nettes ; 135/226 vrais K4 restent mal classés.

## Tester des combinaisons plus efficaces que les veto

Les autres prédictions gelées disponibles montrent : S50+S56 **+130** fold 1 mais 222 régressions ; S57 **+102** avec 115 régressions ; S58 **+99** avec 118 ; S59 **+102** avec 120. L'ajout des nouvelles têtes S62+S63 ne change pas fold 1.

Une nouvelle ablation route par **K initial**, en choisissant la politique parmi S50+S54, S50+S56, S57, S58, S59, S62+S63 sur les **trois autres folds**, pour maximiser `corrections − coût×régressions`, minimum 4 unités de gain sur S58 :

| Coût R | Corrections fold1 | Régressions fold1 | Gain net fold1 | Erreurs fold1 | Modifications vs S58 |
|---|---:|---:|---:|---:|---|
| 1.0 | 365 | 222 | **+143** | **2 529** | +44 net / +104 R |
| 1.15 | 303 | 172 | **+131** | **2 541** | +32 net / +54 R |
| 1.35 | 217 | 118 | +99 | 2 573 | 0 |
| 1.5 | 217 | 118 | +99 | 2 573 | 0 |
| 2.0 | 217 | 118 | +99 | 2 573 | 0 |

À coût 1.15, seul le remplacement de la politique des **K initialement prédits 0** par S50+S56 change le fold 1 : 86 corrections nouvelles / 54 anciennes bonnes réponses perdues / 50 mauvaises→mauvaises, soit 190 divergences et +32 net. Par morceau : BN3-154-E +3, Funk2-119-G +1, Funk3-112-C# +2, SS3-98-C **+26** ; **81 % de l'apport vient d'un seul morceau**. Seuls les choix de politiques sont train-fold-only ; le coût 1.15 a été comparé rétrospectivement aux autres coûts. Ne pas affirmer qu'il a été choisi sans regarder fold 1.

## Pourquoi la séparation est difficile

Médiane votes des 36 politiques pour le K initial : **21.5 en régression** contre **16 en correction** ; médiane votes pour K proposé : **7 contre 11** ; chevauchement élevé (81 régressions et 101 corrections avec ≥18 votes pour le K initial). Parmi les 145 mauvaises→mauvaises du fold, la bonne classe apparaît dans les 36 votes dans 79 cas, sans garantie de sélection correcte.

**Conclusion :** empiler d'autres têtes de refus ne résout pas ce petit fold. Augmenter le gain global par réordonnancement K initial peut aggraver fortement les régressions. Prochaine expérience pertinente : tête dédiée K2↔K3, entraînée sur signal acoustique/temporisation et votes, mesurant explicitement les corrections perdues ; protocole sur enregistrements réellement inédits requis pour promotion.

**Scripts/artefacts locaux :** `s65_small_fold_audit.py`, `s65_small_fold_detail.py`, `s65_source_router.py` et `S65_PETIT_FOLD_REPRODUCTIBLE.zip`. Aucun poids du réseau principal modifié.
