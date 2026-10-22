# Résultats vérifiés — H_FP_PARENT_TIMELINE_01, Onset MLP69 contrôle 69D / contrôle 70D / nouvelle tête temporelle 74D

**Verdict : REFUTED_OR_INCOMPLETE, expérience scientifique rejetée. AUCUNE PROMOTION.** La règle de réfutation [#140](https://github.com/Andriamarosoa/note/issues/140) a été appliquée sans changer le critère préenregistré [#142](https://github.com/Andriamarosoa/note/issues/142).

## Provenance et validation réelle

- Branch : `codex/note-fp-temporal-discriminator-74d-256`.
- [Workflow GitHub #39775357133](https://github.com/Andriamarosoa/note/actions/runs/39775357133) : `smoke` SUCCESS (tests + ABI), puis 15 vrais réseaux `StandardScaler+MLPClassifier(64,32)` **entraînés**, 5 folds 3 TRAIN / 1 CAL / 1 OUTER, 69D/70D/74D same TRAIN/same seed, cuts figés uniquement CAL.
- `train_three_ablations_and_replay_80_original_wavs` **FAIL SCIENTIFIQUE EXPLICITE**, et NON panne du pipeline : les 80 WAV signés, 306536 décisions strictement causales PCM256, ont été rejoués et le `report.json` complet produit. Le programme lève délibérément `AssertionError: H_FP_PARENT_TIMELINE_01 refuted or incomplete ... DO NOT promote`.
- [Artefact complet #12424079451](https://github.com/Andriamarosoa/note/actions/runs/39775357133/artifacts/12424079451) : `trained/report.json`, cinq checkpoints et métadonnées SHA256 par contrôle (15 au total), `replayed/report.json` avec 80 bilans par piste et toutes régressions.
- Anciens états Onset inchangés prouvés par les comptes exacts S613 **TP8099/FP4600/FN3168** ; trois têtes ErrorPrediction MLP69 **TP8054/FP4411/FN3213** ; quatre têtes FNRecovery **TP8160/FP4605/FN3107**. Les cinq autres caractéristiques et le MIDI public ne changent pas.

## Résultats RÉELS du replay audio, et NON proxy de candidats

| Version native256 sur exactement 80 WAV | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Meilleur quatre-têtes gelé (avant l'intervention) | 8160 | 4605 | 3107 | **67.9094541 %** |
| Contrôle 69D, appris indépendamment | 8127 | 4519 | 3140 | **67.9713963 %** |
| Contrôle 70D, contrôle de la confiance MLP69 | 8137 | 4549 | 3130 | 67.9413852 % |
| **Tête temporelle 74D, testée** | **7948** | **4290** | **3319** | **67.6281642 %** |

Bilan 74D contre 4 têtes gelées : **315 FP supprimés, mais 212 TP également perdus**, FN +212, F1 **−0.2812898 point**. La baisse des FP est réelle ; l'objectif conjoint FP et F1 annoncé avant l'expérience échoue. 74D fait certes moins de FP que les deux contrôles, mais **son F1 est pire que chacun d'eux et que le système de référence**.

Le contrôle 69D a une légère amélioration de **+0.0619423 point F1** contre le 4-têtes ; **86 FP évités pour 33 TP sacrifiés**. C'est le meilleur résultat F1 **de cette expérience**, pas une preuve de généralisation externe ni une promotion automatique.

### Où la prédiction temporelle échoue-t-elle ?

Agrégation depuis le `replayed/report.json`, sans modification de seuils après OUTER :

| Composition | F1 quatre-têtes | F1 contrôle 69D | F1 temporel 74D | Δ TP temporel | Δ FP temporel |
|---|---:|---:|---:|---:|---:|
| BN1 | 67.4867 % | 67.5508 % | 67.2206 % | −34 | −53 |
| Funk2 | 71.5730 % | 71.7436 % | 71.7143 % | −13 | −33 |
| Jazz1 | 71.1151 % | 71.1613 % | 71.1151 % | 0 | 0 |
| **Rock3** | **68.4358 %** | **68.4358 %** | **67.3175 %** | **−127** | **−132** |
| SS1 | 61.6949 % | 61.7423 % | 61.8443 % | −38 | −97 |

Sur les 212 TP perdus par temporal74, **127 (59.9%) viennent de Rock3**, malgré 132 FP supprimés sur cette famille. Le calibrage du pli Rock3 provenait de SS1 : le seuil temporel 0.05 avait supprimé **13 TP et 37 FP sur CAL**, mais a finalement supprimé **127 TP et 132 FP** en métriques réellement réappariées sur Rock3 OUTER (**128 TP et 131 FP** sur le proxy événementiel). Cela démontre une **instabilité CAL → OUTER** pour cette variante, précisément sur les attaques légitimes de cette famille.

Par texture : temporel 74D perd **208 TP et retire 270 FP dans les 40 accompagnements**, contre seulement 4 TP perdus pour 45 FP retirés dans les solos. **25 pistes régressées** face aux quatre-têtes (22 accompagnements, 3 solos) ; contrôle69 : 13 pistes régressées (12 accompagnements, 1 solo).

### Pourquoi le diagnostic initial n'est plus suffisant

L'audit descriptif précédent démontrait **une corrélation** : sur 4 411 FP tri-têtes, 782 ont une attaque récente d'un parent harmonique dans les 80 ms strictement passés, mais ce groupe contient également **678 vraies attaques**. La nouvelle expérience vérifie qu'un réseau MLP74 avec ces quatre âges de parents supprimait effectivement davantage de FP — mais supprimait simultanément **trop de TP** pour améliorer la fonction réelle.

**Règle #140 : l'hypothèse H_FP_PARENT_TIMELINE_01 selon laquelle cette information temporelle serait suffisante pour améliorer l'arbitrage FP sans dégrader F1 est REFUTÉE ou INCOMPLÈTE.** Ce test ne démontre pas que toute chronologie harmonique serait inutile ; seulement que ces quatre âges, avec cette représentation, apprentissage, CAL et MLP, **ne fournissent pas la discrimination annoncée**. Pas de réglage opportuniste de threshold selon OUTER, pas de réutilisation de 74D en production ni de claims F1 fictifs.

### Pistes de contre-audit autorisées, non corrigées dans cette série

1. Auditer **les TP supprimés et FP écartés par hauteur**, contexte polyphonique et attaques simultanées, notamment les 127 TP Rock3 perdus, avant une nouvelle hypothèse. Mesurer si les caractéristiques des TP Rock3 ressemblent aux FP vus dans la composition CAL SS1.
2. Étudier l'erreur de généralisation de composition et de texture (accompagnement vs solo), plutôt qu'une règle générale « présence de parent harmonique = veto » ; le veto naïf événementiel avait déjà été rejeté à F1 66.2356 %.
3. Envisager des ensembles d'experts avec **sélecteur appris indépendamment sur TRAIN/CAL**, ou des nouvelles preuves acoustiques de fondamentale/harmonique ; toute proposition doit comporter une prédiction quantitative pré-enregistrée et un **véritable replay WAV 256** sans lookahead/hold.

**Caveat de généralisation** : les références historiques S613/S701 ont vu partiellement GuitarSet FIT80 à l'entraînement ; les modèles nouveaux respectent leurs splits composition-OOF, mais aucun F1 ici ne vaut une validation de toute la chaîne sur un corpus complètement extérieur.

**MIDI officiel et cinq autres caractéristiques strictement inchangés ; aucun poids entraîné1024.**
