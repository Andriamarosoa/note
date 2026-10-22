# Résultat définitif — H_FN_NEGCOVERAGE_01 — REJETÉ POUR PROMOTION

## Provenance

- Hypothèse et seuils préenregistrés AVANT expérience : [issue #141](https://github.com/Andriamarosoa/note/issues/141).
- Exécution [GitHub Actions #39765038340](https://github.com/Andriamarosoa/note/actions/runs/39765038340), commit initial `6b5add756886a64c516f05a7cf23f899dfdf0305`, terminée le 22 octobre 2026 à 11:10 UTC : **conclusion failure scientifique attendue**, smoke SUCCESS, entraînement SUCCESS, replay audio réel effectué en entier et gate préenregistrée FALSE.
- Artifact réel complet `ONSET_FN_NEGCOVERAGE_01_ALL_TRAIN_NEGATIVE_REAL_WAV` id **12421031963**, `trained/report.json`, `replayed/report.json`, 5 poids joblib + 5 métadonnées.
- Vérification indépendante des SHA256 des cinq poids : **5/5 conformes** aux métadonnées, groupes TRAIN/CAL/OUTER **5/5 disjoints**, 15 MLP effectivement entraînés avec couverture 100 % négatifs TRAIN, aucune fuite GT OUTER à l'entraînement.
- 80 WAV GuitarSet, **306536 décisions** toutes les 256 échantillons, quatre têtes interrogées avant sélection, aucun lookahead/hold. Les trois têtes précédentes sont reproduites exactement TP8054 FP4411 FN3213.

## Résultats strictement comparables

| Configuration | TP | FP | FN | F1 | Précision | Rappel |
|---|---:|---:|---:|---:|---:|---:|
| S613 principal | 8099 | 4600 | 3168 | 67.587416 % | 63.776675 % | 71.882489 % |
| 3 têtes gelées | 8054 | 4411 | 3213 | 67.874600 % | 64.612916 % | 71.483092 % |
| 4 têtes précédent meilleur, run #39759373185 | **8160** | 4605 | 3107 | **67.909454 %** | 63.924794 % | 72.423893 % |
| **4 têtes H_FN_NEGCOVERAGE_01, run #39765038340** | **8100** | **4500** | **3167** | **67.876147 %** | **64.285714 %** | **71.891364 %** |

- Contre le précédent 4 têtes : **−60 TP, −105 FP, +60 FN**, F1 **−0.033307 point**. FP baissent effectivement, mais le rappel se dégrade assez pour faire perdre le gain F1 : hypothèse **partiellement soutenue sur la baisse des FP, insuffisante pour expliquer/améliorer le compromis F1**.
- Contre les 3 têtes : **+46 TP, +89 FP, −46 FN** ; gain F1 **+0.001547 point** (trop faible pour promotion).
- **135** NoteOn proposés par FNRecovery, **68** propositions appariées à GT final (les appariements ne correspondent pas simplement aux gains nets de TP à cause des déplacements).
- Comparaison *piste à piste* avec l'ancien 4 têtes : **32 améliorées, 30 régressées, 18 inchangées**. Accompagnement 14 améliorées/21 régressées/5 inchangées ; solo 18 améliorées/9 régressées/13 inchangées.
- Versus l'ancien 4 têtes : accompagnement **−54 TP / −69 FP**, solo **−6 TP / −36 FP**. Les faux positifs sont mieux contenus, mais les vraies récupérations perdues en accompagnement restent le problème dominant.

## Cinq folds, seuils CAL gelés

| OUTER | CAL | Politique | Seuil | Activé | TP | FP | FN | F1 |
|---|---|---|---:|---|---:|---:|---:|---:|
| BN1 | Funk2 | cautious | 0.99 | Oui | 1213 | 672 | 488 | 67.651980 % |
| Funk2 | Jazz1 | cautious | 0.95 | Oui | 1753 | 676 | 725 | 71.448950 % |
| Jazz1 | Rock3 | moderate | 0.99 | Non | 1196 | 484 | 489 | 71.084695 % |
| Rock3 | SS1 | moderate | 0.99 | Non | 2420 | 1333 | 899 | 68.438914 % |
| SS1 | BN1 | cautious | 0.99 | Oui | 1518 | 1335 | 566 | 61.494835 % |

Checkpoints SHA256 validés :
- BN1 `d07354b1ed97c59d85cabf3a399b4640f530dcb564217484bb978ddb708f198a`
- Funk2 `04a30a9d87eefd0735ad5c3b351172f96f8678bea6e1a65a2f0e79311067413c`
- Jazz1 `5b118937670d955f51afc7159cb435f1ff2d6da367b59862f72d2c2922f2cb90`
- Rock3 `db9a1a2c5d1c6133124af2de061cbdf2f5835fc3b5162300d2ffc084a571b752`
- SS1 `a6615914cda2c479b56c750464a1ecb3088f76d47c1ecb613ba740817253c9d0`

## Verdict causal et prochaines pistes d'audit, SANS TUNING

**GATE PRÉENREGISTRÉE = FALSE : PAS DE PROMOTION.** Le programme a correctement levé `AssertionError: H_FN_NEGCOVERAGE_01 predeclared real-WAV F1/FP gate failed: audit hypothesis refuted or incomplete; DO NOT promote`. Ce n'est pas un bug d'exécution à contourner.

Le modèle voit bien plus de diversité des négatifs ; les FP diminuent de 105 par rapport au 4 têtes précédent, mais les TP diminuent aussi de 60, entraînant une baisse F1. L'hypothèse selon laquelle le manque de négatifs serait la cause *suffisante* du mauvais compromis est **réfutée**. Une contribution partielle à la réduction des FP est plausible, non causalement isolée des effets d'augmentation de taille d'entraînement et de répétition positive.

Pour la prochaine hypothèse indépendante, **auditer** (sans modifier les seuils à partir de OUTER) les 60 TP perdus et 105 FP éliminés : répartitions acoustiques, comp/solo, collisions d'événements, distribution/calibration des scores de probabilité, décalage entre CAL et OUTER, particulièrement Funk2. Séparer le nombre d'exemples négatifs du changement de ratio/duplication dans une expérience contrôlée avant d'attribuer une cause.

Limites inchangées : S613 a historiquement vu 20/80 et S701 48/80 morceaux FIT80 ; ce n'est PAS une validation externe indépendante. Les poids S708 historiquement hop1024 ne sont pas employés. Le temps CPU p95/p99 et la latence matérielle ne sont PAS mesurés ; 256/44100 = 5.805 ms est la cadence, pas la latence. Aucune modification du MIDI officiel ni des cinq autres axes.
