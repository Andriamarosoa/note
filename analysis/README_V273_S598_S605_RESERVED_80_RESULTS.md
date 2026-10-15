# S598–S605 — 84,91 % de F1 sur cinq œuvres réservées

**Objectif de 80 % atteint sur le test S605 : 84,91 % de F1 NoteOn ±80 ms, contre 73,51 % pour Basic Pitch par défaut sur les mêmes cinq œuvres.** Le modèle et ses paramètres ont été figés avant les WAV test ; les prédictions ont été archivées avant l’accès aux annotations. Ce résultat concerne 120 secondes, 626 vraies notes MIDI40–88 et une chaîne hors ligne. Il ne certifie pas le Live, NoteOff, Exact-K ni une performance universelle.

[PR #41](https://github.com/Andriamarosoa/note/pull/41) · [Première CI réservée, réussie](https://github.com/Andriamarosoa/note/actions/runs/38685400020) · [Protocole publié avant exposition](README_V273_S598_S605_BOUCLES_PREREGISTRATION.md) · [Contrat du candidat figé](v273_s602_selected_candidate_freeze.json)

## Résultat principal indépendant

Hauteur MIDI exacte, appariement un à un, tous les faux positifs et toutes les notes manquées. Extraits fixés à 5–29 s ; aucun morceau remplacé après résultat. Les annotations humaines Notes_ décrivent le jeu réel ; les partitions Sco_ ne sont pas utilisées. Sur 633 attaques annotées dans les extraits, 6 sont hors de la plage MIDI40–88 et un unisson exactement simultané est fusionné : 626 notes évaluées.

| Politique | TP | FP | FN | Précision % | Rappel % | F1 % |
|---|---:|---:|---:|---:|---:|---:|
| Basic Pitch par défaut | 501 | 236 | 125 | 67,98 | 80,03 | **73,51** |
| S600 décodeur global | 490 | 164 | 136 | 74,92 | 78,27 | **76,56** |
| S601 filtre appris — candidat principal | 484 | 30 | 142 | 94,16 | 77,32 | **84,91** |

**Gain du candidat principal : +11,40 points de F1, avec 206 faux positifs de moins et 17 vrais positifs de moins.** Le gain provient surtout de la suppression des fausses notes ; il ne faut pas le présenter comme une amélioration du rappel. Le rappel passe de 80,03 % à 77,32 %.

## Résultats par œuvre et précision temporelle

| Œuvre | Notes vraies | BP défaut F1 % | Décodeur global F1 % | Filtre appris F1 % |
|---|---:|---:|---:|---:|
| 02_Sonata_vn_vn | 82 | 76,74 | 83,54 | 89,47 |
| 18_Nocturne_vn_fl_tpt | 92 | 74,31 | 80,60 | 90,17 |
| 33_Elise_tpt_tpt_hn_tbn | 150 | 73,89 | 76,57 | 86,83 |
| 40_Miserere_fl_fl_ob_cl_bn | 106 | 69,32 | 67,84 | 75,00 |
| 44_K515_vn_vn_va_va_vc | 196 | 74,02 | 76,73 | 83,80 |

Les cinq œuvres progressent en F1 face à BP par défaut, mais **Miserere reste à 75,00 %**. L’objectif franchi est le F1 micro de la cohorte complète, pas un minimum de 80 % sur chaque œuvre.

| Tolérance | BP défaut F1 % | Décodeur global F1 % | Filtre appris F1 % |
|---|---:|---:|---:|
| ±30 ms | 57,23 | 62,03 | 70,70 |
| ±50 ms | 66,32 | 71,72 | 80,88 |
| ±80 ms | 73,51 | 76,56 | 84,91 |

Le candidat dépasse également 80 % à ±50 ms, mais pas à ±30 ms. Ces tolérances d’appariement ne sont pas des mesures de latence.

| Registre MIDI | BP défaut F1 % | Décodeur global F1 % | Filtre appris F1 % |
|---|---:|---:|---:|
| Grave 40–59 | 73,20 | 72,40 | 82,43 |
| Médium 60–71 | 77,44 | 82,31 | 89,31 |
| Aigu 72–88 | 67,52 | 70,99 | 79,87 |

## Corrections, régressions et erreurs restantes

Par identité de note vraie, S601 **récupère 38 notes et en perd 55** par rapport à BP par défaut. Il retire 214 faux positifs et en ajoute 8 ; 22 faux positifs restent communs. Les notes gagnées/perdues sont publiées individuellement, sans se limiter au gain net.

| Œuvre | Notes récupérées | Notes perdues |
|---|---:|---:|
| 02_Sonata_vn_vn | 5 | 3 |
| 18_Nocturne_vn_fl_tpt | 7 | 10 |
| 33_Elise_tpt_tpt_hn_tbn | 10 | 4 |
| 40_Miserere_fl_fl_ob_cl_bn | 2 | 23 |
| 44_K515_vn_vn_va_va_vc | 14 | 15 |

Les 55 notes BP correctes perdues figurent bien dans la banque de candidats : **52 sont rejetées sous le seuil figé 0,5**, et **3 sont perdues par dédoublonnage ou concurrence d’appariement**. Le code ne perd donc plus ces propositions au raccordement à V88 ; le filtre peut néanmoins les rejeter. Miserere concentre 23 pertes pour 2 gains.

Les **142 notes manquées** du candidat se décomposent en **20 sans candidat à ±80 ms**, **106 dont tous les candidats voisins sont sous 0,5** et **16 liées au dédoublonnage ou à l’appariement**. Aucun seuil n’a été retouché après ce constat.

Le dédoublonnage retenu est de 100 ms à hauteur identique, avec priorité à la probabilité apprise. **Aucune vraie paire de réattaques de même hauteur, sur le même instrument, à moins de 100 ms n’est présente dans ce test.** Ce résultat ne valide donc pas les réattaques rapides, et cette règle peut les supprimer.

[Audit individuel](evidence/v273-s598-s605/S605_individual_note_audit.json) · [Probabilités des 55 pertes et vérification complète](evidence/v273-s598-s605/S605_offline_verification.json) · [Diagnostic de toutes les FN](evidence/v273-s598-s605/S605_primary_FN_diagnosis.json)

## Boucles exécutées, sans confondre développement et test

1. **S598 — conserver les horodatages BP.** Neuf unions BP direct + S585, sur les cinq URMP déjà exposés et les deux piano S591. Meilleur F1 URMP : **74,08 %** contre 73,54 %. Une variante atteint exactement 80 % sur les deux piano exposés : ce score de développement ne satisfait pas le test indépendant demandé.
2. **S600 — notes courtes et seuils BP.** Les 880 anciennes attaques BP sont reproduites exactement en hauteur et en temps. Une grille de 36 décodeurs atteint **79,72 %** sur les cinq anciens URMP. Sur les **37 enregistrements de développement, 22 titres, 5 540 notes**, BP défaut donne **73,27 %** et le meilleur décodeur global **78,37 %**. Paramètres retenus : attaque 0,6, activité 0,4, longueur minimale 5 frames ; le décodeur officiel exclut les durées ≤5 frames.
3. **S601 — filtre de candidats appris.** Banque de **33 232 candidats**, **168 caractéristiques audio**, 7 649 candidats positifs et 2 895 candidats ambigus exclus de l’apprentissage. Un ExtraTrees de 160 arbres est évalué en cinq folds groupés par titre musical : aucun arrangement du titre évalué n’entre dans son apprentissage. Les 90 combinaisons de seuil/distance/décalage sont toutes conservées. Le meilleur choix hors fold atteint **83,30 %**, précision 91,31 %, rappel 76,59 %. Il reste un résultat de développement puisque la sélection utilise ces scores.
4. **S602–S605 — première validation réservée.** Apprentissage final unique sur les 37 sources, publication du poids et du contrat, inférence sur les cinq WAV réservés, gel, puis ouverture de leurs annotations dans un second job. **84,91 % au premier test**, sans nouveau réglage.

Les données de développement ne comprennent aucun des cinq titres K515, Miserere, Elise, Nocturne et Sonata, quelle que soit leur instrumentation. Les deux autres arrangements de Nocturne et Miserere sont mis à l’écart. La séparation porte sur les titres de ce corpus, pas sur des compositeurs tous inédits ni sur l’ensemble des corpus de préentraînement de Basic Pitch.

Le filtre utilise les probabilités de notes, d’attaques et de contours de **Basic Pitch 0.4.0 figé**, les voisinages temporels et fréquentiels, les durées candidates et les propositions des décodeurs. Aucune identité de fichier, de titre ou d’instrument annoncé ne fait partie des 168 entrées. Les candidats conservent l’horloge native de BP. **S601 est une nouvelle chaîne expérimentale ; ce n’est pas un remplacement de la tête S585 dans S80.**

## Traçabilité et reproduction

- Préenregistrement : commit `23338df09cb4f1ed06e4be7d80bc8f3274154ed0`, avant ouverture du nouveau corpus.
- Implémentation et premières boucles : `af5f65da3671efc19790f4d85a65d8c45c5507c4`.
- Candidat figé avant test : `b4693934d41511040f773fc6d97f2c5f11d96e01`.
- CI : **38685400020**, première et unique tentative, deux jobs réussis. Prédictions figées à **05:12:19 UTC**, artefact créé à **05:12:22 UTC**, lancement de la commande de score ouvrant les annotations à **05:12:42 UTC**, résultat à **05:12:47 UTC**.
- Les deux ZIP téléchargés correspondent exactement aux SHA256 GitHub. Le rejeu local reproduit l’intégralité du rapport ; **45 contrôles indépendants d’appariement maximal** confirment les comptes, pour 3 politiques × 5 sources × 3 tolérances.
- Poids publiés : `model/experimental/S601_URMP_candidate_gate_xz.joblib`, SHA256 `f925fa0487abe299b7ef1978d2163bfcfbea8fade8e43a105e22bbf7636d6cfb`. Compression sans perte vérifiée sur les 160 arbres, sans nouvel apprentissage.
- Prédictions figées : SHA256 `4393b83f01fb7f2cf4a173b5b406e2f8b9f4a524156e514b90fc27f54bf227e5`.

Les données préparées proviennent du miroir URMP déjà épinglé en S592. Audio mono PCM16 à 16 kHz, rééchantillonné à 44,1 kHz pour la chaîne ; ce rééchantillonnage ne recrée pas la bande perdue. La diversité des instruments de développement ne constitue pas une certification instrument par instrument. Le nom préparé du violoncelle dans K515 est conservé sans prétendre vérifier l’instrumentation de la distribution originale.

Le workflow [S602–S605](../.github/workflows/v273-s602-s605-reserved-candidate-gate.yml) donne les dépendances exactes. Pour reproduire les étapes de développement :

```bash
PYTHONPATH=.:src python -m scripts.s600_bp_native_development extract
PYTHONPATH=.:src python -m scripts.s600_bp_native_development labels
PYTHONPATH=.:src python -m scripts.s600_bp_native_development sweep --report model/S600/decoder_sweep_train37.json
PYTHONPATH=.:src python -m scripts.s601_bp_candidate_gate build
PYTHONPATH=.:src python -m scripts.s601_bp_candidate_gate cv
```

Ces commandes refusent de réécrire les résultats immuables. Après restauration des artefacts S604/S605, la vérification hors réseau ne refait aucune inférence ni apprentissage :

```bash
PYTHONPATH=.:src python -m scripts.audit_s605_reserved_result --out verification.json
```

[Scores complets](evidence/v273-s598-s605/S605_first_reserved_score.json) · [Prédictions](evidence/v273-s598-s605/S604_frozen_predictions.json) · [Vérité après gel](evidence/v273-s598-s605/S605_truth_after_freeze.json) · [Provenance CI](evidence/v273-s598-s605/S605_CI_provenance.json) · [Toutes les 36 variantes BP](evidence/v273-s598-s605/S600_all_36_trials_train37.json.gz) · [Toutes les 90 variantes du filtre](evidence/v273-s598-s605/S601_all_90_decoders_grouped_cv.json.gz) · [Audit des gains/pertes hors fold](evidence/v273-s598-s605/S601_OOF_paired_audit.json.gz)

## Portée et suite

**Conserver S601 comme candidat hors ligne ayant franchi 80 % sur cette validation réservée.** Les cinq œuvres sont maintenant exposées : toute optimisation future à partir de leurs erreurs devra être évaluée sur une autre cohorte.

Le résultat ne vaut pas 80 % de rappel, de notes MIDI complètes ou d’accords exacts. Le filtre est non causal et utilise du contexte futur ; NoteOff, longues tenues, réattaques rapides, microphone et latence bout en bout restent à valider. Les performances piano historiques ne sont pas remplacées par ce score URMP.

**Aucune modification du Live historique, d’Exact-K ou de S80 ; aucune promotion de freeze_local_combo.** Leurs fichiers exécutables et leurs poids n’ont pas changé. Leur rejeu intégral n’a pas été effectué. Les prochaines validations doivent traiter les 55 pertes, le rappel de Miserere et les réattaques, puis une intégration causale séparée.
