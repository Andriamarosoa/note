# NOTE Onset PCM256 — H_FP_CAL_GRID_04 : résolution des seuils de veto

**Branch :** `codex/note-fp-track-sampling-fine-cal256`. Hypothèse préenregistrée dans [issue #145](https://github.com/Andriamarosoa/note/issues/145), suivant [règle #140](https://github.com/Andriamarosoa/note/issues/140).

## Pourquoi les deux runs précédents ont échoué

Le [workflow #39900882849](https://github.com/Andriamarosoa/note/actions/runs/39900882849) a passé TOUS les tests, entraîné 5 MLP69 natifs 256 par politique (`track_uniform` et `track_class`) et REJOUÉ les 80 vrais WAV GuitarSet, 306536 décisions hop256 par politique. Les deux échecs sont **SCIENTIFIQUES**, conformément à un garde préenregistré. Les deux reproduisent la référence des quatre têtes **TP8160 / FP4605 / FN3107 / F1 67.9094541%**, sans battre le contrôle précédent FP69 **TP8127 / FP4519 / FN3140 / F1 67.9713963%**.

Les deux avaient **exactement 0.0 pour chacun des cinq seuils CAL choisis** : les dix réseaux furent totalement désactivés au décodage, malgré leur véritable entraînement et des checkpoints sauvegardés, par un choix de seuil sur CAL. L'expérience n'est donc PAS une réfutation complète de la pondération des pistes : cette politique n'a jamais pris de décision de veto.

En CAL uniquement, le seuil non nul le plus petit de l'ancienne grille était **0.05**, mais son taux de TP supprimés dépasse la garde maximum 2.5 % sur toutes les partitions. Par exemple, `track_uniform` en CAL Funk2 supprime 56 TP / 1770 (3.16 %), et `track_class` en CAL Funk2 en supprime 115. L'algorithme choisit `cut=0`, seule décision admissible. Un audit local exploratoire avec les 10 checkpoints déjà figés (sans utiliser GT OUTER pour le choix) suggère que les seuils utiles pourraient être très petits (`0.00001` à `0.01`), **sans promettre de gain et sans substituer ses calculs à la validation CI originale scikit-learn 1.4.2**.

## Une nouvelle expérience isolée, **ZÉRO nouvel entraînement**

Ne changer **QUE la granularité de seuils candidats évalués sur CAL**, à partir des dix anciens réseaux 69D réellement entraînés à 256 et leurs SHA256, dans **deux jobs indépendants**. L'ancienne grille reste incluse, ainsi que les nouveaux seuils `0,0.00001,0.00003,0.00005,0.0001,0.0002,0.0003,0.0005,0.0007,0.001,0.0015,0.002,0.003,0.004,0.005,0.0075,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045`, puis `0.05` et seuils initiaux au-dessus.

Maintenir STRICTEMENT les anciennes contraintes CAL : **≥5 FP évités, ≤2.5 % de TP perdus, F1 CAL proxy amélioré**, comparer même 16 pistes CAL, choisir maximum F1 CAL, puis minimum de TP perdus. **Aucune vérité terrain OUTER** utilisée par `recalibrate_onset_fp_track_sampling256.py`. Les cinq splits par composition et les 80 véritables WAV restent exactement identiques ; la seule sortie modifiée des anciens checkpoints est `calibrated_cut`, métadonnées CAL et rapport. Les poids appris `control69`, les 70D/74D signés et les quatre anciennes têtes sont TOUS copiés et SHA-vérifiés sans réentraînement.

Après CAL : replay intégral des vrais 80 WAV GuitarSet originaux, avec **256 nouveaux échantillons par décision (5.805ms @44.1kHz)**, aucune future frame, aucun délai volontaire ni poids S708 formé1024. Recalculer TP/FP/FN/F1 pour chaque bras et montrer régressions par piste, même si l'expérience échoue. Les cinq autres axes indépendants du projet et le MIDI de production restent inchangés.

## Réfutation écrite AVANT ce replay

Chaque bras `track_uniform` et `track_class` est évalué séparément. L'expérience n'est **scientifiquement réussie** que si le vrai F1 du replay > **67.9713963%** ET FP réels < **4519**, tout en préservant S613/S701/ErrorPrediction69/FNRecovery et les contrôles historiques 70D/74D à valeurs identiques. Si le seuil redevient zéro, ou que le score final n'atteint pas les objectifs, **`H_FP_CAL_GRID_04 REFUTED_OR_INCOMPLETE`** : archiver et auditer le compromis TP/FP; NE PAS promouvoir ni réajuster un seuil en consultant OUTER.

**Transparence scientifique :** FIT80 OUTER a DÉJÀ été utilisé dans d'autres expériences historiques ; « choix sur CAL uniquement » n'efface pas cette réutilisation adaptative au niveau du programme de recherche. Toute promotion officielle exigera un corpus externe de compositions/audio réellement indépendant et une mesure end-to-end p95/p99 (la cadence de 5.805ms n'est pas une telle mesure).

## Fichiers / preuves

- `scripts/recalibrate_onset_fp_track_sampling256.py` : vérifie provenance et SHA d'origine des 10 MLP69; grille préréglée; copie de poids bit-à-bit; vérification des anciennes règles CAL; metadata et audit des 45 seuils.
- `tests/test_onset_fp_fine_cal256.py` : invariants de la grille, poids jamais refittés, ancienne garde CAL, 256.
- `.github/workflows/onset-fp-fine-cal256.yml` : deux bras indépendants avec téléchargements des checkpoints immuables du run #39900882849, tests et replay WAV scientifique à seuil fin, rapport même sur échec.
- `series/ONSET_FP_FINE_CAL256/resume.md` et `resultat.md` : historique, réfutation et scores signés.

**Statut à la création : nouveau score non encore validé, aucune promotion.**
