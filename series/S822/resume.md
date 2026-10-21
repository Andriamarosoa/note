---
serie: S822
statut: termine
type: experience
reference: S708
branche: codex/note-s822-s810-style-midi256-ablation
run_id: 39566085159
commit_sha: c503ec99cc3dd92a81dd2172504bbdf59d6e59a4
promotion: false
---
# S822 — Rejouer l'architecture acoustique et supervisée S810 à 256 échantillons

## Objectif

Tester si le gain S809→S810 SHADOW 1024 (+8,44 points Active Exact-K) vient d'une **tête indépendante entraînée** et de l'**agrégation 61 variables**, et non uniquement de la cadence 1024. Garder **feature/decision/metric = 256 échantillons**, sur les 80 WAV GuitarSet FIT80 original, 249411 trames actives, sans hold/lookahead.

## Hypothèse

En maintenant la même cadence et les mêmes splits d'une part et le même algorithme LogisticRegression de l'autre, la comparaison contrôlée entre les 23 coordonnées acoustiques brutes et leur résumé S810 en 61 variables mesure l'apport de l'agrégation. Le résultat peut être meilleur ou pire. La classe K0 est **exclue de cet entraînement S822** : les modèles sont ajustés uniquement sur les trames où le vrai K est K1..K6. K0 est également exclu du dénominateur Active Exact-K ; cette tête ne constitue donc pas un décodeur K0 complet.

## Modifications

Nouveaux modèles de recherche `src/causal_note/s822_s810_style_teacher.py`, `scripts/s822_retrain_s810_midi256.py`. Deux têtes privées de classification K1–K6, `StandardScaler -> LogisticRegression`, distinctes des autres axes. Les deux ont les **mêmes 23 entrées physiques** : 22 indices S814 documentés + un indice spectral `182` prédéclaré ; condition `native23` = 23 statistiques moyennes; condition `aggregated61` = **exacte règle d'agrégation S810**, transposée à ces 23 features de la nouvelle DSP. Ce ne sont **pas** les 23 posteriors historiques S795 ; les reproduire à l'identique demanderait des ablations supplémentaires et risquerait de réintroduire des dépendances à un autre enseignant. Un modèle S810 n'est donc pas copié tel quel : seul son **schéma** est réentraîné sur la nouvelle acoustique sans supervision à l'inférence.

## Protocole

5 plis avec 48 tracks train, 16 inner-cal d'une autre composition, 16 outer-test d'une troisième composition ; mêmes pistes pour les deux ablations. Echantillonnage K par piste et poids inverses pour préserver prévalences. Apprendre chaque tête séparément, évaluer `native23`, `aggregated61` et une sélection entre elles sur inner-CAL exclusivement ; aucun choix sur le test. Comparer avec H0 FIT80 142213/249411 et S815 141946/249411, puis par K, undercount/overcount et composition. FIT80 est une cohorte historiquement explorée, non une validation externe.

## Audits obligatoires

SHA des 80 WAV et manifests, parité H0 S704 et horloges 256; aucune donnée CAL32/HELD/TEST. Toutes sorties concernent uniquement Active Exact-K SHADOW, pas le MIDI unifié. Source S708 et cinq autres axes inchangés; aucune latence supplémentaire due à une décision différée, mais temps CPU et latence physique restent à mesurer. Tout résultat ne peut être promu sans tests de non-régression des six axes sur source externe et validation matérielle.

## Statut

Expérience réelle exécutée et vérifiée, résultats conservés comme recherche SHADOW non promue. Voir la clôture détaillée ci-dessous.


## Clôture scientifique — mesure réelle du 21 octobre 2026

**Run GitHub Actions scientifique [#39566085159](https://github.com/Andriamarosoa/note/actions/runs/39566085159) — SUCCESS, commit `c503ec99cc3dd92a81dd2172504bbdf59d6e59a4`.** Les deux jobs (tests et fit80 réel) ont réussi. Modèles signés et détails par piste dans l'[artefact #12329917731](https://github.com/Andriamarosoa/note/actions/runs/39566085159/artifacts/12329917731). 80 WAV originaux 44,1 kHz SHA vérifiés, 306536 trames d'entrée/decision/score à 256 samples, cinq 48 TRAIN / 16 inner-CAL / 16 outer-TEST, **249411 trames GT actives comparables**. Six scores S708 OFFICIELS et LIVE INCHANGÉS.

**Résultats Active Exact-K FIT80 (aucun nouveau flux MIDI):** H0 142213/249411 = **57,019538 %** ; S815 comparateur historique 141946/249411 = **56,912486 %** ; S822 RAW23 135816/249411 = **54,454695 %**, under **77833**, over **35762** ; S822 AGG61 135644/249411 = **54,385733 %**, under **76568**, over **37199** ; politique S822 sélectionnée sur inner-CAL 134958/249411 = **54,110685 %**, under **78259**, over **36194**. La différence AGG61−RAW23 est **−172 bonnes décisions / −0,068962 point**. Le meilleur de ces trois protocoles S822 est **en dessous** de H0 de 6397 décisions correctes, soit **−2,564843 points**. Aucune promotion.

**Ablation par composition (S822 RAW23 / AGG61, bonnes décisions)** : BN1 **28264 / 28144** (calibration choisit AGG61), Funk2 **28056 / 28295** (CAL choisit RAW23), Jazz1 **22626 / 22499** (CAL choisit AGG61), Rock3 **27957 / 28404** (CAL choisit RAW23), SS1 **28913 / 28302** (CAL choisit AGG61). **Dans chacune des cinq compositions**, la variante sélectionnée par inner-CAL est précisément la moins performante des deux au test outer. Cela indique un désaccord calibration/transfert entre compositions, non une permission d'utiliser outer pour sélectionner le meilleur. Les moyennes exactes et les matrices par vrai K sont dans l'artefact ; aucune politique ne doit être rechoisie sur ces résultats a posteriori.

**Interprétation rigoureuse :** L'identité de la formule de `count_features` S810 à 61 variables a été vérifiée en CI, mais les **23 canaux d'entrée S822 viennent de la DSP brute S813 (222 features)** et **ne sont pas les 23 canaux historiques S795**. De plus, la tête S822 apprend sur les **GT K1..K6 actives uniquement**, donc sa sortie n'est pas un détecteur K0 utilisable telle quelle dans un système MIDI complet. L'écart au S810 76,5656% à cadence 1024 ne permet PAS d'isoler causalement l'effet de la cadence, car le jeu des features et l'objectif d'entraînement diffèrent aussi. Le présent test établit seulement que, **sur ce protocole MIDI256 à entrées natives**, reproduire l'agrégation S810 et la régression logistique n'améliore pas Active Exact-K FIT80.

**Verdict S822 : TERMINÉ — RÉGRESSION — REJETÉ (non promu).** Le meilleur comparateur sur FIT80 reste H0 57,019538%; aucune validation générale indépendante, aucun dépassement de 85%, aucune intégration LIVE, aucun surcoût de hold/lookahead, aucune preuve de latence physique. L'expérience FIT80 reste historiquement explorée : avant d'envisager une future variante, il faut tester les véritables canaux équivalents S795 à MIDI256 si faisable causalement **et** évaluer sur une source externe gelée, sans contaminer les 5 plis utilisés.
