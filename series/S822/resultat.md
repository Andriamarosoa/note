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
# S822 — Résultat comparaison S810-style à cadence MIDI256

## Exécution

Workflow GitHub Actions effectué et terminé avec succès sur les 80 pistes signées. Voir le run final et le détail ci-dessous.

## Métriques

| Mesure | H0 FIT80 MIDI256 | S822 RAW23 MIDI256 | S822 AGG61 MIDI256 | CAL sélectionné |
|---|---:|---:|---:|---:|
| Active Exact-K | 57,019538 % | 54,454695 % | 54,385733 % | 54,110685 % |

Comparer 249411 trames GT K>0 pour chaque colonne : interdit de comparer directement le 76,5656 % S810 à 1024 et 62310 frames à S822.

## Six caractéristiques officielles protégées — S708 CAL32

| Caractéristique | S708 officiel CAL32 | S822 SHADOW MIDI256 |
|---|---:|---|
| Onset F1 | 74,2639 % | non modifié, non mesuré sur un nouveau MIDI |
| Exact-K | 62,0455 % | non modifié, non mesuré sur un nouveau MIDI |
| Active Exact-K | 56,2762 % | tête expérimentale seule, FIT80 uniquement |
| Offset accuracy | 77,3292 % | non modifié, non mesuré sur un nouveau MIDI |
| Full Note F1 | 57,1319 % | non modifié, non mesuré sur un nouveau MIDI |
| Frame Pitch F1 | 76,4626 % | non modifié, non mesuré sur un nouveau MIDI |

## Comparaison à S708

S708 officiel CAL32 Active Exact-K 56,2762 %, autres mesures de référence intactes. Aucune unification MIDI, S708 et LIVE inchangés.

## Audits

| Domaine | Statut | Critère |
|---|---|---|
| Données | PASS | 80 pistes PCM SHA et cinq plis 48/16/16 authentifiés |
| Acoustique | PASS | Égalité numérique avec la formule S810, sur 23 canaux S813 différents, sans futur |
| Architecture | PASS | Deux classifieurs indépendants, uniquement K1–K6 Active Exact-K SHADOW |
| Apprentissage | PASS | StandardScaler entraîné sur TRAIN ; inner-CAL distinct, OUTER test jamais utilisé en sélection |
| Métriques | PASS | H0 142213/249411 reproduit; variantes S822 recomptées, sous/sur comptages cohérents |
| LIVE | UNKNOWN | Aucun hold/futur logiciel ; aucune mesure physique de latence ni intégration au LIVE |

## Verdict

Scores réels vérifiés et candidat régressif : S822 non promu. Des gains sur Active Exact-K SHADOW, même s'ils avaient eu lieu, n'auraient pas prouvé l'amélioration du MIDI final.

## Prochaine action

Examiner les vrais canaux S795 et la capacité de généralisation à composition inconnue; tester un futur candidat par protocole préenregistré sur source indépendante. Ne pas optimiser sur les scores FIT80 déjà explorés.


## Clôture scientifique — mesure réelle du 21 octobre 2026

**Run GitHub Actions scientifique [#39566085159](https://github.com/Andriamarosoa/note/actions/runs/39566085159) — SUCCESS, commit `c503ec99cc3dd92a81dd2172504bbdf59d6e59a4`.** Les deux jobs (tests et fit80 réel) ont réussi. Modèles signés et détails par piste dans l'[artefact #12329917731](https://github.com/Andriamarosoa/note/actions/runs/39566085159/artifacts/12329917731). 80 WAV originaux 44,1 kHz SHA vérifiés, 306536 trames d'entrée/decision/score à 256 samples, cinq 48 TRAIN / 16 inner-CAL / 16 outer-TEST, **249411 trames GT actives comparables**. Six scores S708 OFFICIELS et LIVE INCHANGÉS.

**Résultats Active Exact-K FIT80 (aucun nouveau flux MIDI):** H0 142213/249411 = **57,019538 %** ; S815 comparateur historique 141946/249411 = **56,912486 %** ; S822 RAW23 135816/249411 = **54,454695 %**, under **77833**, over **35762** ; S822 AGG61 135644/249411 = **54,385733 %**, under **76568**, over **37199** ; politique S822 sélectionnée sur inner-CAL 134958/249411 = **54,110685 %**, under **78259**, over **36194**. La différence AGG61−RAW23 est **−172 bonnes décisions / −0,068962 point**. Le meilleur de ces trois protocoles S822 est **en dessous** de H0 de 6397 décisions correctes, soit **−2,564843 points**. Aucune promotion.

**Ablation par composition (S822 RAW23 / AGG61, bonnes décisions)** : BN1 **28264 / 28144** (calibration choisit AGG61), Funk2 **28056 / 28295** (CAL choisit RAW23), Jazz1 **22626 / 22499** (CAL choisit AGG61), Rock3 **27957 / 28404** (CAL choisit RAW23), SS1 **28913 / 28302** (CAL choisit AGG61). **Dans chacune des cinq compositions**, la variante sélectionnée par inner-CAL est précisément la moins performante des deux au test outer. Cela indique un désaccord calibration/transfert entre compositions, non une permission d'utiliser outer pour sélectionner le meilleur. Les moyennes exactes et les matrices par vrai K sont dans l'artefact ; aucune politique ne doit être rechoisie sur ces résultats a posteriori.

**Interprétation rigoureuse :** L'identité de la formule de `count_features` S810 à 61 variables a été vérifiée en CI, mais les **23 canaux d'entrée S822 viennent de la DSP brute S813 (222 features)** et **ne sont pas les 23 canaux historiques S795**. De plus, la tête S822 apprend sur les **GT K1..K6 actives uniquement**, donc sa sortie n'est pas un détecteur K0 utilisable telle quelle dans un système MIDI complet. L'écart au S810 76,5656% à cadence 1024 ne permet PAS d'isoler causalement l'effet de la cadence, car le jeu des features et l'objectif d'entraînement diffèrent aussi. Le présent test établit seulement que, **sur ce protocole MIDI256 à entrées natives**, reproduire l'agrégation S810 et la régression logistique n'améliore pas Active Exact-K FIT80.

**Verdict S822 : TERMINÉ — RÉGRESSION — REJETÉ (non promu).** Le meilleur comparateur sur FIT80 reste H0 57,019538%; aucune validation générale indépendante, aucun dépassement de 85%, aucune intégration LIVE, aucun surcoût de hold/lookahead, aucune preuve de latence physique. L'expérience FIT80 reste historiquement explorée : avant d'envisager une future variante, il faut tester les véritables canaux équivalents S795 à MIDI256 si faisable causalement **et** évaluer sur une source externe gelée, sans contaminer les 5 plis utilisés.
