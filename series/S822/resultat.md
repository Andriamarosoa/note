---
serie: S822
statut: en_cours
type: experience
reference: S708
branche: codex/note-s822-s810-style-midi256-ablation
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S822 — Résultat comparaison S810-style à cadence MIDI256

## Exécution

Workflow GitHub Actions à exécuter sur audio original SHA, 5 folds et 80 pistes signées. Aucun gain déclaré par anticipation.

## Métriques

| Mesure | H0 FIT80 MIDI256 | S822 RAW23 MIDI256 | S822 AGG61 MIDI256 | CAL sélectionné |
|---|---:|---:|---:|---:|
| Active Exact-K | 57,019538 % | NON MESURÉ | NON MESURÉ | NON MESURÉ |

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
| Données | UNKNOWN | 80 PCM signés et 5×(48/16/16) |
| Acoustique | UNKNOWN | Agrégation S810 exacte sur 23 features **différentes** mais causales MIDI256 |
| Architecture | UNKNOWN | Deux modèles indépendants K, autres cinq axes intacts |
| Apprentissage | UNKNOWN | Normalisation FIT uniquement et CAL séparée |
| Métriques | UNKNOWN | 249411, 142213 H0, vérité GT K1–K6 |
| LIVE | UNKNOWN | Aucun futur/hold ; latence matérielle non évaluée |

## Verdict

En attente des scores authentifiés. Promotion=false. Des gains sur les sorties Active Exact-K seules ne prouvent aucune amélioration MIDI finale.

## Prochaine action

Mesurer les 5 folds et examiner sous/surcomptages par K; comparer RAW23 à AGG61 sur le même MIDI256, puis rejeter toute architecture régressive. Ne pas optimiser sur les scores de test.
